import torch
import time
import pickle
import zmq  # For network communication
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
import threading
from queue import Queue

class GPUPrefillServer:
    """GPU server handles prefill requests and returns KV cache"""
    
    def __init__(self, model_id, port=5555):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("####The device is "+ str(self.device))
        self.port = port
        
        # Load model once on GPU
        print(f"Loading model on {self.device} for prefill...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device
        )
        
        # Setup ZeroMQ
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        
    def serialize_kv_cache(self, past_key_values):
        """Efficiently serialize KV cache for network transfer"""
        if isinstance(past_key_values, DynamicCache):
            # Extract key and value tensors from DynamicCache
            serialized_kv = []
            for layer_idx in range(len(past_key_values.key_cache)):
                key_cpu = past_key_values.key_cache[layer_idx].cpu().numpy()
                value_cpu = past_key_values.value_cache[layer_idx].cpu().numpy()
                serialized_kv.append((key_cpu, value_cpu))
        else:
            # Handle tuple format (older transformers versions)
            serialized_kv = []
            for layer_kv in past_key_values:
                key_cpu = layer_kv[0].cpu().numpy()
                value_cpu = layer_kv[1].cpu().numpy()
                serialized_kv.append((key_cpu, value_cpu))
        return pickle.dumps(serialized_kv)
    
    def prefill_request(self, prompt):
        """Process prefill request and return serialized KV cache"""
        timestamps = {}
        timestamps["prefill_start"] = time.time()
        
        # Tokenize and run prefill
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        
        with torch.no_grad():
            # Create empty cache for initial prefill
            past_key_values = DynamicCache()
            outputs = self.model(input_ids, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            logits = outputs.logits
            
            # Get first token
            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).item()
        
        timestamps["prefill_end"] = time.time()
        timestamps["serialize_start"] = time.time()
        
        # Serialize KV cache for transfer
        serialized_kv = self.serialize_kv_cache(past_key_values)
        
        timestamps["serialize_end"] = time.time()
        
        result = {
            "next_token_id": next_token_id,
            "kv_cache": serialized_kv,
            "prompt_length": input_ids.shape[-1],
            "timestamps": timestamps
        }
        
        return result
    
    def run_server(self):
        """Main server loop"""
        print(f"GPU Prefill Server listening on port {self.port}")
        
        while True:
            # Wait for request
            message = self.socket.recv_json()
            prompt = message["prompt"]
            
            # Process prefill
            result = self.prefill_request(prompt)
            
            # Send response
            self.socket.send_pyobj(result)

class CPUDecodeServer:
    """CPU server handles decode requests using received KV cache"""
    
    def __init__(self, model_id, gpu_server_address="tcp://localhost:5555"):
        self.model_id = model_id
        self.device = "cpu"
        self.gpu_server_address = gpu_server_address
        
        # Load model on CPU
        print(f"Loading model on {self.device} for decode...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map=self.device
        )
        
        # Setup ZeroMQ client
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(gpu_server_address)
        
    def deserialize_kv_cache(self, serialized_kv):
        """Deserialize KV cache from network transfer"""
        kv_data = pickle.loads(serialized_kv)
        
        # Create a new DynamicCache and populate it
        past_key_values = DynamicCache()
        
        for layer_idx, (key_np, value_np) in enumerate(kv_data):
            key_tensor = torch.from_numpy(key_np).to(self.device)
            value_tensor = torch.from_numpy(value_np).to(self.device)
            
            # Update the cache with the key-value tensors
            past_key_values.update(key_tensor, value_tensor, layer_idx)
            
        return past_key_values
    
    def full_generation(self, prompt, max_new_tokens=20):
        """Complete generation: prefill on GPU + decode on CPU"""
        timestamps = {}
        timestamps["request_start"] = time.time()
        
        # 1. Send prefill request to GPU server
        request = {"prompt": prompt}
        self.socket.send_json(request)
        
        # 2. Receive prefill result
        prefill_result = self.socket.recv_pyobj()
        timestamps["prefill_received"] = time.time()
        
        # 3. Deserialize KV cache
        timestamps["deserialize_start"] = time.time()
        past_key_values = self.deserialize_kv_cache(prefill_result["kv_cache"])
        next_token_id = prefill_result["next_token_id"]
        prompt_length = prefill_result["prompt_length"]
        timestamps["deserialize_end"] = time.time()
        
        # 4. Decode stage on CPU
        timestamps["decode_start"] = time.time()
        generated_tokens = [next_token_id]
        next_token_tensor = torch.tensor([[next_token_id]], device=self.device)
        
        total_decode_time = 0.0
        
        with torch.no_grad():
            for step in range(max_new_tokens - 1):
                step_start = time.time()
                
                outputs = self.model(next_token_tensor, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
                logits = outputs.logits
                
                next_token_logits = logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1).item()
                
                step_end = time.time()
                total_decode_time += (step_end - step_start)
                
                if next_token_id == self.tokenizer.eos_token_id:
                    break
                    
                generated_tokens.append(next_token_id)
                next_token_tensor = torch.tensor([[next_token_id]], device=self.device)
        
        timestamps["decode_end"] = time.time()
        
        # Generate final text
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Calculate metrics
        metrics = self.calculate_metrics(timestamps, prefill_result["timestamps"], 
                                       len(generated_tokens), total_decode_time)
        
        return {
            "generated_text": prompt + generated_text,
            "metrics": metrics,
            "timestamps": timestamps
        }
    
    def calculate_metrics(self, decode_timestamps, prefill_timestamps, num_tokens, total_decode_time):
        """Calculate end-to-end performance metrics"""
        
        # Prefill metrics (from GPU server)
        ttft_gpu = prefill_timestamps["prefill_end"] - prefill_timestamps["prefill_start"]
        serialization_time = prefill_timestamps["serialize_end"] - prefill_timestamps["serialize_start"]
        
        # Network + deserialization
        network_time = decode_timestamps["prefill_received"] - decode_timestamps["request_start"] - ttft_gpu
        deserialization_time = decode_timestamps["deserialize_end"] - decode_timestamps["deserialize_start"]
        
        # Decode metrics
        decode_wall_time = decode_timestamps["decode_end"] - decode_timestamps["decode_start"]
        ttop = total_decode_time / num_tokens if num_tokens > 0 else 0
        
        # End-to-end
        e2e_time = decode_timestamps["decode_end"] - decode_timestamps["request_start"]
        
        return {
            "ttft_gpu": ttft_gpu,
            "serialization_time": serialization_time,
            "network_transfer_time": network_time,
            "deserialization_time": deserialization_time,
            "ttop_cpu": ttop,
            "decode_wall_time": decode_wall_time,
            "e2e_time": e2e_time,
            "num_tokens": num_tokens,
            "transfer_overhead_pct": ((network_time + serialization_time + deserialization_time) / e2e_time) * 100
        }

def get_prompt() -> str:
    input_filename = "./input.txt"
    with open(input_filename, 'r', encoding='utf-8') as f:
        user_prompt = f.read()
    system_prompt = ""
    prompt = system_prompt + user_prompt
    print("\nThe input prompt is\n")
    print(prompt)
    print("==================")
    return prompt

# Example usage
def run_experiment():
    model_id = "/home/dourlin/Develop/Model_repo/Mistral-7B-v0.1"
    
    # Start GPU server in background (in production, this would be separate process/machine)
    gpu_server = GPUPrefillServer(model_id)
    server_thread = threading.Thread(target=gpu_server.run_server, daemon=True)
    server_thread.start()
    
    time.sleep(2)  # Give server time to start
    
    # Create CPU client
    cpu_client = CPUDecodeServer(model_id)
    
    # Get prompt from file
    prompt = get_prompt()
    result = cpu_client.full_generation(prompt, max_new_tokens=20)
    
    print("\n--- Generated Text ---")
    print(result["generated_text"])
    
    print("\n--- Performance Metrics ---")
    metrics = result["metrics"]
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

if __name__ == "__main__":
    run_experiment()