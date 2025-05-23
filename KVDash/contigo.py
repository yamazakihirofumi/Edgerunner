#!/usr/bin/env python3
import torch
import time
import pickle
import zmq
import threading
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from concurrent.futures import ThreadPoolExecutor
import sys

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

class GPUPrefillServer:
    """GPU server that pre-loads model and handles prefill requests"""
    
    def __init__(self, model_id, port=5555):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.port = port
        
        print(f"[GPU SERVER] Initializing on {self.device}...")
        self.load_model()
        self.setup_network()
        
    def load_model(self):
        """Pre-load model and tokenizer to GPU memory"""
        print(f"[GPU SERVER] Loading model on {self.device} for prefill...")
        start_time = time.time()
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device
        )
        
        load_time = time.time() - start_time
        print(f"[GPU SERVER] Model loaded in {load_time:.2f} seconds")
        
    def setup_network(self):
        """Setup ZeroMQ socket"""
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{self.port}")
        print(f"[GPU SERVER] Listening on port {self.port}")
        
    def serialize_kv_cache(self, past_key_values):
        """Efficiently serialize KV cache for network transfer"""
        if isinstance(past_key_values, DynamicCache):
            serialized_kv = []
            for layer_idx in range(len(past_key_values.key_cache)):
                key_cpu = past_key_values.key_cache[layer_idx].cpu().numpy()
                value_cpu = past_key_values.value_cache[layer_idx].cpu().numpy()
                serialized_kv.append((key_cpu, value_cpu))
        else:
            serialized_kv = []
            for layer_kv in past_key_values:
                key_cpu = layer_kv[0].cpu().numpy()
                value_cpu = layer_kv[1].cpu().numpy()
                serialized_kv.append((key_cpu, value_cpu))
        return pickle.dumps(serialized_kv)
    
    def process_prefill(self, prompt):
        """Process prefill request and return serialized KV cache"""
        timestamps = {}
        timestamps["prefill_start"] = time.time()
        
        # Tokenize and run prefill
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        
        with torch.no_grad():
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
        
        prefill_time = timestamps["prefill_end"] - timestamps["prefill_start"]
        serialize_time = timestamps["serialize_end"] - timestamps["serialize_start"]
        
        print(f"[GPU SERVER] Prefill: {prefill_time:.4f}s, Serialize: {serialize_time:.4f}s")
        
        return {
            "next_token_id": next_token_id,
            "kv_cache": serialized_kv,
            "prompt_length": input_ids.shape[-1],
            "timestamps": timestamps
        }
    
    def run_server(self):
        """Main server loop - waits for requests and processes them"""
        print(f"[GPU SERVER] Ready to accept requests")
        
        while True:
            try:
                # Wait for request
                message = self.socket.recv_json()
                prompt = message["prompt"]
                
                print(f"[GPU SERVER] Received request with {len(prompt)} chars")
                
                # Process prefill
                result = self.process_prefill(prompt)
                
                # Send response
                self.socket.send_pyobj(result)
                print(f"[GPU SERVER] Response sent")
                
            except KeyboardInterrupt:
                print(f"[GPU SERVER] Shutting down...")
                break
            except Exception as e:
                print(f"[GPU SERVER] Error: {e}")

class CPUDecodeClient:
    """CPU client that loads model in parallel and handles decode"""
    
    def __init__(self, model_id, gpu_server_address="tcp://localhost:5555"):
        self.model_id = model_id
        self.device = "cpu"
        self.gpu_server_address = gpu_server_address
        
        # Shared state between threads
        self.model_loaded = False
        self.kv_cache_ready = False
        self.prefill_result = None
        self.model = None
        self.tokenizer = None
        
        print(f"[CPU CLIENT] Initializing...")
        
    def load_model_async(self):
        """Load model on CPU in background thread"""
        print(f"[CPU CLIENT] Starting model loading on {self.device}...")
        start_time = time.time()
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,
            device_map=self.device
        )
        
        load_time = time.time() - start_time
        print(f"[CPU CLIENT] Model loaded in {load_time:.2f} seconds")
        self.model_loaded = True
        
    def query_gpu_server_async(self, prompt):
        """Send query to GPU server in background thread"""
        print(f"[CPU CLIENT] Sending query to GPU server...")
        start_time = time.time()
        
        # Setup ZeroMQ client
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect(self.gpu_server_address)
        
        # Send request
        request = {"prompt": prompt}
        socket.send_json(request)
        
        # Receive response
        self.prefill_result = socket.recv_pyobj()
        
        query_time = time.time() - start_time
        print(f"[CPU CLIENT] GPU query completed in {query_time:.2f} seconds")
        self.kv_cache_ready = True
        
        socket.close()
        context.term()
        
    def deserialize_kv_cache(self, serialized_kv):
        """Deserialize KV cache from network transfer"""
        kv_data = pickle.loads(serialized_kv)
        
        past_key_values = DynamicCache()
        for layer_idx, (key_np, value_np) in enumerate(kv_data):
            key_tensor = torch.from_numpy(key_np).to(self.device)
            value_tensor = torch.from_numpy(value_np).to(self.device)
            past_key_values.update(key_tensor, value_tensor, layer_idx)
            
        return past_key_values
    
    def wait_for_ready(self):
        """Wait for both model loading and KV cache to be ready"""
        print(f"[CPU CLIENT] Waiting for model loading and KV cache...")
        
        while not (self.model_loaded and self.kv_cache_ready):
            if self.model_loaded and not self.kv_cache_ready:
                print(f"[CPU CLIENT] Model loaded, waiting for KV cache...")
            elif not self.model_loaded and self.kv_cache_ready:
                print(f"[CPU CLIENT] KV cache ready, waiting for model...")
            time.sleep(0.1)
            
        print(f"[CPU CLIENT] Both model and KV cache ready - starting decode!")
        
    def run_decode(self, max_new_tokens=20):
        """Run decode stage once everything is ready"""
        timestamps = {}
        timestamps["decode_start"] = time.time()
        
        # Deserialize KV cache
        past_key_values = self.deserialize_kv_cache(self.prefill_result["kv_cache"])
        next_token_id = self.prefill_result["next_token_id"]
        prompt_length = self.prefill_result["prompt_length"]
        
        # Decode stage
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
                    print(f"[CPU CLIENT] EOS token at step {step + 1}")
                    break
                    
                generated_tokens.append(next_token_id)
                next_token_tensor = torch.tensor([[next_token_id]], device=self.device)
        
        timestamps["decode_end"] = time.time()
        
        # Generate final text
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return {
            "generated_tokens": generated_tokens,
            "generated_text": generated_text,
            "total_decode_time": total_decode_time,
            "num_tokens": len(generated_tokens),
            "timestamps": timestamps
        }
    
    def full_generation(self, prompt, max_new_tokens=20):
        """Complete generation with parallel loading and querying"""
        overall_start = time.time()
        
        # Start both tasks in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks
            model_future = executor.submit(self.load_model_async)
            query_future = executor.submit(self.query_gpu_server_async, prompt)
            
            # Wait for both to complete
            model_future.result()
            query_future.result()
        
        parallel_time = time.time() - overall_start
        print(f"[CPU CLIENT] Parallel phase completed in {parallel_time:.2f} seconds")
        
        # Wait for ready state
        self.wait_for_ready()
        
        # Run decode
        decode_result = self.run_decode(max_new_tokens)
        
        overall_time = time.time() - overall_start
        
        # Calculate comprehensive metrics
        metrics = self.calculate_metrics(decode_result, parallel_time, overall_time)
        
        return {
            "generated_text": prompt + decode_result["generated_text"],
            "metrics": metrics,
            "decode_result": decode_result
        }
    
    def calculate_metrics(self, decode_result, parallel_time, overall_time):
        """Calculate comprehensive performance metrics"""
        prefill_timestamps = self.prefill_result["timestamps"]
        
        ttft_gpu = prefill_timestamps["prefill_end"] - prefill_timestamps["prefill_start"]
        serialization_time = prefill_timestamps["serialize_end"] - prefill_timestamps["serialize_start"]
        
        decode_wall_time = decode_result["timestamps"]["decode_end"] - decode_result["timestamps"]["decode_start"]
        ttop = decode_result["total_decode_time"] / decode_result["num_tokens"] if decode_result["num_tokens"] > 0 else 0
        
        return {
            "ttft_gpu": ttft_gpu,
            "serialization_time": serialization_time,
            "parallel_loading_time": parallel_time,
            "ttop_cpu": ttop,
            "decode_wall_time": decode_wall_time,
            "overall_e2e_time": overall_time,
            "num_tokens": decode_result["num_tokens"],
            "tokens_per_second": decode_result["num_tokens"] / decode_wall_time if decode_wall_time > 0 else 0
        }

def run_server(model_id):
    """Run GPU prefill server"""
    server = GPUPrefillServer(model_id)
    try:
        server.run_server()
    except KeyboardInterrupt:
        print("\n[SERVER] Shutting down gracefully...")

def run_client(model_id, gpu_server_address):
    """Run CPU decode client"""
    client = CPUDecodeClient(model_id, gpu_server_address)
    
    # Get prompt from file
    prompt = get_prompt()
    
    # Run full generation with parallel loading
    result = client.full_generation(prompt, max_new_tokens=20)
    
    print("\n" + "="*50)
    print("GENERATION COMPLETE")
    print("="*50)
    print(result["generated_text"])
    
    print("\n" + "="*50)
    print("PERFORMANCE METRICS")
    print("="*50)
    metrics = result["metrics"]
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

def main():
    parser = argparse.ArgumentParser(description="Distributed GPU Prefill + CPU Decode System")
    parser.add_argument("-s", "--server", action="store_true", help="Run as GPU prefill server")
    parser.add_argument("-c", "--client", action="store_true", help="Run as CPU decode client")
    parser.add_argument("--model", default="/home/dourlin/Develop/Model_repo/Mistral-7B-v0.1", help="Model path")
    parser.add_argument("--gpu-server", default="tcp://localhost:5555", help="GPU server address")
    
    args = parser.parse_args()
    
    if args.server:
        print("Starting GPU Prefill Server...")
        run_server(args.model)
    elif args.client:
        print("Starting CPU Decode Client...")
        run_client(args.model, args.gpu_server)
    else:
        print("Please specify -s for server or -c for client")
        sys.exit(1)

if __name__ == "__main__":
    main()