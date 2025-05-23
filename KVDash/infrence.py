import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time 
import os 

model_id = "/home/dourlin/Develop/Model_repo/Mistral-7B-v0.1"

device = "cpu"
print(f"Using device: {device}")

num_cores_to_use = 4 # Set number of cores to use here
torch.set_num_threads(num_cores_to_use) 

# --- Prepare Input ---
input_filename = "./input.txt"


with open(input_filename, 'r', encoding='utf-8') as f:
    user_prompt = f.read()
system_prompt = ""
prompt = system_prompt + user_prompt
#prompt.append(prompt_read)
print("\nThe input prompt is\n")
print(prompt)
print("==================")



timestamps ={}
timestamps["start"] = time.time()
# --- Load Tokenizer and Model ---
print("Loading tokenizer...")
# The tokenizer converts your text into numbers (tokens) the model understands.
tokenizer = AutoTokenizer.from_pretrained(model_id)
timestamps["tokenizer"] = time.time()


print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # torch_dtype=torch.bfloat16, # Optional: Use bfloat16 for less memory
    device_map="auto" # Automatically use available devices (GPU > CPU)
)
timestamps["model"] = time.time()
print("Model loaded.")



input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
prompt_length = input_ids.shape[-1]


max_new_tokens = 20
# --- Manual Generation Loop with Profiling ---
print("\nStarting generation with profiling...")
generated_ids = input_ids
total_decode_time = 0.0
prefill_time = 0.0
past_key_values = None # To store the KV cache

# Use no_grad for inference - speeds up and saves memory
with torch.no_grad():
    print("Running prefill stage...")
    timestamps["prefill_start"] = time.time() # Stamp the time

    outputs = model(input_ids, past_key_values=None, use_cache=True)
    past_key_values = outputs.past_key_values # Get the initial KV cache
    logits = outputs.logits # Get the predictions

    # Get the very next token (greedy approach)
    next_token_logits = logits[:, -1, :]
    next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
    generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

    timestamps["prefill_end"] = time.time() # Mark end of prefill
    
    # 2. Decode Stage (Loop)
    print("Running decode stage (token by token)...")
    timestamps["decode_start"] = time.time() # Stamp the time

    for step in range(max_new_tokens - 1): # -1 because we generated one token already
        start_decode_step = time.time()

        # Run the model with only the *new* token and the *past* KV cache
        outputs = model(next_token_id, past_key_values=past_key_values, use_cache=True)
        logits = outputs.logits
        past_key_values = outputs.past_key_values # Update KV cache

        end_decode_step = time.time()
        total_decode_time += (end_decode_step - start_decode_step)

        # Get the next token
        next_token_logits = logits[:, -1, :] # Logits for the *new* token
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

        # Check for End-Of-Sentence token
        if next_token_id.item() == tokenizer.eos_token_id:
            print(f"EOS token generated at step {step + 1}. Stopping.")
            break

        # Append the new token
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

    timestamps["decode_end"] = time.time() # Stamp the time

print(f"Decode stage (total) took: {total_decode_time:.4f} seconds")
num_decoded_tokens = generated_ids.shape[-1] - prompt_length
avg_decode_time = total_decode_time / num_decoded_tokens
print(f"Generated {num_decoded_tokens} tokens at an average of {avg_decode_time:.4f} seconds/token.")

generated_text = tokenizer.decode(generated_ids[0, prompt_length:], skip_special_tokens=True)

print("\n--- Output ---")
print(prompt + generated_text)

print("\n--- Timing Analysis ---")
print(f"Prefill start: {timestamps['prefill_start']:.4f}")
print(f"Prefill end: {timestamps['prefill_end']:.4f}")
print(f"Decode start: {timestamps['decode_start']:.4f}")
print(f"Decode end: {timestamps['decode_end']:.4f}")

print("\n--- Performance Metrics ---")

# TTFT - Time To First Token (prefill time)
ttft = timestamps["prefill_end"] - timestamps["prefill_start"]
print(f"TTFT (Time To First Token): {ttft:.4f} seconds")

# TTOP - Time To Output Per token (average decode time per token)
ttop = total_decode_time / num_decoded_tokens if num_decoded_tokens > 0 else 0
print(f"TTOP (Time To Output Per token): {ttop:.4f} seconds/token")

# Additional metrics
prefill_time = ttft
decode_time_total = timestamps["decode_end"] - timestamps["decode_start"]
print(f"Prefill time: {prefill_time:.4f} seconds")
print(f"Total decode time: {decode_time_total:.4f} seconds")
print(f"Generated {num_decoded_tokens} tokens")

print("--------------")