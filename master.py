import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
#from hellaswag import render_example, iterate_examples
#----
import socket
import pickle
import sys




# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x




@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        return logits
        ''' 
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
        '''

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        
        #use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer


# ----------------------------------------------------------------------------------------
num_return_sequences = 1 #5 
max_length = 30 

model = GPT.from_pretrained('gpt2')
model.eval() # put model to evaluation model so param not change
# we just use CPU to infrance now
model.to('cuda')


# ----------------------------------------------------------------------------------------

import tiktoken
import numpy as np



#Split line for me to play aroud master workers set up
# ----------------------------------------------------------------------------------------
num_return_sequences = 5 
max_length = 30 

# Socket communication functions
def send_tensor(sock, tensor):
    # Serialize tensor to bytes
    data = pickle.dumps(tensor)
    # Send data size first
    size = len(data)
    sock.sendall(size.to_bytes(8, byteorder='big'))
    # Send actual data
    sock.sendall(data)

def receive_tensor(sock):
    # Get data size first
    size_bytes = sock.recv(8)
    size = int.from_bytes(size_bytes, byteorder='big')
    # Receive data in chunks
    data = b''
    while len(data) < size:
        chunk = sock.recv(min(size - len(data), 4096))
        if not chunk:
            raise RuntimeError("Socket connection broken")
        data += chunk
    # Deserialize back to tensor
    return pickle.loads(data)

# Get execution mode from command line
if len(sys.argv) < 2:
    print("Usage: python script.py [master|worker]")
    sys.exit(1)

mode = sys.argv[1]

# Hard-coded network settings
MASTER_IP = '192.168.1.80'
WORKER_IP = '192.168.1.70'
PORT = 16543

# Common initialization code
import tiktoken
import numpy as np

enc = tiktoken.get_encoding('gpt2') # Use gpt2 encoder
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)  # After get tokens, create a torch tensor

# Replicate these tokens as we plan to run it 5 times
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (5,8) Find 9th token

if mode == 'master':
    print("Running in master mode...")
    
    # Load model
    model = GPT.from_pretrained('gpt2')
    model.eval()  # put model to evaluation mode so param not change
    model.to('cuda')
    
    # Move tokens to GPU
    x = tokens.to('cuda')
    
    # Generate first half of tokens
    first_half = max_length // 2
    
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # First generation phase on master
    while x.size(1) < tokens.size(1) + first_half:  # Generate first_half new tokens
        with torch.no_grad():
            logits = model(x)  # Fixed: unpack tuple
            #logits, _ = model(x)  # Fixed: unpack tuple
            logits = logits[:, -1, :]  # (B : vocab_size), take the logits at the last position
            
            probs = F.softmax(logits, dim=-1)  # Last logit pass through softmax get probability
            
            # Do top-k sampling of 50 (hugging face pipeline default)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            
            ix = torch.multinomial(topk_probs, 1)  # (B,1)
            
            xcol = torch.gather(topk_indices, -1, ix)  # Gather the corresponding indices
            x = torch.cat((x, xcol), dim=1)  # Append to the sequence
    
    print(f"Master generated {x.size(1) - tokens.size(1)} new tokens")
    
    # Connect to worker and send tokens
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"Connecting to worker at {WORKER_IP}:{PORT}...")
    sock.connect((WORKER_IP, PORT))
    
    # Send current tokens to worker
    print("Sending tokens to worker...")
    send_tensor(sock, x.cpu())  # Send CPU tensor to avoid GPU memory issues
    
    # Receive completed tokens from worker
    print("Waiting for worker to complete generation...")
    final_x = receive_tensor(sock)
    sock.close()
    
    # Decode and print results
    for i in range(num_return_sequences):
        tokens_list = final_x[i, :].tolist()
        decoded = enc.decode(tokens_list)
        print(">", decoded)

elif mode == 'worker':
    print("Running in worker mode...")
    
    # Load model
    model = GPT.from_pretrained('gpt2')
    model.eval()  # put model to evaluation mode so param not change
    model.to('cuda')
    
    # Set up server socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((WORKER_IP, PORT))
    server_socket.listen(1)
    
    print(f"Worker listening on {WORKER_IP}:{PORT}")
    conn, addr = server_socket.accept()
    print(f"Connected to master at {addr}")
    
    # Receive tokens from master
    print("Receiving tokens from master...")
    x = receive_tensor(conn)
    x = x.to('cuda')
    
    initial_len = x.size(1)
    print(f"Received sequence with {initial_len} tokens, continuing generation...")
    
    # Continue generating up to max_length
    torch.manual_seed(42)  # Use same seed for reproducibility
    torch.cuda.manual_seed(42)
    
    while x.size(1) < initial_len + max_length - (initial_len - tokens.size(1)):
        with torch.no_grad():
            logits = model(x)  # Fixed: unpack tuple
            #logits, _ = model(x)  # Fixed: unpack tuple
            logits = logits[:, -1, :]
            
            probs = F.softmax(logits, dim=-1)
            
            # Do top-k sampling of 50
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            
            ix = torch.multinomial(topk_probs, 1)
            
            xcol = torch.gather(topk_indices, -1, ix)
            x = torch.cat((x, xcol), dim=1)
    
    new_tokens = x.size(1) - initial_len
    print(f"Worker generated {new_tokens} additional tokens")
    
    # Send completed tokens back to master
    print("Sending completed sequence back to master...")
    send_tensor(conn, x.cpu())
    conn.close()
    server_socket.close()

else:
    print("Invalid mode. Use 'master' or 'worker'.")
    sys.exit(1)