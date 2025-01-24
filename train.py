import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoTokenizer
# import gradio as gr
from tqdm import tqdm
from typing import Optional, Tuple


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        # Calculate RMS
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return self.weight * x_norm

class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_position_embeddings = max_position_embeddings
        self.dim = dim

    def forward(self, x, seq_len):
        # Create position embeddings for the sequence length
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Concatenate to match dimension
        emb = torch.cat((freqs, freqs), dim=-1)
        # Shape: [seq_len, dim]
        return emb

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # Reshape cos and sin to match q and k shapes
    # q and k shape: [batch_size, num_heads, seq_len, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim]
    sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim]

    # Expand cos and sin to match batch size and num heads
    cos = cos.expand(q.shape[0], q.shape[1], -1, -1)
    sin = sin.expand(k.shape[0], k.shape[1], -1, -1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class LlamaSdpaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.num_key_value_heads = config.n_head // 3  # Reduced KV heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim)

    def forward(self, x, attention_mask=None):
        B, T, C = x.size()

        # Project to q, k, v
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_key_value_heads, self.head_dim)

        # Repeat k, v for each query head group
        k = k.repeat_interleave(self.num_key_value_groups, dim=2)
        v = v.repeat_interleave(self.num_key_value_groups, dim=2)

        # Transpose for attention
        q = q.transpose(1, 2)  # (B, num_heads, T, head_dim)
        k = k.transpose(1, 2)  # (B, num_heads, T, head_dim)
        v = v.transpose(1, 2)  # (B, num_heads, T, head_dim)

        # Apply rotary embeddings
        rotary_emb = self.rotary_emb(x, T)
        cos, sin = rotary_emb.cos(), rotary_emb.sin()
        q, k = apply_rotary_pos_emb(q, k, cos, sin, None)

        # Scaled dot product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        # scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # if attention_mask is not None:
        #     scores = scores + attention_mask

        # attn = F.softmax(scores, dim=-1)
        # out = torch.matmul(attn, v)

        out = F.scaled_dot_product_attention(q, k, v, is_causal = True)

        # Reshape and project back
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.n_embd, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = LlamaRMSNorm(config.n_embd)
        self.self_attn = LlamaSdpaAttention(config)
        self.post_attention_layernorm = LlamaRMSNorm(config.n_embd)
        self.mlp = LlamaMLP(config)

    def forward(self, x):
        # Pre-norm architecture
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x

# Replace the tiktoken encoder with HuggingFace tokenizer
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        with open('/content/drive/My Drive/ERAV3/Assign13/input.txt', 'r') as f:
            text = f.read()

        # Use HuggingFace tokenizer instead of tiktoken
        tokens = tokenizer.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        print(f'loaded {len(self.tokens)} tokens')
        print(f'Vocabulary size: {len(tokenizer)}')

        # Calculate actual batches per epoch
        self.n_batches = (len(self.tokens) - 1) // (B * T)
        print(f'1 epoch = {self.n_batches} batches')

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T

        # If we're near the end, reset position
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0

        # Get the chunk of tokens for this batch
        chunk = self.tokens[self.current_position:self.current_position + B * T + 1]

        # Ensure we have enough tokens
        if len(chunk) < B * T + 1:
            self.current_position = 0
            chunk = self.tokens[:B * T + 1]

        # Create inputs and targets
        x = chunk[:B*T].view(B, T)
        y = chunk[1:B*T+1].view(B, T)

        # Advance position
        self.current_position += B * T

        return x, y


@dataclass
class SmolLM2Config:
    block_size: int = 2048  # max_position_embeddings
    vocab_size: int = 49152
    n_layer: int = 30  # num_hidden_layers
    n_head: int = 9   # num_attention_heads
    n_embd: int = 576  # hidden_size
    intermediate_size: int = 1536
    num_key_value_heads: int = 3  # Specific KV heads
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    initializer_range: float = 0.041666666666666664
    use_cache: bool = True

class SmolLM2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.n_layer)])
        self.norm = LlamaRMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.embed_tokens.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(self, idx, targets=None):
        B, T = idx.size()

        # Get embeddings
        x = self.embed_tokens(idx)

        # Forward through layers
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to block_size
            idx_cond = idx[:, -self.config.block_size:]
            # get predictions
            logits, _ = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

torch.set_float32_matmul_precision('high')
model = SmolLM2(SmolLM2Config())
model.to(device)
# model = torch.compile(model)

# Update optimizer settings
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-3,  # learning_rate from config
    betas=(0.9, 0.95),  # adam_beta1, adam_beta2
    eps=1e-8,  # adam_eps
    weight_decay=0.01,  # weight_decay
    fused=True  # torch_adam_is_fused
)

# Training settings
total_steps = 2000000  # train_steps from config
micro_batch_size = 8  # micro_batch_size from config
sequence_length = 2048  # sequence_length from config
warmup_steps = 2000  # lr_warmup_steps
lr_decay_start = 1600000  # lr_decay_starting_step
lr_decay_steps = 400000  # lr_decay_steps


def get_model_summary(model):
    """Print model summary including parameter count"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\nModel Summary:")
    print("=" * 50)
    print(f"Model Type: {model.__class__.__name__}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print("\nLayer-wise parameters:")
    print("-" * 50)

    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"{name}: {params:,} parameters")

        # For nested modules (like layers)
        if hasattr(module, "_modules"):
            for sub_name, sub_module in module.named_children():
                sub_params = sum(p.numel() for p in sub_module.parameters())
                if sub_params > 0:  # Only show if has parameters
                    print(f"  └─ {sub_name}: {sub_params:,} parameters")

    print("=" * 50)
    return total_params, trainable_params

get_model_summary(model)

train_loader = DataLoaderLite(B=4, T=32)

# Calculate number of epochs
# total_tokens = len(train_loader.tokens)
# batches_per_epoch = total_tokens // (4 * 32)
# total_epochs = 5000 / batches_per_epoch
# print(f'\nTraining for approximately {total_epochs:.2f} epochs')
# print(f'Total tokens: {total_tokens:,}')
# print(f'Batches per epoch: {batches_per_epoch}')
# print(f'Total steps: 5,000\n')

# Continue with training loop
# optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4)

# Calculate total epochs and steps per epoch
# total_steps = 5000
# steps_per_epoch = batches_per_epoch
# num_epochs = total_steps // steps_per_epoch
# remaining_steps = total_steps % steps_per_epoch

# print(f"Training for {num_epochs} full epochs plus {remaining_steps} steps")
# print(f"Steps per epoch: {steps_per_epoch}\n")

def save_checkpoint(model, optimizer, step, loss, checkpoint_dir='checkpoints'):
    """Save model checkpoint"""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_step_{step}.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"\nCheckpoint saved at step {step} to {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['step'], checkpoint['loss']

# Update the generate_prediction function to use the new tokenizer
def generate_prediction(model, device, max_tokens=100):
    """Generate sample prediction"""
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_tokens = model.generate(context, max_new_tokens=max_tokens)[0].tolist()
    return tokenizer.decode(generated_tokens)

# Training settings
total_steps = 5050
prediction_interval = 500  # Generate prediction every 500 steps
checkpoint_interval = 1000  # Save checkpoint every 1000 steps
checkpoint_dir = 'checkpoints'

# Create checkpoint directory if it doesn't exist
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Load checkpoint if exists
start_step = 0
if os.path.exists(checkpoint_dir):
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_step_')]
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        start_step, last_loss = load_checkpoint(model, optimizer, checkpoint_path)
        print(f"Resuming training from step {start_step}")

# Training loop
step = start_step
try:
    while step < total_steps:
        t0 = time.time()
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
          logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1-t0)*1000
        tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)

        # Print progress
        if step % 10 == 0:
            print(f"Step {step}/{total_steps} | Loss: {loss.item():.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec: .2f}")

        # Generate prediction at intervals
        if step % prediction_interval == 0:
            print("\n=== Generating Sample Text ===")
            model.eval()
            with torch.no_grad():
                sample_text = generate_prediction(model, device)
                print(f"Generated text at step {step}:")
                print(sample_text)
                print("============================\n")
            model.train()

        # Save checkpoint at intervals
        if step % checkpoint_interval == 0:
            save_checkpoint(model, optimizer, step, loss.item(), checkpoint_dir)

        step += 1

except KeyboardInterrupt:
    print("\nTraining interrupted by user")
    # Save checkpoint on interruption
    save_checkpoint(model, optimizer, step, loss.item(), checkpoint_dir)
    print("Checkpoint saved. You can resume training later.")

# Save final checkpoint
save_checkpoint(model, optimizer, step, loss.item(), checkpoint_dir)
print("\nTraining completed!")

# Generate final sample
print("\n=== Final Generated Sample ===")
model.eval()
with torch.no_grad():
    sample_text = generate_prediction(model, device)
    print(sample_text)
print("===========================")
