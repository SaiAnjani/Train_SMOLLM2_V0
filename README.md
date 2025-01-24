# SmolLM2: A Lightweight Language Model Implementation

This is an implementation of a LLaMA-style language model with grouped-query attention, designed for efficient text generation while maintaining good performance.

## Model Architecture

The model is based on the LLaMA architecture with several optimizations:

### Key Components:

1. **Attention Mechanism**
   - Grouped-Query Attention (GQA)
   - 9 query heads, 3 key/value heads (3:1 ratio)
   - Rotary Positional Embeddings (RoPE)
   - Scaled dot-product attention with causal masking

2. **Model Configuration**
   ```python
   block_size: 2048          # Max sequence length
   vocab_size: 49152         # Vocabulary size
   n_layer: 30              # Number of transformer layers
   n_head: 9                # Number of attention heads
   n_embd: 576             # Embedding dimension
   intermediate_size: 1536  # MLP intermediate size
   num_key_value_heads: 3   # Number of KV heads
   ```

3. **Architecture Details**
   - RMSNorm for layer normalization
   - SiLU activation in MLP blocks
   - Weight tying between embedding and output layers
   - Pre-norm transformer architecture

## Dependencies
    requirements.txt
    torch
    transformers
    gradio


## Installation

1. Clone the repository
    - git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

2. Install dependencies:
    - pip install -r requirements.txt

3. Run the training script:
    - python train.py

    
    Training features:
    - Automatic mixed precision (bfloat16)
    - Checkpointing every 1000 steps
    - Sample generation every 500 steps
    - Progress monitoring with loss and speed metrics
    - Resume training from checkpoints

    Training parameters:
    - Learning rate: 3e-3
    - AdamW optimizer (β1=0.9, β2=0.95)
    - Weight decay: 0.01
    - Batch size: 8
    - Sequence length: 2048

    ## Inference

    Run the Gradio interface:

4. Run the Gradio app:
    - python app.py
    
The interface provides:
    - Text generation from prompts
    - Adjustable parameters:
    - Temperature (0.1 - 2.0)
    - Max length (10 - 500 tokens)
    - Top-k sampling (1 - 100)


## Project Structure

├── model.py # Core model implementation
├── train.py # Training script
├── app.py # Gradio interface
├── requirements.txt # Dependencies
├── Dockerfile # Container configuration
└── README.md # Documentation



## Model Details

1. **Attention Implementation**
   ```python
   out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
   ```
   - Uses PyTorch's optimized SDPA
   - Automatic causal masking
   - Flash Attention when available

2. **Training Optimizations**
   - Gradient accumulation
   - Mixed precision training
   - Efficient checkpointing
   - Performance monitoring

3. **Generation Features**
   - Temperature scaling
   - Top-k sampling
   - Configurable sequence length
   - Memory-efficient generation

