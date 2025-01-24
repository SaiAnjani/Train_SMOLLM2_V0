import torch
from transformers import AutoTokenizer
from model import SmolLM2, SmolLM2Config
import gradio as gr

# Initialize model and tokenizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
model = SmolLM2(SmolLM2Config())

# Load trained weights
checkpoint = torch.load('checkpoints/checkpoint_step_5000.pt', map_location=device)  # Adjust path as needed
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

def generate_text(prompt, max_length=100, temperature=0.7, top_k=50):
    """Generate text from a prompt"""
    # Tokenize the prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_length,
            temperature=temperature,
            top_k=top_k
        )
    
    # Decode and return the generated text
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text

# Gradio interface
def gradio_interface(prompt, max_length, temperature, top_k):
    return generate_text(prompt, int(max_length), float(temperature), int(top_k))

iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Enter your prompt here..."),
        gr.Slider(minimum=10, maximum=500, value=100, step=10, label="Max Length"),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Top K"),
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="SmolLM2 Text Generation",
    description="Generate text using the SmolLM2 model"
)

# For Hugging Face deployment
app = gr.mount_gradio_app(app, iface)

if __name__ == "__main__":
    iface.launch() 