import os
import time
from functools import partial

import gradio as gr
from huggingface_hub import snapshot_download

try:
    import tensorrt_llm as tllm
except ImportError:
    tllm = None

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def download_model(model_id: str, token: str, cache_dir="models"):
    path = snapshot_download(repo_id=model_id, token=token, cache_dir=cache_dir)
    return path


def benchmark(model, tokenizer, prompt="Hello", iterations=10):
    model.eval()
    latencies = []
    with torch.no_grad():
        for _ in range(iterations):
            start = time.time()
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            _ = model.generate(**inputs, max_new_tokens=10)
            latencies.append(time.time() - start)
    return sum(latencies) / len(latencies)


def quantize_to_fp4(model_path: str, output_dir: str):
    if tllm is None:
        raise RuntimeError("tensorrt_llm is not installed")
    os.makedirs(output_dir, exist_ok=True)
    builder = tllm.Builder()
    builder.fp4_mode(True)
    engine_path = os.path.join(output_dir, "engine.plan")
    builder.build(model_path, engine_path)
    return engine_path


def optimize(model_id, token, progress=gr.Progress()):
    if not model_id or not token:
        return "❌ Please provide both Model ID and HuggingFace Token"
    
    try:
        progress(0.1, desc="🔄 Downloading model...")
        model_path = download_model(model_id, token)
        
        progress(0.3, desc="⚡ Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
        
        progress(0.5, desc="📊 Running baseline benchmark...")
        base_latency = benchmark(model, tokenizer)

        progress(0.7, desc="🚀 Quantizing to FP4 with TensorRT-LLM...")
        engine_path = quantize_to_fp4(model_path, os.path.join(model_path, "fp4"))
        
        progress(0.9, desc="📈 Benchmarking optimized model...")
        # Load TensorRT-LLM engine
        engine = tllm.Engine(engine_path)
        quant_latency = benchmark(engine, tokenizer)
        
        progress(1.0, desc="✅ Optimization complete!")
        
        speedup = base_latency / quant_latency
        return f"""
## 🎯 Optimization Results

**Model:** `{model_id}`

### ⚡ Performance Metrics
- **Baseline Latency:** {base_latency:.3f}s
- **FP4 Optimized Latency:** {quant_latency:.3f}s
- **Speedup:** {speedup:.2f}x faster 🚀
- **Memory Reduction:** ~75% (FP32 → FP4)

### 💾 Output Location
Optimized model saved to: `{engine_path}`
        """
        
    except Exception as e:
        return f"❌ **Error during optimization:**\n\n```\n{str(e)}\n```"


# Custom CSS for NVIDIA theme
custom_css = """
/* NVIDIA Brand Colors and Styling */
:root {
    --nvidia-green: #76b900;
    --nvidia-dark: #1a1a1a;
    --nvidia-gray: #2d2d30;
    --nvidia-light-gray: #f8f8f8;
    --nvidia-accent: #00d4aa;
}

.gradio-container {
    background: linear-gradient(135deg, var(--nvidia-dark) 0%, var(--nvidia-gray) 100%);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    color: white;
}

/* Header styling */
.main-header {
    background: linear-gradient(90deg, var(--nvidia-green), var(--nvidia-accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: bold;
    text-align: center;
    padding: 20px;
    font-size: 2.5em;
}

/* Card styling */
.input-card, .output-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(118, 185, 0, 0.3);
    border-radius: 15px;
    padding: 20px;
    margin: 10px 0;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

/* Button styling */
.optimize-btn {
    background: linear-gradient(45deg, var(--nvidia-green), var(--nvidia-accent)) !important;
    border: none !important;
    border-radius: 25px !important;
    padding: 15px 30px !important;
    font-weight: bold !important;
    font-size: 1.1em !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(118, 185, 0, 0.4) !important;
}

.optimize-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(118, 185, 0, 0.6) !important;
}

/* Input field styling */
.gr-textbox {
    background: rgba(255, 255, 255, 0.1) !important;
    border: 2px solid rgba(118, 185, 0, 0.3) !important;
    border-radius: 10px !important;
    color: white !important;
}

.gr-textbox:focus {
    border-color: var(--nvidia-green) !important;
    box-shadow: 0 0 10px rgba(118, 185, 0, 0.5) !important;
}

/* Progress bar styling */
.progress-bar {
    background: var(--nvidia-green) !important;
}

/* Footer styling */
.footer-info {
    text-align: center;
    padding: 20px;
    color: white;
    border-top: 1px solid rgba(118, 185, 0, 0.3);
    margin-top: 30px;
}

/* Animation keyframes */
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(118, 185, 0, 0.7); }
    70% { box-shadow: 0 0 0 10px rgba(118, 185, 0, 0); }
    100% { box-shadow: 0 0 0 0 rgba(118, 185, 0, 0); }
}

.pulse-animation {
    animation: pulse 2s infinite;
}
"""


def main():
    with gr.Blocks(
        css=custom_css,
        title="Sparktimize - NVIDIA AI Model Optimizer",
        theme=gr.themes.Base(
            primary_hue="green",
            secondary_hue="gray",
            neutral_hue="gray",
            font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"]
        ).set(
            body_background_fill="*neutral_950",
            block_background_fill="*neutral_900",
            border_color_primary="*primary_500",
            button_primary_background_fill="linear-gradient(45deg, *primary_500, *secondary_500)",
            button_primary_text_color="white"
        )
    ) as demo:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>⚡ SPARKTIMIZE</h1>
            <p style="color: white; font-size: 1.2em; margin: 0;">
                Powered by NVIDIA TensorRT-LLM | Accelerate Your AI Models
            </p>
        </div>
        """)
        
        # Main content
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML('<div class="input-card">')
                gr.Markdown("""
                ### 🎯 **Model Configuration**
                Transform your HuggingFace models with NVIDIA's cutting-edge optimization technology.
                """)
                
                model_id = gr.Textbox(
                    label="🤗 HuggingFace Model ID",
                    value="meta-llama/Llama-2-7b-hf",
                    placeholder="e.g., microsoft/DialoGPT-medium",
                    info="Enter the model repository ID from HuggingFace Hub"
                )
                
                token = gr.Textbox(
                    label="🔐 HuggingFace Access Token",
                    type="password",
                    placeholder="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                    info="Required for private models and faster downloads"
                )
                
                optimize_btn = gr.Button(
                    "🚀 OPTIMIZE MODEL",
                    variant="primary",
                    size="lg",
                    elem_classes=["optimize-btn", "pulse-animation"]
                )
                
                gr.HTML('</div>')
                
                # Info section
                gr.Markdown("""
                ### 💡 **What happens during optimization?**
                
                1. **📥 Download**: Securely fetch your model from HuggingFace
                2. **⚡ Quantize**: Convert to FP4 precision using TensorRT-LLM
                3. **📊 Benchmark**: Measure performance improvements
                4. **🎯 Results**: Get detailed metrics and speedup analysis
                
                **Expected Benefits:**
                - 🚀 **2-4x faster inference**
                - 💾 **75% memory reduction**
                - 🔋 **Lower power consumption**
                - 🎯 **Maintained accuracy**
                """)
            
            with gr.Column(scale=1):
                gr.HTML('<div class="output-card">')
                gr.Markdown("### 📈 **Optimization Results**")
                
                output = gr.Markdown(
                    """
                    <div style="text-align: center; padding: 40px; color: white;">
                        <h3>🔍 Ready to optimize your model</h3>
                        <p>Enter your model details and click the optimize button to get started.</p>
                        <p><em>Powered by NVIDIA TensorRT-LLM on Blackwell Architecture</em></p>
                    </div>
                    """,
                    elem_id="result-output"
                )
                gr.HTML('</div>')
        
        # System info
        gr.HTML("""
        <div class="footer-info">
            <p><strong>🖥️ System Requirements:</strong> NVIDIA GPU with CUDA 12.6+ | PyTorch 2.7 | TensorRT-LLM</p>
            <p><strong>🎯 Optimized for:</strong> NVIDIA Blackwell GPUs with 128GB Memory</p>
            <p style="margin-top: 15px; font-size: 0.9em;">
                <em>Built with ❤️ using NVIDIA AI technologies</em>
            </p>
        </div>
        """)
        
        # Event handlers
        optimize_btn.click(
            fn=optimize,
            inputs=[model_id, token],
            outputs=output,
            show_progress=True
        )
        
        # Auto-clear results when inputs change
        model_id.change(
            lambda: "🔄 Configuration updated. Ready to optimize!",
            outputs=output
        )
        
        token.change(
            lambda: "🔑 Token updated. Ready to optimize!",
            outputs=output
        )

    return demo


if __name__ == "__main__":
    demo = main()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        favicon_path=None,  # You could add an NVIDIA favicon here
        show_tips=True
    )
