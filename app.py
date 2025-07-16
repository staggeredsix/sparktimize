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


def optimize(model_id, token):
    model_path = download_model(model_id, token)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    base_latency = benchmark(model, tokenizer)

    engine_path = quantize_to_fp4(model_path, os.path.join(model_path, "fp4"))
    # Load TensorRT-LLM engine
    engine = tllm.Engine(engine_path)
    quant_latency = benchmark(engine, tokenizer)
    return f"Base latency: {base_latency:.3f}s, FP4 latency: {quant_latency:.3f}s"


def main():
    with gr.Blocks() as demo:
        gr.Markdown("## Sparktimize: HuggingFace model optimizer")
        model_id = gr.Textbox(label="Model ID", value="meta-llama/Llama-2-7b-hf")
        token = gr.Textbox(label="HF Token", type="password")
        output = gr.Textbox(label="Result")
        optimize_btn = gr.Button("Optimize")
        optimize_btn.click(partial(optimize), inputs=[model_id, token], outputs=output)
    demo.launch()


if __name__ == "__main__":
    main()
