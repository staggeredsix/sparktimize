# Sparktimize

This project provides a simple pipeline to fetch a model from the HuggingFace Hub, quantize it to fp4 using TensorRT-LLM and run benchmarks before and after quantization. A Gradio UI allows triggering these steps from a browser.

## Features

- **Model download**: Provide a Hugging Face access token and a model ID to fetch models securely.
- **Quantization**: Use the `tensorrt_llm` package to convert models to fp4.
- **Benchmarking**: Measure generation latency before and after quantization.
- **Gradio UI**: Start a simple web interface to run the pipeline with one click.
- **Docker support**: Use the provided `Dockerfile` to build a GPU-enabled container for both `amd64` and `arm64`.

This repository is intended for experimentation with NVIDIA Blackwell GPUs with 128&nbsp;GB of memory. It assumes PyTorch 2.7 and TensorRT‑LLM are installed.

## Quick start

```bash
# Install dependencies from NVIDIA's PyPI
./scripts/setup.sh

# Run the Gradio interface
python app.py
```

### Quantize a vision language model

Use `quantize_vlm.py` to download and quantize a model using UnsLoTH or
bitsandbytes. Provide a Hugging Face model ID and access token:

```bash
python quantize_vlm.py --model-id my/model --token <HF_TOKEN> --method unsloth
```

## NVIDIA AI Workbench

The `workbench-project.yml` file configures this repo as an NVIDIA AI Workbench project so it can be launched in a containerized environment with GPU access.
