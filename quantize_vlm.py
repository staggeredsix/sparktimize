import os
import argparse

from huggingface_hub import snapshot_download
import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig

try:
    from unsloth import FastLanguageModel
except ImportError:
    FastLanguageModel = None


def download_model(model_id: str, token: str, cache_dir="models"):
    return snapshot_download(repo_id=model_id, token=token, cache_dir=cache_dir)


def quantize_with_unsloth(model_path: str):
    if FastLanguageModel is None:
        raise RuntimeError("unsloth is not installed")
    model, processor = FastLanguageModel.from_pretrained(model_path, trust_remote_code=True)
    model = FastLanguageModel.quantize(model)
    out_dir = os.path.join(model_path, "unsloth")
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    if processor:
        processor.save_pretrained(out_dir)
    return out_dir


def quantize_with_bitsandbytes(model_path: str):
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config)
    out_dir = os.path.join(model_path, "bnb4")
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize a vision language model")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--method", choices=["unsloth", "bitsandbytes"], default="unsloth")
    args = parser.parse_args()

    path = download_model(args.model_id, args.token)
    if args.method == "unsloth":
        out = quantize_with_unsloth(path)
    else:
        out = quantize_with_bitsandbytes(path)
    print(f"Quantized model saved to {out}")
