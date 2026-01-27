"""
GGUF Conversion Script
Converts HuggingFace models to GGUF format for llama.cpp
"""

import argparse
import os
import subprocess
import sys
from huggingface_hub import snapshot_download


def check_llama_cpp():
    """Check if llama.cpp is available."""
    llama_cpp_path = os.environ.get("LLAMA_CPP_PATH", "./llama.cpp")

    convert_script = os.path.join(llama_cpp_path, "convert_hf_to_gguf.py")
    quantize_binary = os.path.join(llama_cpp_path, "llama-quantize")

    if not os.path.exists(convert_script):
        print(f"Error: convert_hf_to_gguf.py not found at {convert_script}")
        print("Please clone llama.cpp and set LLAMA_CPP_PATH environment variable:")
        print("  git clone https://github.com/ggerganov/llama.cpp")
        print("  export LLAMA_CPP_PATH=/path/to/llama.cpp")
        return None, None

    if not os.path.exists(quantize_binary):
        print(f"Warning: llama-quantize not found at {quantize_binary}")
        print("Please build llama.cpp first:")
        print("  cd llama.cpp && make")
        return convert_script, None

    return convert_script, quantize_binary


def download_model(model_name: str, local_dir: str) -> str:
    """Download model from HuggingFace."""
    print(f"Downloading {model_name}...")
    path = snapshot_download(
        repo_id=model_name,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    print(f"Model downloaded to: {path}")
    return path


def convert_to_gguf(
    model_path: str,
    output_path: str,
    convert_script: str,
) -> str:
    """Convert HuggingFace model to GGUF format (FP16)."""
    print(f"Converting {model_path} to GGUF...")

    cmd = [
        sys.executable,
        convert_script,
        model_path,
        "--outfile", output_path,
        "--outtype", "f16",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error during conversion: {result.stderr}")
        raise RuntimeError("GGUF conversion failed")

    print(f"Converted to: {output_path}")
    return output_path


def quantize_gguf(
    input_path: str,
    output_path: str,
    quantize_binary: str,
    quant_type: str = "Q4_K_M",
) -> str:
    """Quantize GGUF model."""
    print(f"Quantizing to {quant_type}...")

    cmd = [quantize_binary, input_path, output_path, quant_type]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error during quantization: {result.stderr}")
        raise RuntimeError("GGUF quantization failed")

    print(f"Quantized model saved to: {output_path}")
    return output_path


def convert_and_quantize(
    model_name: str,
    output_dir: str,
    quant_types: list = None,
    keep_fp16: bool = False,
):
    """
    Full pipeline: download, convert to GGUF, and quantize.

    Args:
        model_name: HuggingFace model name
        output_dir: Output directory for GGUF files
        quant_types: List of quantization types (default: ["Q4_K_M", "Q5_K_M"])
        keep_fp16: Keep the FP16 GGUF file (default: False)
    """
    if quant_types is None:
        quant_types = ["Q4_K_M", "Q5_K_M"]

    convert_script, quantize_binary = check_llama_cpp()
    if convert_script is None:
        return

    os.makedirs(output_dir, exist_ok=True)

    # Download model
    model_dir = os.path.join(output_dir, "hf_model")
    model_path = download_model(model_name, model_dir)

    # Convert to GGUF (FP16)
    model_basename = model_name.split("/")[-1]
    fp16_path = os.path.join(output_dir, f"{model_basename}-f16.gguf")
    convert_to_gguf(model_path, fp16_path, convert_script)

    # Quantize
    if quantize_binary:
        for quant_type in quant_types:
            quant_output = os.path.join(
                output_dir,
                f"{model_basename}-{quant_type}.gguf"
            )
            quantize_gguf(fp16_path, quant_output, quantize_binary, quant_type)

        # Remove FP16 if not needed
        if not keep_fp16 and os.path.exists(fp16_path):
            os.remove(fp16_path)
            print(f"Removed intermediate FP16 file: {fp16_path}")
    else:
        print("Skipping quantization (llama-quantize not available)")
        print(f"FP16 GGUF saved at: {fp16_path}")

    print("\nConversion complete!")


def main():
    parser = argparse.ArgumentParser(description="Convert model to GGUF format")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./models/gguf",
        help="Output directory"
    )
    parser.add_argument(
        "--quant-types",
        type=str,
        nargs="+",
        default=["Q4_K_M", "Q5_K_M"],
        help="Quantization types"
    )
    parser.add_argument(
        "--keep-fp16",
        action="store_true",
        help="Keep FP16 GGUF file"
    )

    args = parser.parse_args()

    convert_and_quantize(
        model_name=args.model,
        output_dir=args.output,
        quant_types=args.quant_types,
        keep_fp16=args.keep_fp16,
    )


if __name__ == "__main__":
    main()
