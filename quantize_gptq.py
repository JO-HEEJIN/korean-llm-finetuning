"""
GPTQ Quantization Script
Quantizes LLM models to 4-bit using AutoGPTQ
"""

import argparse
import os
import torch
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig


def load_calibration_data(tokenizer, num_samples=128, max_length=512):
    """
    Load calibration data for quantization.
    Uses a simple set of Korean text samples.
    """
    calibration_texts = [
        "인공지능은 현대 기술의 핵심입니다.",
        "오늘 날씨가 매우 좋습니다.",
        "프로그래밍은 논리적 사고가 필요합니다.",
        "한국어 자연어 처리 기술이 발전하고 있습니다.",
        "기계 학습은 데이터를 기반으로 패턴을 학습합니다.",
        "딥러닝 모델은 대량의 데이터가 필요합니다.",
        "양자화는 모델 크기를 줄이는 기술입니다.",
        "GPU는 병렬 연산에 최적화되어 있습니다.",
    ]

    # Repeat samples to reach num_samples
    while len(calibration_texts) < num_samples:
        calibration_texts = calibration_texts * 2
    calibration_texts = calibration_texts[:num_samples]

    calibration_data = []
    for text in calibration_texts:
        tokenized = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding="max_length"
        )
        calibration_data.append(tokenized.input_ids)

    return calibration_data


def quantize_model(
    model_name: str,
    output_dir: str,
    bits: int = 4,
    group_size: int = 128,
    desc_act: bool = False,
    num_samples: int = 128,
):
    """
    Quantize a model using GPTQ.

    Args:
        model_name: HuggingFace model name or path
        output_dir: Directory to save quantized model
        bits: Quantization bits (default: 4)
        group_size: Group size for quantization (default: 128)
        desc_act: Use descending activation order (default: False)
        num_samples: Number of calibration samples (default: 128)
    """
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Preparing calibration data...")
    calibration_data = load_calibration_data(tokenizer, num_samples)

    print("Configuring quantization...")
    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=group_size,
        desc_act=desc_act,
        damp_percent=0.1,
    )

    print(f"Loading model {model_name}...")
    model = AutoGPTQForCausalLM.from_pretrained(
        model_name,
        quantize_config=quantize_config,
        trust_remote_code=True,
    )

    print("Starting quantization...")
    model.quantize(calibration_data)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving quantized model to {output_dir}...")
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Quantization complete!")
    print(f"Quantized model saved to: {output_dir}")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Quantize LLM with GPTQ")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Model name or path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./models/gptq-4bit",
        help="Output directory"
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        choices=[2, 3, 4, 8],
        help="Quantization bits"
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="Group size for quantization"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=128,
        help="Number of calibration samples"
    )

    args = parser.parse_args()

    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("Warning: CUDA not available. GPTQ requires GPU.")
        return

    quantize_model(
        model_name=args.model,
        output_dir=args.output,
        bits=args.bits,
        group_size=args.group_size,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()
