"""
Inference script for fine-tuned Korean LLM.
Supports both LoRA adapter loading and merged model inference.
"""

import argparse
import yaml
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextStreamer,
)
from peft import PeftModel


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_bnb_config(config: dict) -> BitsAndBytesConfig:
    """Create BitsAndBytes configuration for 4-bit quantization."""
    quant_config = config.get("quantization", {})
    compute_dtype = getattr(torch, quant_config.get("bnb_4bit_compute_dtype", "float16"))

    return BitsAndBytesConfig(
        load_in_4bit=quant_config.get("load_in_4bit", True),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=quant_config.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=quant_config.get("bnb_4bit_use_double_quant", True),
    )


def load_model_for_inference(
    config: dict,
    adapter_path: str = None,
    use_4bit: bool = True
):
    """
    Load model for inference.

    Args:
        config: Configuration dictionary
        adapter_path: Path to LoRA adapter (if None, loads base model only)
        use_4bit: Whether to use 4-bit quantization

    Returns:
        model, tokenizer
    """
    model_config = config.get("model", {})
    base_model = model_config.get("base_model", "Qwen/Qwen2.5-3B")
    trust_remote_code = model_config.get("trust_remote_code", True)

    print(f"Loading base model: {base_model}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model
    if use_4bit:
        bnb_config = create_bnb_config(config)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=trust_remote_code,
        )

    # Load LoRA adapter if provided
    if adapter_path:
        print(f"Loading LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        print("LoRA adapter merged successfully")

    model.eval()
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    do_sample: bool = True,
    stream: bool = False,
) -> str:
    """
    Generate a response from the model.

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        do_sample: Whether to use sampling
        stream: Whether to stream output

    Returns:
        Generated text
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    if stream:
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    else:
        streamer = None

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        )

    if not stream:
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the generated part (after the prompt)
        response = generated_text[len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):]
        return response.strip()

    return ""


def format_prompt(instruction: str, template: str) -> str:
    """Format the instruction using the inference template."""
    return template.format(instruction=instruction)


def interactive_mode(model, tokenizer, config: dict):
    """Run interactive chat mode."""
    prompt_config = config.get("prompt", {})
    inference_template = prompt_config.get(
        "inference_template",
        "### 질문:\n{instruction}\n\n### 답변:\n"
    )

    print("\n" + "=" * 50)
    print("한국어 고객 상담 AI - 대화 모드")
    print("종료하려면 'quit' 또는 'exit'를 입력하세요.")
    print("=" * 50 + "\n")

    while True:
        try:
            user_input = input("고객: ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                print("대화를 종료합니다.")
                break

            if not user_input:
                continue

            prompt = format_prompt(user_input, inference_template)

            print("상담사: ", end="", flush=True)
            generate_response(
                model,
                tokenizer,
                prompt,
                max_new_tokens=256,
                temperature=0.7,
                stream=True,
            )
            print("\n")

        except KeyboardInterrupt:
            print("\n대화를 종료합니다.")
            break


def batch_inference(
    model,
    tokenizer,
    config: dict,
    questions: list,
) -> list:
    """
    Run batch inference on a list of questions.

    Args:
        model: The language model
        tokenizer: The tokenizer
        config: Configuration dictionary
        questions: List of questions to process

    Returns:
        List of (question, response) tuples
    """
    prompt_config = config.get("prompt", {})
    inference_template = prompt_config.get(
        "inference_template",
        "### 질문:\n{instruction}\n\n### 답변:\n"
    )

    results = []
    for question in questions:
        prompt = format_prompt(question, inference_template)
        response = generate_response(
            model,
            tokenizer,
            prompt,
            max_new_tokens=256,
            temperature=0.7,
        )
        results.append((question, response))

    return results


def main():
    parser = argparse.ArgumentParser(description="Inference with fine-tuned Korean LLM")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Path to LoRA adapter directory"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt for inference (if not provided, enters interactive mode)"
    )
    parser.add_argument(
        "--no_4bit",
        action="store_true",
        help="Disable 4-bit quantization"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Load model
    model, tokenizer = load_model_for_inference(
        config,
        adapter_path=args.adapter_path,
        use_4bit=not args.no_4bit,
    )

    if args.prompt:
        # Single prompt mode
        prompt_config = config.get("prompt", {})
        inference_template = prompt_config.get(
            "inference_template",
            "### 질문:\n{instruction}\n\n### 답변:\n"
        )

        formatted_prompt = format_prompt(args.prompt, inference_template)
        print(f"\n입력: {args.prompt}")
        print("\n응답: ", end="", flush=True)

        generate_response(
            model,
            tokenizer,
            formatted_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            stream=True,
        )
        print()
    else:
        # Interactive mode
        interactive_mode(model, tokenizer, config)


if __name__ == "__main__":
    main()
