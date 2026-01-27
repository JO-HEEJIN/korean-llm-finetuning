"""
GPTQ Model Inference Script
Run inference on GPTQ quantized models
"""

import argparse
import time
import torch
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM


def load_gptq_model(model_path: str):
    """Load GPTQ quantized model."""
    print(f"Loading GPTQ model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoGPTQForCausalLM.from_quantized(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        use_safetensors=True,
    )

    print("Model loaded successfully")
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> tuple:
    """
    Generate response from the model.

    Returns:
        tuple: (response_text, generation_time, tokens_per_second)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]

    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    end_time = time.time()
    generation_time = end_time - start_time

    output_length = outputs.shape[1] - input_length
    tokens_per_second = output_length / generation_time if generation_time > 0 else 0

    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

    return response, generation_time, tokens_per_second


def interactive_mode(model, tokenizer):
    """Interactive chat mode."""
    print("\n" + "=" * 50)
    print("GPTQ Model Interactive Mode")
    print("Type 'quit' or 'exit' to end")
    print("=" * 50 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                print("Exiting...")
                break

            if not user_input:
                continue

            response, gen_time, tps = generate_response(model, tokenizer, user_input)

            print(f"AI: {response}")
            print(f"[{gen_time:.2f}s, {tps:.1f} tokens/s]\n")

        except KeyboardInterrupt:
            print("\nExiting...")
            break


def main():
    parser = argparse.ArgumentParser(description="GPTQ Model Inference")
    parser.add_argument(
        "--model",
        type=str,
        default="./models/gptq-4bit",
        help="Path to GPTQ model"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt (if not provided, enters interactive mode)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum new tokens"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Error: CUDA is required for GPTQ inference")
        return

    model, tokenizer = load_gptq_model(args.model)

    if args.prompt:
        response, gen_time, tps = generate_response(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(f"\nPrompt: {args.prompt}")
        print(f"Response: {response}")
        print(f"\nGeneration time: {gen_time:.2f}s")
        print(f"Speed: {tps:.1f} tokens/s")
    else:
        interactive_mode(model, tokenizer)


if __name__ == "__main__":
    main()
