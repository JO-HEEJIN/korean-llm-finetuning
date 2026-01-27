"""
GGUF Model Inference Script
Run inference on GGUF models using llama-cpp-python
"""

import argparse
import time
from llama_cpp import Llama


def load_gguf_model(
    model_path: str,
    n_ctx: int = 2048,
    n_gpu_layers: int = -1,
    verbose: bool = False,
) -> Llama:
    """
    Load GGUF model.

    Args:
        model_path: Path to GGUF file
        n_ctx: Context window size
        n_gpu_layers: Number of layers to offload to GPU (-1 for all)
        verbose: Enable verbose output

    Returns:
        Llama model instance
    """
    print(f"Loading GGUF model from {model_path}...")

    model = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=verbose,
    )

    print("Model loaded successfully")
    return model


def generate_response(
    model: Llama,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    stop: list = None,
) -> tuple:
    """
    Generate response from the model.

    Returns:
        tuple: (response_text, generation_time, tokens_per_second)
    """
    if stop is None:
        stop = ["</s>", "<|im_end|>", "<|endoftext|>"]

    start_time = time.time()

    output = model(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
        echo=False,
    )

    end_time = time.time()
    generation_time = end_time - start_time

    response = output["choices"][0]["text"].strip()
    tokens_generated = output["usage"]["completion_tokens"]
    tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0

    return response, generation_time, tokens_per_second


def interactive_mode(model: Llama):
    """Interactive chat mode."""
    print("\n" + "=" * 50)
    print("GGUF Model Interactive Mode")
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

            response, gen_time, tps = generate_response(model, user_input)

            print(f"AI: {response}")
            print(f"[{gen_time:.2f}s, {tps:.1f} tokens/s]\n")

        except KeyboardInterrupt:
            print("\nExiting...")
            break


def main():
    parser = argparse.ArgumentParser(description="GGUF Model Inference")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to GGUF model file"
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
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=2048,
        help="Context window size"
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=-1,
        help="GPU layers (-1 for all, 0 for CPU only)"
    )

    args = parser.parse_args()

    model = load_gguf_model(
        args.model,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
    )

    if args.prompt:
        response, gen_time, tps = generate_response(
            model, args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(f"\nPrompt: {args.prompt}")
        print(f"Response: {response}")
        print(f"\nGeneration time: {gen_time:.2f}s")
        print(f"Speed: {tps:.1f} tokens/s")
    else:
        interactive_mode(model)


if __name__ == "__main__":
    main()
