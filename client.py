"""
API Client example for On-Premise LLM Server
Compatible with OpenAI Python SDK
"""

import os
from typing import Optional, List
from openai import OpenAI


def create_client(
    base_url: str = "http://localhost:8000/v1",
    api_key: Optional[str] = None
) -> OpenAI:
    """Create OpenAI-compatible client"""
    return OpenAI(
        base_url=base_url,
        api_key=api_key or os.getenv("API_KEY", "dummy-key")
    )


def chat_completion(
    client: OpenAI,
    messages: List[dict],
    model: str = "default",
    temperature: float = 0.7,
    max_tokens: int = 256
) -> str:
    """Send chat completion request"""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content


def text_completion(
    client: OpenAI,
    prompt: str,
    model: str = "default",
    temperature: float = 0.7,
    max_tokens: int = 256
) -> str:
    """Send text completion request"""
    response = client.completions.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].text


def stream_chat_completion(
    client: OpenAI,
    messages: List[dict],
    model: str = "default",
    temperature: float = 0.7,
    max_tokens: int = 256
):
    """Stream chat completion response"""
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def main():
    # Example usage
    print("On-Premise LLM Client Example")
    print("=" * 50)

    # Create client
    client = create_client(
        base_url="http://localhost:8000/v1",
        api_key=os.getenv("API_KEY")
    )

    # Chat completion example
    print("\n1. Chat Completion:")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is machine learning?"}
    ]

    try:
        response = chat_completion(client, messages)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")

    # Text completion example
    print("\n2. Text Completion:")
    prompt = "The benefits of artificial intelligence include"

    try:
        response = text_completion(client, prompt)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")

    # Streaming example
    print("\n3. Streaming Chat Completion:")
    try:
        print("Response: ", end="", flush=True)
        for token in stream_chat_completion(client, messages):
            print(token, end="", flush=True)
        print()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
