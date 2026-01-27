"""
Dataset module for Korean LLM fine-tuning.
Handles loading and preprocessing of instruction-tuning datasets.
"""

import json
from typing import Dict, List, Optional, Union
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer


def load_jsonl_data(file_path: str) -> List[Dict]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def format_instruction(
    instruction: str,
    output: str,
    template: str,
    input_text: Optional[str] = None
) -> str:
    """Format a single instruction-output pair using the template."""
    if input_text:
        formatted = template.format(
            instruction=instruction,
            input=input_text,
            output=output
        )
    else:
        formatted = template.format(
            instruction=instruction,
            output=output
        )
    return formatted


def create_prompt_formats(
    examples: Dict[str, List],
    template: str
) -> Dict[str, List]:
    """Create formatted prompts for a batch of examples."""
    formatted_texts = []

    for i in range(len(examples["instruction"])):
        instruction = examples["instruction"][i]
        output = examples["output"][i]
        input_text = examples.get("input", [None] * len(examples["instruction"]))[i]

        formatted = format_instruction(instruction, output, template, input_text)
        formatted_texts.append(formatted)

    return {"text": formatted_texts}


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int = 512
) -> Dataset:
    """Tokenize the dataset for training."""

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
            return_tensors=None
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    return tokenized_dataset


def prepare_dataset(
    config: Dict,
    tokenizer: PreTrainedTokenizer,
    for_training: bool = True
) -> Union[Dataset, tuple]:
    """
    Prepare the dataset for training or evaluation.

    Args:
        config: Configuration dictionary containing data and prompt settings
        tokenizer: HuggingFace tokenizer
        for_training: If True, returns train/eval split. If False, returns full dataset.

    Returns:
        Dataset or tuple of (train_dataset, eval_dataset)
    """
    data_config = config.get("data", {})
    prompt_config = config.get("prompt", {})
    training_config = config.get("training", {})

    template = prompt_config.get("template", "### 질문:\n{instruction}\n\n### 답변:\n{output}")
    max_seq_length = training_config.get("max_seq_length", 512)

    # Load data from local file or HuggingFace
    if "train_file" in data_config and data_config["train_file"]:
        data = load_jsonl_data(data_config["train_file"])
        dataset = Dataset.from_list(data)
    elif "huggingface_dataset" in data_config and data_config["huggingface_dataset"]:
        dataset = load_dataset(data_config["huggingface_dataset"], split="train")
    else:
        raise ValueError("No data source specified in config")

    # Format prompts
    dataset = dataset.map(
        lambda examples: create_prompt_formats(examples, template),
        batched=True,
        remove_columns=dataset.column_names
    )

    # Split dataset if for training
    if for_training:
        validation_split = data_config.get("validation_split", 0.1)
        seed = data_config.get("seed", 42)

        if validation_split > 0:
            split_dataset = dataset.train_test_split(
                test_size=validation_split,
                seed=seed
            )
            train_dataset = split_dataset["train"]
            eval_dataset = split_dataset["test"]
        else:
            train_dataset = dataset
            eval_dataset = None

        return train_dataset, eval_dataset

    return dataset


def get_formatting_func(template: str):
    """
    Return a formatting function for SFTTrainer.

    Args:
        template: Prompt template string

    Returns:
        Formatting function that takes examples and returns formatted text
    """
    def formatting_func(examples):
        texts = []
        instructions = examples["instruction"]
        outputs = examples["output"]
        inputs = examples.get("input", [None] * len(instructions))

        for instruction, output, input_text in zip(instructions, outputs, inputs):
            text = format_instruction(instruction, output, template, input_text)
            texts.append(text)

        return texts

    return formatting_func


if __name__ == "__main__":
    # Test the dataset module
    import yaml

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Load sample data
    data = load_jsonl_data(config["data"]["train_file"])
    print(f"Loaded {len(data)} samples")
    print(f"\nSample data:")
    print(f"Instruction: {data[0]['instruction']}")
    print(f"Output: {data[0]['output'][:100]}...")

    # Test formatting
    template = config["prompt"]["template"]
    formatted = format_instruction(data[0]["instruction"], data[0]["output"], template)
    print(f"\nFormatted prompt:")
    print(formatted)
