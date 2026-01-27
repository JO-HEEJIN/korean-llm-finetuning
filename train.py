"""
Training script for Korean LLM fine-tuning with LoRA.
Uses 4-bit quantization via bitsandbytes for memory efficiency.
"""

import argparse
import os
import yaml
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset

from dataset import load_jsonl_data, get_formatting_func


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


def create_lora_config(config: dict) -> LoraConfig:
    """Create LoRA configuration."""
    lora_config = config.get("lora", {})

    return LoraConfig(
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("lora_alpha", 32),
        lora_dropout=lora_config.get("lora_dropout", 0.05),
        target_modules=lora_config.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
        bias=lora_config.get("bias", "none"),
        task_type=lora_config.get("task_type", "CAUSAL_LM"),
    )


def load_model_and_tokenizer(config: dict, bnb_config: BitsAndBytesConfig):
    """Load the base model and tokenizer with quantization."""
    model_config = config.get("model", {})
    base_model = model_config.get("base_model", "Qwen/Qwen2.5-3B")
    trust_remote_code = model_config.get("trust_remote_code", True)

    print(f"Loading model: {base_model}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=trust_remote_code,
    )

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    return model, tokenizer


def create_training_arguments(config: dict) -> TrainingArguments:
    """Create training arguments from config."""
    train_config = config.get("training", {})

    return TrainingArguments(
        output_dir=train_config.get("output_dir", "./outputs"),
        num_train_epochs=train_config.get("num_train_epochs", 3),
        per_device_train_batch_size=train_config.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 4),
        learning_rate=train_config.get("learning_rate", 2e-4),
        weight_decay=train_config.get("weight_decay", 0.01),
        warmup_ratio=train_config.get("warmup_ratio", 0.03),
        lr_scheduler_type=train_config.get("lr_scheduler_type", "cosine"),
        logging_steps=train_config.get("logging_steps", 10),
        save_steps=train_config.get("save_steps", 100),
        save_total_limit=train_config.get("save_total_limit", 3),
        fp16=train_config.get("fp16", True),
        optim=train_config.get("optim", "paged_adamw_32bit"),
        gradient_checkpointing=train_config.get("gradient_checkpointing", True),
        report_to=train_config.get("report_to", "none"),
        remove_unused_columns=False,
    )


def prepare_training_data(config: dict) -> Dataset:
    """Load and prepare training data."""
    data_config = config.get("data", {})

    if "train_file" in data_config and data_config["train_file"]:
        data = load_jsonl_data(data_config["train_file"])
        dataset = Dataset.from_list(data)
        print(f"Loaded {len(dataset)} samples from {data_config['train_file']}")
    else:
        raise ValueError("No training data file specified in config")

    return dataset


def train(config_path: str, max_steps: int = None):
    """Main training function."""
    # Load configuration
    config = load_config(config_path)
    print("Configuration loaded successfully")

    # Create configurations
    bnb_config = create_bnb_config(config)
    lora_config = create_lora_config(config)

    print(f"LoRA config: r={lora_config.r}, alpha={lora_config.lora_alpha}, dropout={lora_config.lora_dropout}")
    print(f"Target modules: {lora_config.target_modules}")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config, bnb_config)
    print("Model and tokenizer loaded")

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Prepare training data
    dataset = prepare_training_data(config)

    # Get formatting function
    template = config.get("prompt", {}).get("template", "### 질문:\n{instruction}\n\n### 답변:\n{output}")
    formatting_func = get_formatting_func(template)

    # Create training arguments
    training_args = create_training_arguments(config)

    # Override max_steps if specified (for testing)
    if max_steps is not None:
        training_args.max_steps = max_steps
        training_args.num_train_epochs = 1

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        formatting_func=formatting_func,
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=config.get("training", {}).get("max_seq_length", 512),
    )

    print("Starting training...")
    trainer.train()

    # Save the final model
    output_dir = config.get("training", {}).get("output_dir", "./outputs")
    final_model_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Model saved to {final_model_path}")

    return trainer


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Korean LLM with LoRA")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum training steps (for testing)"
    )

    args = parser.parse_args()

    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("Warning: CUDA not available. Training will be slow on CPU.")

    train(args.config, args.max_steps)


if __name__ == "__main__":
    main()
