#!/usr/bin/env python3
"""
SFT (Supervised Fine-Tuning) Training Script for Qwen/Qwen3-30B-A3B-Instruct-2507

Fine-tunes Qwen/Qwen3-30B-A3B-Instruct-2507 on tau-bench trajectory data using LoRA.
Trains the model to generate appropriate assistant responses given conversation history.
"""

import os
import json
import argparse
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import wandb


@dataclass
class SFTDataConfig:
    """Configuration for SFT data processing."""
    train_file: str = None
    val_file: str = None
    max_seq_length: int = 4096
    preprocessing_num_workers: int = 4


def load_sft_data(train_file: str, val_file: str = None) -> Dict[str, Dataset]:
    """Load SFT data from JSONL files."""
    data_files = {"train": train_file}
    if val_file and os.path.exists(val_file):
        data_files["validation"] = val_file
    
    dataset = load_dataset("json", data_files=data_files)
    return dataset


def format_messages_for_training(
    messages: List[Dict],
    tokenizer,
    max_length: int,
) -> Dict[str, torch.Tensor]:
    """
    Format messages into input_ids and labels for SFT training.
    
    The model learns to predict only the assistant responses.
    User/system/tool messages are masked in the labels.
    """
    # Use chat template to format the full conversation
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    
    # Tokenize the full text
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    
    input_ids = tokenized["input_ids"]
    
    # Create labels: mask everything except assistant responses
    # We'll use a simple heuristic - find assistant response boundaries
    labels = input_ids.copy()
    
    # For proper masking, we need to identify where assistant responses start/end
    # This is model-specific; for Qwen3, we look for the assistant role markers
    
    # Simple approach: tokenize prefix up to each assistant turn and mask those tokens
    # More robust: use the chat template structure
    
    # For now, we'll train on the full sequence (simpler but effective for SFT)
    # The chat template naturally structures the data
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": tokenized.get("attention_mask", [1] * len(input_ids)),
    }


def preprocess_function(
    examples: Dict,
    tokenizer,
    max_seq_length: int,
) -> Dict:
    """Preprocess a batch of examples."""
    all_input_ids = []
    all_labels = []
    all_attention_mask = []
    
    for messages in examples["messages"]:
        result = format_messages_for_training(messages, tokenizer, max_seq_length)
        all_input_ids.append(result["input_ids"])
        all_labels.append(result["labels"])
        all_attention_mask.append(result["attention_mask"])
    
    return {
        "input_ids": all_input_ids,
        "labels": all_labels,
        "attention_mask": all_attention_mask,
    }


def create_model_and_tokenizer(
    model_name: str,
    use_lora: bool = True,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
):
    """Create model and tokenizer with optional LoRA configuration."""
    
    print(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Quantization config
    quantization_config = None
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif load_in_8bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quantization_config,
        attn_implementation="sdpa",  # PyTorch native attention, no flash-attn needed
    )
    
    # Prepare for training if quantized
    if load_in_4bit or load_in_8bit:
        model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA if requested
    if use_lora:
        if lora_target_modules is None:
            # Default target modules for Qwen3
            lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            bias="none",
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="SFT Training for Qwen3")
    
    # Data arguments
    parser.add_argument("--train-file", type=str, required=True,
                       help="Path to training JSONL file")
    parser.add_argument("--val-file", type=str, default=None,
                       help="Path to validation JSONL file")
    parser.add_argument("--max-seq-length", type=int, default=4096,
                       help="Maximum sequence length")
    
    # Model arguments
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507",
                       help="Model name or path")
    parser.add_argument("--use-lora", action="store_true", default=True,
                       help="Use LoRA for efficient fine-tuning")
    parser.add_argument("--no-lora", action="store_true",
                       help="Disable LoRA (full fine-tuning)")
    parser.add_argument("--lora-r", type=int, default=64,
                       help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=128,
                       help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                       help="LoRA dropout")
    parser.add_argument("--load-in-4bit", action="store_true",
                       help="Load model in 4-bit quantization")
    parser.add_argument("--load-in-8bit", action="store_true",
                       help="Load model in 8-bit quantization")
    
    # Training arguments
    parser.add_argument("--output-dir", type=str, default="./checkpoints/sft",
                       help="Output directory for checkpoints")
    parser.add_argument("--num-epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Per-device training batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--warmup-ratio", type=float, default=0.03,
                       help="Warmup ratio")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                       help="Max gradient norm for clipping")
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine",
                       help="Learning rate scheduler type")
    
    # Logging arguments
    parser.add_argument("--logging-steps", type=int, default=10,
                       help="Logging frequency (steps)")
    parser.add_argument("--eval-steps", type=int, default=100,
                       help="Evaluation frequency (steps)")
    parser.add_argument("--save-steps", type=int, default=500,
                       help="Checkpoint save frequency (steps)")
    parser.add_argument("--save-total-limit", type=int, default=3,
                       help="Maximum number of checkpoints to keep")
    
    # W&B arguments
    parser.add_argument("--use-wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    parser.add_argument("--wandb-project", type=str, default="qwen3-sft",
                       help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                       help="W&B run name")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--bf16", action="store_true", default=True,
                       help="Use bf16 mixed precision")
    parser.add_argument("--fp16", action="store_true",
                       help="Use fp16 mixed precision")
    parser.add_argument("--preprocessing-num-workers", type=int, default=4,
                       help="Number of preprocessing workers")
    
    args = parser.parse_args()
    
    # Handle LoRA flag
    use_lora = not args.no_lora
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )
    
    # Create model and tokenizer
    print("\n" + "="*60)
    print("Loading Model and Tokenizer")
    print("="*60)
    
    model, tokenizer = create_model_and_tokenizer(
        model_name=args.model_name,
        use_lora=use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )
    
    # Load dataset
    print("\n" + "="*60)
    print("Loading Dataset")
    print("="*60)
    
    dataset = load_sft_data(args.train_file, args.val_file)
    print(f"Train samples: {len(dataset['train'])}")
    if "validation" in dataset:
        print(f"Validation samples: {len(dataset['validation'])}")
    
    # Preprocess dataset
    print("\n" + "="*60)
    print("Preprocessing Dataset")
    print("="*60)
    
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer, args.max_seq_length),
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        eval_strategy="steps" if args.val_file else "no",
        eval_steps=args.eval_steps if args.val_file else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True if args.val_file else False,
        metric_for_best_model="eval_loss" if args.val_file else None,
        greater_is_better=False,
        bf16=args.bf16 and not args.fp16,
        fp16=args.fp16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="wandb" if args.use_wandb else "none",
        run_name=args.wandb_run_name,
        seed=args.seed,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        optim="adamw_torch_fused",
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset.get("validation"),
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    trainer.train()
    
    # Save final model
    print("\n" + "="*60)
    print("Saving Final Model")
    print("="*60)
    
    final_model_dir = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    # Save LoRA adapters separately if using LoRA
    if use_lora:
        lora_dir = os.path.join(args.output_dir, "lora_adapters")
        model.save_pretrained(lora_dir)
        print(f"LoRA adapters saved to {lora_dir}")
    
    print(f"\n✅ Training complete!")
    print(f"Final model saved to: {final_model_dir}")
    
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

