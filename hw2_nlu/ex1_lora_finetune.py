"""Exercise 1: LoRA fine-tune SmolLM-135M for strict medical IE -> JSON.

Splits the 10-sample dataset 8/2 (train/test). Trains a LoRA adapter using TRL
SFTTrainer with completion-only loss, then saves the adapter to outputs/lora_adapter.
"""
import json
import os
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "ex1_medical.json"
OUTPUT_DIR = ROOT / "outputs"
ADAPTER_DIR = OUTPUT_DIR / "lora_adapter"
SPLIT_PATH = OUTPUT_DIR / "ex1_split.json"

MODEL_NAME = "HuggingFaceTB/SmolLM-135M"
SEED = 42

PROMPT_TEMPLATE = (
    "You are a strict medical information extraction engine. "
    "Read the sentence and return a single JSON object with exactly three keys: "
    "\"Drug\", \"Dosage\", \"Adverse_Effect\". "
    "If a field is not mentioned, use the string \"None\". "
    "Output only the JSON object and nothing else.\n\n"
    "Sentence: {text}\n"
    "JSON:\n"
)


def build_prompt(text: str) -> str:
    return PROMPT_TEMPLATE.format(text=text)


def load_split():
    with open(DATA_PATH) as f:
        rows = json.load(f)
    rng = random.Random(SEED)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    test_idx = sorted(idx[:2])
    train_idx = sorted(idx[2:])
    train = [rows[i] for i in train_idx]
    test = [rows[i] for i in test_idx]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(SPLIT_PATH, "w") as f:
        json.dump({"train_idx": train_idx, "test_idx": test_idx,
                   "train": train, "test": test}, f, indent=2)
    return train, test


def main():
    train, test = load_split()
    print(f"Train: {len(train)} | Test: {len(test)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Single-field SFT: prompt + completion + eos as one text. We rely on
    # epoch count + small LoRA to memorize the task; enabling
    # completion_only_loss with this model's tokenizer triggered alignment
    # warnings and corrupted what gets masked.
    train_records = [
        {"text": build_prompt(r["text"]) + r["output"] + tokenizer.eos_token}
        for r in train
    ]
    train_ds = Dataset.from_list(train_records)

    sft_args = SFTConfig(
        output_dir=str(OUTPUT_DIR / "sft_run"),
        num_train_epochs=60,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=3e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        completion_only_loss=False,
        max_length=512,
        seed=SEED,
        bf16=False,
        fp16=False,
        remove_unused_columns=False,
        packing=False,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )
    trainer.train()

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))
    print(f"Adapter saved to {ADAPTER_DIR}")


if __name__ == "__main__":
    main()
