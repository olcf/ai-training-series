#!/usr/bin/env python3
"""Fine-tune a tiny causal language model on generated chat-format data."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    __version__ as transformers_version,
)

try:
    from config import PROJECT_ROOT
except ModuleNotFoundError:
    # module not found because config.py is not a sibling, and certain versions of Python do not look from call path
    PROJECT_ROOT = PROJECT_ROOT = Path(__file__).resolve().parent.parent


DEFAULT_TRAIN_FILE = PROJECT_ROOT / "finetuning" / "generated_train.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "finetuning" / "model_output"
DEFAULT_MODEL_NAME = "sshleifer/tiny-gpt2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune a tiny causal LM on the generated fine-tuning dataset."
    )
    parser.add_argument(
        "--train-file",
        default=str(DEFAULT_TRAIN_FILE),
        help="Input JSONL file containing chat-format training examples",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Base Hugging Face causal LM to fine-tune",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the fine-tuned model will be saved",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device training batch size",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum tokenized sequence length",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate for optimization",
    )
    return parser


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                yield json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_num}: {exc}") from exc


def format_chat_example(messages: list[dict[str, str]]) -> str:
    parts = []
    for message in messages:
        role = message.get("role", "user").strip().lower()
        content = message.get("content", "").strip()
        if not content:
            continue
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


class ChatFineTuneDataset(Dataset):
    """Tokenized dataset for simple causal-LM fine-tuning."""

    def __init__(self, path: Path, tokenizer, max_length: int):
        self.examples = []
        for record in iter_jsonl(path):
            messages = record.get("messages")
            if not isinstance(messages, list):
                continue

            text = format_chat_example(messages)
            if not text:
                continue

            tokenized = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            self.examples.append(
                {
                    "input_ids": tokenized["input_ids"].squeeze(0),
                    "attention_mask": tokenized["attention_mask"].squeeze(0),
                }
            )

        if not self.examples:
            raise ValueError(f"No valid training examples found in {path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.examples[index]


def run(args: argparse.Namespace) -> None:
    train_file = Path(args.train_file)
    output_dir = Path(args.output_dir)

    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")

    print("[ft-train] Starting fine-tuning run")
    print(f"[ft-train] Train file: {train_file}")
    print(f"[ft-train] Base model: {args.model_name}")
    print(f"[ft-train] Output dir: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    dataset = ChatFineTuneDataset(train_file, tokenizer, args.max_length)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # diff in newer API
    if transformers_version.startswith('5'):
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            save_strategy="epoch",
            logging_steps=1,
            report_to="none",
            remove_unused_columns=False,
        )

    else:
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            overwrite_output_dir=True,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            save_strategy="epoch",
            logging_steps=1,
            report_to="none",
            remove_unused_columns=False,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[ft-train] Saved fine-tuned model to {output_dir}")


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
