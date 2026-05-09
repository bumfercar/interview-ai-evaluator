"""
Train the SFT evaluator model with Qwen2.5-7B-Instruct + Unsloth LoRA.

Typical local sanity check:
    python scripts/06_train_sft.py --config configs/sft_train.yaml --dry-run

Typical Colab/GPU training:
    python scripts/06_train_sft.py --config configs/sft_train.yaml

The script converts each SFT sample into ChatML:
    system: instruction
    user: question + answer
    assistant: evaluation JSON

Only assistant responses should contribute to loss. The training path uses
Unsloth's train_on_responses_only helper for that masking.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_CONFIG = Path("configs/sft_train.yaml")
OUTPUT_FIELD_ORDER = ["reasoning", "overall_score", "strengths", "improvements"]

DEFAULT_CONFIG_DICT: dict[str, Any] = {
    "model": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "max_seq_length": 2048,
        "dtype": None,
        "load_in_4bit": True,
    },
    "data": {
        "train_path": "data/splits/train.jsonl",
        "val_path": "data/splits/val.jsonl",
        "test_path": "data/splits/test.jsonl",
        "text_field": "text",
        "max_train_samples": None,
        "max_val_samples": None,
    },
    "output": {
        "dir": "models/lora_adapters/qwen2_5_7b_sft",
        "logging_dir": "models/lora_adapters/qwen2_5_7b_sft/logs",
    },
    "lora": {
        "r": 16,
        "alpha": 16,
        "dropout": 0.05,
        "bias": "none",
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "use_gradient_checkpointing": True,
        "random_state": 42,
    },
    "training": {
        "seed": 42,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 3,
        "learning_rate": 0.0002,
        "warmup_ratio": 0.05,
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "optim": "adamw_8bit",
        "fp16": True,
        "bf16": False,
        "logging_steps": 10,
        "eval_steps": 100,
        "save_steps": 100,
        "save_total_limit": 3,
        "report_to": "none",
    },
}


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        if path == DEFAULT_CONFIG:
            print("PyYAML이 없어 내장 기본 설정으로 dry-run/config 로드를 진행합니다.")
            return DEFAULT_CONFIG_DICT
        raise SystemExit("PyYAML이 설치되어 있지 않아 사용자 config를 읽을 수 없습니다.")

    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: JSON 파싱 실패: {exc}") from exc
    return rows


def maybe_limit(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return rows
    return rows[:limit]


def build_user_content(sample: dict[str, Any]) -> str:
    question = sample["input"]["question"].strip()
    answer = sample["input"]["answer"].strip()
    return f"면접 질문:\n{question}\n\n지원자 답변:\n{answer}"


def ordered_output(output: dict[str, Any]) -> dict[str, Any]:
    return {key: output[key] for key in OUTPUT_FIELD_ORDER}


def build_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
    assistant_json = json.dumps(ordered_output(sample["output"]), ensure_ascii=False, indent=2)
    return [
        {"role": "system", "content": sample["instruction"].strip()},
        {"role": "user", "content": build_user_content(sample)},
        {"role": "assistant", "content": assistant_json},
    ]


def fallback_chatml(messages: list[dict[str, str]]) -> str:
    chunks = []
    for message in messages:
        chunks.append(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>")
    return "\n".join(chunks) + "\n"


def format_sample(sample: dict[str, Any], tokenizer: Any | None = None) -> str:
    messages = build_messages(sample)
    if tokenizer is not None:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return fallback_chatml(messages)


def validate_sample(sample: dict[str, Any]) -> list[str]:
    errors = []
    sample_id = sample.get("id", "<no-id>")
    for field in ["instruction", "input", "output"]:
        if field not in sample:
            errors.append(f"{sample_id}: missing {field}")
    if errors:
        return errors

    if list(sample["output"].keys()) != OUTPUT_FIELD_ORDER:
        errors.append(f"{sample_id}: output field order mismatch")
    score = sample["output"].get("overall_score")
    if not isinstance(score, int) or not 1 <= score <= 10:
        errors.append(f"{sample_id}: invalid overall_score={score}")
    if not sample["input"].get("question") or not sample["input"].get("answer"):
        errors.append(f"{sample_id}: empty input question/answer")
    return errors


def prepare_text_rows(rows: list[dict[str, Any]], tokenizer: Any | None = None) -> list[dict[str, Any]]:
    text_rows = []
    for row in rows:
        errors = validate_sample(row)
        if errors:
            raise ValueError("\n".join(errors[:10]))
        text_rows.append(
            {
                "id": row["id"],
                "text": format_sample(row, tokenizer),
                "metadata": row.get("metadata", {}),
            }
        )
    return text_rows


def print_dry_run(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]) -> None:
    train_text = prepare_text_rows(train_rows)
    val_text = prepare_text_rows(val_rows)
    lengths = [len(row["text"]) for row in train_text + val_text]
    print(f"train samples: {len(train_text)}")
    print(f"val samples: {len(val_text)}")
    print(
        "chat text chars:",
        {
            "min": min(lengths),
            "avg": round(mean(lengths), 1),
            "max": max(lengths),
        },
    )
    print("\n[ChatML sample]")
    print(train_text[0]["text"][:2500])


def run_training(config: dict[str, Any]) -> None:
    try:
        from datasets import Dataset
        from transformers import TrainingArguments, set_seed
        from trl import SFTTrainer
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import train_on_responses_only
    except ImportError as exc:
        raise SystemExit(
            "학습 의존성이 설치되어 있지 않습니다. Colab/GPU 환경에서 pyproject.toml의 학습 의존성을 설치한 뒤 실행하세요.\n"
            f"원인: {exc}"
        ) from exc

    model_cfg = config["model"]
    data_cfg = config["data"]
    lora_cfg = config["lora"]
    train_cfg = config["training"]
    output_cfg = config["output"]

    set_seed(int(train_cfg["seed"]))

    train_rows = maybe_limit(load_jsonl(Path(data_cfg["train_path"])), data_cfg.get("max_train_samples"))
    val_rows = maybe_limit(load_jsonl(Path(data_cfg["val_path"])), data_cfg.get("max_val_samples"))

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["name"],
        max_seq_length=int(model_cfg["max_seq_length"]),
        dtype=model_cfg.get("dtype"),
        load_in_4bit=bool(model_cfg["load_in_4bit"]),
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=int(lora_cfg["r"]),
        target_modules=list(lora_cfg["target_modules"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        bias=lora_cfg["bias"],
        use_gradient_checkpointing=bool(lora_cfg["use_gradient_checkpointing"]),
        random_state=int(lora_cfg["random_state"]),
    )

    train_dataset = Dataset.from_list(prepare_text_rows(train_rows, tokenizer))
    val_dataset = Dataset.from_list(prepare_text_rows(val_rows, tokenizer))

    args = TrainingArguments(
        output_dir=output_cfg["dir"],
        logging_dir=output_cfg["logging_dir"],
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(train_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        num_train_epochs=float(train_cfg["num_train_epochs"]),
        learning_rate=float(train_cfg["learning_rate"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        weight_decay=float(train_cfg["weight_decay"]),
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        optim=train_cfg["optim"],
        fp16=bool(train_cfg["fp16"]),
        bf16=bool(train_cfg["bf16"]),
        logging_steps=int(train_cfg["logging_steps"]),
        evaluation_strategy="steps",
        eval_steps=int(train_cfg["eval_steps"]),
        save_steps=int(train_cfg["save_steps"]),
        save_total_limit=int(train_cfg["save_total_limit"]),
        report_to=train_cfg["report_to"],
        seed=int(train_cfg["seed"]),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field=data_cfg["text_field"],
        max_seq_length=int(model_cfg["max_seq_length"]),
        packing=False,
        args=args,
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    trainer.train()
    trainer.save_model(output_cfg["dir"])
    tokenizer.save_pretrained(output_cfg["dir"])
    print(f"학습 완료. LoRA adapter 저장: {output_cfg['dir']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="SFT train config path")
    parser.add_argument("--dry-run", action="store_true", help="Only validate ChatML formatting")
    parser.add_argument("--sample", type=int, default=3, help="Dry-run sample count")
    args = parser.parse_args()

    config = load_config(Path(args.config))

    if args.dry_run:
        random.seed(int(config["training"]["seed"]))
        data_cfg = config["data"]
        train_rows = load_jsonl(Path(data_cfg["train_path"]))[: args.sample]
        val_rows = load_jsonl(Path(data_cfg["val_path"]))[: args.sample]
        print_dry_run(train_rows, val_rows)
        return

    run_training(config)


if __name__ == "__main__":
    main()
