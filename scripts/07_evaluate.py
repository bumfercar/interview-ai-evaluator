"""
Evaluate baseline or SFT LoRA model on data/splits/test.jsonl.

Examples:
    # Evaluate trained LoRA adapter
    python scripts/07_evaluate.py \
      --adapter /content/drive/MyDrive/AI/models/lora_adapters/qwen2_5_7b_sft \
      --output data/eval/sft_predictions.jsonl

    # Evaluate base model only
    python scripts/07_evaluate.py \
      --baseline \
      --output data/eval/baseline_predictions.jsonl

Metrics:
- JSON parse success rate
- exact score accuracy
- score MAE
- score-band accuracy
- optional Pearson correlation
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_TEST = Path("data/splits/test.jsonl")
DEFAULT_OUTPUT = Path("data/eval/predictions.jsonl")
REQUIRED_OUTPUT_FIELDS = ["reasoning", "overall_score", "strengths", "improvements"]


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_user_content(sample: dict[str, Any]) -> str:
    return (
        f"면접 질문:\n{sample['input']['question'].strip()}\n\n"
        f"지원자 답변:\n{sample['input']['answer'].strip()}"
    )


def build_prompt(sample: dict[str, Any], tokenizer: Any) -> str:
    messages = [
        {"role": "system", "content": sample["instruction"].strip()},
        {"role": "user", "content": build_user_content(sample)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    candidates = []

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    candidates.append(text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        candidates.append(match.group())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def valid_prediction(parsed: dict[str, Any] | None) -> bool:
    if not isinstance(parsed, dict):
        return False
    if not all(field in parsed for field in REQUIRED_OUTPUT_FIELDS):
        return False
    score = parsed.get("overall_score")
    if not isinstance(score, int) or not 1 <= score <= 10:
        return False
    if not isinstance(parsed.get("strengths"), list):
        return False
    if not isinstance(parsed.get("improvements"), list):
        return False
    return True


def score_band(score: int) -> str:
    if score <= 3:
        return "1-3"
    if score <= 6:
        return "4-6"
    if score <= 8:
        return "7-8"
    return "9-10"


def pearson(xs: list[int], ys: list[int]) -> float | None:
    if len(xs) < 2:
        return None
    mean_x = mean(xs)
    mean_y = mean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = sum((x - mean_x) ** 2 for x in xs) ** 0.5
    denom_y = sum((y - mean_y) ** 2 for y in ys) ** 0.5
    if denom_x == 0 or denom_y == 0:
        return None
    return numerator / (denom_x * denom_y)


def load_model(args: argparse.Namespace) -> tuple[Any, Any]:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise SystemExit(
            "평가 의존성이 설치되어 있지 않습니다. Colab에서 transformers, peft, bitsandbytes를 설치한 뒤 실행하세요.\n"
            f"원인: {exc}"
        ) from exc

    quant_config = None
    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )

    if args.adapter and not args.baseline:
        model = PeftModel.from_pretrained(model, args.adapter)

    model.eval()
    return model, tokenizer


def generate_one(sample: dict[str, Any], model: Any, tokenizer: Any, args: argparse.Namespace) -> str:
    import torch

    prompt = build_prompt(sample, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature if args.temperature > 0 else None,
            top_p=args.top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def evaluate(samples: list[dict[str, Any]], model: Any, tokenizer: Any, args: argparse.Namespace) -> list[dict[str, Any]]:
    predictions = []
    for index, sample in enumerate(samples, 1):
        raw_text = generate_one(sample, model, tokenizer, args)
        parsed = extract_json_object(raw_text)
        ok = valid_prediction(parsed)
        true_score = int(sample["output"]["overall_score"])
        pred_score = parsed["overall_score"] if ok else None
        predictions.append(
            {
                "id": sample["id"],
                "true_score": true_score,
                "true_score_band": sample["metadata"]["score_band"],
                "pred_score": pred_score,
                "pred_score_band": score_band(pred_score) if isinstance(pred_score, int) else None,
                "json_ok": ok,
                "raw_prediction": raw_text,
                "parsed_prediction": parsed,
            }
        )
        if index % args.log_every == 0 or index == len(samples):
            print(f"generated {index}/{len(samples)}")
    return predictions


def print_metrics(predictions: list[dict[str, Any]]) -> None:
    total = len(predictions)
    ok_rows = [row for row in predictions if row["json_ok"]]
    json_rate = len(ok_rows) / total if total else 0

    print("[metrics]")
    print(f"  total: {total}")
    print(f"  json_parse_success: {len(ok_rows)}/{total} ({json_rate:.3f})")

    if not ok_rows:
        return

    true_scores = [row["true_score"] for row in ok_rows]
    pred_scores = [row["pred_score"] for row in ok_rows]
    mae = mean(abs(t - p) for t, p in zip(true_scores, pred_scores))
    exact_acc = mean(1 if t == p else 0 for t, p in zip(true_scores, pred_scores))
    band_acc = mean(
        1 if row["true_score_band"] == row["pred_score_band"] else 0
        for row in ok_rows
    )
    corr = pearson(true_scores, pred_scores)

    print(f"  score_mae: {mae:.3f}")
    print(f"  exact_score_accuracy: {exact_acc:.3f}")
    print(f"  score_band_accuracy: {band_acc:.3f}")
    print(f"  pearson: {corr:.3f}" if corr is not None else "  pearson: n/a")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default=str(DEFAULT_TEST), help="Test JSONL path")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Prediction JSONL output path")
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", default=None, help="LoRA adapter path")
    parser.add_argument("--baseline", action="store_true", help="Evaluate base model without adapter")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test samples")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    if not args.baseline and not args.adapter:
        raise SystemExit("--adapter를 지정하거나 --baseline을 사용하세요.")

    samples = load_jsonl(Path(args.test))
    if args.limit is not None:
        samples = samples[: args.limit]

    model, tokenizer = load_model(args)
    predictions = evaluate(samples, model, tokenizer, args)
    write_jsonl(Path(args.output), predictions)
    print_metrics(predictions)
    print(f"saved predictions: {args.output}")


if __name__ == "__main__":
    main()
