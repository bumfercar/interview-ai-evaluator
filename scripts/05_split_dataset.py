"""
Split the processed SFT dataset into train/val/test JSONL files.

The split is stratified by:
    metadata.category + metadata.score_band

Default ratio:
    train : val : test = 8 : 1 : 1

Input:
    data/processed/sft_dataset.jsonl

Outputs:
    data/splits/train.jsonl
    data/splits/val.jsonl
    data/splits/test.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("data/processed/sft_dataset.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/splits")
DEFAULT_SEED = 42


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


def stratify_key(row: dict[str, Any]) -> tuple[str, str]:
    metadata = row["metadata"]
    return metadata["category"], metadata["score_band"]


def split_group(
    rows: list[dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    n = len(rows)
    train_n = round(n * train_ratio)
    val_n = round(n * val_ratio)
    test_n = n - train_n - val_n
    if test_n < 0:
        raise ValueError(f"Invalid split sizes for group size {n}")
    return rows[:train_n], rows[train_n : train_n + val_n], rows[train_n + val_n :]


def split_dataset(
    rows: list[dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[stratify_key(row)].append(row)

    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}

    for key in sorted(groups):
        group = groups[key]
        rng.shuffle(group)
        train_rows, val_rows, test_rows = split_group(group, train_ratio, val_ratio)
        splits["train"].extend(train_rows)
        splits["val"].extend(val_rows)
        splits["test"].extend(test_rows)

    for name in splits:
        splits[name].sort(key=lambda row: row["id"])

    return splits


def counter_by(rows: list[dict[str, Any]], field: str) -> Counter:
    if field == "score":
        return Counter(row["output"]["overall_score"] for row in rows)
    return Counter(row["metadata"][field] for row in rows)


def print_split_stats(splits: dict[str, list[dict[str, Any]]]) -> None:
    for name in ["train", "val", "test"]:
        rows = splits[name]
        print(f"[{name}] {len(rows)}개")
        print("  category:", dict(sorted(counter_by(rows, "category").items())))
        print("  score_band:", dict(sorted(counter_by(rows, "score_band").items())))
        print("  score:", dict(sorted(counter_by(rows, "score").items())))

    print("[strata]")
    for name in ["train", "val", "test"]:
        strata = Counter(stratify_key(row) for row in splits[name])
        values = sorted(strata.items())
        print(f"  {name}:")
        for (category, band), count in values:
            print(f"    {category} / {band}: {count}")


def assert_no_overlap(splits: dict[str, list[dict[str, Any]]]) -> None:
    ids_by_split = {name: {row["id"] for row in rows} for name, rows in splits.items()}
    names = list(ids_by_split)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = ids_by_split[left] & ids_by_split[right]
            if overlap:
                raise ValueError(f"{left}와 {right} split에 중복 ID가 있습니다: {sorted(overlap)[:5]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input SFT dataset JSONL path")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output split directory")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    splits = split_dataset(
        rows,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    assert_no_overlap(splits)

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "train.jsonl", splits["train"])
    write_jsonl(output_dir / "val.jsonl", splits["val"])
    write_jsonl(output_dir / "test.jsonl", splits["test"])

    print_split_stats(splits)
    print(f"저장 완료: {output_dir}")


if __name__ == "__main__":
    main()
