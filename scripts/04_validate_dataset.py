"""
Validate the processed SFT dataset before splitting/training.

Default input:
    data/processed/sft_dataset.jsonl

This script checks:
- JSONL parseability
- required schema and output field order
- score / score_band consistency
- category, score band, question type distributions
- text length statistics
- duplicate question/answer/output patterns
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_INPUT = Path("data/processed/sft_dataset.jsonl")

EXPECTED_OUTPUT_ORDER = ["reasoning", "overall_score", "strengths", "improvements"]
EXPECTED_METADATA_FIELDS = {
    "source_qa_id",
    "category",
    "question_type",
    "score_band",
    "topic_tag",
    "answer_score",
    "label_source",
    "needs_mentor_review",
    "generated_by",
    "reviewed",
    "version",
}
EXPECTED_CATEGORIES = {
    "백엔드": 600,
    "프론트엔드": 400,
    "데이터·AI/ML": 600,
    "인프라·DevOps": 400,
}
EXPECTED_SCORE_BANDS = {
    "1-3": 500,
    "4-6": 500,
    "7-8": 500,
    "9-10": 500,
}


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows = []
    errors = []
    if not path.exists():
        return rows, [f"입력 파일이 없습니다: {path}"]

    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                errors.append(f"{path}:{line_no}: JSON 파싱 실패: {exc}")
    return rows, errors


def expected_band(score: int) -> str:
    if score <= 3:
        return "1-3"
    if score <= 6:
        return "4-6"
    if score <= 8:
        return "7-8"
    return "9-10"


def validate_row(row: dict[str, Any], index: int) -> list[str]:
    errors = []
    prefix = row.get("id", f"line_{index}")

    for field in ["id", "metadata", "instruction", "input", "output"]:
        if field not in row:
            errors.append(f"{prefix}: 필수 필드 누락: {field}")
            return errors

    metadata = row["metadata"]
    if not isinstance(metadata, dict):
        errors.append(f"{prefix}: metadata가 객체가 아님")
        return errors

    missing_meta = EXPECTED_METADATA_FIELDS - set(metadata)
    if missing_meta:
        errors.append(f"{prefix}: metadata 필드 누락: {sorted(missing_meta)}")

    input_obj = row["input"]
    if not isinstance(input_obj, dict):
        errors.append(f"{prefix}: input이 객체가 아님")
    else:
        for field in ["question", "answer"]:
            if not isinstance(input_obj.get(field), str) or not input_obj[field].strip():
                errors.append(f"{prefix}: input.{field}가 비어 있거나 문자열이 아님")

    output = row["output"]
    if not isinstance(output, dict):
        errors.append(f"{prefix}: output이 객체가 아님")
        return errors

    if list(output.keys()) != EXPECTED_OUTPUT_ORDER:
        errors.append(f"{prefix}: output 필드 순서 오류: {list(output.keys())}")

    score = output.get("overall_score")
    if not isinstance(score, int) or not 1 <= score <= 10:
        errors.append(f"{prefix}: overall_score 범위 오류: {score}")
    else:
        if metadata.get("answer_score") != score:
            errors.append(f"{prefix}: metadata.answer_score와 output.overall_score 불일치")
        if metadata.get("score_band") != expected_band(score):
            errors.append(f"{prefix}: score_band와 overall_score 불일치")

    if not isinstance(output.get("reasoning"), str) or len(output["reasoning"].strip()) < 30:
        errors.append(f"{prefix}: reasoning이 너무 짧거나 문자열이 아님")

    for field in ["strengths", "improvements"]:
        value = output.get(field)
        if not isinstance(value, list) or len(value) < 2:
            errors.append(f"{prefix}: output.{field}는 최소 2개 리스트여야 함")
        elif not all(isinstance(item, str) and item.strip() for item in value):
            errors.append(f"{prefix}: output.{field}에 빈 항목이 있음")

    return errors


def length_stats(values: list[str]) -> dict[str, float | int]:
    lengths = [len(value) for value in values]
    if not lengths:
        return {"min": 0, "avg": 0, "max": 0}
    return {
        "min": min(lengths),
        "avg": round(mean(lengths), 1),
        "max": max(lengths),
    }


def print_counter(title: str, counter: Counter, expected: dict[str, int] | None = None) -> list[str]:
    warnings = []
    print(title)
    for key, count in sorted(counter.items()):
        suffix = ""
        if expected and key in expected and expected[key] != count:
            suffix = f"  EXPECTED {expected[key]}"
            warnings.append(f"{title} {key}: expected {expected[key]}, got {count}")
        print(f"  {key}: {count}{suffix}")
    if expected:
        for key, count in expected.items():
            if key not in counter:
                warnings.append(f"{title} {key}: expected {count}, got 0")
                print(f"  {key}: 0  EXPECTED {count}")
    return warnings


def analyze(rows: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    errors = []
    warnings = []

    for index, row in enumerate(rows, 1):
        errors.extend(validate_row(row, index))

    if errors:
        return errors, warnings

    categories = Counter(row["metadata"]["category"] for row in rows)
    score_bands = Counter(row["metadata"]["score_band"] for row in rows)
    scores = Counter(row["output"]["overall_score"] for row in rows)
    question_types = Counter(row["metadata"]["question_type"] for row in rows)
    topic_tags = Counter(row["metadata"]["topic_tag"] for row in rows)

    questions = [row["input"]["question"] for row in rows]
    answers = [row["input"]["answer"] for row in rows]
    reasonings = [row["output"]["reasoning"] for row in rows]
    strength_tuples = [tuple(row["output"]["strengths"]) for row in rows]
    improvement_tuples = [tuple(row["output"]["improvements"]) for row in rows]

    print(f"총 샘플: {len(rows)}")
    if len(rows) != 2000:
        warnings.append(f"총 샘플 수가 2000이 아님: {len(rows)}")

    warnings.extend(print_counter("[직군별]", categories, EXPECTED_CATEGORIES))
    warnings.extend(print_counter("[점수 구간별]", score_bands, EXPECTED_SCORE_BANDS))
    print_counter("[점수별]", scores)
    print_counter("[질문 유형별]", question_types)
    print("[topic_tag 상위 15개]")
    for tag, count in topic_tags.most_common(15):
        print(f"  {tag}: {count}")

    print("[길이 통계]")
    print(f"  question: {length_stats(questions)}")
    print(f"  answer: {length_stats(answers)}")
    print(f"  reasoning: {length_stats(reasonings)}")

    duplicate_report = {
        "question_unique": len(set(questions)),
        "answer_unique": len(set(answers)),
        "reasoning_unique": len(set(reasonings)),
        "strengths_unique": len(set(strength_tuples)),
        "improvements_unique": len(set(improvement_tuples)),
    }
    print("[중복/다양성]")
    for key, value in duplicate_report.items():
        print(f"  {key}: {value}")

    by_band = defaultdict(list)
    for row in rows:
        by_band[row["metadata"]["score_band"]].append(row)

    print("[점수 구간별 다양성]")
    for band in ["1-3", "4-6", "7-8", "9-10"]:
        band_rows = by_band[band]
        band_answers = {row["input"]["answer"] for row in band_rows}
        band_reasonings = {row["output"]["reasoning"] for row in band_rows}
        band_strengths = {tuple(row["output"]["strengths"]) for row in band_rows}
        band_improvements = {tuple(row["output"]["improvements"]) for row in band_rows}
        answer_lengths = [len(row["input"]["answer"]) for row in band_rows]
        print(
            f"  {band}: answers={len(band_answers)}, reasoning={len(band_reasonings)}, "
            f"strengths={len(band_strengths)}, improvements={len(band_improvements)}, "
            f"answer_avg_len={round(mean(answer_lengths), 1)}"
        )

        if len(band_reasonings) < len(band_rows) * 0.25:
            warnings.append(f"{band}: reasoning 다양성이 낮음 ({len(band_reasonings)}/{len(band_rows)})")
        if len(band_strengths) < len(band_rows) * 0.25:
            warnings.append(f"{band}: strengths 다양성이 낮음 ({len(band_strengths)}/{len(band_rows)})")
        if len(band_improvements) < len(band_rows) * 0.25:
            warnings.append(f"{band}: improvements 다양성이 낮음 ({len(band_improvements)}/{len(band_rows)})")

    for left, right in [("1-3", "4-6"), ("4-6", "7-8"), ("7-8", "9-10")]:
        overlap = {
            row["input"]["answer"] for row in by_band[left]
        } & {
            row["input"]["answer"] for row in by_band[right]
        }
        print(f"[답변 overlap] {left} vs {right}: {len(overlap)}")
        if overlap:
            warnings.append(f"{left}와 {right} 사이 동일 답변 {len(overlap)}개")

    needs_review = sum(bool(row["metadata"].get("needs_mentor_review")) for row in rows)
    print(f"[멘토 검수 필요 플래그] {needs_review}/{len(rows)}")
    if needs_review != len(rows):
        warnings.append("needs_mentor_review가 false인 샘플이 있음")

    return errors, warnings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="SFT dataset JSONL path")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures")
    args = parser.parse_args()

    path = Path(args.input)
    rows, parse_errors = load_jsonl(path)
    if parse_errors:
        for error in parse_errors[:20]:
            print(f"ERROR: {error}")
        raise SystemExit(f"JSONL 파싱 실패: {len(parse_errors)}개")

    errors, warnings = analyze(rows)

    if warnings:
        print("[WARNINGS]")
        for warning in warnings:
            print(f"  - {warning}")

    if errors:
        print("[ERRORS]")
        for error in errors[:50]:
            print(f"  - {error}")
        raise SystemExit(f"검증 실패: {len(errors)}개 오류")

    if args.strict and warnings:
        raise SystemExit(f"strict 검증 실패: {len(warnings)}개 경고")

    print("검증 통과")


if __name__ == "__main__":
    main()
