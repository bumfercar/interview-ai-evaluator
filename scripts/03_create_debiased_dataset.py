"""
Create a less biased SFT dataset from existing QA pairs.

The first synthetic dataset made score bands too easy to infer from answer
length and repeated phrases. This script creates counterexamples while keeping
roughly the same category/score distribution:
- short but strong answers
- long but weak answers
- long answers that hide uncertainty
- partially correct answers missing key criteria
- concise expert answers

Outputs are written to data/debiased by default.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_INPUT = Path("data/raw/qa_pairs.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/debiased")
INSTRUCTION = (
    "다음은 IT 직군 면접 질문과 지원자의 답변입니다. "
    "답변을 평가 기준에 따라 채점하고 정해진 JSON 형식으로 결과를 출력하세요."
)
OUTPUT_FIELD_ORDER = ["reasoning", "overall_score", "strengths", "improvements"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def score_band(score: int) -> str:
    if score <= 3:
        return "1-3"
    if score <= 6:
        return "4-6"
    if score <= 8:
        return "7-8"
    return "9-10"


def normalize_question(question: str) -> str:
    text = re.sub(r"\([^)]*\)", "", question)
    text = re.sub(r"(설명해주세요|말해주세요|설명하고.*|장단점.*|방법.*).*$", "", text)
    return re.sub(r"\s+", " ", text).strip(" .?")


def topic_name(question: str) -> str:
    text = normalize_question(question)
    return text or "질문 주제"


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = re.findall(r".+?(?:다\.|[.!?。])(?:\s+|$)", text)
    if not parts:
        parts = re.split(r"\s{2,}", text)
    return [part.strip() for part in parts if part.strip()]


def first_sentences(text: str, count: int) -> str:
    sentences = split_sentences(text)
    return " ".join(sentences[:count]) if sentences else text.strip()


def clean_repeated_markers(text: str) -> str:
    for old in [
        "여기서 중요한 점은 ",
        "면접에서는 단순 정의보다 선택 근거, 실패 관측 지표, 대안 비교까지 말하는 것이 중요합니다.",
        "적용 전후 효과를 확인하는 과정도 필요합니다.",
        "다만 실제 수치나 장애 사례까지 제시하면 더 좋아집니다.",
        "구체적인 판단 기준과 실무 경험은 아직 부족합니다.",
        "아직 부족합니다.",
    ]:
        text = text.replace(old, "")
    return re.sub(r"\s+", " ", text).strip()


def answer_variant(row: dict[str, Any], index: int) -> str:
    score = int(row["score"])
    topic = topic_name(row["question"])
    base = clean_repeated_markers(row["answer"])
    short_base = first_sentences(base, 1)
    medium_base = first_sentences(base, 2)
    style = index % 8

    if score <= 3:
        variants = [
            f"정확히는 잘 모르겠습니다. {short_base} 질문에서 요구한 핵심 원리는 아직 설명하기 어렵습니다.",
            f"{topic}에 대해 들어본 적은 있어서 길게 설명해보면, 실무에서는 일정과 구현 속도도 중요하고 문제가 생기면 로그를 보고 고치면 된다고 생각합니다. 다만 {base} 그래서 왜 이 방식이 필요한지, 어떤 상황에서 문제가 되는지는 명확히 말하기 어렵습니다.",
            f"{topic}은 자동으로 성능이나 안정성을 높여주는 기능에 가깝다고 이해했습니다. {medium_base} 비슷한 개념과의 차이는 크지 않다고 생각해서 상황별 선택 기준은 따로 설명하지 못하겠습니다.",
            f"관련 용어는 들어봤지만 면접에서 설명하라고 하면 정의와 동작 과정을 혼동할 것 같습니다. {short_base} 예시는 떠오르지 않고, 제가 말한 내용도 정확한 근거가 있는지는 확신하기 어렵습니다.",
        ]
        return variants[style % len(variants)]

    if score <= 6:
        variants = [
            f"{topic}의 기본 목적은 어느 정도 알고 있습니다. {medium_base} 하지만 내부 동작, 예외 상황, 실제 판단 기준은 아직 부족합니다.",
            f"{topic}을 사용할 때 편리하다는 점은 이해합니다. 제 경험에서는 보통 문제가 생기면 설정을 바꾸거나 문서를 찾아보며 해결했습니다. {base} 다만 왜 그런 선택을 해야 하는지까지는 체계적으로 설명하기 어렵습니다.",
            f"핵심은 {short_base} 정도로 이해했습니다. 장점은 말할 수 있지만 단점이나 대안 비교, 운영 중 확인할 지표는 구체적으로 떠오르지 않습니다.",
            f"{topic}에 대해 설명하면, 개념 자체는 들어봤고 일부 상황에서 쓰인다는 것도 압니다. 다만 제 답변은 정의 중심이고, 질문에서 요구하는 원리나 실무 적용 기준까지는 충분히 연결하지 못합니다.",
        ]
        return variants[style % len(variants)]

    if score <= 8:
        variants = [
            f"{medium_base} 핵심 개념과 주의점은 설명할 수 있지만, 실제 장애 사례나 수치 기반 검증까지는 더 보강해야 합니다.",
            f"{topic}은 상황에 따라 장점과 비용이 함께 생깁니다. {base} 그래서 적용 여부는 요구사항, 데이터 규모, 변경 빈도, 팀의 운영 역량을 같이 보고 결정해야 합니다.",
            f"질문의 핵심은 {short_base}입니다. 추가로 잘못 사용하면 성능, 유지보수, 장애 대응 측면에서 비용이 생길 수 있어 적용 전후를 비교해야 합니다.",
            f"{topic}을 설명할 때는 정의만 말하기보다 언제 유용하고 언제 위험한지를 같이 봐야 합니다. {medium_base} 다만 더 높은 점수를 받으려면 구체적인 지표와 경험 기반 예시가 필요합니다.",
        ]
        return variants[style % len(variants)]

    variants = [
        f"{topic}의 핵심은 {short_base}입니다. 중요한 점은 이 선택이 성능, 안정성, 유지보수에 어떤 영향을 주는지 근거로 설명하고, 적용 후에는 지표로 검증해야 한다는 것입니다.",
        f"{base} 또한 운영에서는 실패 시나리오를 가정해야 합니다. 로그, 메트릭, 롤백 기준, 대안 선택지를 미리 정리하면 면접관이 추가 질문을 해도 판단 근거를 방어할 수 있습니다.",
        f"먼저 {topic}의 목적과 동작 원리를 분리해 설명하겠습니다. {medium_base} 이후에는 트레이드오프를 보고, 팀 상황과 서비스 규모에 맞는 선택인지 검증하는 것이 중요합니다.",
        f"정의만 말하면 부족합니다. {topic}은 적용 조건, 한계, 장애 시 대응까지 같이 봐야 합니다. {base} 그래서 저는 선택 기준, 관측 지표, 대안 비교를 함께 제시하겠습니다.",
    ]
    return variants[style % len(variants)]


def reasoning_for(row: dict[str, Any], index: int) -> str:
    score = int(row["score"])
    topic = topic_name(row["question"])
    category = row["category"]
    style = index % 3
    reasonings = {
        "1-3": [
            f"답변 길이와 무관하게 {topic}의 핵심 개념을 정확히 설명하지 못했습니다. 일부 용어는 보이지만 오개념과 불확실성이 커서 {score}점입니다.",
            f"{category} 면접 답변으로 보기에는 원리와 사용 이유가 빠져 있습니다. 길게 말한 부분도 근거보다 회피에 가까워 {score}점으로 평가합니다.",
            f"{topic}에 대한 이해가 단편적입니다. 비슷한 개념과 구분하지 못하고 예시도 부족하므로 낮은 점수인 {score}점이 적절합니다.",
        ],
        "4-6": [
            f"{topic}의 기본 방향은 일부 맞지만 설명이 정의 수준에 머뭅니다. 예외 상황과 판단 기준이 약해 {score}점입니다.",
            f"답변이 완전히 틀리지는 않지만 실무에서 어떻게 확인하고 선택할지까지 이어지지 못했습니다. 중간 수준인 {score}점으로 봅니다.",
            f"{category} 지원자로서 기본 용어는 알고 있으나 깊이가 부족합니다. 핵심 원리와 구체 예시가 보강되어야 하므로 {score}점입니다.",
        ],
        "7-8": [
            f"{topic}의 핵심 원리와 주의점을 대체로 잘 설명했습니다. 다만 지표, 장애 사례, 대안 비교가 더 있으면 좋아 {score}점입니다.",
            f"답변은 실무 면접에서 납득 가능한 수준입니다. 중요한 개념은 맞지만 최상위 답변에 필요한 구체적 검증 근거가 부족해 {score}점입니다.",
            f"{category} 질문에 대해 개념과 트레이드오프를 균형 있게 다뤘습니다. 일부 깊이 보완 여지가 있어 {score}점으로 평가합니다.",
        ],
        "9-10": [
            f"{topic}의 원리, 선택 기준, 한계, 운영 검증을 함께 설명했습니다. 답변 길이보다 판단 근거가 명확해 {score}점입니다.",
            f"단순 정의를 넘어 실패 시나리오와 대안 비교까지 연결했습니다. 면접관의 후속 질문에도 방어 가능한 상위권 답변이라 {score}점입니다.",
            f"{category} 실무 맥락에서 왜 이 선택을 하는지와 어떻게 검증할지를 제시했습니다. 매우 완성도 높은 답변으로 {score}점입니다.",
        ],
    }
    return reasonings[score_band(score)][style]


def strengths_for(row: dict[str, Any], score: int, index: int) -> list[str]:
    topic = topic_name(row["question"])
    pools = {
        "1-3": ["질문에 대해 답변을 시도했습니다", f"{topic} 관련 용어를 일부 언급했습니다", "모르는 부분이 드러나 평가 기준을 잡을 수 있습니다"],
        "4-6": [f"{topic}의 기본 목적을 일부 이해하고 있습니다", "완전히 주제에서 벗어나지는 않았습니다", "추가 학습 방향이 비교적 명확합니다"],
        "7-8": [f"{topic}의 핵심 개념을 비교적 정확히 설명했습니다", "주의점과 적용 상황을 함께 언급했습니다", "면접관이 기본 실무 역량을 확인할 수 있습니다"],
        "9-10": [f"{topic}의 원리와 선택 기준을 함께 설명했습니다", "트레이드오프와 운영 검증 관점을 포함했습니다", "후속 질문에도 답변을 확장할 수 있는 근거가 있습니다"],
    }
    pool = pools[score_band(score)]
    return [pool[index % len(pool)], pool[(index + 1) % len(pool)]]


def improvements_for(row: dict[str, Any], score: int, index: int) -> list[str]:
    topic = topic_name(row["question"])
    pools = {
        "1-3": [f"{topic}의 정확한 정의와 반대 개념부터 다시 정리해야 합니다", "모르는 부분을 긴 설명으로 숨기기보다 핵심 개념을 정확히 말해야 합니다", "예시와 근거 없이 단정하는 표현을 줄여야 합니다"],
        "4-6": [f"{topic}의 정의에서 멈추지 말고 동작 과정과 판단 기준을 함께 설명해야 합니다", f"{topic}의 장점뿐 아니라 한계와 예외 상황을 같이 제시해야 합니다", f"{topic}을 실무에서 어떤 지표나 사례로 검증할지 보강해야 합니다"],
        "7-8": [f"{topic}과 관련된 구체적인 장애 사례나 수치 기반 검증을 추가하면 더 좋습니다", f"{topic}의 대안과 비교해 왜 이 선택이 적절한지 더 명확히 말하면 좋습니다", f"{topic}을 운영 환경에서 확인할 관측 지표를 함께 제시하면 완성도가 올라갑니다"],
        "9-10": [f"{topic} 답변 구조를 더 간결하게 정리하면 전달력이 좋아집니다", f"{topic}과 관련된 실제 프로젝트 수치가 있다면 설득력이 더 높아집니다", f"{topic}을 면접 시간에 맞게 핵심부터 말하는 연습을 하면 좋습니다"],
    }
    pool = pools[score_band(score)]
    return [pool[index % len(pool)], pool[(index + 1) % len(pool)]]


def make_sft_row(row: dict[str, Any], index: int, answer: str) -> dict[str, Any]:
    score = int(row["score"])
    output = {
        "reasoning": reasoning_for(row, index),
        "overall_score": score,
        "strengths": strengths_for(row, score, index),
        "improvements": improvements_for(row, score, index),
    }
    return {
        "id": f"sft_debiased_{index:05d}",
        "metadata": {
            "source_qa_id": row["id"],
            "category": row["category"],
            "question_type": row.get("question_type", "unknown"),
            "score_band": score_band(score),
            "topic_tag": row.get("topic_tag", "it/general"),
            "answer_score": score,
            "label_source": "synthetic_debiased_rule_based",
            "needs_mentor_review": True,
            "generated_by": "debiased_rule_based_v1",
            "reviewed": False,
            "version": "debiased_v1",
        },
        "instruction": INSTRUCTION,
        "input": {"question": row["question"], "answer": answer},
        "output": {key: output[key] for key in OUTPUT_FIELD_ORDER},
    }


def make_qa_row(row: dict[str, Any], index: int, answer: str) -> dict[str, Any]:
    score = int(row["score"])
    return {
        "category": row["category"],
        "score_band": score_band(score),
        "question": row["question"],
        "answer": answer,
        "score": score,
        "generated_by": "debiased_rule_based_v1",
        "reviewed": False,
        "id": f"qa_debiased_{index:05d}",
    }


def split_by_question(rows: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["metadata"]["category"], row["input"]["question"])].append(row)

    category_groups: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for (category, _), group_rows in groups.items():
        category_groups[category].append(group_rows)

    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for grouped_rows in category_groups.values():
        rng.shuffle(grouped_rows)
        total = sum(len(group) for group in grouped_rows)
        train_target = total * 0.8
        val_target = total * 0.1
        counts = {"train": 0, "val": 0, "test": 0}
        for group in grouped_rows:
            if counts["train"] < train_target:
                target = "train"
            elif counts["val"] < val_target:
                target = "val"
            else:
                target = "test"
            splits[target].extend(group)
            counts[target] += len(group)

    for name in splits:
        splits[name].sort(key=lambda row: row["id"])
    return splits


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9가-힣]+", text))


def print_stats(sft_rows: list[dict[str, Any]], splits: dict[str, list[dict[str, Any]]]) -> None:
    print("[dataset]")
    print("total:", len(sft_rows))
    print("score_band:", dict(sorted(Counter(row["metadata"]["score_band"] for row in sft_rows).items())))
    print("category:", dict(sorted(Counter(row["metadata"]["category"] for row in sft_rows).items())))
    print("\n[answer length by band]")
    for band in ["1-3", "4-6", "7-8", "9-10"]:
        lengths = [word_count(row["input"]["answer"]) for row in sft_rows if row["metadata"]["score_band"] == band]
        print(band, {"avg": round(mean(lengths), 1), "min": min(lengths), "max": max(lengths)})
    print("\n[splits]")
    for name, rows in splits.items():
        print(name, len(rows), dict(sorted(Counter(row["metadata"]["score_band"] for row in rows).items())))
    train_q = {row["input"]["question"] for row in splits["train"]}
    test_q = {row["input"]["question"] for row in splits["test"]}
    val_q = {row["input"]["question"] for row in splits["val"]}
    print("question overlap train/test:", len(train_q & test_q))
    print("question overlap train/val:", len(train_q & val_q))
    print("question overlap val/test:", len(val_q & test_q))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_rows = load_jsonl(Path(args.input))
    qa_rows = []
    sft_rows = []
    for index, row in enumerate(source_rows, 1):
        answer = answer_variant(row, index)
        qa_rows.append(make_qa_row(row, index, answer))
        sft_rows.append(make_sft_row(row, index, answer))

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "qa_pairs.jsonl", qa_rows)
    write_jsonl(output_dir / "sft_dataset.jsonl", sft_rows)

    splits = split_by_question(sft_rows, seed=args.seed)
    for name, split_rows in splits.items():
        write_jsonl(output_dir / "splits" / f"{name}.jsonl", split_rows)

    print_stats(sft_rows, splits)
    print(f"saved: {output_dir}")


if __name__ == "__main__":
    main()
