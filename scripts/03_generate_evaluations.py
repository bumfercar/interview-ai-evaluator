"""
Generate SFT evaluation samples from raw QA pairs.

Input:
    data/raw/qa_pairs.jsonl

Output:
    data/processed/sft_dataset.jsonl

The current phase uses deterministic evaluation text instead of an external LLM.
This gives us a stable first SFT dataset for JSON-format learning and score
calibration. Mentor-reviewed samples can later replace or augment these outputs.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("data/raw/qa_pairs.jsonl")
DEFAULT_OUTPUT = Path("data/processed/sft_dataset.jsonl")
DEFAULT_SEEDS = Path("data/seeds/seed_questions.jsonl")

INSTRUCTION = (
    "다음은 IT 직군 면접 질문과 지원자의 답변입니다. "
    "답변을 평가 기준에 따라 채점하고 정해진 JSON 형식으로 결과를 출력하세요."
)

QUESTION_TYPE_KEYWORDS = {
    "IT 맥락 인성·사고방식": [
        "코드 리뷰",
        "의견",
        "충돌",
        "장애",
        "문서화",
        "온콜",
        "커뮤니케이션",
        "협업",
        "일정",
        "품질",
        "피드백",
        "레거시",
        "기술 부채",
        "우선순위",
        "요구사항",
    ],
    "CS 기초": [
        "TCP",
        "HTTP",
        "HTTPS",
        "DNS",
        "프로세스",
        "스레드",
        "메모리",
        "정렬",
        "BFS",
        "DFS",
        "해시",
        "데드락",
        "운영체제",
        "네트워크",
        "인덱스",
        "트랜잭션",
        "정규화",
        "분류",
        "회귀",
        "Precision",
        "Recall",
        "F1",
        "AUC",
    ],
}

TOPIC_KEYWORDS = [
    ("database/index", ["인덱스"]),
    ("database/transaction", ["트랜잭션", "ACID", "격리"]),
    ("network/tcp", ["TCP", "handshake"]),
    ("network/http", ["HTTP", "HTTPS", "REST"]),
    ("backend/spring", ["Spring", "JPA", "Bean", "Filter", "Interceptor"]),
    ("backend/cache", ["Redis", "캐시"]),
    ("frontend/react", ["React", "Hook", "Virtual DOM", "state", "props"]),
    ("frontend/browser", ["브라우저", "DOM", "렌더링", "CORS", "Storage"]),
    ("frontend/security", ["XSS", "CSRF", "Same-Origin"]),
    ("ml/evaluation", ["Accuracy", "Precision", "Recall", "F1", "AUC", "Confusion"]),
    ("ml/training", ["오버피팅", "Gradient", "Loss", "Regularization", "Dropout"]),
    ("ml/llm", ["LLM", "RAG", "Transformer", "Attention", "토큰", "Embedding"]),
    ("devops/container", ["Docker", "컨테이너", "Kubernetes", "Pod"]),
    ("devops/operations", ["CI/CD", "모니터링", "로깅", "배포", "장애", "온콜"]),
    ("devops/network", ["로드 밸런서", "DNS", "방화벽", "CDN", "VPN"]),
]


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
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def load_seed_index(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = load_jsonl(path)
    index = {row["question"].strip(): row for row in rows}
    for row in rows:
        index.setdefault(normalize_question(row["question"]), row)
    return index


def normalize_question(question: str) -> str:
    text = re.sub(r"\([^)]*\)", "", question)
    text = re.sub(r"(핵심 원리와 필요한 이유|실무에서 주의할 점|장점과 한계|문제가 발생했을 때 어떻게 확인할지).*$", "", text)
    text = re.sub(r"(은|는)\s*왜\s+.*$", "", text)
    text = re.sub(r"(에 대해|이 무엇|가 무엇|을 사용하는 이유|를 사용하는 이유|의 차이|가 필요한 이유|이 필요한 이유).*$", "", text)
    text = re.sub(r"(설명하고|설명해주세요|말해주세요|장단점.*|방법.*).*$", "", text)
    return re.sub(r"\s+", " ", text).strip(" .?")


def find_seed(question: str, seed_index: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    exact = seed_index.get(question.strip())
    if exact:
        return exact
    normalized = normalize_question(question)
    if normalized in seed_index:
        return seed_index[normalized]
    for key, seed in seed_index.items():
        if len(key) >= 8 and (key in normalized or normalized in key):
            return seed
    return None


def detect_question_type(question: str, seed: dict[str, Any] | None = None) -> str:
    if seed:
        return seed.get("question_type", "직무 전문")
    for question_type, keywords in QUESTION_TYPE_KEYWORDS.items():
        if any(keyword.lower() in question.lower() for keyword in keywords):
            return question_type
    return "직무 전문"


def detect_topic_tag(question: str, category: str, seed: dict[str, Any] | None = None) -> str:
    if seed:
        return seed.get("topic_tag", "it/general")
    lowered = question.lower()
    for tag, keywords in TOPIC_KEYWORDS:
        if any(keyword.lower() in lowered for keyword in keywords):
            return tag
    return {
        "백엔드": "backend/general",
        "프론트엔드": "frontend/general",
        "데이터·AI/ML": "ml/general",
        "인프라·DevOps": "devops/general",
    }.get(category, "it/general")


def score_band(score: int) -> str:
    if score <= 3:
        return "1-3"
    if score <= 6:
        return "4-6"
    if score <= 8:
        return "7-8"
    return "9-10"


def find_issue(answer: str, score: int) -> str:
    if score <= 3:
        if any(word in answer for word in ["무조건", "항상", "거의 같은", "상관없", "자동"]):
            return "핵심 개념을 단순화하거나 다른 개념과 혼동하고 있습니다"
        return "질문에서 요구한 핵심 개념 설명이 부족합니다"
    if score <= 6:
        if any(word in answer for word in ["정도", "어렵", "헷갈", "부족"]):
            return "기본 방향은 일부 맞지만 설명이 피상적입니다"
        return "개념 정의는 있으나 동작 원리와 근거가 부족합니다"
    if score <= 8:
        return "핵심 개념은 적절히 설명했지만 더 깊은 운영 사례가 보강되면 좋습니다"
    return "개념, 원리, 트레이드오프를 균형 있게 설명했습니다"


def evidence_marker(answer: str, score: int) -> str:
    if score <= 3:
        if any(word in answer for word in ["무조건", "항상", "거의 같은", "상관없", "자동"]):
            return "단정적인 표현으로 잘못된 일반화를 하고 있습니다"
        if any(word in answer for word in ["헷갈", "잘 모르", "떠오르지"]):
            return "불확실성이 드러나며 근거 제시가 부족합니다"
        return "핵심 용어는 언급했지만 설명의 방향이 맞지 않습니다"
    if score <= 6:
        if any(word in answer for word in ["정도", "어렵", "부족", "헷갈"]):
            return "스스로 한계를 드러내며 깊이 있는 설명으로 이어지지 못했습니다"
        return "정의 중심 답변에 머물러 판단 기준이 약합니다"
    if score <= 8:
        if any(word in answer for word in ["주의", "트레이드오프", "비용", "지표"]):
            return "주의점이나 비용을 함께 언급해 기본 실무 감각을 보여줍니다"
        return "핵심 개념은 맞지만 근거와 예시가 더 필요합니다"
    if any(word in answer for word in ["엣지", "롤백", "지표", "메트릭", "장애", "대안"]):
        return "실패 시나리오와 운영 검증 관점까지 포함했습니다"
    return "개념 설명과 선택 기준을 함께 제시했습니다"


def make_reasoning(pair: dict[str, Any], metadata: dict[str, Any], index: int) -> str:
    score = int(pair["score"])
    category = pair["category"]
    question = pair["question"]
    topic = human_topic(question)
    issue = find_issue(pair["answer"], score)
    evidence = evidence_marker(pair["answer"], score)
    question_type = metadata["question_type"]
    topic_tag = metadata["topic_tag"]

    variants = {
        1: [
            f"{category}의 {question_type} 질문인 '{shorten(question)}'에서 {topic}의 핵심 정의가 틀렸습니다. {evidence}. {issue}. 기본 개념 확인 단계에서 탈락에 가까워 {score}점으로 평가합니다.",
            f"{topic_tag} 주제 답변으로 보기에는 개념 간 구분이 거의 되지 않습니다. {evidence}. 질문이 요구한 원리와 필요성을 설명하지 못해 {score}점이 적절합니다.",
        ],
        2: [
            f"{topic}에 관한 용어는 일부 등장하지만 설명 방향이 잘못되었습니다. {evidence}. 면접관이 추가 질문을 하면 오개념이 드러날 가능성이 높아 {score}점으로 평가합니다.",
            f"{category} 지원자의 답변으로는 기초 이해가 매우 약합니다. {issue}. 단편적인 표현은 있으나 근거와 예시가 없어 {score}점 수준입니다.",
        ],
        3: [
            f"{topic}을/를 들어본 흔적은 있지만 정확한 동작 원리나 사용 이유를 설명하지 못했습니다. {evidence}. 일부 키워드 언급을 감안해도 낮은 수준이라 {score}점입니다.",
            f"답변이 질문의 방향을 부분적으로 따라가지만 핵심 설명이 빠져 있습니다. {issue}. 기본기를 다시 확인해야 하는 답변으로 {score}점으로 봅니다.",
        ],
        4: [
            f"{topic}의 기본 목적을 어렴풋이 알고 있지만 정의 중심 설명에도 미치지 못하는 부분이 있습니다. {evidence}. 실무 적용 기준이 없어 {score}점으로 평가합니다.",
            f"{question_type} 질문에 대한 답변이 피상적입니다. {issue}. 면접관이 동작 과정이나 예시를 요구하면 답변이 부족할 가능성이 커 {score}점입니다.",
        ],
        5: [
            f"{topic}의 큰 방향은 일부 맞지만 설명이 일반론에 머뭅니다. {evidence}. 장단점과 적용 상황을 구체화하지 못해 중간 이하인 {score}점으로 평가합니다.",
            f"답변은 자연스럽지만 {topic_tag} 관점의 핵심 판단 기준이 부족합니다. 개념 정의는 일부 있으나 실무 근거가 약해 {score}점입니다.",
        ],
        6: [
            f"{topic}에 대해 기본 개념은 말했지만 깊이가 제한적입니다. {evidence}. 구체적인 예시와 트레이드오프가 보강되면 합격선에 가까워질 수 있어 {score}점입니다.",
            f"{category} 면접에서 완전히 틀린 답은 아니지만, 원리와 장애 상황 설명이 부족합니다. {issue}. 현재 수준은 {score}점으로 판단합니다.",
        ],
        7: [
            f"{topic}의 핵심 개념을 대체로 정확히 설명했습니다. {evidence}. 다만 실제 프로젝트 사례나 수치 기반 검증이 부족해 좋은 답변의 하단인 {score}점입니다.",
            f"{topic_tag} 주제에서 필요한 기본 원리와 주의점을 언급했습니다. 그러나 대안 비교와 엣지 케이스 설명이 제한적이어서 {score}점으로 평가합니다.",
        ],
        8: [
            f"{category} 질문에 대해 {topic}의 원리와 주의점을 균형 있게 설명했습니다. {evidence}. 실무 예시가 조금 더 구체적이면 9점 이상도 가능하지만 현재는 {score}점입니다.",
            f"답변이 면접관을 납득시킬 정도로 완결되어 있습니다. {issue}. 다만 운영 지표나 실패 사례가 더 있으면 상위권으로 올라갈 수 있어 {score}점입니다.",
        ],
        9: [
            f"{topic}의 개념, 동작 원리, 트레이드오프를 함께 설명했습니다. {evidence}. 구체적 수치나 실제 장애 사례가 아주 풍부한 수준은 아니지만 상위권 답변으로 {score}점입니다.",
            f"{question_type} 질문에 대해 단순 정의를 넘어 선택 기준과 운영 관점을 제시했습니다. {evidence}. 면접관이 추가 질문을 해도 방어 가능한 답변이라 {score}점입니다.",
        ],
        10: [
            f"{topic}에 대한 이해가 매우 깊습니다. {evidence}. 원리, 한계, 운영 검증, 대안까지 연결되어 있어 탑티어 답변으로 {score}점이 적절합니다.",
            f"{category} 지원자 답변으로서 {topic_tag} 주제의 핵심과 실무 판단 기준을 모두 갖췄습니다. {evidence}. 멘토가 보더라도 수정이 거의 필요 없는 {score}점 답변입니다.",
        ],
    }
    return variants[score][index % len(variants[score])]


def make_strengths(pair: dict[str, Any], metadata: dict[str, Any], index: int) -> list[str]:
    score = int(pair["score"])
    topic = human_topic(pair["question"])
    topic_tag = metadata["topic_tag"]
    if score <= 3:
        return [
            "질문에 대해 최소한 답변을 시도했습니다",
            f"{topic} 관련 용어를 일부 언급해 평가 가능한 단서는 있습니다",
        ]
    if score <= 6:
        options = [
            f"{topic}의 기본 목적을 일부 이해하고 있습니다",
            f"{topic_tag} 주제에서 완전히 벗어나지는 않았습니다",
            "답변이 짧고 구어체라 면접 상황의 자연스러움은 있습니다",
        ]
        return [options[index % len(options)], options[(index + 1) % len(options)]]
    if score <= 8:
        options = [
            f"{topic}의 핵심 개념을 비교적 정확히 설명했습니다",
            "장점이나 주의점을 함께 언급해 실무 적용 가능성을 보여줍니다",
            "개념 설명이 질문의 의도와 대체로 잘 맞습니다",
            f"{topic_tag} 관점에서 필요한 기본 판단 기준을 일부 제시했습니다",
        ]
        return [options[index % len(options)], options[(index + 2) % len(options)]]
    options = [
        f"{topic}의 원리와 필요성을 명확히 설명했습니다",
        "트레이드오프와 운영 관점까지 함께 다뤄 답변의 깊이가 좋습니다",
        "장애나 실패 상황을 고려하는 태도가 드러납니다",
        f"{topic_tag} 주제에서 상위권 답변에 필요한 선택 기준을 제시했습니다",
    ]
    return [
        options[index % len(options)],
        options[(index + 1) % len(options)],
    ]


def make_improvements(pair: dict[str, Any], metadata: dict[str, Any], index: int) -> list[str]:
    score = int(pair["score"])
    topic = human_topic(pair["question"])
    question_type = metadata["question_type"]
    if score <= 3:
        options = [
            f"{topic}의 정확한 정의와 동작 원리를 다시 학습해야 합니다",
            "비슷한 개념과의 차이를 예시로 구분해 설명하는 연습이 필요합니다",
            "틀린 내용을 확신 있게 말하기보다 모르는 범위를 명확히 인정하고 기본 정의부터 정리해야 합니다",
        ]
        return [options[index % len(options)], options[(index + 1) % len(options)]]
    if score <= 6:
        options = [
            f"{topic}이 실제로 어떻게 동작하는지 단계별로 설명해야 합니다",
            "장단점, 적용 상황, 주의사항을 구체적인 예시와 함께 보강해야 합니다",
            f"{question_type} 질문에서는 정의뿐 아니라 왜 필요한지와 언제 쓰면 안 되는지도 함께 말해야 합니다",
        ]
        return [options[index % len(options)], options[(index + 1) % len(options)]]
    if score <= 8:
        options = [
            f"{topic}을/를 실제 프로젝트나 장애 대응 사례와 더 구체적으로 연결하면 좋습니다",
            f"{topic}의 대안 기술이나 엣지 케이스까지 비교하면 상위권 답변이 됩니다",
            f"{topic}에 대한 운영 지표나 검증 방법을 덧붙이면 답변의 신뢰도가 더 높아집니다",
            f"{question_type} 관점에서 본인의 경험 범위와 판단 근거를 조금 더 명확히 제시하면 좋습니다",
        ]
        return [options[index % len(options)], options[(index + 2) % len(options)]]
    options = [
        f"{topic} 답변 구조를 조금 더 간결하게 정리하면 전달력이 더 좋아집니다",
        f"{topic}에 대해 가능하다면 수치나 실제 경험을 덧붙여 설득력을 높일 수 있습니다",
        f"멘토 검수 관점에서는 {topic} 사례의 맥락과 본인의 역할을 더 명확히 쓰면 좋습니다",
        f"{question_type} 질문에서는 이미 높은 수준이므로, 마지막에는 핵심 결론을 한 문장으로 정리하면 더 좋습니다",
    ]
    return [
        options[index % len(options)],
        options[(index + 2) % len(options)],
    ]


def human_topic(question: str) -> str:
    text = re.sub(r"\([^)]*\)", "", question)
    text = re.sub(r"(은|는)\s*왜\s+.*$", "", text)
    text = re.sub(r"(에 대해|이 무엇|가 무엇|을 사용하는 이유|를 사용하는 이유|의 차이|가 필요한 이유|이 필요한 이유).*$", "", text)
    text = re.sub(r"(설명하고|설명해주세요|말해주세요|장단점.*|방법.*).*$", "", text)
    text = text.strip(" .?")
    return text[:45] or "해당 개념"


def shorten(text: str, limit: int = 60) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def build_sample(pair: dict[str, Any], index: int, seed_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    score = int(pair["score"])
    category = pair["category"]
    question = pair["question"].strip()
    answer = pair["answer"].strip()
    band = pair.get("score_band") or score_band(score)
    seed = find_seed(question, seed_index)
    metadata = {
        "source_qa_id": pair.get("id", f"qa_{index:05d}"),
        "category": category,
        "question_type": detect_question_type(question, seed),
        "score_band": band,
        "topic_tag": detect_topic_tag(question, category, seed),
        "answer_score": score,
        "label_source": "synthetic_rule_based",
        "needs_mentor_review": True,
        "generated_by": "rule_based_eval_v2",
        "reviewed": False,
        "version": "v2",
    }

    return {
        "id": f"sft_{index:05d}",
        "metadata": metadata,
        "instruction": INSTRUCTION,
        "input": {
            "question": question,
            "answer": answer,
        },
        "output": {
            "reasoning": make_reasoning(pair, metadata, index),
            "overall_score": score,
            "strengths": make_strengths(pair, metadata, index),
            "improvements": make_improvements(pair, metadata, index),
        },
    }


def validate_samples(samples: list[dict[str, Any]]) -> list[str]:
    errors = []
    required_output = ["reasoning", "overall_score", "strengths", "improvements"]
    for sample in samples:
        output = sample.get("output", {})
        if list(output.keys()) != required_output:
            errors.append(f"{sample.get('id')}: output field order mismatch")
        score = output.get("overall_score")
        if not isinstance(score, int) or not 1 <= score <= 10:
            errors.append(f"{sample.get('id')}: invalid overall_score={score}")
        if not output.get("reasoning"):
            errors.append(f"{sample.get('id')}: empty reasoning")
        if not isinstance(output.get("strengths"), list) or not output["strengths"]:
            errors.append(f"{sample.get('id')}: empty strengths")
        if not isinstance(output.get("improvements"), list) or not output["improvements"]:
            errors.append(f"{sample.get('id')}: empty improvements")
    return errors


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_stats(samples: list[dict[str, Any]], output_path: Path) -> None:
    categories = Counter(sample["metadata"]["category"] for sample in samples)
    bands = Counter(sample["metadata"]["score_band"] for sample in samples)
    scores = Counter(sample["output"]["overall_score"] for sample in samples)
    qtypes = Counter(sample["metadata"]["question_type"] for sample in samples)

    print(f"저장 완료: {output_path}")
    print(f"총 샘플: {len(samples)}")
    print("[직군별]")
    for key, value in categories.items():
        print(f"  {key}: {value}")
    print("[점수 구간별]")
    for key, value in sorted(bands.items()):
        print(f"  {key}: {value}")
    print("[점수별]")
    for key, value in sorted(scores.items()):
        print(f"  {key}: {value}")
    print("[질문 유형별]")
    for key, value in qtypes.items():
        print(f"  {key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input QA JSONL path")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output SFT JSONL path")
    parser.add_argument("--seeds", default=str(DEFAULT_SEEDS), help="Seed question JSONL path")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    seed_index = load_seed_index(Path(args.seeds))
    pairs = load_jsonl(input_path)
    samples = [build_sample(pair, index, seed_index) for index, pair in enumerate(pairs, 1)]

    errors = validate_samples(samples)
    if errors:
        for error in errors[:20]:
            print(f"ERROR: {error}")
        raise SystemExit(f"Validation failed with {len(errors)} errors")

    write_jsonl(output_path, samples)
    print_stats(samples, output_path)


if __name__ == "__main__":
    main()
