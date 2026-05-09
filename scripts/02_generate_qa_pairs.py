"""
02_generate_qa_pairs.py
시드 질문 × 점수 등급 조합으로 QA 쌍 생성 프롬프트를 만들고,
ChatGPT 응답을 파싱해 data/raw/qa_pairs.jsonl에 저장.

[사용법]
1단계 - 프롬프트 파일 생성:
    python scripts/02_generate_qa_pairs.py --gen-prompts

2단계 - ChatGPT 응답 저장:
    data/raw/qa_responses/ 폴더에 응답 파일 저장
    파일명 형식: {role_key}_{score_band}.json  (예: backend_1-3.json)

3단계 - 응답 파싱 & 합치기:
    python scripts/02_generate_qa_pairs.py --parse
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

SEEDS_PATH = Path("data/seeds/seed_questions.jsonl")
PROMPTS_DIR = Path("data/raw/qa_prompts")
RESPONSES_DIR = Path("data/raw/qa_responses")
OUTPUT_PATH = Path("data/raw/qa_pairs.jsonl")

# 직군별 목표 QA 쌍 수 (총 2,000건)
ROLE_TARGETS = {
    "백엔드":        600,
    "프론트엔드":    400,
    "데이터·AI/ML":  600,
    "인프라·DevOps": 400,
}

ROLE_KEYS = {
    "백엔드":        "backend",
    "프론트엔드":    "frontend",
    "데이터·AI/ML":  "data_ai",
    "인프라·DevOps": "devops",
}

SCORE_BANDS = ["1-3", "4-6", "7-8", "9-10"]

SCORE_DESCRIPTIONS = {
    "1-3": (
        "매우 부족한 수준. 면접 불합격.\n"
        "- 핵심 개념을 잘못 이해하거나 전혀 모르는 상태\n"
        "- 틀린 정보를 자신 있게 말하거나, 전혀 엉뚱한 내용으로 답변\n"
        "- '잘 모르겠지만', '아마도', '~인 것 같습니다' 같은 불확실한 표현 남발\n"
        "- 예시나 근거 없음. 개념과 개념을 혼동\n"
        "- 예: 'TCP는 빠른 전송을 위해 연결 없이 데이터를 보내는 방식입니다' (TCP/UDP 혼동)"
    ),
    "4-6": (
        "보통 수준. 합격선 미달.\n"
        "- 개념은 대략 알지만 정확하지 않거나 피상적\n"
        "- 교과서적인 정의만 말하고 왜 필요한지, 어떻게 동작하는지 설명 못 함\n"
        "- 실무 경험이나 구체적 예시 전혀 없음\n"
        "- 답변이 두루뭉술하고 면접관이 '그래서요?' 라고 되물을 수준\n"
        "- 예: '인덱스는 조회를 빠르게 해주는 기능입니다. 자주 조회하는 컬럼에 사용합니다'"
    ),
    "7-8": (
        "좋은 수준. 합격선 이상.\n"
        "- 핵심 개념을 정확히 이해하고 명확하게 설명\n"
        "- 동작 원리나 이유를 설명할 수 있음\n"
        "- 간단한 실무 예시나 경험을 한두 가지 언급\n"
        "- 면접관이 납득할 수 있는 완결된 답변\n"
        "- 예: '인덱스는 B-Tree 구조로 정렬된 별도 공간입니다. 조회는 빠르지만 INSERT/UPDATE 시 "
        "인덱스도 갱신되므로 쓰기 성능이 저하됩니다. 실무에서는 카디널리티가 높은 컬럼 위주로 사용합니다'"
    ),
    "9-10": (
        "매우 뛰어난 수준. 탑티어.\n"
        "- 개념을 깊이 이해하고 내부 동작 원리까지 설명\n"
        "- 트레이드오프, 한계, 주의사항을 스스로 언급\n"
        "- 실무에서 겪은 구체적 상황이나 수치 기반 예시 포함\n"
        "- 엣지케이스나 대안 방법까지 언급하며 면접관을 납득시키는 수준\n"
        "- 예: '인덱스는 B-Tree 기반으로 O(log n) 탐색이 가능합니다. 다만 복합 인덱스는 "
        "선두 컬럼 기준으로만 탐색되므로 컬럼 순서가 중요합니다. 실제로 주문 서비스에서 "
        "user_id + created_at 복합 인덱스를 설계했는데, user_id 단독 조회는 인덱스를 타지만 "
        "created_at 단독 조회는 풀스캔이 발생해 별도 인덱스를 추가했습니다'"
    ),
}


def load_seeds_by_role() -> dict:
    by_role = defaultdict(list)
    with open(SEEDS_PATH, encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            by_role[q["category"]].append(q)
    return by_role


# 점수 구간별 답변 최소 길이 및 작성 지침
BAND_RULES = {
    "1-3":  {"min_chars": 100, "guide": "핵심 개념을 틀리거나 오해한 내용을 포함해 100~250자로 작성. 짧고 틀린 답이 자연스러움."},
    "4-6":  {"min_chars": 150, "guide": "개념은 대략 맞지만 피상적이고 예시 없음. 150~300자로 작성."},
    "7-8":  {"min_chars": 200, "guide": "정확한 설명 + 간단한 예시나 실무 언급. 200~400자로 작성."},
    "9-10": {"min_chars": 250, "guide": "정확한 설명 + 구체적 예시 + 트레이드오프 또는 엣지케이스 언급. 250~500자로 작성."},
}


def build_prompt(role: str, score_band: str, seed_questions: list, target_count: int) -> str:
    score_desc = SCORE_DESCRIPTIONS[score_band]
    lo, hi = score_band.split("-")
    rule = BAND_RULES[score_band]

    seed_list = "\n".join(
        f'{i+1}. [{q["question_type"]}] {q["question"]}'
        for i, q in enumerate(seed_questions)
    )

    prompt = f"""당신은 국내 IT 기업 시니어 개발자 출신 면접관입니다.
{role} 직군 지원자의 면접 QA 쌍을 정확히 {target_count}개 생성하세요.

## 점수 기준: {score_band}점 / 10점 만점
{score_desc}

## 답변 작성 필수 규칙
1. 길이: {rule["guide"]}
2. 위 점수 기준의 특징을 답변에 충실히 반영할 것
   - {score_band}점 답변이 다른 점수 구간 답변과 명확히 구별되어야 함
3. 자연스러운 한국어 구어체 (실제 면접처럼 말하는 투)
4. score는 생성한 답변의 품질을 보고 {lo}~{hi} 사이에서 직접 판단해 배정
5. 질문은 아래 시드를 그대로 쓰거나 변형하거나 유사한 새 질문을 만들어도 됨

## 참고 시드 질문
{seed_list}

## 출력 형식 (JSON 배열만, 다른 텍스트 없이)
[
  {{
    "question": "면접 질문",
    "answer": "지원자 답변",
    "score": {lo}
  }},
  ...
]
정확히 {target_count}개 생성."""

    return prompt


def gen_prompts():
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    by_role = load_seeds_by_role()

    print("프롬프트 파일 생성 중...\n")
    for role, total in ROLE_TARGETS.items():
        per_band = total // len(SCORE_BANDS)  # 각 등급당 목표 수
        role_key = ROLE_KEYS[role]
        seeds = by_role[role]

        for band in SCORE_BANDS:
            prompt = build_prompt(role, band, seeds, per_band)
            filename = PROMPTS_DIR / f"{role_key}_{band}.txt"
            filename.write_text(prompt, encoding="utf-8")
            print(f"  ✅ {filename.name}  ({per_band}개 요청)")

    print(f"\n총 {len(ROLE_TARGETS) * len(SCORE_BANDS)}개 프롬프트 파일 생성됨")
    print(f"저장 위치: {PROMPTS_DIR}/")
    print("\n[다음 단계]")
    print("1. 각 .txt 파일 내용을 ChatGPT에 붙여넣기")
    print(f"2. 응답을 {RESPONSES_DIR}/{{role_key}}_{{score_band}}.json 으로 저장")
    print("   예: data/raw/qa_responses/backend_1-3.json")
    print("3. python scripts/02_generate_qa_pairs.py --parse 실행")


def parse_responses():
    if not RESPONSES_DIR.exists():
        print(f"❌ 응답 폴더가 없습니다: {RESPONSES_DIR}")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_pairs = []
    errors = []
    role_key_to_label = {v: k for k, v in ROLE_KEYS.items()}

    for filepath in sorted(RESPONSES_DIR.glob("*.json")):
        # 파일명에서 role_key, score_band 파싱
        stem = filepath.stem  # e.g. "backend_1-3"
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            errors.append(f"파일명 형식 오류: {filepath.name}")
            continue

        role_key, score_band = parts
        role_label = role_key_to_label.get(role_key)
        if not role_label:
            errors.append(f"알 수 없는 직군 키: {role_key} ({filepath.name})")
            continue

        try:
            raw = filepath.read_text(encoding="utf-8").strip()
            # JSON 배열 추출 (앞뒤 텍스트 제거)
            match = re.search(r'\[.*\]', raw, re.DOTALL)
            if not match:
                errors.append(f"JSON 배열 없음: {filepath.name}")
                continue
            pairs = json.loads(match.group())
        except Exception as e:
            errors.append(f"파싱 실패 {filepath.name}: {e}")
            continue

        lo_int = int(score_band.split("-")[0])
        hi_int = int(score_band.split("-")[1])
        min_len = BAND_RULES.get(score_band, {}).get("min_chars", 100)

        valid = 0
        for pair in pairs:
            if not all(k in pair for k in ["question", "answer", "score"]):
                continue
            # 구간별 최소 길이 필터
            if len(pair["answer"]) < min_len:
                continue
            # 최대 길이 필터
            if len(pair["answer"]) > 700:
                continue
            # 점수 범위 보정
            score = max(lo_int, min(hi_int, int(pair["score"])))
            all_pairs.append({
                "category": role_label,
                "score_band": score_band,
                "question": pair["question"].strip(),
                "answer": pair["answer"].strip(),
                "score": score,
                "generated_by": "gpt-4o",
                "reviewed": False,
            })
            valid += 1
        print(f"  {filepath.name}: {valid}/{len(pairs)}개 유효")

    # ID 부여 후 저장
    for i, pair in enumerate(all_pairs):
        pair["id"] = f"qa_{i+1:05d}"

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    # 통계
    from collections import Counter
    print(f"\n총 {len(all_pairs)}개 QA 쌍 저장 → {OUTPUT_PATH}")
    print("\n[직군별]")
    for role, cnt in Counter(p["category"] for p in all_pairs).items():
        print(f"  {role}: {cnt}개")
    print("\n[점수 등급별]")
    for band, cnt in sorted(Counter(p["score_band"] for p in all_pairs).items()):
        print(f"  {band}점: {cnt}개")

    if errors:
        print("\n[오류]")
        for e in errors:
            print(f"  ❌ {e}")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--gen-prompts", action="store_true", help="ChatGPT용 프롬프트 파일 생성")
    group.add_argument("--parse", action="store_true", help="ChatGPT 응답 파일 파싱 & 합치기")
    args = parser.parse_args()

    if args.gen_prompts:
        gen_prompts()
    else:
        parse_responses()


if __name__ == "__main__":
    main()
