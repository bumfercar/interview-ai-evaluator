"""
01_make_seed_questions.py
시드 질문 200개를 GPT-4o로 자동 생성하여 data/seeds/seed_questions.jsonl에 저장.

실행:
    python scripts/01_make_seed_questions.py --config configs/data_gen.yaml
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

# ── 직군별 직무 전문 토픽 ────────────────────────────────────────────────────
ROLE_SPECIFIC_TOPICS = {
    "백엔드": [
        "RESTful API 설계", "데이터베이스 인덱스 최적화", "동시성 처리(스레드/비동기)",
        "캐싱 전략(Redis)", "MSA vs 모놀리식", "JWT 인증·인가", "SQL vs NoSQL",
        "트랜잭션과 격리 수준", "서버 사이드 렌더링", "배치 처리 설계",
    ],
    "프론트엔드": [
        "Virtual DOM 동작 원리", "브라우저 렌더링 파이프라인", "CSR vs SSR vs SSG",
        "웹 성능 최적화(LCP·CLS·FID)", "상태 관리(Redux·Zustand·Recoil)",
        "CORS 동작 원리", "웹 접근성(WCAG)", "CSS-in-JS vs CSS Modules",
        "번들러(Webpack·Vite) 동작 원리", "TypeScript 타입 시스템",
    ],
    "데이터·AI/ML": [
        "과적합 방지 기법", "경사 하강법 변형(Adam·SGD)", "트랜스포머 어텐션 메커니즘",
        "데이터 전처리·피처 엔지니어링", "모델 평가 지표(F1·AUC·PR 곡선)",
        "배치 정규화 vs 레이어 정규화", "파이프라인 설계(MLOps)", "A/B 테스트 설계",
        "클래스 불균형 처리", "분산 학습(DDP·FSDP)",
    ],
    "인프라·DevOps": [
        "컨테이너와 VM 차이(Docker)", "쿠버네티스 스케줄링", "CI/CD 파이프라인 설계",
        "IaC(Terraform·Ansible)", "모니터링·알림 체계(Prometheus·Grafana)",
        "블루-그린 vs 카나리 배포", "네트워크 보안(VPC·보안그룹)", "로그 집계(ELK)",
        "오토스케일링 전략", "장애 복구(DR) 설계",
    ],
}

# ── IT 맥락 인성 상황 ────────────────────────────────────────────────────────
IT_ATTITUDE_SITUATIONS = [
    "코드 리뷰에서 동료의 코드에 심각한 버그를 발견했을 때",
    "기술 부채가 쌓인 레거시 코드를 유지보수해야 할 때",
    "예상치 못한 프로덕션 장애가 발생했을 때",
    "팀원과 기술 스택 선택에 대해 의견이 충돌할 때",
    "마감 기한이 촉박해 코드 품질과 속도 사이에서 선택해야 할 때",
    "새로운 기술을 빠르게 학습해 프로젝트에 적용해야 할 때",
    "페어 프로그래밍에서 실력 차이가 크게 나는 동료와 협업할 때",
    "본인이 설계한 아키텍처가 성능 문제로 비판받을 때",
]


def build_prompt(role: str, category: str, count: int, score_band: str = None) -> str:
    """GPT-4o에게 전달할 시드 질문 생성 프롬프트를 구성합니다."""

    role_topics = ROLE_SPECIFIC_TOPICS.get(role, [])

    if category == "CS 기초":
        topic_hint = "자료구조, 알고리즘, 운영체제, 네트워크, 데이터베이스 중 실제 면접에서 자주 나오는 심층 주제"
    elif category == "직무 전문":
        topic_hint = f"{role} 직군 실무에서 중요한 주제: {', '.join(role_topics[:6])}"
    else:  # IT 맥락 인성
        situations = random.sample(IT_ATTITUDE_SITUATIONS, min(count, len(IT_ATTITUDE_SITUATIONS)))
        topic_hint = f"IT 현업 상황: {', '.join(situations)}"

    prompt = f"""당신은 국내 IT 기업 면접관입니다.
{role} 직군 지원자에게 실제로 출제하는 "{category}" 면접 질문 {count}개를 생성하세요.

조건:
1. 각 질문은 서로 다른 주제를 다룰 것
2. 실제 면접에서 나올 법한 구체적이고 기술적인 질문일 것
3. 주제 힌트: {topic_hint}
{"4. IT 환경(코드 리뷰·기술 의사결정·장애 대응·학습 방식 등) 맥락이 반드시 포함될 것 (일반 인성 질문 제외)" if category == "IT 맥락 인성·사고방식" else ""}

출력 형식 (JSON 배열만 출력, 다른 텍스트 없음):
[
  "질문 내용 1",
  "질문 내용 2",
  ...
]"""
    return prompt


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(min=1, max=10),
    retry=retry_if_exception_type(Exception),
)
def call_gpt(client: OpenAI, model: str, prompt: str, max_tokens: int) -> list[str]:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.9,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content.strip()
    # JSON 객체로 감싸서 반환하는 경우 대응
    parsed = json.loads(raw)
    if isinstance(parsed, list):
        return parsed
    # {"questions": [...]} 같은 형태 대응
    for v in parsed.values():
        if isinstance(v, list):
            return v
    raise ValueError(f"예상치 못한 GPT 응답 형식: {raw[:200]}")


def generate_seed_questions(config: dict, client: OpenAI) -> list[dict]:
    """config에 정의된 분포에 따라 시드 질문을 생성합니다."""
    model = config["openai"]["model"]
    seeds_cfg = config["seed_questions"]
    roles_cfg = config["roles"]
    cats_cfg = config["categories"]

    # 직군 × 카테고리 조합별 목표 건수 계산
    plan = []
    for role_key, role_info in roles_cfg.items():
        role_label = role_info["label"]
        role_count = seeds_cfg["per_role"][role_key]  # e.g. 60
        for cat_key, cat_info in cats_cfg.items():
            cat_label = cat_info["label"]
            cat_count = round(role_count * cat_info["ratio"])
            if cat_count > 0:
                plan.append({
                    "role_key": role_key,
                    "role": role_label,
                    "category": cat_label,
                    "count": cat_count,
                })

    print(f"[계획] {len(plan)}개 조합, 총 목표: {sum(p['count'] for p in plan)}개 질문")

    all_questions = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for i, item in enumerate(plan, 1):
        role = item["role"]
        category = item["category"]
        count = item["count"]

        print(f"  [{i:02d}/{len(plan)}] {role} / {category} — {count}개 생성 중...", end=" ", flush=True)

        prompt = build_prompt(role, category, count)

        # JSON 배열을 직접 반환하도록 프롬프트 수정 (response_format=json_object 대응)
        wrapped_prompt = prompt + '\n\n반드시 {"questions": [...]} 형태의 JSON으로 출력하세요.'

        try:
            client_response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": wrapped_prompt}],
                max_tokens=config["openai"]["max_tokens_qa"],
                temperature=config["openai"]["temperature"],
                response_format={"type": "json_object"},
            )
            raw = client_response.choices[0].message.content.strip()
            parsed = json.loads(raw)

            questions = []
            if isinstance(parsed, list):
                questions = parsed
            else:
                for v in parsed.values():
                    if isinstance(v, list):
                        questions = v
                        break

            total_prompt_tokens += client_response.usage.prompt_tokens
            total_completion_tokens += client_response.usage.completion_tokens

            for q in questions[:count]:
                all_questions.append({
                    "role": role,
                    "category": category,
                    "question": q.strip(),
                })

            print(f"완료 ({len(questions[:count])}개)")

        except Exception as e:
            print(f"실패: {e}")

        # Rate limit 대응
        time.sleep(0.3)

    print(f"\n[토큰 사용량] prompt={total_prompt_tokens:,} / completion={total_completion_tokens:,}")
    print(f"[예상 비용] ${(total_prompt_tokens * 2.5 + total_completion_tokens * 10) / 1_000_000:.4f}")

    return all_questions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_gen.yaml")
    args = parser.parse_args()

    # 설정 로드
    with open(args.config) as f:
        config = yaml.safe_load(f)

    random.seed(config["general"]["seed"])

    # OpenAI 클라이언트
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY가 .env에 없습니다.")
    client = OpenAI(api_key=api_key)

    # 시드 질문 생성
    questions = generate_seed_questions(config, client)

    # 저장
    seeds_dir = Path(config["general"]["seeds_dir"])
    seeds_dir.mkdir(parents=True, exist_ok=True)
    output_path = seeds_dir / "seed_questions.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for i, q in enumerate(questions):
            q["id"] = f"seed_{i+1:04d}"
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    # 결과 통계
    print(f"\n{'='*50}")
    print(f"총 생성된 시드 질문: {len(questions)}개")
    print(f"저장 위치: {output_path}")
    print("\n[직군별 분포]")
    from collections import Counter
    role_dist = Counter(q["role"] for q in questions)
    for role, cnt in sorted(role_dist.items()):
        print(f"  {role}: {cnt}개")
    print("\n[카테고리별 분포]")
    cat_dist = Counter(q["category"] for q in questions)
    for cat, cnt in sorted(cat_dist.items()):
        print(f"  {cat}: {cnt}개")


if __name__ == "__main__":
    main()
