# AI 면접 평가 모델 (캡스톤 AI 파트)

> **이 파일의 역할**: Claude Code가 이 프로젝트의 컨텍스트·결정 사항·규칙을 즉시 파악하기 위한 단일 진입점. 코드 작성 전 반드시 이 파일 전체를 읽을 것.

---

## 1. 프로젝트 개요

현직자 멘토와 취업 준비생 멘티를 연결하는 **멘토–멘티 매칭 플랫폼**의 *AI 면접 평가 모델* 파트. 멘티가 모의 면접을 진행하면 AI가 답변을 평가지(JSON)로 자동 생성하고, 멘토는 검수·수정만 수행한다. 멘토 수정 데이터는 다시 모델 학습(DPO)에 사용되어, 운영될수록 평가가 멘토 기준에 가까워지는 자기개선 구조이다.

**팀 프로젝트이며, 본인 담당 범위는 AI 파인튜닝 + 추론 백엔드(FastAPI).**

## 2. 본 코드베이스의 목표 (Phase 체크리스트)

- [ ] **Phase 1** — 합성 데이터 생성 파이프라인 + 약 2,000건 IT 면접 평가 데이터
- [ ] **Phase 2** — SFT 학습 코드 (Qwen2.5-7B + Unsloth LoRA)
- [ ] **Phase 3** — 평가 코드 (baseline vs SFT, 4지표 비교)
- [ ] **Phase 4** — 추론 백엔드 (FastAPI, 팀 프론트와 통합)
- [ ] **Phase 5** — DPO 학습 코드 (운영 데이터 수집 후, 현재는 placeholder)

## 3. 기술 스택

| 영역 | 선택 |
|---|---|
| 베이스 모델 | **Qwen2.5-7B-Instruct** (Apache 2.0) |
| 학습 라이브러리 | Unsloth + transformers + trl + peft |
| 양자화 | 4bit (bitsandbytes) |
| 학습 환경 | Colab T4 16GB (무료 우선, 부족 시 Pro 결제) |
| 합성 데이터 생성 | 시드 질문 + 점수 구간별 프롬프트 + deterministic 보정 스크립트, 추후 OpenAI API로 품질 보강 |
| 백엔드 | FastAPI + uvicorn |
| 데이터 형식 | JSONL + datasets |
| Python | 3.10+ |

## 4. 디렉터리 구조

```
.
├── CLAUDE.md                # 이 파일
├── .env                     # API 키 (절대 커밋 금지)
├── .gitignore
├── pyproject.toml
├── configs/
│   ├── data_gen.yaml        # 합성 데이터 생성 설정 (분포·등급별 비율)
│   ├── sft_train.yaml       # SFT 하이퍼파라미터
│   └── eval.yaml            # 평가 설정
├── data/
│   ├── seeds/               # 시드 질문 풀
│   ├── raw/                 # QA 프롬프트·응답·병합 원본
│   ├── processed/           # 검수·정제 후 최종 데이터
│   └── splits/              # train/val/test 분리본
├── scripts/
│   ├── 01_make_seed_questions.py
│   ├── 02_generate_qa_pairs.py      # QA 프롬프트 생성 + 응답 파싱
│   ├── 03_refine_qa_responses.py    # 현재 1차 QA 데이터 보정/재생성
│   ├── 03_generate_evaluations.py   # TODO: 답변 평가 JSON 생성
│   ├── 04_validate_dataset.py       # SFT 데이터셋 검증
│   ├── 05_split_dataset.py
│   ├── 06_train_sft.py
│   ├── 07_evaluate.py
│   └── 08_compare_baseline.py
├── notebooks/
│   └── sft_colab.ipynb      # Colab T4 학습용
├── models/
│   └── lora_adapters/       # 학습된 LoRA 어댑터 (Drive 백업 필수)
├── api/
│   ├── main.py              # FastAPI 엔트리
│   ├── inference.py         # 모델 로딩·생성
│   └── schemas.py           # Pydantic 스키마
└── tests/
    └── test_data_format.py
```

## 5. 데이터 스키마

### 5.1 SFT 샘플 형식 (JSONL)

```json
{
  "id": "synth_be_cs_00012",
  "metadata": {
    "category": "백엔드",
    "question_type": "CS",
    "score_band": "4-6",
    "topic_tag": "database/index",
    "generated_by": "synthetic_refine_v1",
    "reviewed": true,
    "version": "v1"
  },
  "instruction": "다음은 IT 직군 면접 질문과 지원자의 답변입니다. 답변을 평가 기준에 따라 채점하고 정해진 JSON 형식으로 결과를 출력하세요.",
  "input": {
    "question": "...",
    "answer": "..."
  },
  "output": {
    "reasoning": "...",
    "overall_score": 6,
    "strengths": ["...", "..."],
    "improvements": ["...", "..."]
  }
}
```

**중요 — 출력 순서 고정**: `reasoning` → `overall_score` → `strengths` → `improvements` (CoT 효과로 점수 calibration 향상).

### 5.2 DPO 샘플 형식 (Phase 5, placeholder)

```json
{
  "prompt": "<instruction + 질문 + 답변 ChatML 직렬화>",
  "chosen": "<멘토가 수정한 평가지 JSON 문자열>",
  "rejected": "<모델이 원래 출력한 평가지 JSON 문자열>"
}
```

## 6. 데이터 구성 (Phase 1 목표)

총 약 **2,000건** — *양보다 질* 원칙. 현재 1차 QA 원본은 `data/raw/qa_responses/*.json`과 `data/raw/qa_pairs.jsonl`에 생성되어 있다.

현재 데이터 생성 방식은 다음과 같다.

1. `data/seeds/seed_questions.jsonl` 기반으로 직군별 시드 질문을 구성한다.
2. `scripts/02_generate_qa_pairs.py --gen-prompts`로 점수 구간별 프롬프트를 만든다.
3. 현재는 API 비용과 시간 제약 때문에 `scripts/03_refine_qa_responses.py`로 deterministic QA 응답을 생성·보정한다.
4. `scripts/02_generate_qa_pairs.py --parse`로 raw 응답을 `data/raw/qa_pairs.jsonl`에 병합한다.

주의: 현재 QA 데이터는 학습 파이프라인 검증과 1차 SFT 준비를 위한 **규칙 기반 합성 데이터**에 가깝다. 최종 품질을 높이려면 멘토 검수 또는 GPT-4o/상위 모델 기반 재작성으로 일부 샘플을 보강해야 한다.

### 6.1 직군 분포 (4개)

| 직군 | 비율 | 건수 |
|---|---|---|
| 백엔드 | 30% | 600 |
| 프론트엔드 | 20% | 400 |
| 데이터·AI/ML | 30% | 600 |
| 인프라·DevOps | 20% | 400 |

### 6.2 질문 카테고리 (3개)

| 카테고리 | 비율 | 비고 |
|---|---|---|
| CS 기초 | 40% | 자료구조·알고리즘·OS·네트워크·DB |
| 직무 전문 | 40% | 직군별 도구·언어·아키텍처 지식 |
| **IT 맥락 인성·사고방식** | 20% | **반드시 IT 환경(코드 리뷰·기술 의사결정·장애 대응·학습·페어 프로그래밍 등) 맥락 포함**. 일반 인성 질문(예: "본인의 장단점은?", "10년 후 모습은?")은 **제외** |

### 6.3 점수 등급 (4구간, 균등 강제)

`1-3 / 4-6 / 7-8 / 9-10` — 각 25% (총 500건씩). GPT-4o 합성은 5~8점에 몰리는 편향이 있으므로 등급별로 **별도 프롬프트**로 강제 생성.

현재 병합 결과 기준:

| 구분 | 건수 |
|---|---:|
| 총 QA 쌍 | 2,000 |
| 백엔드 | 600 |
| 프론트엔드 | 400 |
| 데이터·AI/ML | 600 |
| 인프라·DevOps | 400 |
| 1-3 / 4-6 / 7-8 / 9-10 | 각 500 |

### 6.4 현재 QA 데이터 품질 메모

- 점수 구간과 직군 분포는 프로젝트 목표와 맞춘 상태.
- 질문 주제와 답변이 어긋나지 않도록 `03_refine_qa_responses.py`에서 주요 키워드별 답변 profile을 둔다.
- `1-3`은 명확한 개념 혼동, `4-6`은 피상적 이해, `7-8`은 핵심 설명+주의점, `9-10`은 원리+트레이드오프+운영 관점으로 분리한다.
- 다만 deterministic 생성 특성상 문장 패턴 반복이 남아 있다. SFT 전에 최소 100~200건 표본 검수, 가능하면 각 직군/점수 구간별 20~30건씩 멘토 또는 사람이 수정하는 것이 좋다.
- 멘토 기반 모델이 목표이므로 합성 데이터는 “초기 형식 학습” 용도이고, 실제 정렬은 멘토 수정 데이터로 DPO 또는 추가 SFT를 수행하는 방향이 맞다.

## 7. 핵심 설계 결정 (Why)

### 왜 Qwen2.5-7B-Instruct?
한국어 7B급 상위권(LogicKor) + 코드·기술 문서 학습 비중 높아 IT 답변 정확성 판단에 유리 + JSON 구조화 출력 안정 + Unsloth 공식 지원으로 T4 4bit QLoRA 가능 + Apache 2.0.

### 왜 SFT → DPO 두 단계 (PPO 아닌)?
- **SFT 선행**: 사전학습 모델은 "면접 평가자" 역할을 모름. 출력 형식·평가 항목·점수 분포를 먼저 학습. DPO를 cold-start에 직접 적용하면 학습 불안정.
- **DPO 선택**: PPO는 별도 보상 모델(reward model) 학습이 필요해 무겁지만, DPO는 (chosen, rejected) 페어만으로 학습 가능. 멘토 검수 단계에서 페어가 자연스럽게 생성되는 우리 서비스 구조와 적합.

### 왜 IT 직군 특화?
도메인마다 평가 기준이 다름. IT는 답변의 옳고 그름이 비교적 명확해 합성 데이터 라벨 검증 가능. 동일 파이프라인을 다른 직군으로 확장 가능한 구조로 설계. **인성 질문도 IT 맥락(코드 리뷰·기술 의사결정·장애 대응 등)으로 한정**.

### 왜 양보다 질 (~2,000건)?
LoRA는 일부 파라미터만 업데이트해 라벨 노이즈에 민감(LIMA 등 연구). 무작위 양 증가 대신 메타데이터(직군·카테고리·점수 등급)로 분포 통제한 고품질 데이터 우선. 시드 질문은 실제 면접 후기·기술 자료 기반, 합성 평가지는 무작위 표본 수동 검수.

### 왜 다항목 점수 분리 안 함?
멘토가 매번 4개 항목(기술/구조/깊이/명료성)을 검수하면 부담 ↑ → 참여율 ↓ → DPO 학습 신호 손실. **종합 점수 1개 + 텍스트 피드백** 구조가 멘토 부담 최소화 + 학습 신호 풍부의 균형점.

## 8. 학습 환경 및 자원 제약 대응

### 메모리 통제 (T4 16GB)
- `max_seq_len = 2048` (99%+ 샘플 커버)
- `per_device_train_batch_size = 2`, `gradient_accumulation_steps = 4` (실효 배치 8)
- `gradient_checkpointing = True`
- 4bit 양자화 + Unsloth → VRAM ~10GB

### 데이터 길이 통제
- 합성 데이터 생성 시 답변을 **200~500자(한글)** 수준으로 제한 (GPT-4o 프롬프트에 명시)
- 전처리에서 토큰 길이 1,800 초과 샘플 필터

### 세션 위험 대응
- `save_steps = 100`, `save_total_limit = 3`
- LoRA 어댑터(~50MB)만 Google Drive에 저장
- `resume_from_checkpoint`로 중단 시 재개 가능

### 예상 학습 시간
2,000건 × 3 에폭 ≈ **약 25분** (T4 기준).

## 9. SFT 코드 작성 체크리스트 (반드시 확인)

1. **Loss masking** — 응답(평가지)에만 loss 적용. Unsloth `train_on_responses_only` 또는 trl `response_template` 정확히 지정. **빠뜨리면 모델이 질문까지 학습해서 망함.**
2. **ChatML 변환** — Qwen2.5 chat template 사용. `system`=instruction, `user`=질문+답변, `assistant`=`json.dumps(output, ensure_ascii=False, indent=2)`.
3. **검증셋 분리** — `train : val : test = 8 : 1 : 1`.
4. **하이퍼파라미터 1차 시작값**:
   ```python
   learning_rate = 2e-4
   num_train_epochs = 3
   lora_r = 16
   lora_alpha = 16
   lora_dropout = 0.05
   target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"]
   warmup_ratio = 0.05
   weight_decay = 0.01
   lr_scheduler_type = "cosine"
   seed = 42
   ```
5. **재현성** — `transformers.set_seed(42)` + torch/numpy/random 모두 시드 고정.
6. **학습 중 자동 검증** — `eval_steps = 100`마다 검증셋 5~10개를 실제 generate해서:
   - JSON 파싱 성공률
   - 필수 필드(`reasoning`, `overall_score`, `strengths`, `improvements`) 누락 여부
   - `overall_score` 1~10 범위 여부
   
   loss는 줄어드는데 출력 JSON이 깨지는 케이스가 흔함. 위 3가지가 95% 이상이어야 통과.

## 10. 평가 지표 (Phase 3)

`test/` 200건에 대해 baseline(zero-shot Qwen2.5) vs SFT 모델을 같은 프롬프트로 비교.

### 핵심 4지표
1. **JSON 파싱 성공률** — 출력이 valid JSON이고 필수 필드가 모두 있는가
2. **점수 MAE** — 예측 점수와 정답 점수의 평균 절대 오차
3. **점수 Pearson 상관계수** — 예측·정답 점수가 같은 방향으로 움직이는가
4. **점수 등급 정확도** — 예측이 정답 등급(1-3/4-6/7-8/9-10)에 들어오는 비율

### 보조 지표
- **LLM-as-judge** — GPT-4o가 두 모델 출력을 비교해 win rate 산출

### 운영 단계 지표
- **멘토 수정률** — 운영 중 멘토가 평가지를 수정하는 빈도. DPO 학습의 진짜 정렬 효과를 측정하는 지표.

## 11. 코딩 규칙

- 한글 주석 OK, 변수·함수명은 영어 (snake_case)
- API 키·모델 가중치·데이터는 **절대 git 커밋 금지**, `.gitignore` 철저히
- 합성 데이터 비용 추적: API 호출 횟수 + 토큰 사용량 stdout 로그
- 모든 무작위성은 `seed = 42`로 고정
- 한 스크립트는 한 가지 일만 (책임 명확히, idempotent하게)
- 새 의존성 추가 시 `pyproject.toml`에 명시 + 버전 핀

## 12. 자주 쓰는 명령

### 환경 설정
```bash
pip install -U "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "trl<0.9.0" "peft<0.12.0" accelerate bitsandbytes datasets
pip install openai python-dotenv pyyaml fastapi uvicorn pydantic
```

### 합성 데이터 생성 파이프라인
```bash
python scripts/01_make_seed_questions.py --config configs/data_gen.yaml
python scripts/02_generate_qa_pairs.py --gen-prompts
python scripts/03_refine_qa_responses.py
python scripts/02_generate_qa_pairs.py --parse

# 다음 단계 TODO: QA 쌍을 SFT 평가 데이터로 변환
# python scripts/03_generate_evaluations.py --config configs/data_gen.yaml
python scripts/04_validate_dataset.py --input data/processed/sft_dataset.jsonl
python scripts/05_split_dataset.py --input data/processed/sft_dataset.jsonl
```

### 학습
```bash
# 로컬 형식 확인: python scripts/06_train_sft.py --config configs/sft_train.yaml --dry-run
# Colab/GPU 학습: python scripts/06_train_sft.py --config configs/sft_train.yaml
# Colab: notebooks/sft_colab.ipynb 실행
```

### 평가
```bash
python scripts/07_evaluate.py --model models/lora_adapters/checkpoint-XXX --test data/splits/test.jsonl
python scripts/08_compare_baseline.py
```

### 백엔드 실행
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

## 13. 작업 우선순위

현재는 **Phase 1 (합성 데이터 생성)** 진행 중. 현재 QA 원본 2,000건은 생성 완료했고, 다음 목표는 멘토가 검수하기 쉬운 평가 JSON 데이터셋으로 변환하는 것이다.

1. `configs/data_gen.yaml` 작성 (분포·등급별 비율 명시)
2. `scripts/01_make_seed_questions.py` — 시드 질문 200개 생성·큐레이팅 (실제 면접 후기·기술 블로그 기반, 사람이 일부 직접 작성/검수)
3. `scripts/02_generate_qa_pairs.py --gen-prompts` — 시드 + 점수 등급 조합으로 QA 프롬프트 생성
4. `scripts/03_refine_qa_responses.py` — 현재 1차 응답 JSON 생성·보정 (2,000건)
5. `scripts/02_generate_qa_pairs.py --parse` — `data/raw/qa_pairs.jsonl` 병합
6. `scripts/03_generate_evaluations.py` — TODO: QA 쌍을 평가 JSON(`reasoning`, `overall_score`, `strengths`, `improvements`)으로 변환
7. `scripts/04_validate_dataset.py` — 분포·JSON 파싱 가능 여부·길이·중복·점수 일관성 검증
8. `scripts/05_split_dataset.py` — train/val/test 분리
9. **멘토 또는 사람이 100~200건 표본 수동 검수** ← 가장 중요. 라벨 품질이 모델 품질을 결정.

### 멘토 기반 학습으로 이어가는 권장 흐름

1. 1차 SFT는 합성 데이터로 모델의 역할, JSON 출력 형식, 점수 구간 감각을 먼저 학습시킨다.
2. 서비스 화면에서는 AI 평가지를 그대로 확정하지 않고 멘토가 `overall_score`, `reasoning`, `strengths`, `improvements`를 수정할 수 있게 한다.
3. 수정 전 모델 출력은 `rejected`, 멘토 수정본은 `chosen`으로 저장한다.
4. 운영 데이터가 충분히 쌓이면 DPO로 “우리 멘토들이 선호하는 평가 방식”에 맞춘다.
5. 멘토 수정률이 높은 질문 유형이나 점수 구간은 합성 데이터도 추가 보강한다.

## 14. DO / DON'T

✅ **DO**
- 한 스크립트가 끝나면 결과 통계를 stdout에 출력 (분포·길이·비용 등)
- API 호출은 retry + rate limit 대응 (`tenacity` 등)
- 모든 중간 산출물은 `data/raw/`에 보관 (재현·디버깅용)
- 시드 질문은 실제 면접 자료 기반으로 큐레이팅, GPT 단독 생성 X
- 합성 평가지 무작위 표본 수동 검수

❌ **DON'T**
- API 키 코드에 하드코딩 (반드시 `.env` + `python-dotenv`)
- 일반 인성 질문 합성 데이터에 포함 (예: "장단점은?", "10년 후 모습?") — IT 맥락 없으면 제외
- 다항목 점수 분리 (`technical_accuracy`, `structure` 등) 추가 — 의도적으로 단순화한 구조
- LoRA 어댑터·데이터 git 커밋
- `max_seq_len > 2048` (T4 OOM 위험)
- baseline 평가 빠뜨리기 (전후 비교가 캡스톤 핵심 지표)

## 15. 참고 자료

- Unsloth: https://github.com/unslothai/unsloth
- Qwen2.5 모델 카드: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- LIMA (양보다 질 근거): https://arxiv.org/abs/2305.11206
- DPO 논문: https://arxiv.org/abs/2305.18290
- TRL DPO 문서: https://huggingface.co/docs/trl/dpo_trainer
- TRL SFTTrainer: https://huggingface.co/docs/trl/sft_trainer

---

## 현재 진행 상황

- [x] 기획 및 설계 (중간 보고서 제출 완료)
- [x] Phase 1-1 — 시드 질문 및 QA 원본 2,000건 생성 (`data/raw/qa_pairs.jsonl`)
- [x] Phase 1-2 — QA 쌍을 SFT 평가 JSON 데이터셋으로 변환 (`data/processed/sft_dataset.jsonl`)
- [x] Phase 1-2.5 — SFT 데이터셋 형식·분포·중복 검증 스크립트 작성
- [x] Phase 1-3 — train/val/test 분리 (`data/splits/*.jsonl`)
- [ ] Phase 1-4 — 멘토/사람 표본 검수 후 라벨 품질 보정
- [x] Phase 2-0 — SFT 학습 설정 및 ChatML 변환 dry-run 작성 (`configs/sft_train.yaml`, `scripts/06_train_sft.py`)
- [ ] Phase 2-1 — Colab/GPU 환경에서 작은 샘플 SFT 실행
- [ ] 이후 Phase 2~5 순차 진행
