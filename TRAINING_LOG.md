# SFT 학습 과정 기록

이 문서는 AI 면접 평가 모델을 실제로 학습하면서 진행한 과정, 발생한 문제, 해결 방법, 현재 결과를 기록하기 위한 파일이다. 나중에 Notion이나 발표 자료로 옮길 때 바로 참고할 수 있도록 너무 어렵지 않은 표현으로 정리했다.

## 1. 학습 목표

이번 학습의 목표는 면접 질문과 지원자 답변을 입력했을 때, 모델이 정해진 형식의 평가 JSON을 생성하도록 만드는 것이다.

모델이 출력해야 하는 형태는 다음과 같다.

```json
{
  "reasoning": "점수 판단 이유",
  "overall_score": 7,
  "strengths": ["답변의 장점"],
  "improvements": ["개선할 점"]
}
```

단순히 자연어 피드백을 생성하는 것이 아니라, 백엔드와 프론트엔드에서 바로 활용할 수 있도록 구조화된 JSON을 안정적으로 출력하는 것이 중요하다.

## 2. 사용한 학습 방식

이번에는 SFT(Supervised Fine-Tuning)를 사용했다.

SFT는 모델에게 입력과 정답 출력 예시를 함께 보여주면서, 특정 작업을 따라 하도록 학습시키는 방식이다. 이 프로젝트에서는 다음과 같은 예시를 학습시켰다.

```text
입력:
면접 질문 + 지원자 답변

출력:
reasoning, overall_score, strengths, improvements가 포함된 평가 JSON
```

SFT를 먼저 선택한 이유는 아직 실제 멘토가 수정한 운영 데이터가 충분하지 않기 때문이다. 먼저 합성 데이터로 기본 평가자 역할을 학습시키고, 이후 서비스에서 멘토 수정 데이터가 쌓이면 DPO 같은 선호도 학습으로 발전시키는 방향이 적절하다고 판단했다.

## 3. 사용한 모델과 환경

학습에 사용한 기본 모델은 Qwen2.5-7B-Instruct이다.

```text
base model: Qwen2.5-7B-Instruct
학습 방식: SFT
경량화 방식: LoRA + 4bit
실행 환경: Google Colab T4
저장 위치: Google Drive
```

Mac에는 별도 GPU가 없기 때문에 실제 학습은 Colab 무료 T4 GPU에서 진행했다.

## 4. 경량화 방식

7B 모델 전체를 그대로 학습하면 무료 Colab T4에서는 메모리와 시간이 부담된다. 그래서 다음과 같은 경량화 방식을 사용했다.

### 4bit 로딩

```yaml
load_in_4bit: true
```

기본 모델을 4bit로 압축해서 GPU 메모리에 올렸다. 이 덕분에 T4에서도 7B 모델을 사용할 수 있었다.

### LoRA

전체 모델 파라미터를 전부 학습하지 않고, 작은 adapter만 학습했다.

실제 학습 로그에서는 다음처럼 확인되었다.

```text
Trainable parameters = 40,370,176 of 7,655,986,688
0.53% trained
```

즉 전체 모델의 약 0.53%만 학습했다. 이것이 학습 메모리를 크게 줄여주는 핵심이다.

### Unsloth

Unsloth를 사용해 LoRA 학습 속도와 메모리 사용량을 최적화했다.

```text
Unsloth - 2x faster free finetuning
```

다만 경량화를 하더라도 7B 모델 자체가 크기 때문에 학습 시간이 아주 짧아지는 것은 아니다. 경량화의 가장 큰 의미는 "작은 GPU에서도 학습이 가능하게 만드는 것"에 가깝다.

## 5. 작은 샘플 학습

처음부터 전체 데이터를 학습하지 않고, 먼저 작은 샘플로 학습 파이프라인이 정상 작동하는지 확인했다.

사용한 설정은 다음과 같다.

```text
train samples: 200
val samples: 50
epochs: 3
GPU: Colab T4
```

학습 로그 요약:

```text
Num examples = 200
Num Epochs = 3
Total steps = 75
Total batch size = 8
Trainable parameters = 40,370,176
```

학습 시간은 약 16분 정도 걸렸다.

## 6. 학습 결과

작은 샘플 학습은 정상적으로 완료되었다.

최종 로그:

```text
train_loss = 0.3739
eval_loss = 0.9444
학습 완료. LoRA adapter 저장:
/content/drive/MyDrive/AI/models/lora_adapters/qwen2_5_7b_sft
```

학습 loss는 다음처럼 빠르게 감소했다.

```text
1.713
0.6695
0.228
0.08402
0.04314
0.02779
0.02476
```

loss가 빠르게 내려간 것은 모델이 학습 데이터를 잘 따라가고 있다는 의미다. 다만 200개만 사용했기 때문에 과적합 가능성도 있다. 그래서 loss만 보고 성능이 좋다고 판단하면 안 되고, 실제 생성 결과와 test 평가 지표를 함께 봐야 한다.

## 7. 평가 결과

학습된 LoRA adapter를 사용해 test 데이터 중 10개 샘플을 평가했다.

실행 명령:

```python
!python scripts/07_evaluate.py \
  --adapter /content/drive/MyDrive/AI/models/lora_adapters/qwen2_5_7b_sft \
  --test data/splits/test.jsonl \
  --output data/eval/sft_predictions_sample.jsonl \
  --limit 10
```

평가 결과:

```text
total: 10
json_parse_success: 10/10 (1.000)
score_mae: 0.700
exact_score_accuracy: 0.400
score_band_accuracy: 1.000
pearson: 0.244
```

## 8. 평가 결과 해석

### JSON 형식 안정성

```text
json_parse_success: 1.000
```

10개 중 10개 모두 JSON 파싱에 성공했다. 이 프로젝트에서는 매우 중요한 결과다. 백엔드와 연동하려면 모델 출력이 사람이 읽기 좋은 문장만 되어서는 안 되고, 프로그램이 읽을 수 있는 JSON이어야 하기 때문이다.

### 점수 오차

```text
score_mae: 0.700
```

모델이 예측한 점수와 정답 점수의 평균 차이가 0.7점이라는 뜻이다. 10점 척도에서 작은 샘플 기준으로는 나쁘지 않은 결과다.

### 정확 점수 일치율

```text
exact_score_accuracy: 0.400
```

정확히 같은 점수를 맞힌 비율은 40%였다. 면접 평가는 7점과 8점처럼 가까운 점수의 차이가 애매할 수 있기 때문에, 정확 점수 일치율만으로 판단하기는 어렵다.

### 점수대 정확도

```text
score_band_accuracy: 1.000
```

점수대는 100% 맞췄다. 예를 들어 7점과 8점이 완전히 같은 점수는 아니어도 같은 7-8점 구간이면 실무적으로는 비슷한 수준의 답변으로 볼 수 있다. 현재 프로젝트에서는 이 지표가 중요하다.

### Pearson 상관계수

```text
pearson: 0.244
```

점수 흐름의 상관관계는 낮게 나왔다. 하지만 샘플 수가 10개뿐이라 아직 의미 있게 판단하기 어렵다. 최소 50개나 100개 이상으로 다시 평가해야 한다.

## 9. 학습 중 발생한 문제와 해결

### PEFT 버전 문제

처음에는 다음 오류가 발생했다.

```text
TypeError: LoraConfig.__init__() got an unexpected keyword argument 'target_parameters'
```

원인은 Colab에 설치된 `peft` 버전이 현재 Unsloth와 맞지 않았기 때문이다. 기존에 너무 낮은 버전으로 고정되어 있던 의존성을 수정했다.

### TrainingArguments 인자명 변경

다음 오류도 발생했다.

```text
TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'
```

Transformers 버전에 따라 `evaluation_strategy` 대신 `eval_strategy`를 사용하는 경우가 있어서, 현재 설치된 버전에 맞는 인자를 자동 선택하도록 수정했다.

### SFTConfig 호환 문제

다음 오류가 발생했다.

```text
TypeError: SFTConfig.__init__() got an unexpected keyword argument 'push_to_hub_token'
```

최신 TRL에서 `SFTTrainer` 내부 설정 방식이 바뀌면서 생긴 문제였다. 해결을 위해 `SFTConfig`를 직접 만들고, 현재 설치된 버전이 받을 수 있는 인자만 넘기도록 수정했다.

### padding_free와 packing 충돌

다음 오류가 발생했다.

```text
ValueError: When `padding_free=True` without packing, `max_length` is not enforced.
```

우리는 일반 SFT 방식으로 `packing=False`를 사용하고 있었는데, 최신 TRL 기본값이 `padding_free=True`라 충돌했다. 그래서 `padding_free=False`를 명시해 해결했다.

## 10. 현재 상태

현재까지 확인된 것은 다음과 같다.

```text
데이터 생성 완료
SFT 데이터셋 생성 완료
train/val/test 분리 완료
Colab T4 학습 실행 성공
LoRA adapter 저장 성공
샘플 10개 평가 성공
JSON 출력 안정성 확인
```

즉, 학습과 평가 파이프라인은 정상 작동한다.

아직 최종 성능이 검증된 것은 아니다. 지금 결과는 200개 샘플로 학습한 작은 실험이므로, 전체 학습 전 중간 확인 단계로 봐야 한다.

## 11. 다음 실험 계획

200개 학습에 약 16분이 걸렸기 때문에, 전체 1600개 train 데이터를 3 epoch로 학습하면 T4 무료 환경에서는 시간이 꽤 오래 걸릴 수 있다.

그래서 다음 실험은 다음처럼 진행하는 것이 적절하다.

```text
1차 실험: 800개 / 1 epoch / max_seq_length 1024
2차 실험: 1600개 / 1 epoch
3차 실험: 결과가 부족하면 1600개 / 2 epoch
```

현재 추천 설정:

```yaml
model:
  max_seq_length: 1024

data:
  max_train_samples: 800
  max_val_samples: 100

training:
  num_train_epochs: 1
```

이렇게 하면 전체 데이터를 바로 오래 돌리기 전에, 시간 대비 성능이 어느 정도 나오는지 확인할 수 있다.

## 12. 앞으로 확인할 지표

다음 평가에서는 10개가 아니라 최소 50개 또는 100개 샘플을 확인해야 한다.

중요하게 볼 지표는 다음과 같다.

```text
json_parse_success
score_mae
score_band_accuracy
pearson
```

특히 이 프로젝트에서는 다음 순서로 중요하다.

```text
1. JSON 형식이 깨지지 않는가
2. 점수대가 크게 틀리지 않는가
3. 개선점과 장점이 답변 내용과 맞는가
4. 점수 흐름이 정답 데이터와 비슷한가
```

loss는 참고 지표일 뿐이고, 실제 서비스에서는 JSON 안정성과 평가 내용의 품질이 더 중요하다.

## 13. Base Qwen과 SFT 모델 비교

처음 Base Qwen을 평가했을 때는 모델이 JSON을 만들기는 했지만, 우리가 원하는 스키마를 따르지 않았다. 예를 들어 `reasoning`, `overall_score`, `strengths`, `improvements` 대신 `총점`, `평가항목`, `채점` 같은 다른 필드를 사용했다.

이 결과만으로는 Base 모델의 점수 판단력이 낮다고 말하기 어렵다. JSON 형식 문제는 프롬프트를 강하게 주면 어느 정도 해결될 수 있기 때문이다. 그래서 Base Qwen에도 동일하게 강한 출력 규칙을 주고 다시 평가했다.

Base Qwen에 추가한 핵심 규칙은 다음과 같다.

```text
반드시 JSON 객체 하나만 출력
reasoning, overall_score, strengths, improvements 필드만 사용
overall_score는 1부터 10까지의 정수
Markdown 코드블록이나 추가 설명 금지
```

이후 같은 test 샘플 50개를 기준으로 Base Qwen과 SFT LoRA 모델을 비교했다.

| 모델 | JSON 성공률 | MAE | 정확 점수 일치율 | 점수대 정확도 | Pearson |
|---|---:|---:|---:|---:|---:|
| Base Qwen + 강한 프롬프트 | 50/50 (100%) | 1.940 | 8.0% | 32.0% | 0.724 |
| SFT LoRA 모델 | 46/50 (92%) | 0.870 | 39.1% | 87.0% | 0.902 |

각 지표의 의미는 다음과 같다.

```text
MAE: 정답 점수와 모델 점수의 평균 오차
정확 점수 일치율: 모델이 정답 점수를 정확히 맞춘 비율
점수대 정확도: 1-3, 4-6, 7-8, 9-10 구간을 맞춘 비율
Pearson: 좋은 답변에는 높은 점수, 부족한 답변에는 낮은 점수를 주는 흐름을 얼마나 잘 따라가는지
```

비교 결과, Base Qwen도 강한 프롬프트를 주면 JSON 형식은 맞출 수 있었다. 하지만 점수 판단에서는 SFT 모델이 더 안정적이었다.

핵심 개선은 다음과 같다.

```text
평균 점수 오차: 1.94점 -> 0.87점
점수대 정확도: 32% -> 87%
정확 점수 일치율: 8% -> 39.1%
Pearson: 0.724 -> 0.902
```

즉 SFT 모델은 Base Qwen보다 정답 점수에 더 가깝게 채점했고, 답변 수준의 높고 낮음을 더 잘 따라갔다. 다만 SFT 모델의 JSON 성공률이 92%로 나온 것은 추가로 확인해야 한다. 이전 100개 평가에서는 JSON 성공률이 100%였기 때문에, 강한 평가 프롬프트와 학습된 출력 패턴이 일부 충돌했을 가능성이 있다.

현재 결론은 다음과 같다.

```text
Base Qwen은 프롬프트만으로 출력 형식은 맞출 수 있었지만 점수 판단은 불안정했다.
SFT 모델은 평균 점수 오차와 점수대 정확도에서 더 좋은 결과를 보였다.
따라서 SFT는 단순히 JSON 형식을 맞춘 것이 아니라, 면접 답변의 점수 판단력도 개선한 것으로 볼 수 있다.
```

단, 이 결과가 데이터 편향 없이 완전히 일반화된 성능을 의미한다고 보기는 아직 어렵다. 이후에는 사람이 직접 만든 새로운 답변과 더 어려운 예외 케이스로 추가 검증해야 한다.

## 14. 백엔드/프론트 연동 관점

이 모델은 최종적으로 백엔드에서 호출할 수 있는 AI 평가 엔진으로 사용될 예정이다.

예상 입력:

```json
{
  "question": "Spring에서 트랜잭션 전파 옵션을 설명해주세요.",
  "answer": "..."
}
```

예상 출력:

```json
{
  "reasoning": "지원자 답변의 수준을 평가한 이유",
  "overall_score": 7,
  "strengths": ["개념의 핵심을 설명함"],
  "improvements": ["구체적인 예시가 부족함"]
}
```

프론트엔드는 사용자의 답변을 입력받고, 백엔드는 AI 평가 API를 호출한 뒤 결과를 저장하거나 화면에 반환하면 된다.

## 15. 메모

이번 실험에서 확인한 가장 중요한 점은 "학습이 실제로 돌아간다"는 것이다. 처음부터 최종 성능을 목표로 하기보다, 먼저 작은 데이터로 학습과 평가 흐름이 끝까지 이어지는지 확인한 것이 의미 있었다.

다음 단계에서는 학습 데이터 수를 늘리고, epoch와 max sequence length를 조절하면서 시간 대비 성능을 비교해야 한다.

## 16. 데이터 편향 점검과 보정

기존 데이터의 평가 성능이 높게 나온 뒤, 모델이 실제 내용을 이해한 것인지 아니면 데이터의 쉬운 패턴을 학습한 것인지 확인했다.

점검 결과 기존 데이터에는 다음 문제가 있었다.

```text
1-3점 답변 평균 길이: 27 words
4-6점 답변 평균 길이: 40 words
7-8점 답변 평균 길이: 52 words
9-10점 답변 평균 길이: 80 words
답변 길이만 사용한 점수대 예측 정확도: 97.5%
train/test 동일 질문 overlap: 153개
```

즉 기존 데이터는 점수대가 올라갈수록 답변이 길어지는 경향이 강했고, 모델이 내용보다 길이와 반복 표현을 보고 점수를 맞혔을 가능성이 있었다.

이를 줄이기 위해 `scripts/03_create_debiased_dataset.py`를 추가하고, `data/debiased/` 아래에 새 데이터셋을 생성했다.

보정 방향은 다음과 같다.

```text
짧지만 핵심을 잘 말하는 고득점 답변 추가
길지만 핵심 키워드가 없거나 틀린 저득점 답변 추가
일부는 맞지만 판단 기준과 예외 상황이 부족한 중간 점수 답변 추가
본인이 잘 모르는 내용을 긴 말로 숨기는 답변 추가
같은 질문이 train/test에 동시에 들어가지 않도록 question 단위 split 적용
```

새 데이터셋 점검 결과는 다음과 같다.

```text
총 샘플: 2000
train: 1604
val: 204
test: 192
질문 overlap train/test: 0
답변 길이만 사용한 점수대 예측 정확도: 33.3%
검증 스크립트 통과
```

새 데이터의 점수대별 평균 길이는 다음과 같다.

```text
1-3점: 43.0 words
4-6점: 45.3 words
7-8점: 51.2 words
9-10점: 72.8 words
```

9-10점 답변이 여전히 평균적으로 길기는 하지만, 저득점에도 긴 답변이 포함되고 고득점에도 짧은 답변이 포함되도록 만들어 길이만으로 점수대를 맞히기 어렵게 했다.

새 학습용 설정 파일은 다음과 같다.

```text
configs/sft_train_debiased.yaml
```

이 설정은 `data/debiased/splits/train.jsonl`, `val.jsonl`, `test.jsonl`을 사용한다.

현재 결론은 다음과 같다.

```text
기존 데이터는 길이와 반복 표현 편향이 강했다.
보정 데이터는 질문 overlap을 제거했고, 길이만으로 점수대를 맞히는 정확도를 97.5%에서 33.3%로 낮췄다.
따라서 다음 학습은 기존 데이터가 아니라 data/debiased 기준으로 진행하는 것이 더 적절하다.
```

