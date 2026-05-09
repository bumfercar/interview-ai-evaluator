# Colab SFT 실행 가이드

이 프로젝트는 로컬 Mac에서 데이터 생성/검증까지 하고, 실제 SFT 학습은 Colab T4 GPU에서 실행하는 흐름을 기준으로 한다.

## 1. Colab 런타임

Colab에서 다음을 설정한다.

```text
런타임 > 런타임 유형 변경 > 하드웨어 가속기: T4 GPU
```

GPU 확인:

```bash
!nvidia-smi
```

## 2. Google Drive 마운트

학습 결과는 Colab 세션이 종료되어도 남도록 Drive에 저장한다.

```python
from google.colab import drive
drive.mount("/content/drive")
```

## 3. GitHub repo 가져오기

```bash
%cd /content
!git clone <YOUR_GITHUB_REPO_URL> AI
%cd /content/AI
```

이미 clone한 repo를 갱신할 때:

```bash
%cd /content/AI
!git pull
```

## 4. 의존성 설치

Colab 노트북 코드 셀에서 실행한다. 이전에 예전 버전의 `peft`, `trl`을 설치했다면 런타임을 재시작한 뒤 다시 설치하는 것이 가장 깔끔하다.

```bash
!pip install -U "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install -U --no-deps peft trl accelerate bitsandbytes datasets
!pip install -U pyyaml scipy scikit-learn
```

설치 후에는 다음을 권장한다.

```text
런타임 > 런타임 다시 시작
```

다시 시작한 뒤 Drive 마운트와 repo 이동을 다시 실행한다.

## 5. 데이터 확인

```bash
!python scripts/04_validate_dataset.py --input data/processed/sft_dataset.jsonl
!wc -l data/splits/train.jsonl data/splits/val.jsonl data/splits/test.jsonl
```

정상 기준:

```text
train: 1600
val: 200
test: 200
```

## 6. ChatML 변환 dry-run

```bash
!python scripts/06_train_sft.py --config configs/sft_train.yaml --dry-run --sample 2
```

여기서 `system`, `user`, `assistant` 형태가 보이면 정상이다.

## 7. Drive 저장 경로 설정

Colab에서 학습 결과를 Drive에 저장하려면 `configs/sft_train.yaml`의 output 경로를 Colab용으로 바꾼다.

```python
import pathlib
import yaml

path = pathlib.Path("configs/sft_train.yaml")
cfg = yaml.safe_load(path.read_text())

cfg["output"]["dir"] = "/content/drive/MyDrive/AI/models/lora_adapters/qwen2_5_7b_sft"
cfg["output"]["logging_dir"] = "/content/drive/MyDrive/AI/models/lora_adapters/qwen2_5_7b_sft/logs"

# 작은 샘플 학습으로 먼저 확인할 때
cfg["data"]["max_train_samples"] = 200
cfg["data"]["max_val_samples"] = 50

# Unsloth T4 안정 설정
cfg["lora"]["dropout"] = 0.0
cfg["lora"]["use_gradient_checkpointing"] = "unsloth"

path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
```

전체 학습을 할 때는 `max_train_samples`, `max_val_samples`를 `None`으로 바꾼다.

```python
import pathlib
import yaml

path = pathlib.Path("configs/sft_train.yaml")
cfg = yaml.safe_load(path.read_text())
cfg["data"]["max_train_samples"] = None
cfg["data"]["max_val_samples"] = None
path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
```

## 8. 작은 샘플 학습

먼저 100~200개 정도로 학습이 도는지 확인한다.

```bash
!python scripts/06_train_sft.py --config configs/sft_train.yaml
```

확인할 것:

- OOM 없이 시작되는지
- loss가 출력되는지
- checkpoint가 Drive 경로에 저장되는지

## 9. 전체 학습

작은 샘플이 성공하면 `max_train_samples`, `max_val_samples`를 `null`로 되돌리고 다시 실행한다.

```bash
!python scripts/06_train_sft.py --config configs/sft_train.yaml
```

결과물:

```text
/content/drive/MyDrive/AI/models/lora_adapters/qwen2_5_7b_sft
```

이 폴더가 이후 평가/추론 백엔드에서 사용할 LoRA adapter다.

## 10. 학습 결과 평가

학습이 끝나면 test 데이터 200개로 자동 평가한다.

작게 먼저 확인:

```bash
!python scripts/07_evaluate.py \
  --adapter /content/drive/MyDrive/AI/models/lora_adapters/qwen2_5_7b_sft \
  --test data/splits/test.jsonl \
  --output data/eval/sft_predictions_sample.jsonl \
  --limit 20
```

전체 test 평가:

```bash
!python scripts/07_evaluate.py \
  --adapter /content/drive/MyDrive/AI/models/lora_adapters/qwen2_5_7b_sft \
  --test data/splits/test.jsonl \
  --output data/eval/sft_predictions.jsonl
```

확인할 핵심 지표:

```text
json_parse_success
score_mae
exact_score_accuracy
score_band_accuracy
pearson
```

좋은 신호 예시:

```text
json_parse_success가 높다
score_mae가 낮다
score_band_accuracy가 높다
```

주의: loss가 내려가도 JSON 출력이 깨지면 실제 서비스에는 쓰기 어렵다. 그래서 학습 후에는 반드시 `07_evaluate.py`로 test 평가를 확인한다.

## GitHub에 올릴 것

```text
CLAUDE.md
COLAB_GUIDE.md
pyproject.toml
.env.example
configs/
scripts/
data/seeds/
data/raw/
data/processed/
data/splits/
```

## GitHub에 올리면 안 되는 것

```text
.env
models/
checkpoints/
*.safetensors
*.bin
*.pt
*.pth
wandb/
```
