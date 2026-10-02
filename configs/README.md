# 설정 파일 안내

일반 실행에서는 [panda.yaml](panda.yaml) 또는 [planar.yaml](planar.yaml)만 편집한다.
`make panda-view` 등 기존 실행 명령과 기본 설정 경로는 바뀌지 않았다.
학습 모델은 YAML이 아니라 실행 설정이 지정한 `models/`의 별도 산출물이다.

## 실행 설정 — 이 디렉터리

| 파일 | 용도 |
|---|---|
| [panda.yaml](panda.yaml) | Panda IK·제어·안전성·RDF 모델 경로·시각화 설정 |
| [planar.yaml](planar.yaml) | 평면 로봇 실행·안전성·실험 설정 |

두 실행 YAML의 위치는 유지했다. 따라서 YAML 기준 상대경로인 로봇 asset·모델
경로도 그대로다. 학습·보관 폴더의 설정은 실행 시 자동으로 합쳐지거나 로드되지 않는다.

## 오프라인 학습 설정 — `training/`

| 파일 | 용도 |
|---|---|
| [panda_rdf_training.yaml](training/panda_rdf_training.yaml) | 학습 CLI 기본값: BP10(축당 9차), 200회. 전처리·검증 CLI도 이 파일을 읽음 |
| [panda_rdf_training_full_bp24.yaml](training/panda_rdf_training_full_bp24.yaml) | 전체 링크 fresh fit용 BP24(축당 23차), 800회 설정 |
| [panda_rdf_exterior_sampling.yaml](training/panda_rdf_exterior_sampling.yaml) | 선택적 외부 표본 풀·배치 설정. BP24 전체 학습과 함께 명시적으로 지정 |

`--training-config`를 생략하면 계속 **BP10/200회**가 기본이다. 이번 분리는 기본값을
BP24로 바꾸거나 현재 배포 모델을 다시 학습하는 작업이 아니다.
별도 학습에서는 `--training-config configs/training/<파일명>`을 지정한다.
외부 표본 설정은 `--exterior-sampling-config`로 별도로 전달한다.
기존 모델을 보존하려면 항상 새 `--output` 경로를 선택한다.
자세한 명령은 [Panda RDF 구현 상세](../docs/panda_rdf_implementation.md)를 참고한다.

## 과거 실험 보관 — `archive/`

| 파일 | 보관 이유 |
|---|---|
| [panda_rdf_training_bp24.yaml](archive/panda_rdf_training_bp24.yaml) | link4 BP24/200회 비교 |
| [panda_rdf_training_bp24_400.yaml](archive/panda_rdf_training_bp24_400.yaml) | link4 BP24/400회 비교 |
| [panda_rdf_training_bp24_800.yaml](archive/panda_rdf_training_bp24_800.yaml) | 기존 배치 prefix를 보존한 800회 비교 |
| [panda_rdf_failure_study_20260929.json](archive/panda_rdf_failure_study_20260929.json) | link6 라벨 교체·link0/hand 국소 가중치 실험 프로토콜 |
| [panda_rdf_link6_sign_20260930.json](archive/panda_rdf_link6_sign_20260930.json) | link6 결정론적 부호 라벨 재학습 프로토콜 |
| [panda_rdf_hand_targeted_20260930.json](archive/panda_rdf_hand_targeted_20260930.json) | hand 취약 영역 보강·체크포인트 연속 학습 프로토콜 |
| [panda_rdf_hand_weight4_20260930.json](archive/panda_rdf_hand_weight4_20260930.json) | hand 보강점 가중치 4 비교 프로토콜 |

일반 시뮬레이션에는 필요 없지만, 과거 실험 재현 및 일부 회귀 테스트에서 사용하므로
삭제하지 않고 보관한다. `bp24_800`과 `full_bp24`는 현재 수치가 같아도 실험 목적이
달라 합치지 않았다. JSON 프로토콜에는 입력·모델 해시와 고정 조건이 포함되어 있으므로
새 실험용으로 원본을 수정하지 않는다.

2026-10-02 정리에서는 이동한 10개 파일의 내용과 SHA-256을 보존했다.
`outputs/`의 원본 로그·코드 사본·실험 설정·해시는 수정하지 않았다. 따라서 그 안의
과거 `configs/<파일명>` 경로를 현재 checkout에서 사용할 때에는 위의 새 위치로
바꿔야 한다. 보관 폴더로 옮겼다고 과거 실패 후보가 승인되는 것은 아니다.

## GPU 의존성

예전 수동 설치용 `panda_rdf_gpu_requirements.txt`는 제거했다. GPU용 PyTorch는
`pyproject.toml`의 `cuda` 선택 의존성과 `uv.lock`으로 관리한다.

```bash
uv sync --locked --extra cuda
make panda-view
```

직접 GPU 명령을 실행할 때에는 `uv run --locked --extra cuda ...`를 사용한다.
CPU 실행은 `make panda-view PANDA_RDF_DEVICE=cpu`로 선택할 수 있다.
