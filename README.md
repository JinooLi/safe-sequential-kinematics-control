# 안전 역기구학 기반 CLF 로봇 팔

Panda 7자유도 팔의 안전 자세 IK, CLF-CBF 관절 속도 제어, RDF 자기 충돌 평가를
연구하는 저장소다. 평면 3자유도 로봇의 비교 실험도 포함한다.

**구현을 처음 읽을 때는 [현재 구현 지도](docs/current/README.md)부터 시작한다.**
전체 실행 흐름을 먼저 설명하고, 핵심 기능이 받는 입력과 반환하는 출력을 연결한다.

| 문서 | 용도 |
|---|---|
| [현재 구현 지도](docs/current/README.md) | 현재 코드의 실행 흐름, 기능별 입력·출력, 결과 파일 |
| [코드 의존성 구조](docs/current/dependencies.md) | 계층별 import 방향, 공통 타입, 파일 이동과 호출 변경 |
| [기존 연구 기록](docs/history/README.md) | 설계 이유, 이전 요구사항, 상세 수식, 실험·성능 측정 기록 |
| [문서 관리 기준](docs/README.md) | 현재 설명과 과거 기록의 구분 및 갱신 방법 |
| [설정 파일 안내](configs/README.md) | 실행·학습·재현용 설정의 위치 |

## 실행 시작

```bash
uv sync --locked
# CPU에서 direct 경로를 짧게 확인한다. 수렴 성능 검증용 실행은 아니다.
make panda-direct PANDA_RDF_DEVICE=cpu ARGS="--duration 0.1"
make help
```

기본 입력은 [Panda 설정](configs/panda.yaml)이다. 위 명령의 `direct`는 IK를 생략하며,
RDF를 사용하는 경우 설정과 호환되고 검증을 통과한 학습 산출물이 필요하다.
실행 결과는 새 `outputs/panda/<run-id>/` 디렉토리에 저장한다.

`make panda-view`는 현재 YAML의 모드로 계산·저장한 뒤 기록을 재생한다.
Panda 실행 Make 명령은 기본적으로 CUDA를 선택하므로, CPU를 쓰려면
`PANDA_RDF_DEVICE=cpu`를 명시한다. GUI에는 display가 필요하다.
세부 옵션의 기준은 [Makefile](Makefile)과 각 명령의 `--help`다.

Panda의 Direct/Sequential 몬테카를로 비교는 다음 명령으로 실행한다.

```bash
make panda-compare ARGS="--dry-run" # 최종 설정만 확인
make panda-compare                 # 전용 YAML의 main 설정으로 바로 비교 실험 실행
```

실험 설정은 [전용 YAML](configs/experiments/panda_comparison.yaml)에 모았다.
표본 수, gain·계산 주기·종료 판정, 장면 생성 범위와 결과 경로를 조절한다.
다른 실험 파일은 `PANDA_EXPERIMENT_CONFIG=<파일>`로 선택한다. 결과에는 **원본 실험 YAML과
최종 적용 설정**을 함께 저장한다. 같은 조건으로 재개하려면 YAML의 `output` 또는
`OUTPUT=<기존 디렉토리>`를 지정한다. 실행 예시는
[Panda 실험 프로토콜](docs/experiments/panda_monte_carlo_protocol.md#실행과-재개)에 있다.

저장된 비교 실험은 `panda-replay`로 MuJoCo에서 재생한다. `METHOD`는 `direct` 또는
`sequential`이며 기본값은 `sequential`이다. 당시 저장한 목표·장애물·관절 궤적을 사용한다.

```bash
make panda-replay RESULTS_DIR=outputs/panda_comparison/my_experiment ARGS="--list"
make panda-replay RESULTS_DIR=outputs/panda_comparison/my_experiment CASE_ID=seed2333_0000 METHOD=sequential
```

`ARGS="--dry-run"`으로 GUI 없이 입력 파일과 모델을 확인할 수 있다.
재생은 CUDA 없이 가능하며, 화면을 표시할 수 있는 데스크톱 환경이 필요하다.

기존 긴 README는 [이전 프로젝트 안내](docs/history/project_readme.md)에 보존했다.
