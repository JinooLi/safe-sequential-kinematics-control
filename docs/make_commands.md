# Make 명령 안내

`make` 또는 `make help`로 전체 목록을 본다. 이름의 `panda-`와 `planar-`가 대상 로봇을
나타낸다. `*-view`는 화면 표시를 포함하고, 기본 실행은 headless이다.

## 공통 사용법

```bash
make setup
make panda-sequential-view
make panda-direct ARGS="--duration 0.1"
make planar-run PLANAR_CONFIG=tests/fixtures/planar.yaml ARGS="--duration 0.1"
make panda-run ARGS="--help"
```

| 변수 | 용도 |
|---|---|
| `PANDA_CONFIG` | Panda YAML. 기본 `configs/panda.yaml` |
| `PLANAR_CONFIG` | Planar YAML. 기본 `configs/planar.yaml` |
| `ARGS` | 해당 CLI에 전달할 추가 옵션. 예: `--duration 1 --run-name trial` |
| `TRACE` | 저장된 NPZ 파일 경로 |
| `RESULTS_DIR` | 완료된 planar 비교 실험 디렉터리 |
| `CASE_ID` | 비교 그림에 사용할 case 번호. 0도 사용 가능 |
| `OUTPUT` | 후처리·RDF 명령의 PNG/CSV/모델 출력 경로 |

실행 결과의 디렉터리는 `ARGS="--output-dir ..."`로 지정한다.
`planar-compare-plot-all`도 여러 PNG를 만들므로 `ARGS="--output-dir ..."`를 사용한다.
변수 대신 `ARGS`에 `--trace`, `--results-dir`, `--case-id`, `--output`을 직접 전달해도 된다.
필수 입력이 빠지면 CLI가 필요한 옵션을 안내한다.

명시적인 `*-direct`, `*-sequential` target은 YAML의 제어 모드를 덮어쓴다.
`*-run`, `*-view`는 YAML의 모드를 사용한다. `ARGS`는 마지막에 전달되므로 같은 옵션을
다시 지정하면 그 값이 우선한다. IK worker 수는 Panda YAML의 `safe_ik.workers`에서,
planar 비교 worker 수는 `sequential_experiment.workers` 또는 `ARGS="--workers N"`에서 정한다.

## Panda

| 명령 | 기능 |
|---|---|
| `panda-run` | 설정된 모드로 headless 실행 |
| `panda-view` | 설정된 모드로 MuJoCo viewer 및 최종 그래프 표시 |
| `panda-direct` / `panda-direct-view` | IK를 생략하는 최종 end-effector pose 제어 |
| `panda-sequential` / `panda-sequential-view` | safe IK 후 순차 CLF-CBF 제어 |
| `panda-plot` / `panda-plot-view` | 저장된 Panda trace에서 진단 그래프 저장 / 저장·표시 |

```bash
make panda-plot TRACE=outputs/panda/<run-id>/trajectory.npz OUTPUT=outputs/panda_control.png
```

`panda-plot*`는 제어를 다시 실행하지 않는다. 여기서 필요한 NPZ는 Panda의
`time`, `clf_value`, `clf_slack`, `stage_index`, `stage_names`, `barrier_labels`,
`barrier_values`를 가진 파일이다.

## Planar

| 명령 | 기능 |
|---|---|
| `planar-run` / `planar-view` | 설정된 목표·장애물로 단일 궤적 실행 |
| `planar-direct` / `planar-direct-view` | 단일 direct 제어 |
| `planar-sequential` / `planar-sequential-view` | 단일 sequential 제어 |
| `planar-compare` | 무작위 시나리오에서 direct/sequential 쌍 비교 |
| `planar-compare-plot` | 완료된 비교 실험의 case 하나 시각화 |
| `planar-compare-plot-all` | 완료된 비교 실험의 모든 case 시각화 |
| `planar-train-rdf` | Bernstein RDF 학습과 오차 검증 |
| `planar-rdf` / `planar-rdf-view` | capsule/Bernstein 거리장 그래프 저장 / 저장·표시 |
| `planar-record-plot` / `planar-record-view` | 저장된 Recorder 신호 그래프 저장 / 저장·표시 |
| `planar-record-gui` | 저장된 신호를 버튼·시간 슬라이더로 탐색 |
| `planar-record-csv` | 저장된 Recorder NPZ를 CSV로 변환 |

단일 planar 실행은 기존 `run_simulation()`을 CLI로 연결한다. 기본 실행 시간과
초기 관절 자세는 `sequential_experiment.duration`, `initial_q`를 사용한다.
`--duration`, `--initial-q`, `--kinematic`/`--physics`, `--camera`, `--recording-gui`로
실행 조건을 조절할 수 있다. 결과는 새 `outputs/planar/<run-id>/`에 저장한다.
적용 설정·메타데이터와 함께 `trajectory.npz`, `trajectory.csv`, `plots/control.png`가 생성된다.

```bash
make planar-sequential-view PLANAR_CONFIG=tests/fixtures/planar.yaml
make planar-direct PLANAR_CONFIG=tests/fixtures/planar.yaml ARGS="--duration 1 --physics"
make planar-compare PLANAR_CONFIG=tests/fixtures/planar.yaml ARGS="--cases 2 --workers 2 --duration 0.1"
make planar-compare-plot RESULTS_DIR=outputs/sequential_comparison/<run-id> CASE_ID=0
make planar-compare-plot-all RESULTS_DIR=outputs/sequential_comparison/<run-id>
make planar-rdf-view ARGS="--representation bernstein --q 0.2 0.1 -0.3"
make planar-train-rdf OUTPUT=outputs/trial_rdf.npz ARGS="--degree 8 --samples 5000 --seed 2"
make planar-record-gui TRACE=outputs/planar/<run-id>/trajectory.npz
make planar-record-csv TRACE=outputs/planar/<run-id>/trajectory.npz OUTPUT=outputs/export.csv
```

`planar-train-rdf`에서 `OUTPUT`을 생략하면 기존 CLI와 동일하게
`models/planar_bernstein_rdf.npz`를 갱신한다.
Planar record 명령에는 `Recorder` 형식 NPZ를 사용한다. 비교 실험의 case별 NPZ는
필드 구성이 다르므로 `planar-compare-plot*`로 시각화한다.

사용자가 편집한 `configs/planar.yaml`의 `joint5`는 현재 기본 URDF에 없다.
Sequential 실행과 비교 실험은 모델에 존재하는 프레임을 설정하거나 위 예시처럼
고정 fixture를 지정해야 한다. Make 명령이 사용자 설정을 자동 수정하지 않는다.

## 환경·개발과 기존 명령 호환

| 명령 | 기능 |
|---|---|
| `setup` / `setup-runtime` | lock 기준 전체 / runtime 의존성 설치 |
| `build` | sdist와 wheel 생성 |
| `test` | 전체 테스트. `ARGS="-k parallel"`로 필터 가능 |
| `check-format` | isort·Black 검사 |
| `check` | 포맷 검사와 전체 테스트 |
| `format` | import 순서·포맷 수정 |

기존 `run`, `train-rdf`, `visualize-rdf`, `compare-sequential-run`,
`compare-sequential-visualize-all`은 대응하는 새 명령의 별칭으로 유지한다.
기존 `panda-view`도 동일하게 동작한다.
`compare-sequential-visualize`는 과거 호환을 위해 `ARGS="visualize ..."` 형태를
유지한다. 새 `planar-compare-plot`에는 `visualize`를 붙이지 않는다.
