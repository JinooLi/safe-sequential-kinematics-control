# Safe inverse-kinematics CLF robot arm

[아키텍처·파일 구조 개선 설명과 마이그레이션 가이드](docs/architecture.md) · [전체 Make 명령 안내](docs/make_commands.md)

```bash
uv sync --locked  # 패키지와 개발 도구 설치
```

## Panda branch: safe pose IK and trajectory control

The default entry point now controls a fixed-gripper, 7-DoF Franka Emika
Panda. It first searches the target 6D pose with damped resolved-rate IK while
projecting a soft-min capsule-safety gradient into the Moore--Penrose null
space. Pose-converged candidates are retained only when every obstacle and
self-collision barrier is nonnegative. The nearest candidate supplies the
link-endpoint targets for sequential task-space CLFs; the final stage uses an
end-effector position-and-orientation CLF. Hard CBF constraints apply at every
stage.

`trajectory.mode: direct` bypasses safe pose-IK completely and starts the
end-effector pose CLF immediately. This keeps the direct baseline independent
of IK search cost for later comparisons.

In sequential mode, each safe-IK seed ends after a candidate keeps both pose
tolerances and all safety barriers satisfied for the configured hold duration
(3 seconds by default). All configured seeds are evaluated, and the candidate
nearest the initial joint configuration is selected afterward. The first seed
starts at that initial configuration and the remaining seeds are uniform random
joint-limit samples.

IK 초기값 탐색은 `safe_ik.workers`로 병렬화합니다. 현재 기본 설정은 **4개 프로세스**이며,
`1`이면 직렬 실행합니다. 전체 seed 수는 `max_restarts: 10`으로 유지됩니다.
각 프로세스는 독립적인 기구학·안전 모델을 사용하며, 모든 탐색이 끝난 후 기존과 같은
기준으로 후보를 선택합니다. `trajectory.mode: direct`에서는 IK 프로세스를 만들지 않습니다.
자세한 동작은 [병렬 IK 설명](docs/panda_safe_pose_ik.md#parallel-seed-search)에 있습니다.

```bash
make panda-run  # 설정된 모드로 headless
make panda-view # MuJoCo viewer
```

Panda settings live in `configs/panda.yaml`. Defaults use
`0.01 s` integration and `10 s` per random IK seed, followed by a `30 s`,
`20 Hz` trajectory budget. The fixed-gripper model at `assets/robots/panda.xml` is
derived from MuJoCo Menagerie's Apache-2.0 `franka_emika_panda`; unchanged
source assets, license, and provenance are retained under
`vendor/franka_emika_panda/`.

The MuJoCo viewer shows a world-frame triad at the robot base: **red = +x**,
**green = +y**, and **blue = +z**.

Every Panda run saves its applied configuration, environment/code/model metadata,
summary, joint/control/CLF/CBF trace, and plot in a new
`outputs/panda/<run-id>/` directory. Existing run names are never overwritten.
`make panda-view` opens `plots/control.png` after its MuJoCo viewer is closed.
Use `make panda-run ARGS="--mode direct --duration 0.1"` for a short pipeline smoke test.

The Panda `qp` section exposes OSQP's `max_iterations`, `absolute_tolerance`,
and `relative_tolerance`. The run summary distinguishes actual infeasibility
(`qp_infeasible_count`) from an unsolved numerical/iteration-limit termination
(`qp_solver_failure_count`).

The original planar 3-DoF implementation and comparison utilities remain for
regression and reference, but are no longer the default `main.py` path.


## Legacy planar implementation notes

### 모델 일관성 검사

모델 정보는 Pinocchio용 `assets/robots/planar.urdf`, 시뮬레이션용
`assets/robots/planar.xml`, 안전 제약용 `configs/planar.yaml`에 각각 필요하다.
중복을 숨기지 않고, legacy planar 실행을 조립할 때 세 파일의 링크 길이·반지름과 원형
장애물 위치·반지름이 일치하는지 검사한다. 하나라도 다르면 제어를 시작하지
않고 오류를 낸다. 검사는 `src/safe_ik/application/planar_configuration.py`의
`validate_geometry_consistency()`이 담당한다.

### 구성과 의존성 경계

legacy application layer가 YAML 설정을 읽고 QP controller, OSQP solver, Pinocchio
kinematics, recorder를 조립한다. simulation loop는 이 구현체들을 직접 만들지
않고 controller/kinematics/recorder 인터페이스를 인자로 받는다. 따라서
테스트나 실제 로봇 연결에서 MuJoCo·Pinocchio 구현체를 교체할 수 있다.

```text
src/safe_ik/
  domain/        수학·기하·IK·설정 값·공통 계약
  control/       planar/Panda CLF-CBF QP와 solver 계약
  adapters/      OSQP·Pinocchio·MuJoCo 구현
  application/   설정 파싱·조립·실행·CLI·재현 정보
  experiments/   비교 시나리오·병렬 실행·분석·저장
  logging/       SimulationStep 기록·NPZ·CSV
  visualization/ 그래프·거리장·GUI
configs/         실험 YAML
assets/robots/   프로젝트 로봇 XML/URDF
tests/fixtures/ 고정 회귀 시나리오
```

### 생성 파일 정책

실행 중 생성되는 `outputs/*`, Python cache, matplotlib cache는 Git에서
제외한다. 반대로 `models/planar_bernstein_rdf.npz`는 현재 controller 설정이
직접 참조하는 재현 가능한 RDF artifact이므로 버전 관리 대상이다. 루트의
이전 중복 MuJoCo 모델 `mjmodel.xml`은 사용 경로가 없고
`assets/robots/planar.xml`과 중복되어 제거했다.

### 명령 별칭

반복해서 입력할 필요 없이 아래 Makefile 별칭을 사용할 수 있다.

```bash
make help                    # Panda / Planar별 전체 목록
make panda-sequential-view   # Panda safe IK + 순차 제어 + viewer
make panda-direct            # Panda direct headless
make planar-run              # Planar 단일 실행
make planar-view             # Planar 단일 실행 + viewer
make planar-compare          # Planar direct/sequential 무작위 비교
make planar-train-rdf        # Planar RDF 학습
make planar-rdf-view         # Planar 거리장 표시
make check                   # 포맷 검사 + 전체 테스트
```

### Sequential end-effector randomized comparison

Sequential controller의 IK branch 선택, 단계별 CLF, CBF 제약 및 on/off 비교 기준은
[Sequential control algorithm](docs/sequential_control.md)에 정리되어 있다.
실험 결과를 어떻게 해석하고 비교할지는
[Sequential CLF performance analysis](docs/sequential_performance_analysis.md)에
정리되어 있다.

The following command samples one shared target and two shared circular
obstacles per case, then runs direct and sequential control. The runs are
headless; no MuJoCo viewer opens.

Configure every batch constant in `configs/planar.yaml` under
`sequential_experiment`, then run:

```bash
make planar-compare
```

현재 작업 설정의 `joint5`는 기본 planar 모델에 없는 프레임입니다. 사용자 변경값은
보존했으며, 모델에 맞는 프레임을 선택하거나 아래 smoke test의 고정 fixture를 사용하세요.

`sequential_experiment.workers` controls process-level case parallelism. Each
worker runs both methods for one shared case serially; use a value no larger
than the available CPU cores.

Results are saved under `outputs/sequential_comparison/<run-name>/`:

- `config.yaml`, `resolved_config.json`, `metadata.json`: 전체 적용 설정, Git 상태, 환경·소스·모델 해시.

- `summary.csv` / `summary.json`: direct와 sequential의 case별 진단 및
  aggregate analysis.
- `analysis.json`: method-wise success/failure taxonomy and paired metric
  differences for cases where both methods succeeded.
- `paired_metrics.png`: 수렴 시간, control effort, 최소 barrier, 최대 CLF
  slack, 목표 도달 전 nonzero slack 비율의 task-decomposition paired scatter plots.
- `case_###_direct.npz`, `case_###_sequential.npz`: command, CLF slack, QP
  solve time, active stage를 포함한 trajectory.

The experiment-only success tolerance is independent from the controller's
link-switch tolerance.  `sequential_experiment` also contains the random seed,
run duration, initial joint posture, target/obstacle sampling ranges, collision
exclusion margins, controller-mode override, and output path. Target position
and target yaw are sampled independently. Success also
requires the configured yaw tolerance and barrier tolerance.

The configured batch compares:
`Direct EE CLF + CBF` versus `Sequential CLF + CBF`. Both use the same
configured yaw weight. To smoke-test the output pipeline without changing YAML,
temporarily override the batch size and run duration:

```bash
make planar-compare ARGS="--config tests/fixtures/planar.yaml --cases 1 --workers 1 --duration 0.1"
```

Render a saved case without a GUI to compare both final end-effector paths,
the obstacles, target, and final arm poses:

```bash
make planar-compare-plot ARGS="\
  --results-dir outputs/sequential_comparison/run_YYYYMMDD_HHMMSS \
  --case-id 17"
```

## 거리장 기반 CBF 안전 제약

`src/safe_ik/domain/safety.py`는 각 링크를 capsule로 외접하는 2D signed distance
field(SDF)를 제공한다. 현재 URDF/MuJoCo 링크는 원통이므로 평면 투영의 끝은
평평하지만, capsule은 그 형상을 포함한다. 즉, 실제 기하보다 보수적인 안전
모델이다.

제어기는 `configs/planar.yaml`의 `safety`와 `obstacle`을 읽어 각
링크--원형 장애물 쌍에 대해 다음 CBF 제약을 QP에 추가한다.

$$
\frac{\partial h(q)}{\partial q}\,\dot{q} \geq -\alpha h(q)
$$

$$
h(q) = f_{\mathrm{link}}(\mathrm{obstacle\_center}, q)
        - \mathrm{obstacle\_radius}
        - \mathrm{safe\_distance}
$$

`safe_distance`는 링크와 장애물의 물리 반지름 외에 추가할 여유거리다.
Bernstein RDF에서는 검증 데이터의 최악 근사오차보다 큰 값을 이 여유에
포함해 보수성을 유지한다. `link_lengths`와 `link_radius`는 현재 반드시
URDF/MuJoCo와 동일하게 유지해야 한다.

CLF relaxation slack은 `delta >= 0`만 제약하며, `QP_ctrl.Q`의 비용 가중치로
크기를 억제한다. CBF 안전 제약은 이 slack으로 완화되지 않는다.

`ee_obj: [x, y]`와 `ee_orientation: yaw`는 각각 end-effector의 목표 위치와
목표 yaw[rad]를 정의한다. 평면 CLF task는 `[x, y, yaw]`이며 Pinocchio 6D
spatial Jacobian에서는 `[J_x, J_y, J_angular_z] = J[[0, 1, 5], :]`를 사용한다.
특히 `J[2, :]`는 linear z 속도이므로 yaw 제어에 사용하면 안 된다. yaw 오차는
실험 성공 판정과 IK branch 거리에서는 shortest-angle wrap을 사용한다. CLF의 yaw
항은 wrap 대신 `1 - cos(yaw - yaw_target)`를 사용해 ±pi에서 미분 불가능한 점을
피한다.

### CLF QP

목표 수렴에는 단일 CLF-QP 제약

\[
\dot V - \delta \le -k_V V
\]

을 사용한다. 링크별 CBF와 joint-limit CBF는 hard constraint로 유지되며
CLF slack으로 완화되지 않는다.

기준 구현의 검증은 다음으로 실행한다.

```bash
uv run python -m unittest discover -s tests -v
```

### Bernstein RDF 학습과 사용

다음 명령은 capsule 기준 SDF로 링크별 2D Bernstein 계수를 학습하고
`models/planar_bernstein_rdf.npz`에 저장한다.

```bash
uv run python -m safe_ik.application.train_rdf \
  --degree 12 --samples 12000 --seed 0
```

학습 출력의 `Validation max error`보다 큰 RDF 오차 여유를
`safety.safe_distance`에 반영한 뒤, `safety.representation: bernstein`으로
설정한다. 현재 저장된 모델의 sampled max error는 약 `0.024857 m`이며
설정값 `0.08 m`는 기존 `0.05 m` clearance와 반올림한 `0.03 m` RDF 여유를
합친 값이다.

`BernsteinSDF2D.sample_level_set(count, level)`은 무작위 seed를 고른 후
해석적 gradient를 쓰는 Newton projection으로 정확한 level-set 점을 만든다.
이는 surface 시각화나 표면점 기반 비교용이며, 제어기는 매 주기 장애물의
중심점을 RDF에 직접 질의하므로 표면점 샘플링을 기다리지 않는다.

### 거리장 시각화

현재 설정의 RDF, 로봇 표면(`f=0` 실선), 안전 level set
(`f=safe_distance` 점선), 장애물을 하나의 PNG로 렌더링한다.

```bash
uv run python -m safe_ik.application.visualize_rdf \
  --q 0.2 -0.4 0.1 --output outputs/distance_field.png
```

기준 capsule과 학습된 RDF를 비교하려면 `--representation capsule` 또는
`--representation bernstein`을 지정한다. 화면에 바로 표시하려면 `--show`를
추가한다.

### 시뮬레이션 목표 표시

MuJoCo 시뮬레이션에는 충돌 계산에 포함되지 않는 초록색 구체 `goal_marker`가
표시된다. 위치는 별도 XML 값이 아니라 `configs/planar.yaml`의
`ee_obj: [x, y]`를 실행 시 직접 사용하므로, 목표를 바꾸려면 이 설정만 수정하면
된다.

### 장애물 비활성화 비교

`configs/planar.yaml`의 각 `obstacle` 항목에 있는 `enabled`를 `false`로
바꾸면 해당 장애물은 CBF 제약, MuJoCo 장면, 거리장 PNG에서 모두 제외된다.
초기 자세와 목표를 유지한 채 장애물이 있을 때와 없을 때를 비교할 수 있다.

### Barrier 기록과 시각화

`run_simulation(..., recorder=Recorder())`은 매 제어 step을 typed
`SimulationStep`으로 기록한다. end-effector 위치 `ee`와 world-frame yaw
`ee_yaw`[rad]는 NPZ/CSV/PNG 및 Recorder GUI에서 각각 확인할 수 있다. 그 안의
`SafetyDiagnostics`가 각 장애물--링크 barrier를 이름과 함께 보유한다. 순서는
`obs1_link1`, `obs1_link2`, ... 이며, 각 성분은 다음과 같다.

```text
h = f_link(obstacle_center, q) - obstacle_radius - safe_distance
```

양수는 안전 여유가 남았음을, `0`은 CBF 경계를, 음수는 여유 침범을 뜻한다.
생성되는 CSV에는 위 이름으로 각각의 열이 생기고, PNG plot 및 `show_gui()`의
`barrier` 항목에는 `h=0` 점선이 함께 표시된다.

관절 범위도 `safety.joint_limits`에서 동일한 zeroing-CBF로 제약한다. 각 유한
범위 `[q_min, q_max]`에 대해 `q_i-q_min`과 `q_max-q_i`를 barrier로 두므로,
QP는 각각 하한 밖과 상한 밖으로 향하는 속도를 제한한다. 범위가 없는 관절은
`null`로 지정하며, CBF 이득은 `safety.joint_limit_alpha`로 조정한다.

### Kinematic CBF simulation

기본 시뮬레이션은 QP의 결정변수 `u = q_dot`를 직접 적분한다. 따라서 CBF
제약을 평가한 속도와 실제 시뮬레이션 속도가 동일하며, velocity actuator의
추종 지연으로 barrier가 침범하는 문제를 피한다. actuator 동역학을 포함한
실험이 필요하면 `run_simulation(..., kinematic=False)`를 사용한다.
