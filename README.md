# safe inverse kinematics CLBF(?)


## 주의점

### 모델 일관성 검사

모델 정보는 Pinocchio용 `settings/urdf.xml`, 시뮬레이션용
`settings/mujoco.xml`, 안전 제약용 `settings/controller.yaml`에 각각 필요하다.
중복을 숨기지 않고, `main.py` 시작 시 세 파일의 링크 길이·반지름과 원형
장애물 위치·반지름이 일치하는지 검사한다. 하나라도 다르면 제어를 시작하지
않고 오류를 낸다. 검사는 `src/application/configuration.py`의
`validate_geometry_consistency()`이 담당한다.

### 구성과 의존성 경계

`main.py`만 YAML 설정을 읽고 QP controller, OSQP solver, Pinocchio
kinematics, recorder를 조립한다. simulation loop는 이 구현체들을 직접 만들지
않고 controller/kinematics/recorder 인터페이스를 인자로 받는다. 따라서
테스트나 실제 로봇 연결에서 MuJoCo·Pinocchio 구현체를 교체할 수 있다.

```text
src/
  domain/        SDF/RDF, CBF 기하, typed state·diagnostics·interfaces
  control/       CLF-CBF QP 구성과 solver interface
  adapters/      OSQP, Pinocchio 등 외부 라이브러리 구현체
  application/   설정 검증, RDF 학습, MuJoCo simulation orchestration
  logging/       SimulationStep 저장·CSV·plot·GUI
```

### 생성 파일 정책

실행 중 생성되는 `outputs/*`, Python cache, matplotlib cache는 Git에서
제외한다. 반대로 `models/planar_bernstein_rdf.npz`는 현재 controller 설정이
직접 참조하는 재현 가능한 RDF artifact이므로 버전 관리 대상이다. 루트의
이전 중복 MuJoCo 모델 `mjmodel.xml`은 사용 경로가 없고
`settings/mujoco.xml`과 중복되어 제거했다.

### 명령 별칭

반복해서 입력할 필요 없이 아래 Makefile 별칭을 사용할 수 있다.

```bash
make help            # 가능한 별칭 확인
make run             # MuJoCo 시뮬레이션
make test            # 전체 테스트
make train-rdf       # RDF 학습
make visualize-rdf   # 거리장 PNG
make visualize-clf   # CLF q-space 등고면 PNG
make visualize-clbf  # end-effector CLBF 3D 단면 PNG
```

각 시각화의 추가 인자는 `ARGS`로 전달한다. 예를 들어 창을 열고 CLBF 해상도를
높이려면 다음처럼 실행한다.

```bash
make visualize-clbf ARGS="--show --resolution 160"
```

### Sequential end-effector randomized comparison

The following command samples one shared target and two shared circular
obstacles per case, then runs the same case with sequential end-effector
control disabled and enabled.  The runs are headless; no MuJoCo viewer opens.

Configure every batch constant in `settings/controller.yaml` under
`sequential_experiment`, then run:

```bash
make compare-sequential ARGS="run"
```

`sequential_experiment.workers` controls process-level case parallelism. Each
worker runs one paired sequential-off/on case serially; use a value no larger
than the available CPU cores (the checked-in default is `4`).

Results are saved under `outputs/sequential_comparison/<run-name>/`:

- `summary.csv` / `summary.json`: four outcome groups and final metrics.
- `case_###_sequential_off.npz`, `case_###_sequential_on.npz`: compact
  trajectories for later inspection.

The experiment-only success tolerance is independent from the controller's
link-switch tolerance.  `sequential_experiment` also contains the random seed,
run duration, initial joint posture, target/obstacle sampling ranges, collision
exclusion margins, controller-mode override, and output path. Target position
and target yaw are sampled independently. Success also
requires the configured yaw tolerance and barrier tolerance.

Render a saved case without a GUI to compare both final end-effector paths,
the obstacles, target, and final arm poses:

```bash
make compare-sequential ARGS="visualize \
  --results-dir outputs/sequential_comparison/run_YYYYMMDD_HHMMSS \
  --case-id 17"
```

## 거리장 기반 CBF 안전 제약

`src/domain/safety.py`는 각 링크를 capsule로 외접하는 2D signed distance
field(SDF)를 제공한다. 현재 URDF/MuJoCo 링크는 원통이므로 평면 투영의 끝은
평평하지만, capsule은 그 형상을 포함한다. 즉, 실제 기하보다 보수적인 안전
모델이다.

제어기는 `settings/controller.yaml`의 `safety`와 `obstacle`을 읽어 각
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

`QP_ctrl.d`는 CLF relaxation의 허용 비율이다. QP에는
`0 <= delta <= d * V` 제약이 추가되며, 값이 클수록 목표 추종 CLF를 더 많이
완화할 수 있다. CBF 안전 제약은 이 slack으로 완화되지 않는다.

`ee_obj: [x, y]`와 `ee_orientation: yaw`는 각각 end-effector의 목표 위치와
목표 yaw[rad]를 정의한다. 평면 CLF task는 `[x, y, yaw]`이며 Pinocchio 6D
spatial Jacobian에서는 `[J_x, J_y, J_angular_z] = J[[0, 1, 5], :]`를 사용한다.
특히 `J[2, :]`는 linear z 속도이므로 yaw 제어에 사용하면 안 된다. yaw 오차는
`[-pi, pi)`로 wrap하여 ±pi 경계에서 짧은 회전 방향을 택한다.

### CLF / CLBF QP 선택

`settings/controller.yaml`의 `control_mode`로 목표 수렴 constraint를 선택한다.

```yaml
control_mode: clf   # 기존 CLF-QP
# 또는
control_mode: clbf  # 최종 누적 CLBF-QP
```

`clf` 모드에서는 기존 제약

\[
\dot V - \delta \le -k_V V
\]

을 사용한다. `clbf` 모드에서는 **end-effector와 활성 장애물 사이의** barrier만
설정 순서대로 하나씩 누적한다. 링크별 CBF의 \(h\)와 혼동하지 않도록, 여기서는
unsafe 영역에서 양수가 되는 다음 barrier를 사용한다.

\[
b_i=r_i+r_{ee}+d_{safe}-\lVert p_{ee}-p_{obs,i}\rVert,
\qquad B_i=\max(b_i,0)^2-\epsilon_i,
\qquad W_0=V,\qquad W_k=W_{k-1}+\lambda_kB_k+\kappa_k
\]

두 장애물이 활성화된 기본 설정에서는 \(W_1,W_2\)를 만들며, 중간 \(W_1\)은
기록용이고 마지막 \(W_2\)만으로 단 하나의 CLBF QP 행을 추가한다.

\[
\nabla_qW_M\dot q\le-\gamma W_M
\]

CLBF 행에는 slack을 허용하지 않는다. 기존 링크별 CBF는 CLBF mode에서도 그대로
hard constraint로 유지한다. `clbf.barriers`의 첫 항은 \(\epsilon_1,\lambda_1,\kappa_1\)를
직접 지정한다. \(B_i\)는 구성상 항상 \(B_i\ge-\epsilon_i\)를 만족한다. 이후
`auto_from_previous: true` 항은 \(\epsilon_i=\epsilon_1\)로 두고, 논문의 재귀식과 해당
장애물의 \(c_{i,3}\)로 \(\lambda_i,\kappa_i\)를 자동 계산한다. recorder는 `clf`와
`clbf` 항목으로 각각 \(V\), \(W_1,\ldots,W_M\)를 기록한다.

기준 구현의 검증은 다음으로 실행한다.

```bash
uv run python -m unittest discover -s tests -v
```

### Bernstein RDF 학습과 사용

다음 명령은 capsule 기준 SDF로 링크별 2D Bernstein 계수를 학습하고
`models/planar_bernstein_rdf.npz`에 저장한다.

```bash
uv run python -m src.application.train_rdf \
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
uv run python -m src.application.visualize_rdf \
  --q 0.2 -0.4 0.1 --output outputs/distance_field.png
```

기준 capsule과 학습된 RDF를 비교하려면 `--representation capsule` 또는
`--representation bernstein`을 지정한다. 화면에 바로 표시하려면 `--show`를
추가한다.

### q-space CLF 등고면 시각화

3자유도 로봇에서는 CLF를 forward kinematics로 관절 공간에 직접 대입한
`V(q1, q2, q3)`를 만들 수 있다. 아래 명령은 기본 관절 범위
`[-pi, pi]^3`에서 `V(q)=1`인 2차원 등고면을 3D PNG로 저장한다.

```bash
uv run python -m src.application.visualize_clf \
  --level 1 --resolution 36 --output outputs/clf_level_set.png
```

`--q-min`, `--q-max`으로 세 관절에 공통으로 적용할 표시 범위, `--level`로
원하는 CLF 값을 바꿀 수 있다. 이 CLF는 controller와 동일하게
`V=2(||p_ee(q)-p_goal||^2 + wrap(yaw(q)-yaw_goal)^2)`를 사용한다.
`--show`를 더하면 회전 가능한 matplotlib 창으로도 확인할 수 있다.

### End-effector CLBF 단면 시각화

다음 명령은 최종 CLBF \(W_2\)를 task state \([x,y,\mathrm{yaw}]\)의 세 가지
2차원 조합 \((x,y)\), \((x,\mathrm{yaw})\), \((y,\mathrm{yaw})\)에 대해 3D surface로
그린다. 각 패널에서 표시하지 않는 한 축은 목표값에 고정된다. 빨간 선은 \(W_2=0\),
점은 목표 task state다.

```bash
uv run python -m src.application.visualize_clbf --show
```

기본 출력 경로는 `outputs/clbf_task_slices.png`다.

### 시뮬레이션 목표 표시

MuJoCo 시뮬레이션에는 충돌 계산에 포함되지 않는 초록색 구체 `goal_marker`가
표시된다. 위치는 별도 XML 값이 아니라 `settings/controller.yaml`의
`ee_obj: [x, y]`를 실행 시 직접 사용하므로, 목표를 바꾸려면 이 설정만 수정하면
된다.

### 장애물 비활성화 비교

`settings/controller.yaml`의 각 `obstacle` 항목에 있는 `enabled`를 `false`로
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
