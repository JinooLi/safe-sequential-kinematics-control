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
`SimulationStep`으로 기록한다. 그 안의 `SafetyDiagnostics`가 각 장애물--링크
barrier를 이름과 함께 보유한다. 순서는 `obs1_link1`, `obs1_link2`, ... 이며,
각 성분은 다음과 같다.

```text
h = f_link(obstacle_center, q) - obstacle_radius - safe_distance
```

양수는 안전 여유가 남았음을, `0`은 CBF 경계를, 음수는 여유 침범을 뜻한다.
생성되는 CSV에는 위 이름으로 각각의 열이 생기고, PNG plot 및 `show_gui()`의
`barrier` 항목에는 `h=0` 점선이 함께 표시된다.
