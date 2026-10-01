# 안전 역기구학 기반 CLF 로봇 팔

[아키텍처·파일 구조 개선 설명과 마이그레이션 가이드](docs/architecture.md) · [전체 Make 명령 안내](docs/make_commands.md)

```bash
uv sync --locked  # 패키지와 개발 도구 설치
```

## Panda: 안전 자세 IK와 궤적 제어

기본 진입점은 그리퍼가 고정된 7자유도 Franka Emika Panda를 제어한다.
먼저 감쇠된 속도 기반 IK로 목표 6차원 자세를 탐색하면서,
soft-min 안전 기울기를 Moore–Penrose 영공간에 투영한다.
자세가 수렴해도 모든 장애물·자기 충돌 장벽이 음수가 아니어야 후보로 유지한다.
가장 가까운 후보에서 순차 작업 공간 CLF의 링크 끝점 목표를 만들고,
마지막 단계에서는 말단 장치 위치·방향 CLF를 사용한다.
모든 단계에 완화 불가능한 CBF 제약을 적용한다.

자기 충돌 기본값은 시각화 메시 기반 RDF(`safety.self_collision_model: rdf`)이며,
닫힌 손가락은 손 형상에 합친다. 외부 장애물은 기존 캡슐–구 모델을 유지한다.
호환되는 학습 산출물이 독립적인 표면 근처 오차 10 mm 검증을 통과해야
IK나 궤적 제어에 사용할 수 있다. 산출물이 없거나 검증에 실패했거나
설정과 호환되지 않으면 실행을 중단한다. 캡슐 자기 충돌을 사용하려면
`safety.self_collision_model: capsule`을 명시해야 한다.
IK 탐색 모델만 따로 선택하려면 `safe_ik.self_collision_model: capsule` 또는
`rdf`를 지정한다. 생략/null이면 `safety.self_collision_model`을 따른다.
예를 들어 IK는 capsule, 궤적 제어는 RDF로 설정할 수 있다. 이 경우 찾은
후보를 제어용 RDF로 재검사해 통과한 후보만 사용하며, 기존 CBF 제약은 유지한다.
직렬·병렬 탐색 모두 적용되고 `make panda-run`/`make panda-view`로 그대로 실행한다.
물리적 여유 30 mm와 RDF 오차 여유 10 mm를 합산해 장벽에서 40 mm를 차감한다.
기존 5 mm 설정의 산출물은 학습 구와 설정이 달라 **그대로 제어에 사용할 수 없다**.
새 설정과 호환되는 전체 링크 산출물의 거리·미분 검증이 필요하다.
[구현 진행 상황](docs/panda_rdf_implementation_status.md)과
[구현 상세](docs/panda_rdf_implementation.md)를 참조한다.

```bash
make panda-train-rdf                         # 오프라인 학습·보고서 생성; 실패 시 종료 코드 2
make panda-validate-rdf                      # 계수 재사용, 오프라인 검증 재실행
make panda-validate-rdf ARGS="--refresh-queries" # 재학습 없이 질의 밀도·시드 변경 반영
make panda-inspect-mesh                      # 원본 visual mesh의 열린 경계 3D HTML
make panda-prepare-mesh                      # 공식 voxel 전처리 + 원본/재구성 비교 HTML
```

Mesh 진단 결과는 `outputs/panda_mesh_inspection/index.html`에 저장한다.
파일을 브라우저로 열면 링크별 열린 경계(빨강), 비다양체 연결(주황),
면 방향 불일치(보라)를 회전·확대해서 볼 수 있다. 외부 CDN이나 서버는 필요 없다.
경계를 선택하면 해당 영역으로 확대된다. 진단용 복사본에서 같은 위치의 정점을
묶어 UV/재질 경계의 중복을 제거하지만, 원본 mesh나 학습 artifact는 변경하지 않는다.
표시된 경계는 실제 물리적 구멍과 동일하지 않으며, 이 도구는 안전성 인증이 아니다.

학습은 이제 원본 Visual mesh를 공식 RDF 방식의 128³ voxel SDF와 Marching
Cubes로 재구성한 뒤 수행한다. 10 mm 검증 기준도 **재구성 mesh**에 적용한다.
거리 정확도는 기준 거리 `d_mesh >= 0`인 외부·표면 점만 평가하며, 합격 판정은
`0 <= d_mesh <= 60 mm`에서 수행한다. 내부 점은 통계에서 제외하지만 signed
학습 데이터와 미분·경계 검사는 유지한다. 평가 영역 정책만 바뀌고 구·정규화
설정은 같은 산출물에 한해 계수를 유지한 재검증이 가능하다. `error_margin`
변경은 학습 구도 바꾸므로 `--validate-only`로 호환성 검사를 우회할 수 없다.
원본↔재구성 표면 차이는 양방향 통계와 HTML로 보고하며 별도 합격 임계값은 없다.
`panda-prepare-mesh`의 결과는 `outputs/panda_mesh_reconstruction/reference.npz`와
`reference.meshes.html`이고, RDF 계수 학습이나 제어 승인을 수행하지 않는다.
일반 학습도 산출물 옆에 `.meshes.html`을 생성한다. 기존 학습 파일은 구형
스키마이므로 새 기준의 재학습이 필요하다. 먼저 `OUTPUT=outputs/.../candidate.npz`로
별도 저장하면 기존 후보와 보고서를 보존할 수 있다.

`trajectory.mode: direct`는 안전 자세 IK를 생략하고 말단 장치 자세 CLF를
바로 시작한다. 이후 비교에서 직접 제어 기준선에 IK 탐색 비용이 섞이지 않게 한다.
RDF 산출물 검증과 궤적 초기 안전 검사는 직접 제어 모드에도 적용한다.

순차 모드에서는 후보가 자세 허용오차와 모든 안전 장벽 조건을 설정된 유지
시간(기본 3초) 동안 만족하면 해당 시드의 탐색을 끝낸다. 모든 설정 시드를
평가한 뒤 초기 관절 자세와 가장 가까운 후보를 선택한다. 첫 시드는 초기
자세 자체에서 시작하고 나머지는 관절 한계 안에서 균일하게 뽑는다.

IK 초기값 탐색은 `safe_ik.workers`로 병렬화한다. 현재 기본 설정은
**5개 프로세스**이며, 1이면 직렬 실행한다. 전체 시드 수는
`max_restarts: 10`이다. 각 프로세스는 독립적인 기구학·안전 모델을 사용하며,
탐색 완료 후 동일한 기준으로 후보를 선택한다. 직접 제어 모드에서는 IK
프로세스를 만들지 않는다. [병렬 IK 설명](docs/panda_safe_pose_ik.md#parallel-seed-search)을 참조한다.

```bash
make panda-run  # CUDA 의존성 설치 + GPU RDF, 화면 없이 실행
make panda-view # GPU RDF + 진행률 → 전체 계산·저장 → MuJoCo 재생 → 그래프
make panda-view PANDA_RDF_DEVICE=cpu # CUDA 없이 CPU 참조 경로 실행
make panda-gpu-view # 명시적인 CUDA 실행 명령도 유지
```

Panda run/view Make 명령의 기본값은 `PANDA_RDF_DEVICE=cuda`다.
`uv run --locked --extra cuda`와 `--rdf-device cuda`를 사용하므로 pull 후 별도
`pip install torch` 없이 잠긴 CUDA 의존성을 설치한다. `cpu`를 명시하면 CUDA extra를
설치하지 않는다. 이 Make 변수는 YAML의 `safety.rdf.runtime_device`보다 우선하며,
직접 `uv run safe-ik ...`를 실행할 때는 YAML 기본값(cpu)을 따른다.
NVIDIA CUDA가 필요하며, 사용할 수 없으면 CPU로 자동 대체하지 않는다.
모델 재학습은 필요 없다. 현재 capsule IK는 그대로이며 RDF를 사용하는 안전성 평가만
GPU로 옮긴다. [구현·성능·검증](docs/panda_rdf_runtime_optimization.md#9-runtime-cuda-rdf-추가--2026-10-01)을 참조한다.

RTX 4070 Linux 컴퓨터에서는 저장소의 `panda` 브랜치에서 다음처럼 실행한다.
모델과 메시가 저장소에 포함되어 있어 학습 결과를 별도로 복사할 필요는 없다.

```bash
git pull --ff-only origin panda
make panda-view
```

Make는 PATH의 uv를 우선 사용하고, 없으면 `~/.local/bin/uv`를 찾는다.
다른 위치에 설치했을 때만 `UV=/설치경로/uv`를 전달한다.
NVIDIA 드라이버와 데스크톱 display는 필요하며 uv가 설치하는 대상이 아니다.
첫 CUDA 설치는 수 GB 다운로드가 발생한다. SSH의 display 없는 셸에서는
`make panda-run ARGS="--mode direct --duration 0.1"`로 먼저 동작을 확인할 수 있다.
서버에서 학습 등 다른 실행이 진행 중이면 해당 checkout을 pull하지 않는다.

Panda 실행은 `tqdm`으로 IK의 완료 시드·현재 반복 수와 궤적 계산 스텝을 표시한다.
병렬 탐색과 실패 후 자세 전용 진단도 별도로 표시한다. `panda-view`는 계산 중
뷰어를 열지 않고, 계산·저장 후 원래 시간 간격으로 기록을 재생한다. 재생 중에는
IK·RDF·QP를 다시 계산하지 않는다. 뷰어를 닫으면 진단 그래프가 열리고, 그래프까지
닫으면 종료한다. `ARGS="--no-progress"`로 진행 표시를 끌 수 있다.

Panda 설정은 `configs/panda.yaml`에 있다. IK는 `safe_ik.iteration_number`로
시드당 최대 반복 횟수(기본 1000회)를 지정하며, 실제 실행 시간 제한은 아니다.
내부 수치 간격은 기존의 0.01을 유지한다. 이후 궤적 예산은 60초,
제어 주파수는 20 Hz다.
`assets/robots/panda.xml`의 고정 그리퍼 모델은 MuJoCo Menagerie의
Apache-2.0 `franka_emika_panda`에서 가져왔다. 원본 자료, 라이선스, 출처는
`vendor/franka_emika_panda/`에 유지한다.

MuJoCo 뷰어는 로봇 베이스에 월드 좌표축을 표시한다.
**빨강은 +x, 초록은 +y, 파랑은 +z**다.

각 Panda 실행은 적용 설정, 환경·코드·모델 메타데이터, 요약,
관절·제어·CLF·CBF 이력, 그래프를 새로운 `outputs/panda/<run-id>/`에
저장한다. 기존 실행 이름을 덮어쓰지 않는다. `make panda-view`는 MuJoCo
뷰어를 닫은 뒤 `plots/control.png`를 연다.
짧은 동작 확인에는 `make panda-run ARGS="--mode direct --duration 0.1"`을
사용한다. RDF 기본 설정에서는 검증을 통과한 산출물이 필요하다.

Panda의 `qp` 설정은 OSQP의 `max_iterations`, `absolute_tolerance`,
`relative_tolerance`를 제공한다. 실행 요약은 실제 실행 불가능한 QP
(`qp_infeasible_count`)와 수치 오류·반복 한도 때문에 해를 구하지 못한 종료
(`qp_solver_failure_count`)를 구분한다.

기존 평면 3자유도 구현과 비교 도구는 회귀 검증·참고용으로 유지하지만,
기본 `main.py` 실행 경로는 아니다.

## 기존 평면 로봇 구현 안내

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

### 순차 말단 장치 제어의 무작위 비교

Sequential controller의 IK branch 선택, 단계별 CLF, CBF 제약 및 on/off 비교 기준은
[순차 제어 알고리즘](docs/sequential_control.md)에 정리되어 있다.
실험 결과를 어떻게 해석하고 비교할지는
[순차 CLF 성능 해석](docs/sequential_performance_analysis.md)에
정리되어 있다.

다음 명령은 사례마다 공유 목표 하나와 원형 장애물 두 개를 뽑고,
직접 제어와 순차 제어를 실행한다. 화면 없이 실행하며 MuJoCo 뷰어는 열지 않는다.

`configs/planar.yaml`의 `sequential_experiment`에 일괄 실험 상수를
설정한 뒤 실행한다:

```bash
make planar-compare
```

현재 작업 설정의 `joint5`는 기본 planar 모델에 없는 프레임입니다. 사용자 변경값은
보존했으며, 모델에 맞는 프레임을 선택하거나 아래 smoke test의 고정 fixture를 사용하세요.

`sequential_experiment.workers`는 사례별 프로세스 병렬 실행 수를 정한다.
각 작업자는 공유 사례 하나의 두 방법을 직렬로 실행한다.
사용 가능한 CPU 코어 수 이하로 설정한다.

결과는 `outputs/sequential_comparison/<run-name>/`에 저장한다:

- `config.yaml`, `resolved_config.json`, `metadata.json`: 전체 적용 설정, Git 상태, 환경·소스·모델 해시.

- `summary.csv` / `summary.json`: direct와 sequential의 case별 진단 및
  집계 분석.
- `analysis.json`: 방법별 성공·실패 분류와 두 방법이 모두 성공한 사례의
  지표 차이.
- `paired_metrics.png`: 수렴 시간, 제어 노력, 최소 장벽, 최대 CLF 완화량,
  목표 도달 전 완화 변수가 0이 아닌 비율을 사례별로 짝지어 비교한 산점도.
- `case_###_direct.npz`, `case_###_sequential.npz`: 제어 명령, CLF 완화량,
  QP 풀이 시간, 활성 단계를 포함한 궤적.

실험 성공 판정용 허용오차는 제어기의 링크 전환 허용오차와 별개다.
`sequential_experiment`에는 난수 시드, 실행 시간, 초기 관절 자세,
목표·장애물 표본 범위, 충돌 제외 여유, 제어 모드 재정의, 출력 경로도 들어 있다.
목표 위치와 목표 요각은 독립적으로 뽑는다. 성공하려면 설정한 요각 및
장벽 허용오차도 만족해야 한다.

설정된 실험은 직접 말단 장치 CLF + CBF(`Direct EE CLF + CBF`)와
순차 CLF + CBF(`Sequential CLF + CBF`)를 비교한다. 두 방법은 같은
요각 가중치를 사용한다. YAML을 바꾸지 않고 출력 파이프라인을 간단히
검사하려면 사례 수와 실행 시간을 임시로 지정한다:

```bash
make planar-compare ARGS="--config tests/fixtures/planar.yaml --cases 1 --workers 1 --duration 0.1"
```

저장된 사례를 GUI 없이 그려 두 방법의 최종 말단 장치 경로, 장애물,
목표, 최종 팔 자세를 비교할 수 있다:

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

### 장벽값 기록과 시각화

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

### 기구학 기반 CBF 시뮬레이션

기본 시뮬레이션은 QP의 결정변수 `u = q_dot`를 직접 적분한다. 따라서 CBF
제약을 평가한 속도와 실제 시뮬레이션 속도가 동일하며, velocity actuator의
추종 지연으로 barrier가 침범하는 문제를 피한다. actuator 동역학을 포함한
실험이 필요하면 `run_simulation(..., kinematic=False)`를 사용한다.
