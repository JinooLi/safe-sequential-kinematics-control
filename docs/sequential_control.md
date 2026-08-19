# Sequential end-effector control

## 목적

이 제어기는 최종 목표 관절 자세 `q*`를 직접 추종하지 않는다. 대신 목표 pose에서
계산한 안전한 IK configuration의 각 링크 끝점 위치를 task 목표로 사용한다. 3-DOF
평면 로봇에서는 다음 세 목표를 만든다.

```text
목표 pose (p*, yaw*)
        │
        ▼
안전한 IK 해 q* 선택
        │
        ├─ 1단계: joint2  → q*의 link1 끝점 위치
        ├─ 2단계: joint3  → q*의 link2 끝점 위치
        └─ 3단계: ee_joint → q*의 link3 끝점 위치 (= p*) + yaw*
```

각 단계의 task frame은 `sequential_end_effector.frame_names`에 정의한다. 기본
설정은 `[joint2, joint3, ee_joint]`이다.

## 1. 목표 configuration 선택

최종 목표 position `p*`, 목표 yaw `yaw*`, 링크 길이 `(l1, l2, l3)`에서 마지막
링크를 제외한 손목 위치를 먼저 계산한다.

$$
p_w = p^* - l_3
\begin{bmatrix}\cos(yaw^*)\\\sin(yaw^*)\end{bmatrix}
$$

base--elbow--wrist의 2-link 기하로 elbow-up 및 elbow-down 두 해를 계산한다.
각 해는 다음 조건을 만족해야 한다.

- 목표 pose에 도달한다.
- finite joint limit을 만족한다.
- 모든 링크 capsule이 활성 장애물과 `safe_distance`를 포함한 안전 여유를
  만족한다.
- 비인접 링크인 link1과 link3의 중심선이 교차하지 않는다. 이 검사는 목표 IK
  후보를 제외하기 위한 binary geometry test이며, 실행 중 CBF 제약은 아니다.

둘 이상의 안전한 해가 있으면 제어 시작 시 현재 관절 자세와 wrapped joint-angle
거리가 가장 가까운 해를 선택한다. 안전한 해가 하나도 없으면 controller 생성이
실패하며 목표를 불가능한 것으로 처리한다.

선택된 해의 누적 forward kinematics로 링크 끝점 위치
`[p1*, p2*, p3*]`를 계산한다. 이것이 각 sequential 단계의 position 목표다.

## 2. CLF 구성

활성 frame의 현재 position을 `p`, planar Jacobian을
`J_task = [Jx; Jy; Jyaw]`라 하자. 모든 CLF에는 전역 계수
`safety.clf_weight`가 곱해진다.

### 중간 단계: link1, link2

첫 두 단계는 position-only CLF를 사용한다.

$$
V_i = w_{clf}\,2\lVert p-p_i^*\rVert^2
$$

$$
\nabla_q V_i = w_{clf}\,4(p-p_i^*)^T[J_x;J_y]
$$

### 마지막 단계: link3 / end-effector

마지막 단계에 진입하는 즉시 position과 yaw를 함께 사용한다. position 목표에
먼저 도달한 뒤 yaw를 제어하는 별도 phase는 없다.

$$
V_3 = w_{clf}\,2\left(
  \lVert p-p_3^*\rVert^2 + w_{yaw}\,
  \operatorname{wrap}(yaw-yaw^*)^2
\right)
$$

여기서 `w_yaw`는 `sequential_end_effector.final_yaw_weight`이다. `1`보다
작게 설정하면 마지막 단계에서 position에 비해 yaw의 영향력을 낮춘다.

## 3. CLF-CBF QP

결정변수는 관절 속도와 CLF slack이다.

$$
z = [\dot q^T,\delta]^T
$$

CLF 제약은 다음과 같다.

$$
\nabla_q V\dot q-\delta \le -k_VV,
\qquad 0\le\delta\le dV
$$

안전 제약은 slack으로 완화되지 않는 hard CBF다.

- 모든 활성 장애물--링크 쌍:
  \(\nabla_qh\dot q\ge-\alpha h\)
- finite joint limit의 상한/하한 zeroing CBF
- 관절별 속도 상한 `velocity_limit`

안전 거리장은 `safety.representation`에 따라 exact capsule SDF 또는 Bernstein
RDF를 사용한다.

## 4. 단계 전환

이 절은 `staged` 모드에만 적용된다. 마지막 링크 이전 단계에서는 아래 중 하나가
발생하면 다음 링크 frame으로 전환한다.

1. 활성 frame이 자신의 목표점에서 `goal_tolerance` 이내에 도달
2. QP infeasible
3. 관절 속도가 `velocity_tolerance` 이하이고, 최소 link--obstacle barrier가
   `barrier_tolerance` 이하인 CBF 경계 정지

마지막 `ee_joint`에서는 더 이상 전환하지 않는다.

### Stage 1의 CBF 정지 복구

stage 1에서 최소 barrier가 `barrier_tolerance` 이내이고, 관절 속도가
`velocity_tolerance` 이하인 정지가 `link1_stuck_cycles`회 연속 발생하면, 제어기는
즉시 stage 2로 전환하지 않는다. 먼저 선택된 IK 해의 첫 관절 목표를 같은 물리 자세의
반대 winding으로 바꾼다.

$$
q_{1,\mathrm{reverse}}^* = q_{1,\mathrm{normal}}^* \mp 2\pi
$$

복구 중에는 link1 endpoint position CLF 대신 다음 관절각 CLF를 사용하며, CBF와
joint-limit 제약은 그대로 hard constraint로 유지한다.

$$
V_{\mathrm{reverse}} = w_{clf}w_{\mathrm{reverse}}
\left(q_1-q_{1,\mathrm{reverse}}^*\right)^2
$$

반대 경로에서도 첫 관절이 CBF 경계에서 정지하거나, QP가 infeasible이거나,
`link1_reverse_max_steps` 안에 도달하지 못하면 stage 1을 포기하고 stage 2로
전환한다. 각 전환 사유는 실험 결과의 `switch_reason`에 기록된다.

시뮬레이션 루프는 frame 전환이 발생한 동일 상태에서 새 frame의 kinematics와 QP를
다시 계산한다. 따라서 다음 timestep까지 기다리지 않고 새 단계의 명령을 구한다.

## Sequential off 비교 기준

비교 실험에서 sequential off는 중간 목표점과 IK branch 선택을 제어에 사용하지
않는다. 처음부터 최종 end-effector를 대상으로 다음 CLF를 사용한다.

$$
V_{off} = w_{clf}\,2\left(
  \lVert p-p^*\rVert^2 + \operatorname{wrap}(yaw-yaw^*)^2
\right)
$$

즉 `final_yaw_weight`는 sequential on의 마지막 단계에만 적용되며, off에서는 yaw
가중치가 `1`이다. 두 비교 조건은 동일한 목표 pose, 장애물, 초기 관절 자세 및 hard
CBF/joint-limit 제약을 사용한다. 비교 case는 안전한 IK branch가 하나 이상 존재할
때만 채택하므로, IK 해가 없거나 모든 해가 충돌/교차로 제외되는 목표는 재샘플하며
실험 결과에 포함하지 않는다.

## 관련 설정

```yaml
sequential_end_effector:
  enabled: true
  frame_names: [joint2, joint3, ee_joint]
  goal_tolerance: 0.2 # 목표 위치 허용 오차 [m]
  velocity_tolerance: 0.001 # 정지 판정 관절 속도 norm [rad/s]
  barrier_tolerance: 0.001 # CBF h 경계 근접 판정 [m]
  # 마지막 link3의 position+yaw CLF에서 yaw 오차항에만 곱하는 가중치.
  # 1보다 작게 두면 position보다 yaw의 영향력이 작아진다.
  final_yaw_weight: 0.4
  # link1이 CBF 경계에서 정지하면 q1의 반대 winding(q1* ± 2π)을 시도한다.
  # 이 복구 제어에도 obstacle/joint-limit CBF는 hard constraint로 유지된다.
  link1_reverse_enabled: true
  link1_stuck_cycles: 10 # 연속 정지 횟수 후 반대 방향 회전 시작
  link1_reverse_max_steps: 500 # 반대 방향 회전 실패로 판단할 최대 제어 step 수
  link1_reverse_weight: 1.0 # q1 반대 winding CLF 가중치

safety:
  clf_rate: 1.0
  clf_weight: 5.0
  velocity_limit: 4.0
```

`settings/controller.yaml`이 실제 실행 설정의 기준이다. 값 변경 후에는 sequential
on/off 비교 실험으로 수렴 시간, 최종 position/yaw 오차, 최소 barrier를 함께
확인한다.
