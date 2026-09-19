# 명확한 구현을 위한 프롬프트

## 1. Panda 및 task 정의

### 제어 관절
panda 7-dof만 사용한다. gripper는 고정.

### end-effector frame
gripper tip으로 한다.

### task
task는 6D pose로 한다.

### position/orientation error 표현과 허용오차
position error은 위치의 차의 2-norm으로 하고, 허용 오차는 기존처럼 `controller.yaml`에서 조절할 수 있게 하고, 기존 값을 사용한다.

YAML에서 목표 orientation은 `orientation.approach`, `orientation.up`의 두
카테시안 벡터로 지정한다. 각각 `gripper_tip`의 local +z 축과 local +y 축이
base/world frame에서 향할 방향이다. 입력 벡터의 크기는 무시하고 normalize하며,
`up`은 `approach`에 수직인 평면으로 투영한다. 따라서 두 벡터는 평행하면 안 된다.
이 두 벡터로 만들어진 right-handed rotation matrix를 unit quaternion으로 변환하고,
오차는 아래에 써놓은것과 같이 정의한다.

--- 

여기에서 각 $\mathbf{R}_{d}, \mathbf{R_e}$는, 각 frame의 기저 벡터(좌표계 축)를 column vector로 행렬에 채워 넣은 것이다. 이때, 각 기저 벡터는 base frame 좌표계로 표현되어있다. 이에 대응되는 quaternions를 다음과 같이 표기한다.

$$
\begin{aligned}
\mathcal{Q}_{d}=\left\{\eta_{d},\varepsilon_{d} \right\},\quad \mathcal{Q}_{e}=\left\{\eta_{e},\varepsilon_{e} \right\}
\end{aligned}
$$

이때, $\mathbf{R}_{d}\mathbf{R}_{e}^{\top}$에 대응되는 쿼터니언은 다음과 같다.

$$
\begin{aligned}
\Delta\mathcal{Q}:=\left\{\Delta\eta,\Delta \varepsilon \right\} = \mathcal{Q}_{d}*\mathcal{Q}_{e}^{-1}
\end{aligned}
$$

두 회전이 동일할 때 iff $\Delta\mathcal{Q}=\left\{1,\mathbf{0} \right\}$이다. 따라서, orientation error은 다음과 같이 쓸 수 있다.

$$
\begin{aligned}
e_{O}=\Delta \varepsilon &= \eta_{e}(\mathbf{q})\varepsilon_{d}-\eta_{d}\varepsilon_{e}(\mathbf{q})-\varepsilon_{d}\times \varepsilon_{e}(\mathbf{q})
\end{aligned}
$$

--- 

이때 orientation의 허용 오차는 $e_{O}$의 2-norm이 특정 수 미만이도록 하는 것이고, 이것도 `controller.yaml`에서 조절하도록 하며 기존 값을 사용한다.

### base/world/obstacle/target 좌표계 정의
이는 기존처럼 하면 된다. 카테시안 스페이스에서 정의되도록 한다.

## 2. Safe pose-IK solver 형태
이것은 실제 로봇을 움직이는 것이 아닌 IK를 찾기 위한 것이므로, velocity limit을 적용할 필요는 없다. 
또한, hard CBF도 반드시 따를 필요가 없다. 이를 만족하지 못하는 것은 나중에 걸러질 것이기 때문이다.

따라서, resolved-rate는 위에서 제시한 orientation error과 position error의 합으로부터 가져오도록 한다. 
null-space projection은 거리장의 미분에서 가져오도록 한다. 

### inverse jacobian
damped pseudoinverse를 사용한다. 
그러나 Null space control에서 사용하는 pseudoinverse는 무어 펜로즈 inverse를 사용한다.

### singularity damping 계수
0.001로 두되, `controller.yaml`에서 수정할 수 있도록 한다.

### pose 수렴 gain 및 CLF slack 정책
수렴 gain은 우선 1로 두되, `controller.yaml`에서 수정할 수 있도록 한다. 
CLF slack은 QP를 사용하지 않을 것이므로 없어도 된다.

### safe-IK search iteration period, 최대 iteration, 최대 탐색 시간
각 seed마다 최대 탐색 시간은 10초로 하고, integration period는 0.01초로 한다.
이들 또한 `controller.yaml`에서 수정할 수 있도록 한다.
seed당 최대 탐색 시간과 integration period가 seed당 최대 iteration을 정한다.

## 3. Null-space safety objective $\Phi(q)$
log-sum-exp 즉, soft min을 사용해서 barrier 전체를 하나로 만든다.

### soft-min temperature
1로 설정하고, `controller.yaml`에서 수정할 수 있도록 한다. 

### safety gradient gain
1로 설정, `controller.yaml`에서 수정할 수 있도록 한다. 

### safety objective 활성화 거리
어차피 Null space control이므로 항상 활성화하자.

### 안전 margin 
항상 활성화하므로 마진 자체의 의미가 없다.

## 4. Hard safety constraint 정의
hard safety constraint는 trajectory를 만드는 경우와, 만든 IK들을 검증하는 경우에만 사용한다.
따라서, IK를 찾을 때는 사용하지 않으나, sequential하게 제어하는 상황(trajectory를 만드는 상황)에서는 사용한다.

### 장애물 형상
기존과 비슷하게 sphere 여러개를 지정할 수 있도록 한다.

### Panda link collision
우선 캡슐 근사를 사용한다. 그러나, 이후 이것을 기존의 거리장 기반으로 바꿀 예정이므로 코드 작성에 유의해야한다.

### self-collision pair 목록, 인접 link 제외 규칙
모든 link의 조합을 목록으로 하되, 관절로 이어진 두 link의 조합은 목록에서 제거한다.

### base, 손목, gripper 포함 여부
모두 포함한다.

### 각 barrier의 safety margin과 $\alpha_i$
모든 safety margin은 0.05로 하되, `controller.yaml`에서 바꿀 수 있도록 한다.
모든 $\alpha_i$는 1을 기본으로 두되, `controller.yaml`에서 바꿀 수 있도록 한다. 이때 각각 바꾸는 것이 아닌 모든 $\alpha_i$가 같도록 한다.

### CBF infeasible 시 정지/slack/seed 변경 / 재계획 정책
CBF infeasible 시 CLF의 slack variable을 키워도 안되는 경우이므로, 아주 드문 경우이다. 이때 우선 정지를 하도록 하자.

## 5. 초기 seed와 multi-start
### 초기 seed 생성 방식
무작위 joint sample을 사용한다.

### restart 수와 seed 분포
restart 수는 최대 10으로 하며, 이 또한 `controller.yaml`에서 바꿀 수 있도록 한다. seed의 분포는 uniform 분포를 사용한다.

### 각 seed의 탐색 time/iteration budget
각 seed에는 위에서 정의한 0.01초 integration period와 10초의 탐색 시간을
할당한다. 각각의 값은 `controller.yaml`에서 조절할 수 있어야 한다.

### 후보가 여러개일 때의 선택 기준
position error와 orientation error가 각각 설정된 tolerance 이하이고 모든
barrier가 $h_i\ge0$인 상태를 3초 동안 연속으로 유지한 첫 번째 해를 반환한다.
유지 중 하나라도 위반하면 시간을 0으로 초기화한다. 유지 시간은
`safe_ik.convergence_hold_duration`에서 조절하며, 유지 시간이 끝난 뒤에는 현재
seed의 남은 iteration만 생략한다. 설정된 10개 seed를 모두 평가한 뒤, 초기
configuration과 joint-angle distance가 가장 가까운 safe IK 후보를 선택한다.
이 중 seed 0은 초기 configuration 자체에서 시작하고, 나머지 9개 seed는 joint
limit 안에서 uniform random sample로 생성한다.

## 6. 실패 분류
이것은 제안대로 한다.

## 7. safe trajectory 단계

### CLF 선택
sequential mode에서는 safe IK configuration $q^*$에서 각 link 끝점의 목표
position을 계산하고, link1 끝점부터 link7 끝점까지 다음 task-space CLF로
순차 제어한다.

$$
V_i(q)=\frac{1}{2}\lVert p_i(q)-p_i(q^*)\rVert^2
$$

마지막 stage와 direct mode는 configuration CLF를 사용하지 않고 다음
end-effector pose CLF를 사용한다.

$$
V_{pose}(q)=\frac{1}{2}\lVert p(q)-p_d\rVert^2
+\frac{w_O}{2}\lVert e_O(q)\rVert^2
$$

$w_O$와 intermediate stage position 허용오차는 `panda_controller.yaml`에서
조절할 수 있도록 한다. 모든 stage에서 hard CBF를 적용한다.
direct mode에서는 safe pose-IK 탐색 및 검증을 수행하지 않고 목표 pose로 바로
제어를 시작하며 IK candidate 전환도 사용하지 않는다.

### 다른 safe IK 후보로 전환하는 조건
멈췄는데 성공 조건에 도달하지 못하는 경우.
멈췄음의 조건은 최근 0.5초 동안의 전체 관절 속도 2-norm 평균이 기존에 정의된
정지 판별 임계값을 넘지 못하는 것으로 한다. moving-average window 0.5초는
`controller.yaml`에서 수정할 수 있도록 한다.

### trajectory 성공 조건
기존과 같으며 제안한 대로 joint error, final pose error, 최소 barrier, timeout을 사용한다.

### control period 및 velocity/acceleration limit
trajectory 제어 시간은 30초, integration period는 0.05초로 한다. 즉 기본 제어
주파수는 20Hz이다. 두 값은 `controller.yaml`에서 조절할 수 있도록 한다.
velocity/acceleration limit은 실제 franka emika panda의 각 관절 속도, 가속도 제한의 1/3으로 한다.
이 각각의 상수들은 모두 `controller.yaml`에서 제어 가능해야한다.

## 8. 실험 지표
실험 지표는 제안한 대로 따른다.
