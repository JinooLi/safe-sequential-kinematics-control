# Panda RDF 자기 충돌 확정 명세

기준일: 2026-09-28. 상태: 구현 요구사항이며 구현 완료 보고서가 아니다.
본 문서는 이전 RDF 설계 기록보다 우선한다. 실행 설정은 별도 구현·검증 후 적용한다.

## 1. 목적과 적용 범위

원 논문의 RDF 학습 방법을 활용하고, 연구 범위는 RDF 이후 self-collision 거리 집계와
CBF-QP 구성에 둔다. RDF 학습 알고리즘 자체의 개선은 목표로 하지 않는다.

- Self-collision에 Visual mesh 기반 RDF를 적용한다.
- 외부 장애물은 기존 capsule–sphere 방식으로 유지한다.
- Safe pose-IK null-space 안전 목적, IK 후보 검증, trajectory CBF에 모두 적용한다.
- 기존 capsule self-collision도 선택 가능하게 유지한다. YAML 기본 모델은 `rdf`다.
- RDF 준비 실패 시 capsule로 자동 전환하지 않는다.
- 실행 중 독립적인 mesh collision checker는 추가하지 않는다.
- Mesh 거리 계산은 학습 라벨과 오프라인 정확도 검증에 사용한다.
- 기존 pose CLF와 sequential stage 목적은 이 변경으로 재정의하지 않는다.

## 2. 로봇 형상

Visual mesh를 기준으로 base, arm 링크, hand, 양쪽 finger를 모두 포함한다.
Finger는 모델의 완전 닫힘 한계에 고정하고, 팔의 7-DOF만 제어한다.
Hand와 닫힌 양쪽 finger는 동일한 local 좌표계로 변환해 하나의 RDF 형상으로 합친다.
통합 형상 내부의 의도된 접촉은 별도 충돌 쌍으로 만들지 않는다.

RDF 학습용 형상은 원본 Visual mesh에서 공식 RDF 저장소의 voxel SDF 및
Marching Cubes 전처리로 재구성한다. 구체적인 채택 설정과 검증 정책은
9절의 결정 3·4에 기록한다. 원본 asset과 시뮬레이션 표시 mesh는 바꾸지 않는다.

통합 후 직접 연결된 인접 링크만 제외하고 나머지 모든 링크 쌍을 검사한다.
Base와 통합 hand도 검사에 포함한다. 정확한 mesh 파일, 닫힘 값, 좌표 변환 및
pair 목록은 실제 모델에서 확인하며 이름의 숫자 차이만으로 인접성을 판단하지 않는다.

위치·거리는 m, 시간은 s, 관절 각도는 rad로 표기한다. Gradient는 열벡터다.
Local → world 변환은 다음과 같다.

$$
T_i(q)=\begin{bmatrix}R_i(q)&t_i(q)\\0&1\end{bmatrix},\qquad q\in\mathbb R^7.
$$

## 3. RDF 학습 정책

Li et al., *Representing Robot Geometry as Distance Fields: Applications to
Whole-body Manipulation*의 RDF 학습 및 analytical derivative 방법을 따른다.
채택한 방식은 거리값 대상 recursive ridge regression이다.

$$
f_i(p)=\Psi_i(p)^\top w_i,\qquad
\nabla_pf_i=\left(\frac{\partial\Psi_i}{\partial p}\right)^\top w_i.
$$

Bernstein 기저의 해석적 미분과 좌표 정규화 chain rule을 사용한다.
Gradient loss, Eikonal loss, 독자적인 정규화 계수 탐색은 도입하지 않는다.
학습 알고리즘은 유지하되, 각 링크의 외부 구 r₁까지 학습 영역이 포함되도록 설정하는
변경은 허용하며 원 논문 대비 적용상의 차이로 명시한다.

공식 참조 구현은 `idiap/RDF`의 커밋
`550c13901ca62a659171b305b0d087d88c3fb580`으로 고정한다. 전처리는
`panda_layer/mesh_test.py`, 학습 표본 생성은 `sample_sdf_points.py`,
회귀는 `bf_sdf.py`를 기준으로 한다. 차수와 기저 개수, 좌표 정규화 및
거리 단위를 구분하며, Loss의 합/평균 정의가 다른 정규화 계수를 그대로 옮기지 않는다.

### 이전 독자 학습 설정의 상태

축별 12차, 학습점 50,000개, 학습 분포 60:40, 표면 근처 학습 범위 ±60 mm는
원 논문을 따르기로 변경하면서 보류되었다. 논문 설정으로 확인 없이 재사용하지 않는다.
독립 검증점 5,000개와 검증 분포 60:40은 이후 별도의 오프라인 품질 검사 설정으로
다시 확정되었으며 아래 9절에 적용한다.

## 4. 안전거리와 두 동심 구

물리적 self-collision 안전거리와 RDF 오차 여유는 다음과 같다.

$$
d_{\mathrm{physical}}=0.030,\qquad
\epsilon_{\mathrm{RDF}}=0.005,\qquad
d_{\mathrm{effective}}=0.035\ \mathrm m.
$$

링크별 Visual mesh의 최소 포괄 구 중심을 cᵢ, 반지름을 r_mesh,i로 정한다.

$$
r_{0,i}=r_{\mathrm{mesh},i}+d_{\mathrm{effective}},\qquad \alpha=0.5,
$$

$$
r_{1,i}=\max\left((1+\alpha)r_{0,i},\ r_{\mathrm{mesh},i}+0.060\right),
\qquad \Delta r_i=r_{1,i}-r_{0,i}.
$$

외부 구는 기본 비율뿐 아니라 표면 안팎 60 mm 검사 영역의 포함 조건도 만족해야 한다.
안전거리와 기본 전환 비율은 YAML에서 조절한다. 변경 시 기존 학습 영역과 artifact의
유효성을 다시 확인한다. 5 mm 오차 여유는 경험적 검증 기반이며 전역 오차 상한 증명은 아니다.

## 5. 표면 최댓값과 외부 함수

$$
M_i=\max_{\|x-c_i\|=r_{0,i}}f_i(x).
$$

오프라인에서 다음 방법으로 추정한다.

1. 내부 구 표면에 고르게 배치한 10,000개 점에서 RDF를 평가한다.
2. 값이 큰 영역에서 서로 떨어진 20개 초기점을 선택한다.
3. 각 초기점에서 구 표면 제약 수치 최적화를 수행한다.
4. 샘플 평가와 최적화에서 얻은 값 중 최댓값을 저장한다.

샘플 수와 초기점 수는 조절 가능하게 한다. 저장값은 추정치이며 엄밀한 전역 최대를
보장하지 않는다. 정확한 최적화 기법과 종료 허용오차는 구현자가 정해 기록한다.

링크 첨자를 생략하면 외부 함수는 다음과 같다.

$$
\rho=\|p-c\|,\qquad v=\frac{p-c}{\rho},\qquad F(p)=M+\rho-r_0.
$$

외부 구 표면에서 F = M + Δr이고, 바깥에서도 계속 방사 방향으로 증가한다.
외부 전체를 상수 M + Δr로 고정하지 않는다.

## 6. 스칼라 smoothstep과 정확한 기울기

$$
t=\frac{\rho-r_0}{r_1-r_0},\qquad
\sigma(\rho)=\begin{cases}
0,&\rho\le r_0,\\
3t^2-2t^3,&r_0<\rho<r_1,\\
1,&\rho\ge r_1.
\end{cases}
$$

$$
D(p)=\begin{cases}
f(p),&\rho\le r_0,\\
(1-\sigma)f(p)+\sigma F(p),&r_0<\rho<r_1,\\
F(p),&\rho\ge r_1.
\end{cases}
$$

전환 구간에서는 다음 전체 미분을 사용한다.

$$
\nabla_pD=(1-\sigma)\nabla_pf+\sigma v
+(F-f)\frac{6t(1-t)}{r_1-r_0}v.
$$

내부에서는 ∇f, 외부에서는 v를 사용한다. 구 중심에서는 v를 계산하지 않는다.
스칼라 함수와 무관한 별도 gradient 혼합은 사용하지 않는다.
RDF가 매끄러우면 두 경계에서 값과 1차 미분이 연결된다.

D는 실제 거리의 하한이 아니다. 내부 구가 물체와 유효 안전 여유 영역을 포함하도록
구성하고, 내부 RDF 판정의 정확성을 검증한다. 매끄러움만으로 gradient 방향·크기나
단조성이 보장되지는 않는다. 본 확장은 원 논문 학습법과 구분되는 runtime 설계다.

## 7. 표면 질의점과 거리 집계

### 질의점 표본

$$
N_i=\left\lceil\lambda_{\mathrm{query}}4\pi r_{0,i}^2\right\rceil,
\qquad\lambda_{\mathrm{query}}=500\ \mathrm m^{-2}.
$$

개수는 내부 구의 표면적으로 정하되 실제 점은 Visual mesh 표면에 배치한다.
Mesh 면적 비례 후보를 생성하고 서로 떨어진 점을 선택해 군집을 줄인다.
밀도는 YAML에서 조절한다. 최대 샘플 간격이나 연속 표면 전체의 충돌 누락 방지는
엄밀하게 보장되지 않는다. 이 점들은 RDF 학습점과 별개다.

### 양방향 top-k

각 링크 쌍에 대해 양방향으로 최종 D 값을 평가한다. 방향별 기본 k=10이며,
매 평가마다 현재 값이 가장 작은 점들을 다시 선택한다. Hysteresis는 적용하지 않는다.
설정에서 k를 변경하거나 전체 샘플을 사용할 수 있다.

$$
n=\min(k,N_i)+\min(k,N_j).
$$

선택 교체 시 gradient 불연속은 우선 허용하고 검증 대상으로 남긴다.
동률은 재현 가능한 방식으로 처리하며 세부 규칙을 구현 문서에 기록한다.

### 비정규화 soft-min

$$
\varepsilon_{\mathrm{sm}}=0.005\ \mathrm m,\qquad
\beta=\frac{\log n}{\varepsilon_{\mathrm{sm}}},\qquad
\widetilde D_{ij}=-\frac1\beta\log\sum_{a=1}^n e^{-\beta D_a}.
$$

보통 n=20이며 beta는 약 599.15 m⁻¹이다. 실제 개수에 맞춰 계산한다.
합을 n으로 나누지 않고 stable log-sum-exp를 사용한다.
n=1이면 해당 거리 자체를 사용하고, n=0은 유효하지 않은 입력으로 처리한다.

$$
\min_aD_a-\varepsilon_{\mathrm{sm}}\le\widetilde D_{ij}\le\min_aD_a.
$$

Soft-min 오차는 YAML에서 조절하며 RDF 오차 여유와 별개다.
Soft-min의 보수 오차 5 mm를 barrier에서 다시 차감하지 않는다.

## 8. 상대 Jacobian과 CBF

$$
x_{j,a}=R_js_{j,a}+t_j,\qquad
p_{i\leftarrow j,a}=R_i^\top(x_{j,a}-t_i),
$$

$$
\nabla_qD_a=\left(\frac{\partial p_{i\leftarrow j,a}}{\partial q}\right)^\top
\nabla_pD_i.
$$

질의하는 점과 질의받는 링크의 상대운동을 모두 반영한다. 선택 집합이 고정된 구간에서

$$
w_a=\frac{e^{-\beta D_a}}{\sum_b e^{-\beta D_b}},\qquad
\nabla_q\widetilde D_{ij}=\sum_aw_a\nabla_qD_a.
$$

$$
h_{ij}=\widetilde D_{ij}-0.035,\qquad
\nabla_qh_{ij}^\top\dot q\ge-\gamma h_{ij},\qquad
\gamma=1\ \mathrm s^{-1}.
$$

모든 쌍에 공통 gain을 사용하고 YAML에서 조절한다. 링크 쌍별 hard CBF 제약을
사용하며 CLF slack을 공유하지 않는다.

## 9. 오프라인 검증

모델당 독립 검증점은 5,000개다.

- 외부 학습 구 전체에서 3,000개.
- 재구성 mesh 표면 안팎 60 mm에서 2,000개.
- 학습점과 독립적으로 생성하며 난수 흐름을 분리한다.

### 필수 거리 오차 기준

아래의 d_mesh는 voxel 전처리로 **재구성한 mesh**의 signed distance다.
원본 Visual mesh와 재구성 mesh 사이의 형상 오차와 혼합하지 않는다.

$$
\mathcal V_{\mathrm{near}}
=\{p\in\mathcal V:|d_{\mathrm{mesh}}(p)|\le0.060\},
$$

$$
\max_{p\in\mathcal V_{\mathrm{near}}}
|f(p)-d_{\mathrm{mesh}}(p)|\le0.005\ \mathrm m.
$$

학습 구 전체의 오차도 기록하되, 위 5 mm 통과 기준은 near-surface 점에 적용한다.
이는 유한 검증점의 경험적 기준이며 연속 영역의 오차 상한 증명은 아니다.

### 학습 오차 개선 결정 기록 — 2026-09-28

**결정 1: 최대 절대오차 허용치를 5 mm로 유지한다.**

- 공식 RDF 평가의 `|d| < 0.030 m`는 가까운 평가점을 분류하는 기준이며,
  예측 오차를 30 mm까지 허용한다는 뜻이 아니다.
- 본 명세의 검증 영역 `|d_mesh| <= 0.060 m`와 최대 절대오차 기준
  `0.005 m`는 그대로 유지한다. 허용치를 `0.030 m`로 확대하지 않는다.
- `error_margin = 0.005 m`, `physical_margin = 0.030 m` 및
  유효 안전거리 `0.035 m`를 유지한다. Soft-min 오차 설정도 변경하지 않는다.
- Mesh 부호 판정과 RDF 근사 성능을 개선하여 기존 합격 기준을 만족시키는
  방향으로 진행한다. 검증 실패 artifact의 사용 거부 정책은 유지한다.
- 후속 설계 항목은 사용자 결정 후 순차적으로 기록하며, 필요한 결정이
  완료된 뒤 코드 변경과 재학습을 진행한다.

**결정 2: Visual mesh 기반 학습·검증용 복사본의 보정을 허용한다.**

- 원본 Visual mesh와 시뮬레이션 표시 형상은 보존한다. 별도의 학습·검증용
  복사본에서 정점 병합, 면 방향 정리, 필요한 열린 경계 보정을 허용한다.
- 보정 전후 형상 차이를 검증한다. 의도된 구멍·빈 공간을 무조건 메우거나
  collision mesh/convex hull로 자동 대체하지 않는다. 구체적인 전처리 및
  형상 차이의 보고·판정 정책은 결정 3·4를 따른다.
- 보정 전에 원본 형상의 열린 경계, 비다양체 연결 및 면 방향 불일치를
  직접 확인할 수 있는 3D HTML 진단 뷰어를 제공한다. 링크 및 경계별 확대,
  회전과 가림 해제 표시를 지원한다.
- 뷰어는 UV·재질 경계의 중복 정점으로 생기는 오판을 줄이기 위해 위치를
  소수 8자리(m) 기준으로 묶은 진단용 복사본에서 연결 관계를 검사한다.
  이 처리는 학습 mesh 보정이나 artifact 갱신이 아니다.
- 열린 경계의 연결 성분은 실제 물리적 구멍의 개수와 동일하지 않으며,
  경계가 없다는 사실만으로 자기 교차·부호 정확성·안전성을 인증하지 않는다.

**결정 3: 공식 RDF의 voxel 기반 표면 재구성을 채택한다.**

- 기본 전처리는 `Visual mesh → voxel SDF → Marching Cubes의 0 등위면
  → 재구성 mesh → RDF 학습`이다. 독자적인 국소 hole filling이나
  Poisson 재구성을 기본 방식으로 추가하지 않는다.
- 각 형상을 unit cube로 정규화한 뒤 `voxel_resolution=128`,
  `surface_point_method=scan`, `sign_method=normal`, `scan_count=100`,
  `scan_resolution=400`, `sample_point_count=10000000`,
  `normal_sample_count=100`, `pad=True`를 사용한다. Padding 전 격자는
  `128³`이며, 양수 값을 한 겹 추가한 뒤 0 등위면을 추출한다.
- 공식 코드의 `spacing=(2/128, 2/128, 2/128)`와 원래 스케일·중심 복원
  절차를 참조하고, 복원 후 실제 좌표와 형상 변화량을 기록한다.
  공식 예제에서 주석 처리된 저장 부분은 재현 가능한 오프라인 산출물로 구현한다.
- 원본 참조의 `check_result=False`는 voxel SDF 라이브러리의 선택적 검사
  설정이다. 이를 따르더라도 본 프로젝트의 오프라인 검증 및 실패 artifact
  사용 거부 정책을 끄는 것으로 해석하지 않는다.
- 우리 입력은 현재 Visual mesh와 닫힌 hand/finger 통합 형상이다.
  공식 사전 생성 STL이나 학습 가중치로 조용히 대체하지 않는다.
- 재구성 전후를 HTML에서 비교하고 형상 차이를 보고한다. 재구성 mesh를
  기준으로 한 RDF 오차와 원본 대비 재구성 오차는 서로 구분한다.
  최종 검증 기준 형상 및 재구성 오차의 판정 정책은 결정 4를 따른다.
- 이번 선택은 mesh 전처리 방식에 관한 것이다. 기존 smoothstep 확장,
  스칼라 함수의 정확한 해석적 미분, CBF-QP 및 안전거리 설정은 유지한다.

**결정 4: 재구성 mesh 기준으로 RDF를 검증하고, 원본 대비 형상 차이는 보고한다.**

- 학습 라벨과 독립 검증의 기준 형상은 재구성 mesh로 통일한다.
  위 near-surface 검증점의 최대 절대오차 `0.005 m`를 필수 통과 기준으로 유지한다.
- 원본 Visual mesh와 재구성 mesh 사이의 표면 차이는 양방향 거리 통계와
  비교 HTML로 보고한다. 별도의 형상 차이 자동 합격·불합격 임계값은 추가하지 않는다.
  이 통계는 샘플 기반이며 연속 표면 전체의 최대 차이를 증명하지 않는다.
- RDF 검증을 통과했다는 것은 재구성 mesh에 대한 근사 오차 검사를 통과했다는
  뜻이다. 원본 형상에 대한 총 오차가 5 mm 이하이거나 재구성이 원본을
  보수적으로 감싼다는 보장은 아니다. 큰 형상 변화도 자동으로 탈락시키지
  않으므로, 원본 대비 안전성 해석에는 별도 형상 차이 보고를 함께 사용한다.
- 원본 asset, 시뮬레이션 표시 형상, Visual mesh 기반 표면 질의점 및 포괄 구의
  기존 정의는 유지한다. 안전거리·RDF 오차 여유도 이 선택만으로 변경하지 않는다.
- 유효하지 않은 재구성 결과 및 RDF 검증 실패를 허용하는 결정은 아니다.
  전처리 결과와 검증 기준 형상을 artifact에 식별 가능하게 기록한다.

**결정 5: 사용자 YAML의 `basis_count: 10`을 유지해 link4를 비교한다.**

- BP_10은 축당 Bernstein 기저 10개, 즉 **9차**이며 총 계수는 1,000개다.
  정확히 10차를 뜻하는 `basis_count: 11`로 바꾸지 않는다.
- 우선 `link4`만 BP_8(7차)과 비교한다. 동일한 저장 재구성 mesh, 학습 데이터,
  배치 추출 및 독립 검증점에서 기저 개수만 바꾼다.
- 재구성 mesh 기준 near 최대오차 5 mm, 정규화, 학습 갱신 횟수, 안전거리,
  smoothstep 및 미분 검증 기준은 유지한다. 다른 링크 전체에 대한 합격을
  추론하지 않으며 결과는 기존 artifact와 별도로 보존한다.

### 미분 및 경계 검사

- 같은 RDF의 해석적 gradient와 수치 미분 일치.
- 확장 함수의 gradient와 수치 미분 일치.
- 두 구 경계에서 함수값과 gradient 연속성.
- 제어 구현의 상대 Jacobian과 관절 gradient 일치.

검사는 오프라인에서 수행한다. 실행 시마다 수치 미분 검사를 반복하지 않는다.
수치 허용오차는 구현자가 합리적으로 정하고 근거와 함께 기록한다.

### 추가 통계

- 거리 오차와 부호 오판.
- Gradient 크기·방향 오차.
- 실제 mesh gradient가 정의되지 않거나 모호한 점의 별도 집계.

추가 통계는 우선 합격·불합격 기준에 추가하지 않는다. 실제 거리 gradient 정확도와
동일 모델의 analytic/수치 미분 일치는 서로 다른 검사다.

검증 실패 시 해당 RDF 사용을 중단하고 링크 및 실패 항목·오차를 보고한다.
설정을 검토한 후 별도로 재학습한다. 경고만 출력하고 사용하지 않는다.

## 10. 실행 및 실패 정책

- 기본 self-collision 모델은 RDF. Capsule은 명시적으로 선택할 때만 사용한다.
- Artifact가 없거나 검증에 실패했거나 설정과 호환되지 않으면 사용을 중단한다.
- 불일치 항목과 재생성 방법을 안내한다. 자동 재학습·자동 capsule fallback은 없다.
- 실행 시 오프라인 검증 통과 기록과 artifact–설정 호환성을 확인한다.
- Trajectory 초기 상태의 h가 음수이면 제어를 시작하지 않고 위반 쌍과 값을 보고한다.
- 초기 trajectory 검사와 Safe IK의 임의 seed 처리는 구분한다.
- 제어 중 QP infeasible 또는 수치 실패 시 시뮬레이션을 중단한다.
- 실패 시점, 관절 상태, barrier 값, solver 상태를 저장한다.
- 적분 후 추가 RDF barrier 검사 및 그 결과에 따른 자동 중단은 추가하지 않는다.
- 별도 runtime mesh collision checker는 없다.

이산 적분과 유한 표면 샘플링에 따른 안전 위반 가능성은 남는다.

## 11. 실행 환경, 재현성 및 로그

CPU에서 먼저 구현한다. 계산시간 측정 후 GPU 필요성을 판단한다.
기본 seed는 0이고 설정에서 변경할 수 있다. 학습·검증·query 샘플링에는
서로 구분된 난수 흐름을 사용한다.

기존 실험 로그에 다음을 추가한다.

- 최소 self-collision barrier와 해당 링크 쌍.
- RDF 거리 평가, 샘플 선택, gradient 계산, QP 풀이 시간.
- QP 상태 및 중단 사유.
- 사용한 RDF artifact 식별자와 설정.

Barrier 기록은 제어 시 계산한 값을 활용한다. 적분 후 추가 검사를 로그 명목으로
도입하지 않는다. Safe IK 단계에 QP가 없다면 QP 시간은 해당 없음으로 구분한다.

## 12. 설정·학습 산출물 요구

YAML 등 설정으로 제어할 대상은 self-collision 모델, query 밀도, top-k/전체 모드,
soft-min 오차, 물리적 안전거리, 전환 비율, 공통 CBF gain, 기본 seed 등이다.
RDF 오차 여유 기본 5 mm도 다른 오차와 구분해 명시한다.
정확한 키 이름과 학습/실행 설정의 파일 분리는 구현 세부사항으로 정해 기록한다.

Artifact는 형상 및 닫힘 자세, 좌표계·정규화, 학습 설정, c/r₀/r₁, M 추정 설정·결과,
검증 결과를 식별할 수 있어야 한다. 변경된 설정에 기존 artifact를 조용히 재사용하지 않는다.
관련 항목의 호환성은 의존성을 기준으로 검사하며 모든 runtime 값 변경을 곧바로
재학습 사유로 취급하지 않는다.

## 13. 구현자에게 위임된 사항과 남은 확인

원 논문·공식 구현에 명시된 값이 있으면 우선 따른다. 없는 구현 세부값은 합리적인
기본값을 정하고 근거·출처·실제 사용값을 기록한다. 최적화 종료 허용오차, batch 크기,
샘플링 구현과 artifact 형식 등을 작은 설정마다 다시 질문하지 않는다.

다음은 아직 확인·구체화가 필요한 작업이다.

1. 원 논문·공식 코드 버전과 학습 하이퍼파라미터.
2. Mesh 및 관절 매핑, 닫힘 수치, signed distance 라벨 생성 방법.
3. 최소 포괄 구, 구 표면 최적화, query 샘플 선택의 구체적인 수치 알고리즘.
4. 검증 수치 허용오차와 gradient 정답의 모호한 점 처리.
5. Artifact 형식, 호환성 체크, 로그 스키마와 시간 측정 경계.
6. CPU 성능 측정 및 기존 capsule 대비 비교 실험 구성.

안전 기준·알고리즘·연구 범위를 바꿔야 하는 경우는 사용자에게 다시 확인한다.
현재 제어 주기 등 이번 대화에서 변경하지 않은 기존 설정은 임의로 재정의하지 않는다.

## 14. 참조 기록

- [공식 RDF 표면 재구성 코드](https://github.com/idiap/RDF/blob/550c13901ca62a659171b305b0d087d88c3fb580/panda_layer/mesh_test.py)
- [공식 RDF 학습 표본 생성](https://github.com/idiap/RDF/blob/550c13901ca62a659171b305b0d087d88c3fb580/sample_sdf_points.py)
- [공식 RDF 학습 코드](https://github.com/idiap/RDF/blob/550c13901ca62a659171b305b0d087d88c3fb580/bf_sdf.py)
- [학습 영역 밖 거리처리](chatgpt-conversation://6ab1d84e-558c-83e8-9d81-8f2b2dd3e73b)
- [거리값 오차 학습](chatgpt-conversation://6ab3507d-9ac0-83e8-b7e0-d405c0d37cef)
- [2D 확장 함수 시각화](rdf_lower_bound_visualization.html): 파일명은 이전 실험에서
  유지되었으나 현재 최대값 기반 함수는 하한이 아니다. 정확한 캡슐 거리 예제는
  실제 학습 RDF나 Panda의 안전성을 검증한 결과가 아니다.
