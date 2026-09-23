# Panda RDF 기반 자기 충돌 CBF 설계

## 1. 문서의 범위와 상태

이 문서는 「학습 영역 밖 거리처리」 대화와 후속 논의에서 결정한 설계를 정리한다.
구현 완료 보고서가 아니며, 이 문서 작성으로 기존 제어기나 YAML 설정은 변경되지 않는다.

목표는 링크별 Robot Distance Field(RDF)를 이용하여 self-collision 거리를 평가하고,
그 거리의 관절 미분으로 velocity-level CBF-QP 제약을 구성하는 것이다.
참고 방법론의 task-priority collision avoidance를 그대로 도입하는 것이 아니라,
거리 표현을 활용하여 기존 CLF-CBF-QP에 연결한다.

결정된 구조는 다음과 같다.

1. 링크별 구형 영역에서 RDF를 학습한다.
2. 학습 영역 밖에서는 구 표면으로의 투영과 거리 확장을 사용한다.
3. 구 경계 안쪽의 전환 구간에서 cubic smoothstep으로 두 표현을 연결한다.
4. 충돌 검사 대상 링크 쌍마다 양방향으로 가까운 표면 샘플을 선택한다.
5. 방향별 선택 개수 $k$는 하이퍼파라미터이며, 전체 샘플 사용도 지원하는 설계로 한다.
6. 선택된 거리는 비정규화 log-sum-exp soft-min으로 합친다.
7. 두 링크의 상대운동을 미분하여 링크 쌍별 하나의 hard CBF 제약을 만든다.

외부 장애물 처리, 기존 CLF 및 sequential stage 정의의 변경은 이 문서의 범위가 아니다.

## 2. 기호와 좌표계

관절 상태는 $q\in\mathbb R^7$, 제어 입력은 $u_q=\dot q$로 둔다.
이 문서에서는 모든 gradient를 열벡터로 정의한다.

링크 $i$의 local 좌표를 world 좌표로 변환하는 강체 변환은 다음과 같다.

$$
T_i(q)=
\begin{bmatrix}R_i(q)&t_i(q)\\0&1\end{bmatrix}.
$$

링크 $i$의 물체 영역을 $\mathcal S_i$, 고정된 local 표면 샘플 집합을
$\mathcal P_i=\{s_{i,a}\}_{a=1}^{N_i}$로 정의한다.
샘플은 해당 링크에 고정되며, 관절 운동에 따라 world 좌표만 변한다.

링크 $j$의 샘플을 링크 $i$의 RDF에 질의할 때는 다음 좌표를 사용한다.

$$
x_{j,a}(q)=R_j(q)s_{j,a}+t_j(q),
\qquad
p_{i\leftarrow j,a}(q)
=R_i(q)^\top\bigl(x_{j,a}(q)-t_i(q)\bigr).
$$

거리와 위치의 단위는 m, 관절 각도는 rad, 시간은 s이다.
충돌 검사 대상 링크 쌍의 집합을 $\mathcal C$로 둔다.
구체적인 pair 목록과 인접·고정 링크 제외 규칙은 별도로 확정해야 한다.

## 3. 링크별 RDF와 구형 학습 영역

링크 $i$에 대해 중심 $c_i$, 반지름 $r_i$인 local 구형 영역을 사용한다.

$$
\mathcal B_i=\{p:\|p-c_i\|\le r_i\},
\qquad \mathcal S_i\subset\operatorname{int}\mathcal B_i.
$$

학습 영역과 runtime 확장 경계는 같은 구를 사용한다.
학습된 함수 $f_i(p)$는 물체 밖에서 양수, 표면에서 0, 안에서 음수인
signed distance를 근사한다. 정확한 SDF라는 가정은 하지 않는다.

Bernstein 기반 RDF를 사용하는 경우 일반적인 표현은 다음과 같다.

$$
f_i(p)=\sum_{a=0}^{n_x}\sum_{b=0}^{n_y}\sum_{c=0}^{n_z}
w^{(i)}_{abc}B_a^{n_x}(\xi_x)B_b^{n_y}(\xi_y)B_c^{n_z}(\xi_z),
\qquad
B_a^n(z)=\binom na z^a(1-z)^{n-a}.
$$

여기서 $\xi$는 local 좌표의 affine 정규화 좌표이다.
기저 정규화에 사용하는 bounding box와 실제 학습 샘플 영역인 구는 구분한다.
다항 차수, 학습점 분포·개수, loss 및 정규화 범위는 아직 확정하지 않았다.
$\nabla_p f_i$에는 좌표 정규화의 chain rule도 포함해야 한다.

## 4. 구 표면 투영과 외부 확장

아래에서는 링크 첨자 $i$를 생략한다. 구 중심에서 query까지의 거리를 $\rho$,
방사 방향 단위벡터를 $v$로 정의한다.

$$
\rho=\|p-c\|,\qquad v=\frac{p-c}{\rho},\qquad p^*=c+rv.
$$

구 밖에서 $p^*$는 구 표면의 최근접점이다. 확장 함수는 다음과 같다.

$$
g(p)=f(p^*)+(\rho-r).
$$

구 밖에서는 $\rho-r=\|p-p^*\|$이다.
다음 절의 구 안쪽 전환 구간에서도 같은 식을 연장해서 사용하되,
이때 $\rho-r$는 음수이며 절댓값으로 바꾸지 않는다.

투영 미분과 확장 함수의 gradient는 다음과 같다.

$$
\frac{\partial p^*}{\partial p}
=\frac r\rho(I-vv^\top),
\qquad
\nabla_p g
=\frac r\rho(I-vv^\top)\nabla_p f(p^*)+v.
$$

확장 함수는 $\rho>0$인 전환 구간과 외부에서만 평가한다.
구 중심에서는 원래 RDF만 평가하므로 $v$를 계산할 필요가 없다.

## 5. 구 내부 전환 구간과 매끄러운 혼합

전환 폭 $\delta_i$는 $0<\delta_i<r_i$를 만족한다.
전환 구간을 학습 구 내부에 두어, 학습 영역 밖에서 원래 RDF를 직접 평가하지 않는다.

$$
s=\frac{\rho-(r-\delta)}{\delta},
\qquad
\sigma(\rho)=
\begin{cases}
0,&\rho\le r-\delta,\\
3s^2-2s^3,&r-\delta<\rho<r,\\
1,&\rho\ge r.
\end{cases}
$$

최종 거리 surrogate $D$는 다음과 같다.

$$
D(p)=
\begin{cases}
f(p),&\rho\le r-\delta,\\
(1-\sigma)f(p)+\sigma g(p),&r-\delta<\rho<r,\\
g(p),&\rho\ge r.
\end{cases}
$$

일반 logistic sigmoid 대신 cubic smoothstep을 사용한다.
전환 구간 밖에서 가중치가 정확하게 0 또는 1이고,
양 끝에서 가중치 미분이 0이기 때문이다.

전환 구간에서 정확한 gradient는 다음과 같다.

$$
\sigma_\rho=\frac{6s(1-s)}{\delta},
\qquad
\nabla_p D
=(1-\sigma)\nabla_p f
+\sigma\nabla_p g
+(g-f)\sigma_\rho v.
$$

마지막 항은 가중치가 위치에 의존하므로 생기며, 생략하면 안 된다.
$f$가 충분히 매끄러우면 $D$는 두 경계에서 값과 1차 미분이 연결되는 $C^1$ 함수이다.
다만 이것만으로 뒤의 top-k 선택까지 포함한 전체 함수가 $C^1$이 되는 것은 아니다.

## 6. 양방향 거리 평가와 선택 개수

링크 쌍 $(i,j)\in\mathcal C$에 대해 두 방향을 모두 평가한다.

$$
d_{i\leftarrow j,a}(q)=D_i(p_{i\leftarrow j,a}(q)),
\qquad
d_{j\leftarrow i,b}(q)=D_j(p_{j\leftarrow i,b}(q)).
$$

여기서 가까운 샘플은 원래 다항식 값이 아니라, 확장과 blending까지 적용한
최종 $D_i$ 값이 가장 작은 샘플을 의미한다.

각 방향에서 작은 값 $k$개를 선택한다. 실제 개수는 각각
$\min(k,N_j)$와 $\min(k,N_i)$이며, 선택된 두 방향의 인덱스 집합을
$\mathcal I_k(q)$, 총개수를 $m=|\mathcal I_k(q)|$로 둔다.
충분한 샘플이 있으면 $m=2k$이다.

- 기존 논의의 $k=3$은 방향별 3개, 총 6개에 해당한다.
- 양의 정수 $k$를 하이퍼파라미터로 노출한다.
- 전체 샘플 모드에서는 $m=N_i+N_j$이고 거리 순위에 따른 선택을 하지 않는다.
- $N_i$는 표면 샘플 해상도, $k$는 집계에 사용하는 선택 개수이므로 서로 다르다.

작은 $k$는 선택 이후의 gradient 계산량과 집계량을 줄일 수 있다.
그러나 가까운 샘플을 찾기 위한 전체 거리 평가는 여전히 필요할 수 있으므로,
전체 계산량이 $k$에 비례한다고 가정하지 않는다.

## 7. 비정규화 soft-min과 보수성

선택한 거리들을 $d_a$, $a\in\mathcal I_k(q)$로 표기한다.
$\beta>0$에 대해 링크 쌍의 집계 거리는 다음과 같다.

$$
\widetilde d_{ij}(q)
=-\frac1\beta\log\sum_{a\in\mathcal I_k(q)}e^{-\beta d_a(q)}.
$$

합을 $m$으로 나누는 정규화는 하지 않는다.
$\beta$의 단위는 $\mathrm m^{-1}$이다.
선택 집합이 고정된 구간에서 gradient는 다음과 같다.

$$
w_a=\frac{e^{-\beta d_a}}{\sum_b e^{-\beta d_b}},
\qquad
\nabla_q\widetilde d_{ij}=\sum_a w_a\nabla_q d_a.
$$

수치 구현에서는 overflow와 underflow를 줄이는 stable log-sum-exp를 사용한다.
예를 들어 $d_{\min}=\min_a d_a$로 이동하여 계산할 수 있다.

$$
\widetilde d_{ij}
=d_{\min}-\frac1\beta\log\sum_a e^{-\beta(d_a-d_{\min})}.
$$

비정규화 soft-min의 범위는 다음과 같다.

$$
d_{\min}-\frac{\log m}{\beta}
\le\widetilde d_{ij}\le d_{\min}.
$$

따라서 $k$를 늘리면 계산량뿐 아니라 보수성도 달라진다.
동일한 거리값의 샘플이 많아도 집계 값은 낮아진다.
큰 $\beta$는 최솟값을 더 가깝게 근사하지만 gradient 가중치가 가까운 점에 집중된다.
이는 선택된 surrogate 거리들에 대한 관계이며, 실제 mesh 거리의 하한 보장은 아니다.

### Top-k 선택의 비매끄러움

각 $d_a$가 $C^1$이어도 선택 경계의 $k$번째와 $k+1$번째 샘플이 교체될 때
집계 gradient가 불연속적일 수 있다. 동률을 결정적으로 처리해도 이 문제는 사라지지 않는다.

전체의 고정된 샘플 집합을 사용하면 선택 교체에 의한 비매끄러움은 제거된다.
유한한 $k$를 늘린다고 매끄러움이 단조롭게 개선된다고 주장하지 않는다.
top-k 모드의 비매끄러운 지점에 대한 이론적 CBF 처리는 별도의 검토 항목이다.

## 8. 두 링크의 상대운동 Jacobian

링크 $i$의 원점 world 선속도 Jacobian을 $J_{v,i}$,
world 각속도 Jacobian을 $J_{\omega,i}$로 정의한다.
링크 $j$의 샘플 world 위치 $x_{j,a}$의 Jacobian은 다음과 같다.

$$
J_{x,j,a}
=J_{v,j}-[R_js_{j,a}]_\times J_{\omega,j}.
$$

여기서 $[z]_\times y=z\times y$이다.
링크 $i$ 좌표계에서 본 query point의 상대 Jacobian은 다음과 같다.

$$
J_{p,i\leftarrow j,a}
=\frac{\partial p_{i\leftarrow j,a}}{\partial q}
=R_i^\top\left(
J_{x,j,a}-J_{v,i}
+[x_{j,a}-t_i]_\times J_{\omega,i}
\right).
$$

따라서 관절 공간 거리 gradient는 다음과 같다.

$$
\nabla_q d_{i\leftarrow j,a}
=J_{p,i\leftarrow j,a}^\top
\nabla_pD_i(p_{i\leftarrow j,a}).
$$

반대 방향은 $i,j$를 바꾸어 동일하게 계산한다.
질의받는 링크를 고정 장애물로 취급하거나 한쪽 링크의 Jacobian만 사용하면 안 된다.
사용할 kinematics API의 Jacobian 기준점·좌표계·선속도/각속도 행 순서는 구현 시 확인해야 한다.

## 9. 링크 쌍별 CBF와 QP 연결

링크 쌍별 유효 안전거리를 $d_{\mathrm{safe},ij}$로 두고 다음 barrier를 구성한다.

$$
h_{ij}(q)=\widetilde d_{ij}(q)-d_{\mathrm{safe},ij}.
$$

$h_{ij}\ge0$을 surrogate 기준 허용 조건으로 사용한다.
$\gamma_{ij}>0$인 선형 class-$\mathcal K$ 함수를 사용하면 QP 제약은 다음과 같다.

$$
\nabla_qh_{ij}(q)^\top u_q
\ge-\gamma_{ij}h_{ij}(q),
\qquad (i,j)\in\mathcal C.
$$

표준 상한 부등식 형태로는 다음과 같다.

$$
-\nabla_qh_{ij}(q)^\top u_q
\le\gamma_{ij}h_{ij}(q).
$$

각 링크 쌍에 하나의 제약을 추가하며, CBF에는 CLF slack을 공유하지 않는다.
기존 CLF 제약은 예를 들어 다음 형태를 유지한다.

$$
\nabla_qV(q)^\top u_q\le-c_VV(q)+\delta_{\mathrm{CLF}},
\qquad\delta_{\mathrm{CLF}}\ge0.
$$

QP는 기존 목적함수, 관절 한계 및 입력 제약을 유지하면서 self-collision 제약을 포함한다.
여러 hard 제약이 충돌하면 QP가 infeasible할 수 있으며,
CLF slack만으로 항상 feasible해지는 것은 아니다.

## 10. 설정 인터페이스 제안

아래 항목명은 문서상의 인터페이스 제안이며, 아직 실제 YAML에 적용하지 않았다.
경계 blending, 거리 집계, 선택 개수는 별도 설정으로 관리한다.

| 항목 | 의미 | 상태 |
| --- | --- | --- |
| `nearest_samples_per_direction` | 방향별 선택 개수 $k$ | 양의 정수 또는 전체 사용을 뜻하는 `null` |
| `softmin_beta` | $\beta$, 단위 $\mathrm m^{-1}$ | 양수, 수치 미정 |
| `blend_width` | $\delta_i$, 단위 m | $0<\delta_i<r_i$, 수치 미정 |
| `safety_distance` | $d_{\mathrm{safe},ij}$, 단위 m | 물리적 여유와 오차 여유를 고려하여 결정 |
| `cbf_gain` | $\gamma_{ij}$, 단위 $\mathrm s^{-1}$ | 양수, 수치 미정 |

선택 개수에 대한 설정 예시는 다음과 같다. `3`은 기존 논의의 예시이며 고정값이 아니다.

```yaml
self_collision:
  nearest_samples_per_direction: 3  # null이면 양방향 전체 표면 샘플 사용
```

링크별 구 중심·반지름과 학습 좌표 정규화 정보는 RDF 모델과 일관되게 저장해야 한다.
설정 파일에서 변경할 값과 학습 artifact에 고정할 값의 최종 배치는 구현 전에 확정한다.

## 11. 안전성 해석의 한계

### 거리 확장은 정확한 SDF가 아니다

구 밖에서 정확한 물체 거리 $d_{\mathrm{true}}$를 사용하더라도 삼각부등식에 의해
다음 관계가 성립한다.

$$
d_{\mathrm{true}}(p)
\le d_{\mathrm{true}}(p^*)+\|p-p^*\|.
$$

따라서 확장 식은 실제 거리를 과대평가할 수 있으며,
smooth blending도 이를 자동으로 보수적인 하한으로 만들지 않는다.
최종 함수는 정확한 SDF 대신 smooth extended RDF surrogate로 부른다.

### 검증 오차와 보장된 오차 상한을 구분한다

검증점 집합 $\mathcal P_{\mathrm{val}}$에서 양의 오차를 측정할 수 있다.

$$
\widehat\epsilon_+
=\max_{p\in\mathcal P_{\mathrm{val}}}
\max\bigl(0,D(p)-d_{\mathrm{mesh}}(p)\bigr).
$$

이는 검증점에서의 경험적인 최대 과대평가이지, 연속 영역 전체의 보장된 상한은 아니다.
학습 내부, 전환 구간, 외부 사용 영역을 모두 검증해야 한다.
표면 샘플 사이의 충돌 누락과 이산 제어 주기의 영향도 별도로 평가해야 한다.

유효 안전거리를 아래처럼 분해하는 것은 가능한 설계이지만,
각 오차항의 수치와 산정 방법은 아직 확정하지 않았다.

$$
d_{\mathrm{safe},ij}
=d_{\mathrm{physical},ij}
+\epsilon_{\mathrm{RDF},ij}
+\epsilon_{\mathrm{sampling},ij}
+\epsilon_{\mathrm{discrete},ij}.
$$

고전적인 연속시간 CBF 안전성 논의에는 초기 안전성, 미분 가능성,
제약을 만족시키는 입력의 존재 등이 필요하다.
top-k 선택, 거리 근사, 유한 표면 샘플링, 입력 한계 및 이산 적분이 있는 본 구조에서
물리적인 무충돌이 자동으로 보장된다고 주장하지 않는다.

## 12. 구현 전 남은 결정과 검증 항목

아래는 미확정 항목이며 이 문서가 임의로 값을 결정하지 않는다.

- 학습·검증에 사용할 mesh 종류와 local 좌표계, gripper 포함 범위.
- 충돌 검사 대상 pair 목록과 제외 목록.
- 링크별 구의 중심·반지름, 표면과 전환 구간 사이의 여유.
- RDF 차수, 학습 데이터 생성 방식, 정확도 목표 및 artifact 형식.
- 표면 샘플 수 $N_i$와 배치 방식·최대 간격.
- $k,\beta,\delta_i,d_{\mathrm{safe},ij},\gamma_{ij}$의 초기값과 링크별 override 여부.
- top-k 동률 처리, 비매끄러운 지점의 처리 및 이론적 보장 범위.
- QP 실패 시 처리와 실제 사용 영역의 거리 오차 검증 방식.
- 같은 RDF barrier를 safe pose-IK의 null-space 목적·후보 검증에도 적용할 범위.

최소 검증 항목은 다음과 같다.

1. 원래 RDF, 구 투영, 확장, blending gradient의 유한차분 비교.
2. 두 구 경계에서 값과 gradient의 연속성 확인.
3. 두 링크를 함께 움직였을 때 상대 Jacobian의 유한차분 검증.
4. 고정 선택 집합에서 soft-min gradient 검증과 top-k 교체 지점의 별도 검사.
5. mesh 기준 거리와 surrogate 거리의 오차 및 실제 충돌 누락 검사.
6. $k$ 변화에 따른 거리 평가·선택·gradient·QP 시간의 분리 측정.
7. 최소 실제 거리, 최소 barrier, QP 실패율, 제어 주기 초과율 비교.
8. 전체 샘플 모드를 기준으로 top-k 모드의 계산량과 제어 결과 비교.

## 13. 관련 문서

- [기존 Panda 안전 자세 IK 및 궤적 제어기](panda_safe_pose_ik.md)
- [설계 논의 원문: 학습 영역 밖 거리처리](chatgpt-conversation://6ab1d84e-558c-83e8-9d81-8f2b2dd3e73b)

이 문서의 구형 확장, smoothstep 연결, 양방향 top-k 집계 및 CBF-QP 결합은
프로젝트의 설계안으로 기술한 것이며, 참고 논문의 방법과 동일하다고 주장하지 않는다.
