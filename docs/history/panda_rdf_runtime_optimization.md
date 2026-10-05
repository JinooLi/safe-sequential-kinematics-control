# Panda RDF 자기충돌 계산 최적화

> 기존 설계·구현·실험 기록입니다. 본문의 현재·기본값은 작성 당시 기준이며, 현재 동작은 [구현 지도](../current/README.md)를 참고하세요.

작성 기준: 2026-10-01. 현재 실행 코드와 로컬 측정을 정리한다.
학습 정확도 개선 실험이 아니라 **이미 학습한 RDF를 제어 중 평가하는 비용**이 대상이다.
이후 추가한 RTX 4070 runtime CUDA 구현과 원격 측정은 9절에 별도로 기록한다.

## 1. 현재 적용 결과

같은 모델·같은 64개 자세·같은 top-k 2로 비교한 안전성 평가 1회의 평균 시간이다.
외부 장애물 capsule 평가를 포함하며 전체 IK/궤적 실행 시간은 아니다.

| 실행 정책 | 평균 | p95 | 기준 |
| --- | ---: | ---: | --- |
| 이전 양방향 RDF | 45.24 ms | 48.52 ms | `cbf7e70`의 평가기 |
| 고정 단방향 RDF | 23.84 ms | 25.53 ms | 구 선별 끔 |
| 고정 단방향 + 연속 구 선별 | 19.70 ms | 22.43 ms | 현재 적용 |

구 선별만의 추가 감소는 **17.38%**, 양방향 대비 누적 감소는 **56.45% / 2.30배**다.
이는 이 표본에서의 측정이며 모든 자세에서 같은 속도를 보장하지 않는다.
동일 자세 재사용 등 더 일찍 적용된 최적화는 세 정책 모두에 들어 있으므로, 위 누적 감소에 그 효과까지 포함시킨 것은 아니다.

현재 설정은 다음과 같다.

```yaml
safe_ik:
  self_collision_model: capsule
safety:
  self_collision_model: rdf
  rdf:
    runtime_device: cpu
    query_density: 500.0
    top_k: 2
    sphere_gate: true
    sphere_gate_transition_fraction: 0.5
    softmin_error: 0.005
    physical_margin: 0.010
    error_margin: 0.010
```

이 절은 YAML을 따르는 CPU 참조 실행을 설명한다. 현재 Panda Make 타깃의 장치 선택은 9절을 따른다. **IK 탐색은 capsule**, 궤적 제어의 자기충돌은 RDF이므로 새 구 선별은 우선 궤적 RDF에 적용된다.
IK도 RDF로 선택하면 동일 구현을 사용한다. 외부 장애물은 기존 capsule 계산을 유지한다.
현재 파일에 이미 설정되어 있던 top-k 2와 질의 밀도 500 m⁻²는 이번에 바꾸지 않았다.

## 2. 무엇이 오래 걸렸나

구 선별을 쓰기 전 한 링크 쌍의 계산은 다음 순서였다.

```text
q → FK와 프레임 Jacobian
  → source 링크 표면점들을 world로 변환
  → target 링크 좌표계로 일괄 변환
  → 전체 표면점의 RDF 거리 조회
  → 작은 거리 top-k 선택
  → 선택점만 RDF 공간 gradient 조회
  → 두 링크의 상대운동으로 관절 gradient 계산
  → soft-min과 margin → 쌍별 CBF 값/gradient
```

당시 6,000개 프레임 실행 기록의 RDF 계측 누적 시간은 301.60초였고, 그중 거리 평가 구간은 184.72초(약 61.2%)였다. 이 항목에는 target 좌표계 변환도 포함된다.
전체 프로그램 벽시계 시간이나 GUI 시간으로 해석하지 않는다.
별도 세부 프로파일에서는 거리 조회와 선택점 gradient 조회가 지배적이었고, 좌표 변환 자체는 상대적으로 작았다.

BP24는 축당 24개, 총 13,824개 계수를 사용하는 다항식이다. 구현은 전체 3차원 기저 텐서를 무조건 만드는 대신 NumPy 수축 연산을 사용하지만, 가까운 점에서 다항식과 미분을 평가하는 비용은 여전히 크다. 이 절의 측정과 기본 실행 RDF는 **CPU NumPy**다. 학습 때 RTX 4070을 사용했다는 사실이 제어 시 GPU 실행을 뜻하지 않으며, 선택적인 GPU 실행은 9절에서 별도로 설명한다.

## 3. 지금까지 적용한 최적화

### 3.1 동일 q의 결과 재사용

- `PandaKinematics.update()`는 동일 q에 대한 FK·관절 Jacobian을 재계산하지 않는다.
- `PandaRDFSafety.evaluate()`는 직전 q와 정확히 일치하면 저장된 장벽 평가를 반환한다.
- IK는 한 번 얻은 `BarrierEvaluation`으로 안전 여부와 `barriers.softmin(...)`을 함께 계산한다. soft-min을 위해 FK/RDF를 다시 호출하지 않는다.
- 제어 진단·로그는 `controller.last_barrier`를 재사용한다.

캐시 키는 근사적으로 반올림한 자세가 아니라 정확한 배열 비교다. 다른 q에 이전 gradient를 쓰는 근사 캐시는 아니다. 기록되는 `barrier_q`도 실제 평가 자세이며, 적분 뒤 새 자세를 검사한 것처럼 기록하지 않는다. 모델·질의점·설정은 평가기 생존 기간 동안 고정이라는 전제다. 설정을 바꿨다면 평가기를 새로 구성한다.

### 3.2 좌표 변환과 미분의 배치 처리

표면점 `Q`는 N×3 행렬로 두고 다음처럼 한꺼번에 변환한다.

```python
world = Q @ source_rotation.T + source_origin
local = (world - target_origin) @ target_rotation
```

점마다 Python 반복문으로 Transform을 곱하지 않는다. source의 world 표면점과 점 Jacobian은 한 평가 안에서 여러 쌍이 공유한다.
현재는 구 선별 후 실제로 필요한 source에 대해서만 지연 생성한다.
대부분의 source가 적어도 한 가까운 쌍에 쓰이면 이 지연 생성만으로 얻는 이득은 작을 수 있다.

### 3.3 모든 점은 거리만, 선택점만 gradient

먼저 `query(..., gradient=False)`로 거리 배열을 만들고 top-k를 고른 뒤, 선택점에 대해서만 공간 미분을 계산한다.
선택 인덱스는 매 자세에서 갱신한다. 모든 표면점의 gradient를 먼저 계산한 후 버리는 방식은 사용하지 않는다.

기존 `ExtendedRDF`는 이미 **개별 점이 r1 바깥이면 다항식 대신 해석적 외부 함수**를 사용한다.
따라서 전체 질의점 수를 절반으로 줄여도 고비용 다항식 평가가 절반으로 줄어드는 것은 아니다.
이 외부 함수는 학습 RDF의 연속 확장이지, 실제 mesh 거리의 엄밀한 하한으로 인증된 함수는 아니다.

### 3.4 IK와 궤적의 안전성 backend 분리

`safe_ik.self_collision_model`로 IK만 capsule 또는 RDF를 선택할 수 있다.
현재는 capsule이다. 독립 seed의 병렬 탐색도 지원하지만 작업자는 기구학·평가기·
캐시를 각각 갖는다. backend 변경이나 병렬화의 이득을 위 RDF 단일 평가 표에
중복해서 더하지 않는다. Capsule 선택은 RDF와 수치적으로 같은 판정을 보존하는
최적화가 아니라 다른 형상 근사를 사용하는 선택이다.

### 3.5 양방향에서 고정 단방향으로 변경

사용자 승인으로 `link0 < ... < link7 < hand` 순서의 앞 링크 표면점에서 뒤 링크
RDF로만 조회한다. 예를 들어 `link1 → link4`만 계산하고 `link4 → link1`은 생략한다.
쌍 목록이 역순으로 들어와도 방향이 바뀌지 않는다. 상대운동 미분에는 여전히
**양쪽 링크의 운동**이 모두 들어간다.

자세당 거리 질의점은 이 모델에서 9,207개에서 5,134개로 줄었다. 단방향에서도
source 표면점 전체를 고려하며, 중심 방향 내적으로 절반을 버리는 반구 필터는 없다.

top-k와 비정규화 soft-min은 유지한다.

$$
\widetilde D=-\frac{1}{\beta}\log\sum_{a=1}^{n}e^{-\beta D_a},
\qquad \beta=\frac{\log n}{\varepsilon_{sm}}.
$$

현재 n=2, 오차 예산 0.005m이면 β≈138.63m⁻¹이다. n=1은 그 거리 자체를 쓴다.
이 값은 IK가 여러 **장벽 사이**를 집계할 때의 `safe_ik.softmin_temperature`와 다르다.
단방향화로 실제 선택점 수가 바뀌면 β도 달라진다.

이 변경은 계산값을 그대로 보존하지 않는다. 역방향 표본의 보완이 없어지고, soft-min 표본 수도 바뀐다.
이전 검증에서 쌍별 장벽 차이는 −2.5~+67.803mm였다. 따라서 양방향과 동등한 충돌 검출 보장을 주장하지 않는다.

## 4. 이번 변경: 먼 쌍은 구, 가까운 쌍은 RDF

### 4.1 세 반경을 구별한다

| 기호 | 의미 | 이번에 변경하는가 |
| --- | --- | --- |
| bᵢ | artifact 중심에서 실제 Visual mesh와 질의점을 포함하도록 확인한 반경 | 초기화 때 확인 |
| r₀,ᵢ | 학습 시 물리·오차 여유를 포함한 안쪽 반경 | 보존 |
| r₁,ᵢ | 학습/확장 함수의 바깥쪽, 즉 두 번째 반경 | 보존 |

구 중심은 링크 프레임 원점이 아니라 모델에 저장된 로컬 중심을 world로 옮긴 점이다.

$$
c_i(q)=R_i(q)c_i^{local}+t_i(q),\qquad L_{ij}=\|c_j-c_i\|.
$$

초기화 때 bᵢ는 실제 Visual mesh의 **모든 정점**과 저장 질의점의 최대 반경에
1e−9m 바깥쪽 여유를 더해 정한다. 구는 볼록하므로 모든 삼각형 정점을 포함하면
그 삼각형도 포함한다. 손 형상에는 현재 고정 손가락 형상이 포함된다.
unchecked metadata의 반경만 믿거나 질의점만으로 전체 mesh 포함을 추정하지 않는다.

기존 artifact의 형상·배열 해시·검증 통과 검사는 먼저 수행한다. 실제 형상이 r1에
포함되지 않거나 아래의 양의 전환 여유를 확보하지 못하면 명시적 오류로 중단한다.
임의 fallback, margin 축소, 모델 검증 우회는 하지 않는다.

### 4.2 안전 여유에서 전환폭을 정한다

쌍별로 다음을 정의한다.

$$
m=m_{physical,runtime}+m_{error},\qquad
R_{ij}=r_{1,i}+r_{1,j},
$$
$$
A_{ij}=R_{ij}-b_i-b_j-m>0,\qquad
\Delta_{ij}=\eta A_{ij},\quad 0<\eta<1.
$$

현재 η=0.5이며 `sphere_gate_transition_fraction`으로 정한다.
현재 실행 margin 합은 20mm다. r0/r1은 기존 학습 시 설정을 그대로 유지한다.
실제 모델의 26쌍에서 Δ는 약 99.95~129.59mm다.

| 중심 거리 L | 평가 | RDF 조회 |
| --- | --- | --- |
| L ≥ R | 구 장벽만 | 생략 |
| R−Δ < L < R | 구와 RDF 장벽의 연속 혼합 | 수행 |
| L ≤ R−Δ | 기존 RDF 장벽만 | 수행 |

두 바깥 구가 만나기 시작하면 RDF의 비중이 0부터 증가하고,
더 가까워지면 완전히 기존 RDF로 전환한다. 전환을 위해 일부 겹치는 구간에서는
RDF를 계산해야 하므로, 가중치가 작다고 계산 비용도 비례해서 작아지는 것은 아니다.

### 4.3 장벽을 연속적으로 연결한다

먼 쌍의 저렴한 장벽과 기존 RDF 장벽은 각각 다음과 같다.

$$
h_s=L-b_i-b_j-m,\qquad h_R=\widetilde D_{ij}-m.
$$

실제 포함 구가 분리된 영역에서 L−bᵢ−bⱼ는 두 mesh 간 간격의 하한이다.
전환 구간에서는

$$
h_s\ge A-\Delta=(1-\eta)A>0.
$$

즉, margin을 뺀 구 기준 여유가 이미 양수인 곳에서만 두 표현을 혼합한다.
그보다 가까워져 구 장벽이 0이 될 수 있는 영역에서는 가중치가 이미 1이고,
기존 RDF 계산만 사용한다. 이 조건은 **선별 구간의 기하학적 분리**를 정당화한다.
학습 RDF의 연속 공간 정확도나 전체 로봇 궤적의 안전성을 새로 증명하지는 않는다.

$$
u=\operatorname{clip}\left(\frac{R-L}{\Delta},0,1\right),
\qquad w=3u^2-2u^3,
$$
$$
h=(1-w)h_s+w h_R.
$$

쌍별 CBF 행을 삭제하거나 제약식 양변에 작은 w를 곱하는 방식이 아니다.
먼 쌍도 같은 이름·같은 행 위치에서 구 장벽을 유지한다. 외부 장애물 행은 바꾸지 않는다.

### 4.4 가중치의 미분도 포함한다

기울기를 단순 가중평균만 하면 위 h의 실제 미분과 달라진다.
실제 구현은 다음 세 항을 모두 사용한다.

$$
\nabla_q h=(1-w)\nabla_q h_s+w\nabla_q h_R
 +(h_R-h_s)\nabla_q w.
$$
$$
\nabla_q w=-\frac{6u(1-u)}{\Delta}\nabla_q L
\quad(0<u<1).
$$

구 중심의 회전 이동도 포함한다. [v]×를 외적 행렬이라 하면

$$
J_{c_i}=J_{v_i}-[R_i c_i^{local}]_\times J_{\omega_i},\qquad
\nabla_q L=(J_{c_j}-J_{c_i})^\top\frac{c_j-c_i}{L}.
$$

전환 양 끝에서는 dw/dL=0이므로 값뿐 아니라 gradient도 각각 구/RDF에 연결된다.
단, RDF 쪽 top-k 선택이 바뀌지 않는 구간이라는 전제다. 중심이 정확히 겹치면
구 거리의 방향이 정의되지 않지만 그곳에서는 w=1이므로 구 미분을 사용하지 않는다.
구 분기에서는 0/0을 만들지 않도록 처리한다.

### 4.5 설정의 의미와 되돌리기

- `sphere_gate: false`: 구 선별만 끄고 기존 고정 단방향 RDF로 돌아간다.
  양방향으로 되돌리는 옵션은 아니다. 키가 없는 이전 설정의 기본값도 false다.
- `sphere_gate_transition_fraction`: 허용 범위는 엄격히 0<η<1이다.
  작으면 짧은 구간에서 빠르게 RDF로 바뀌고 가중치 기울기가 더 커질 수 있다.
  크면 전환이 넓어지지만 완전 RDF로 넘어갈 때 남는 구 여유는 작아진다.
  RDF를 계산하기 시작하는 바깥 경계 R은 그대로이므로 η를 바꾼다고 RDF 호출 수가
  직접 줄어드는 것은 아니다.
- 기존 `transition_ratio`: 학습 당시 r1과 개별 RDF 확장 구간을 정하는 별개 설정이다.
  새 gate의 전환폭을 조절하려고 이 값을 바꾸지 않는다.
- `softmin_error`: 가까운 쌍에서 선택점 거리를 집계하는 보수 오차 예산이다.
  구 gate의 전환폭·온도와 관계없는 설정이다.

새 두 키는 runtime-only다. 모델 재학습·질의점 재생성·artifact metadata 수정이 필요 없다.
실행 metadata의 `rdf_sphere_gate` / `safe_ik_rdf_sphere_gate`에 정책,
확인한 포함 반경, r1, η, 쌍별 전환폭과 최소 여유를 기록한다.
해당 backend가 capsule이거나 gate를 끈 경우는 null이다.

## 5. 이번 세부 성능 측정

### 조건

- 로컬 Intel i7-10700 / WSL2, Python 3.11.14, NumPy 2.4.6, SciPy 1.17.1.
- BLAS/OpenMP 각 1 thread. CUDA 성능 측정이나 원격 서버 측정이 아니다.
- 동일한 저장 자세 64개, 정책별 3회씩 총 192회. 정책 순서를 교차해 평가했다.
- 현재 모델, 밀도 500 m⁻², top-k 2, 런타임 자기충돌 26쌍, 동일 외부 장애물.
- 측정마다 q/FK 캐시를 비워 실제 새 자세 평가 비용을 비교했다. 초기화/모델 로딩 제외.
- 엄격한 로더를 통과한 모델 SHA-256:
  `392dd422f0813822a7ac953ec7fe3c2e5211897d6e256bbd37f7c3f90ce08743`.

1절은 실제 `evaluate()` 총시간이다. 아래는 같은 수식을 세분화한 계측 평가기다.
각 평가 결과의 값·gradient가 실제 평가기와 정확히 일치하는지 확인했다.
타이머 삽입 비용과 별도 반복 측정 때문에 두 표의 총시간은 조금 다르다.

| 단계 | 단방향, gate 끔 | 단방향, gate 켬 | 켠 뒤 비중 |
| --- | ---: | ---: | ---: |
| FK·프레임 Jacobian | 0.085 ms | 0.085 ms | 0.44% |
| 구 중심·중심 Jacobian·가중치 | <0.001 ms | 0.153 ms | 0.78% |
| 표면점 → world | 0.059 ms | 0.059 ms | 0.30% |
| 표면점 Jacobian | 0.440 ms | 0.427 ms | 2.18% |
| world → target 좌표 | 0.212 ms | 0.078 ms | 0.40% |
| 전체 질의점 RDF 거리 | 12.675 ms | 11.578 ms | 59.05% |
| top-k 선택 | 0.363 ms | 0.157 ms | 0.80% |
| 선택점 RDF gradient | 5.248 ms | 4.614 ms | 23.53% |
| 상대 Jacobian·chain rule | 0.892 ms | 0.351 ms | 1.79% |
| 쌍별 soft-min·margin | 3.442 ms | 1.346 ms | 6.86% |
| 장벽 혼합·가중치 미분 | 0.007 ms | 0.029 ms | 0.15% |
| 외부 장애물 capsule | 0.661 ms | 0.660 ms | 3.37% |
| 결과 조립·유한성 검사 | 0.026 ms | 0.025 ms | 0.13% |
| 반복문·계측 잔여 | 0.065 ms | 0.043 ms | 0.22% |
| 합계 | 24.177 ms | 19.606 ms | 100% |

각 자세에서 평균 **16.375쌍 구 전용 / 3.625쌍 혼합 / 6쌍 RDF 전용**이었다.
생략된 RDF 쌍 수는 14~17개였고, 질의점 수는 5,134개에서 평균 2,014.66개로
60.76% 줄었다. 그러나 주로 생략되는 것은 원래 저렴한 외부 확장 함수 영역이므로
시간 감소는 17.38%다. 가까운 점의 다항식 거리·gradient는 여전히 전체의 약 82.6%다.
이는 다음 최적화의 병목이 좌표 변환보다는 다항식 평가라는 근거다.

## 6. 검증과 보존 사항

- 단위·통합 테스트: 305개 실행, 287개 통과, 선택 의존성 관련 18개 건너뜀.
- 합성 형상으로 양쪽 전환 경계의 값·gradient 연속성을 검사했다.
- 병진뿐 아니라 두 링크 회전과 로컬 중심 offset을 포함해 유한차분과 비교했다.
  가중치 미분을 빼면 검사가 실패하도록 비상수 혼합 사례도 포함했다.
- 실제 BP24 모델 무작위 5개 자세에서 관절 미분 최대 오차
  2.37e−10m/rad, 상대 점 Jacobian 최대 오차 2.18e−10m/rad.
  해당 검사에서 top-k 교체로 제외된 쌍은 0개였다.
- 저장 64자세의 완전 RDF 영역 **384개 쌍/자세 행은 값·gradient가 bitwise 동일**했다.
  외부 장애물의 값·gradient도 동일했다.
- 가까운 쌍은 기존 계산을 보존하지만 먼 쌍·혼합 구간은 다른 장벽이다.
  이번 표본의 `gated − ungated` 쌍별 차이는 −187.205~0mm였고 부호 전환은 없었다.
  이 차이는 모델 정확도 오차가 아니라 구 하한을 사용하는 정책 차이다.
  다른 자세에서도 차이가 항상 이 범위이거나 항상 음수라는 보장은 없다.
- 형상 누락·포함 실패·전환 여유 부족·잘못된 설정을 거부하는 테스트,
  엄격한 로더와의 결합, 직렬/병렬 IK 일치, 실행 metadata 테스트를 포함했다.
- artifact 파일과 학습 계수·반경·질의점·기존 보고서는 변경하지 않았다.
  자기충돌 제외 쌍과 외부 장애물 정책도 그대로다.

구 gate의 C1 연결이 **전체 제어기의 진동 제거**를 의미하지는 않는다.
top-k 구성 교체, QP 활성 제약 변경, IK/제어 이득, 이산 적분 등은 별도 원인이다.
이번에는 저장 자세 재평가와 미분 검증을 수행했으며 전체 궤적 시뮬레이션이나 GUI
재생으로 진동 감소를 검증하지 않았다. 표본 기반 학습 오차 검증을 연속 공간의
엄밀한 충돌 안전성 보증으로 확대 해석하지 않는다.

## 7. 검토했지만 적용하지 않은 것

| 아이디어 | 상태와 이유 |
| --- | --- |
| 중심 방향 내적이 양수인 반구 점만 평가 | 적용하지 않음. 이전 진단에서 점 수는 약 절반이었지만 고비용 다항식 점 감소는 약 12%, 속도 개선은 약 8.3%였다. 표본상 누락이 없다는 결과는 일반 보증이 아니다. |
| soft-min을 hard min으로 교체 | 적용하지 않음. 별도 진단에서 top-k 2 → 1은 약 15.6% 빨랐지만, 최솟값 점 교체 시 gradient 점프와 최대 5mm의 기존 soft-min 보수 여유 상실이 있다. |
| w를 CBF 식 양변에 곱해 켜고 끄기 | 적용하지 않음. w>0이면 같은 제약이고 w=0에서만 행이 사라지며 실제 장벽 미분도 표현하지 못한다. |
| 먼 쌍을 목록에서 완전히 삭제 | 적용하지 않음. 고정된 행·라벨에서 구 장벽을 유지한다. |
| 더 낮은 질의 밀도, margin 축소, 추가 자기충돌 제외 | 이번 최적화에서 변경하지 않음. 각각 별도 정책 변경이다. |

## 8. 코드·근거·재현

주요 구현:

- [쌍별 구 선별과 미분](../../src/safe_ik/domain/panda_rdf_gate.py)
- [단방향 RDF 평가·캐시·혼합](../../src/safe_ik/domain/panda_rdf_safety.py)
- [설정 검증](../../src/safe_ik/domain/panda_config.py)
- [형상 검증 후 평가기 구성](../../src/safe_ik/application/panda_setup.py)
- [기구학 재사용](../../src/safe_ik/adapters/panda_kinematics.py)
- [IK 평가 재사용](../../src/safe_ik/domain/safe_pose_ik.py)
- [제어 평가 재사용](../../src/safe_ik/control/panda_controller.py)
- [오프라인 관절 미분 검사](../../src/safe_ik/application/panda_rdf_training.py)
- [구 선별 회귀 테스트](../../tests/test_rdf_sphere_gate.py)
- [단방향 회귀 테스트](../../tests/test_rdf_unidirectional.py)

로컬 측정 파일은 `outputs/` 아래에 보존했다. 이 디렉터리는 Git 추적 대상이 아니므로
다른 checkout에 문서만 복사하면 측정 데이터까지 생기지는 않는다.

| 기록 | 위치 |
| --- | --- |
| 이번 3정책/단계별 비교·미분 감사 | `outputs/panda_rdf_sphere_gate_validation_dh9v8o/report.json` |
| 재현 스크립트·측정값/장벽/gradient | 위 디렉터리의 `benchmark.py`, `samples.npz` |
| 입력 64자세 | `outputs/panda_rdf_runtime_profile_RmxZWg/stage_samples.npz` |
| 최초 프로파일 | 위 입력 디렉터리의 `report.json`, `rdf_calls.prof` |
| 이전 단방향 검증 | `outputs/panda_rdf_unidirectional_validation_Uy9yuW/report.json` |
| 이전 단방향 단계별 측정 | `outputs/panda_rdf_unidirectional_stages_jlRyou/report.json` |
| 반구 필터 진단 | `outputs/panda_rdf_hemisphere_diagnostic_SskZ2i/report.json` |

위 별도 실행들의 수치를 섞어 한 실행의 세부 시간처럼 합산하지 않는다.
재현 스크립트는 `cbf7e70`의 양방향 소스, 현재 소스/모델, 기존 64자세를 필요로 한다.

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg \
  .venv/bin/python -m unittest discover -s tests -q

PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg \
  .venv/bin/python outputs/panda_rdf_sphere_gate_validation_dh9v8o/benchmark.py
```

벤치마크 재실행은 그 디렉터리의 측정 보고서/NPZ를 갱신하므로 기존 결과를 보존하려면
스크립트의 출력 위치를 새 디렉터리로 지정한다. 학습·실행 모델 설치는 수행하지 않는다.

## 9. Runtime CUDA RDF 추가 — 2026-10-01

앞 절의 CPU 구현은 참조 경로로 보존하고, 사용자가 승인한 GPU 실행을 선택적으로
추가했다. 학습이 아니라 **매 제어 자세의 RDF 추론**이다. 모델 재학습은 필요 없다.

### 실행 방법

```bash
make panda-view   # 잠긴 CUDA 의존성 설치 → GPU 계산 → 기록 재생·그래프
make panda-run    # GUI 없는 CUDA 실행
make panda-view PANDA_RDF_DEVICE=cpu # CUDA extra 없는 CPU 참조 경로
make panda-gpu-view # 명시적 GPU 타깃도 유지

# 같은 동작을 직접 실행
uv run --locked --extra cuda safe-ik --config configs/panda.yaml --rdf-device cuda
```

`safety.rdf.runtime_device`는 `cpu` 또는 `cuda`이고, `--rdf-device`가 YAML을 덮어쓴다.
RTX 4070 환경에서 pull 뒤 바로 실행하도록 Panda Make 타깃은
`PANDA_RDF_DEVICE=cuda`를 기본값으로 사용한다. `--locked --extra cuda`로 의존성을
준비하고 `--rdf-device cuda`로 YAML을 명시적으로 덮어쓴다. CPU 컴퓨터에서는
`PANDA_RDF_DEVICE=cpu`를 전달하면 CUDA extra 없이 실행한다. 자동 장치 탐지나
조용한 CPU fallback은 없다. 직접 CLI를 부를 때는 YAML의 기본값 cpu가 유지된다.
직접 CLI에서 cuda를 요청했다면 `uv run --extra cuda ...`도 함께 사용한다.
Python 3.11 / NVIDIA CUDA 환경에서 검증했으며, 선택 의존성은 Linux/Windows용
PyTorch 2.8.0+cu128로 `pyproject.toml`과 `uv.lock`에 고정했다.
초기 다운로드는 GPU 라이브러리 때문에 수 GB일 수 있다. 드라이버는 uv가 설치하지 않는다.
CUDA 또는 PyTorch가 없으면 오류를 내며 CPU로 자동 대체하지 않는다.

현재 IK backend는 capsule이므로 위 명령이 IK까지 RDF로 바꾸지는 않는다.
IK를 RDF로 선택하면 CUDA 경로를 함께 사용한다. 병렬 작업자는 `spawn`으로 독립
CUDA 상태를 생성한다. 여러 worker가 같은 GPU를 공유하므로 worker 수를 늘릴수록
빠르다고 보장할 수 없으며, GPU 메모리도 각 프로세스가 별도로 사용한다.

### CPU/GPU 계산 경계

```text
CPU: FK·프레임 Jacobian → 구 선별 → 활성 쌍 목록
                   ↓ 프레임 배열·활성 인덱스만 전송
GPU: 상주 질의점 변환 → 배치 Bernstein/외부 함수 → stable top-k
     → 선택점 공간 미분 → 상대운동 chain rule → 쌍별 soft-min
                   ↓ 쌍별 값·관절 gradient만 회수 (동기화)
CPU: 구/RDF 혼합과 가중치 미분 → 기존 장애물 검사·QP
```

계수·중심·반경·질의점·기저 상수는 평가기 생성 때 GPU에 복사한다.
GPU가 반환한 값으로 같은 구 전환 수식을 사용하고, 원래 CPU 외부 장애물 계산과
QP를 유지한다. 프레임 배열에는 양쪽 링크의 병진·회전 Jacobian이 모두 들어간다.
구 선별로 제외된 쌍은 GPU RDF 배치에도 넣지 않는다. 전부 제외되면 GPU 추론도 생략한다.

활성 쌍들의 점 배열을 padding해 한 배치로 계산한다. 가짜 padding 점은 top-k와
soft-min의 실제 표본 수에서 제외한다. 계수 수가 같은 모델을 지원하며 현재 BP24
9링크가 해당된다. 서로 다른 기저 수를 섞으면 명시적 오류다.

개별 점의 r1 바깥에서는 다항식을 외삽하지 않는다. GPU 배치에서 동적 크기 마스크로
CPU 동기화를 추가하는 대신, 바깥 점의 다항식 입력만 중심의 유한한 placeholder로
바꾼다. 그 계산값은 가중치 0이므로 결과에는 해석적 외부 함수만 기여한다.
이는 바깥의 실제 점 좌표를 바꾸거나 학습된 함수를 clipping하는 것이 아니다.
값·gradient·양쪽 반경 경계에서 CPU 참조와 대조했다.

float64를 유지하며 FP32/TF32 근사, autograd, 추가 학습, 역방향 조회, hard min 전환은
하지 않았다. CUDA Graph/torch.compile도 사용하지 않는다. 배치화만으로 측정한 이득이다.
동률은 NumPy와 같은 source 인덱스 순서로 고르도록 stable argsort를 사용한다.
안정 정렬 의미는 [PyTorch 2.8 argsort 문서](https://docs.pytorch.org/docs/2.8/generated/torch.argsort.html)를 따른다.

### RTX 4070 측정

lab-server의 **i9-13900KF / RTX 4070**, Python 3.11.14, PyTorch 2.8.0+cu128,
NumPy 2.4.6, SciPy 1.17.1에서 같은 모델·저장 64자세·3회 교차 측정했다.
단방향+구 선별, 밀도 500 m⁻², top-k 2, 26쌍, margin 설정을 양쪽에 동일하게 적용했다.
BLAS/OpenMP는 1 thread이며 각 측정의 q/FK 캐시는 비웠다. 초기화·예열은 제외하고
CPU↔GPU 전송, 결과 동기화, CPU 구 선별·혼합·외부 장애물까지 포함한 `evaluate()` 시간이다.

| 장치 | 평균 | p95 |
| --- | ---: | ---: |
| 같은 서버의 CPU NumPy | 11.868 ms | 13.152 ms |
| RTX 4070 CUDA float64 | 1.751 ms | 1.866 ms |

약 **6.78배 / 85.25% 감소**다. 앞 절의 로컬 i7 CPU 19.70ms와 직접 나눈 수치가 아니다.
전체 IK·QP·시뮬레이션 시간이 같은 비율로 감소한다는 뜻도 아니다.
PyTorch tensor allocator peak는 약 71.1MiB이며 CUDA context/driver 메모리는 별도다.

같은 표본에서 장벽값 최대 차이 4.72e−16m, 관절 gradient 최대 차이 8.40e−16,
장벽 부호 전환 0건이었다. float64 연산 순서 차이가 있으므로 bitwise 동일을 요구하지는
않는다. 일반적인 top-k 교체 경계에서의 비매끄러움은 기존 CPU와 마찬가지로 남는다.

추가로 seed 20261001의 관절 범위 내 무작위 64자세를 같은 방법으로 검사했다.
CPU 평균 19.289ms / p95 40.140ms, CUDA 평균 1.985ms / p95 2.585ms였다.
값·관절 gradient 최대 차이는 각각 8.60e−16m / 1.61e−15, 장벽 부호 전환은 0건이다.
자세에 따라 활성 쌍과 점의 내부/외부 함수 비율이 달라지므로 이 시간은 위 저장 자세
표와 별도 결과다. 이 측정의 tensor allocator peak는 약 84.7MiB였다.

### 검증·진단 기록

- 원격 전체 테스트 319개 통과(건너뜀 없음). 실제 CUDA, BP24 거리·미분,
  r0/r1 경계, 구 전환, n=1/top-k/전체 모드, padding, 동률 선택을 포함한다.
- 엄격한 로더와 CUDA 구성, 관절 유한차분, 직렬·spawn worker 일치를 검사했다.
- 로컬 CPU 환경은 319개 중 290개 통과, Torch/CUDA 의존 29개 건너뜀.
- GUI 없는 direct-mode MuJoCo 실행을 CPU/CUDA 각각 0.1초(5 control steps) 확인했다.
  두 실행 모두 failure와 QP 오류가 없고, 기록된 관절각 최대 차이는 2.36e−12rad,
  최종 관절각 차이는 1.57e−11rad였다. 시간 제한으로 끝난 짧은 연결 검사이며
  목표 도달 성공이나 장시간 안정성 검증을 의미하지 않는다.
- CUDA 초기화 실패는 `runtime_initialization_failed` 보고서로 남기며 제어를 시작하지 않는다.
  GPU의 비유한 거리·장벽·미분은 오류로 처리한다. 실행 중 오류도 기존 중단 정책을 따른다.
- 원격 검증은 `outputs/rdf_cuda_runtime_20261001_3cxhDo`의 사본에서 수행했다.
  기존 checkout c726d1f와 학습 산출물은 보존했고 누락된 tqdm은 사본의 `.test-deps`에만
  설치했다. 원래 원격 가상환경은 변경하지 않았다.
- 모델 SHA는 이전 CPU 측정과 동일하며 계수·질의점·반경·검증 보고서를 수정하지 않았다.

`rdf_runtime` 및 `safe_ik_rdf_runtime` metadata에 장치·dtype·PyTorch/CUDA 버전과
GPU 이름을 기록한다. 기존 3열 `rdf_timings` 형식은 유지한다. **CUDA 경로에서는
distance 열이 거리만이 아니라 전송·선택·미분·soft-min을 포함한 배치 전체 wall time**이다.
selection 열은 0, gradient 열은 CPU 구 계산·혼합 시간이다. metadata의 `timing_layout`도
이 차이를 명시하므로 CPU의 세부 단계 시간과 같은 의미로 비교하면 안 된다.

구현과 재현:

- [CUDA 배치 평가기](../../src/safe_ik/adapters/rdf_torch_runtime.py)
- [CPU/CUDA 벤치마크](../../src/safe_ik/application/benchmark_rdf_runtime.py)
- [Torch/CUDA 회귀 테스트](../../tests/test_rdf_torch_runtime.py)
- 로컬 보존: `outputs/panda_rdf_cuda_runtime_validation_dXnSs3/benchmark_saved/report.json`
- 추가 무작위 검사: 같은 디렉터리의 `benchmark_random/report.json`
- 짧은 실행 비교: 같은 디렉터리의 `smoke/{cpu,cuda}/summary.json`, `trajectory.npz`

```bash
uv run --extra cuda python -m safe_ik.application.benchmark_rdf_runtime \
  --config configs/panda.yaml \
  --samples outputs/panda_rdf_runtime_profile_RmxZWg/stage_samples.npz \
  --output-dir outputs/my_new_cuda_benchmark
```

출력 디렉터리는 새 경로여야 한다. `--samples`를 생략하면 고정 seed의 관절 범위 내
64자세를 새로 만들어 비교한다. benchmark는 학습·모델 설치·궤적 제어를 실행하지 않는다.

### Pull 후 실행을 위한 배포 검증

Panda Make 타깃의 CUDA 기본값을 추가한 뒤, 원격 검증 사본에 **새 `.venv`**를 만들고
`uv sync --locked --extra cuda`만으로 82개 패키지를 설치했다. 앞선 검증의
`.test-deps`와 `PYTHONPATH` 보완 없이 실행했으며, 잠긴 tqdm 4.70.1도 설치됐다.
Make 명령 회귀 테스트 4개를 추가한 원격 전체 **323개 테스트가 모두 통과**했다.
로컬 CPU 환경은 323개 중 294개 통과, 선택 Torch/CUDA 테스트 29개 건너뜀이다.

`make panda-view ARGS="--help"`로 표준 타깃의 의존성·진입점·장치 선택을 확인했고,
`make panda-run ARGS="--mode direct --duration 0.1 ..."`로 실제 CUDA 추론과
5개 제어 스텝을 실행했다. failure 및 QP 오류는 없었다. 증거는 원격 검증 사본의
`outputs/deployment_smoke/cuda/`에 있다. 이 검사는 전체 목표 도달이나 실제 GUI 창을
띄우는 검사가 아니며, `panda-view` 재생에는 그래픽 데스크톱/display가 필요하다.
