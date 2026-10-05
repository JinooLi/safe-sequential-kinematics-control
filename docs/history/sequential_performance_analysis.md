# 순차 CLF 성능 해석 계획

> 기존 설계·구현·실험 기록입니다. 본문의 현재·기본값은 작성 당시 기준이며, 현재 동작은 [구현 지도](../current/README.md)를 참고하세요.

## 목적

이 문서는 sequential end-effector CLF 제어가 **어떤 조건에서**, **어떤 이유로**,
그리고 **어떤 대가를 치르며** 유효한지 해석하기 위한 실험 및 분석 계획이다.
단순한 목표 도달 여부보다, 안전·수렴·계산 비용·실패 양상을 함께 기록하는 것을
목표로 한다.

현재 제어기는 목표 pose에서 collision-free IK configuration `q*`를 선택하고, 그
configuration의 링크 끝점 목표 `[p_1^*, p_2^*, p_3^*]`를 순서대로 추종한다.
마지막 링크에서는 position과 yaw를 동시에 제어한다. 안전은 hard CBF와 joint-limit
CBF로 유지한다. 알고리즘 상세는 [순차 제어 알고리즘](sequential_control.md)을
참조한다.

## 1. 먼저 정할 성공 기준

하나의 rollout은 다음을 모두 만족할 때 성공으로 분류한다.

- 최종 end-effector position 오차가 허용 오차 이하
- 최종 yaw 오차가 허용 오차 이하
- 모든 시점에서 최소 barrier 값이 음수가 아님
- 실행 시간 제한 내에 도달

최종 오차는 다음과 같이 둔다.

$$
e_p = \lVert p_3-p^*\rVert,
\qquad
e_\psi = \left|\operatorname{wrap}(\psi-\psi^*)\right|
$$

여기서 `p_3`는 최종 링크 끝점 위치, `\psi`는 end-effector yaw다. 실제 비교
실험에서는 `sequential_experiment`의 position/yaw/barrier tolerance를 이 성공
판정에 사용한다.

## 2. 사례마다 저장할 지표

성공률 하나만 비교하면 성능 차이의 원인을 알기 어렵다. 각 case에 아래 지표를
기록한다.

| 범주 | 지표 | 해석 |
|---|---|---|
| 목표 성능 | 최종 `e_p`, `e_psi`, 성공 여부 | 목표 pose 정확도 |
| 속도 | 수렴 시간, 제어 step 수 | 목표에 얼마나 빨리 도달하는가 |
| 안전 | 전체 구간 최소 barrier, 충돌 여부 | CBF가 유지한 안전 여유 |
| 제어 effort | $\sum_t\lVert\dot q_t\rVert\Delta t$, 최대 $\lVert\dot q\rVert$ | 빠른 수렴의 관절 운동 비용 |
| QP | infeasible 횟수, slack 크기, solve time | 제약 충돌과 온라인 계산 부담 |
| 순차 구조 | 단계별 체류 시간, 전환 횟수·전환 사유 | 어떤 단계에서 막히는가 |
| IK | 선택된 branch, 안전 IK 해 개수 | 초기 자세와 기하가 branch 선택에 미치는 영향 |

특히 **최소 barrier**, **수렴 시간**, **최대 CLF slack**, **목표 도달 전 nonzero
slack 비율**을 함께 제시한다. 마지막 지표는 목표 도달까지 기록된 제어 sample 중
`slack > sequential_experiment.slack_active_threshold`인 sample의 비율이다. 더 빠른
방법이 안전 여유를 더 많이 소모했는지, 또는 CLF 감소 제약을 더 자주 완화했는지를
분리해서 해석할 수 있다.

구현은 각 run directory에 `summary.csv`, `summary.json`, `analysis.json`,
`paired_metrics.png`를 만든다. trajectory NPZ에는 command, CLF slack, QP solve time,
active stage도 저장한다. `analysis.json`의 paired difference는 두 방법이 모두 성공한
case에 대해서만 계산한다.

## 3. 비교 방법과 구성 요소별 효과 분석

동일한 초기 관절 자세, 목표 pose, 장애물 배치를 공유하는 paired experiment로
비교한다. 난수 seed도 고정한다.

| 방법 | 목적 |
|---|---|
| 직접 말단 장치 CLF | 중간 링크 목표가 없는 기본 기준선 |
| 직접 말단 장치 CLF + CBF | 순차 작업 분해가 없는 기준선 |
| 순차 CLF + CBF | 순차 작업 분해의 효과 |

두 방법은 동일한 yaw 가중치와 같은 case를 공유한다. `Direct` 대 `Sequential`의
paired comparison으로 순차 task 분해의 효과를 평가한다.

$$
\Delta T=T_{\mathrm{sequential}}-T_{\mathrm{direct}},
\qquad
\Delta h_{\min}=h_{\min,\mathrm{sequential}}-h_{\min,\mathrm{direct}}
$$

`\Delta T`가 양수여도 성공률 또는 `\Delta h_min`이 개선된다면, 순차 구조가
수렴 시간과 교환하여 feasibility 또는 안전 여유를 얻는다는 해석이 가능하다.

## 4. 필요한 시각화

대표 성공 case와 대표 실패 case를 각각 선정해 아래 그래프를 만든다.

1. 작업공간 궤적: 모든 링크 형상, end-effector path, 장애물, 목표 pose
2. 시간에 따른 링크별 목표 오차

$$
e_i(t)=\lVert p_i(t)-p_i^*\rVert,
\qquad i\in\{1,2,3\}
$$

3. 시간에 따른 yaw 오차와 활성 stage
4. 각 link--obstacle 쌍의 barrier 및 전체 최소 barrier
5. 관절 속도 norm, CLF slack, QP solve time

정상적인 sequential 실행은 `e_1` 감소 후 첫 전환, `e_2` 감소 후 두 번째 전환,
그리고 `e_3`와 yaw 오차 감소의 패턴을 보여야 한다. 반대로 중간 단계에서 barrier가
0 근처에 머물고 속도가 작아진다면 CBF 경계 정지에 의한 전환 또는 실패로 해석한다.

## 5. 실패를 분류하는 기준

실패 case는 하나의 실패율로 합치지 말고 아래처럼 구분한다.

- **IK 해 없음**: 목표 자세에 대해 충돌 없는 IK 해가 처음부터 없음
- **CBF 경계 정지**: 장벽 경계에서 속도가 작아져 다음 단계로 전환됨
- **QP 실행 불가능**: CLF, CBF, 관절 한계, 속도 제약을 동시에 만족할 수 없음
- **시간 초과**: 안전하지만 시간 제한 내 최종 허용오차에 도달하지 못함
- **안전 위반**: 장벽이 음수가 됨

이 분류는 장애물 배치가 어려운 것인지, 순차 목표가 부적절한 것인지, 혹은 QP
제약의 tuning 문제인지를 구분하게 해 준다.

현재 paired 실험은 공정한 on/off 비교를 위해 collision-free IK 해가 하나 이상 있는
목표만 실행한다. 따라서 `IK infeasible`은 실행 실패로 섞지 않고, 목표 재표본화 중
제외된 횟수를 `summary.json`의 `sampling.ik_rejections_before_acceptance`에 별도로
기록한다.

## 6. 파라미터 민감도

현재 알고리즘에서 아래 값은 서로 다른 trade-off를 만든다.

| 파라미터 | 확인할 효과 |
|---|---|
| `safety.clf_rate` | CLF 감소 요구량과 수렴 속도 |
| `safety.clf_weight` | CLF 항의 상대적 영향과 control effort |
| `sequential_end_effector.final_yaw_weight` | direct 및 마지막 단계 position--yaw trade-off |
| `safety.alpha` | barrier 접근 시 회피 반응의 강도 |
| `safety.safe_distance` | 안전 여유와 feasibility trade-off |
| `safety.velocity_limit` | 물리적 속도 제한이 수렴 시간에 미치는 영향 |
| `QP_ctrl.Q` | CLF slack 사용에 대한 비용과 목표 추종--feasibility trade-off |

한 번에 하나의 파라미터만 바꾸고 나머지는 고정한다. 각 값에 대해 여러 seed를
실행하여 중앙값과 사분위 범위, 또는 평균과 신뢰구간을 보고한다. 성공한 case만의
수렴 시간을 보고하면 survivorship bias가 생기므로 성공률도 반드시 병기한다.

## 7. 통계적 비교

동일 case를 두 방법이 공유하므로 paired 분석을 사용한다.

- 성공/실패: paired contingency table과 성공률 차이
- 수렴 시간: 두 방법 모두 성공한 case에 대해 paired difference `\Delta T`
- 최소 barrier와 control effort: paired difference의 중앙값 및 분포
- 실패 원인: 방법별 failure taxonomy 비율

평균만 제시하지 말고 box plot, violin plot 또는 empirical CDF를 함께 제시한다.
어려운 case 일부가 평균을 왜곡하는지 확인할 수 있다.

## 8. 이론적으로 주장할 수 있는 범위

현재 QP의 활성 단계 CLF 제약은 다음 형식이다.

$$
\nabla_q V_i(q)\dot q-\delta\le-k_VV_i(q),
\qquad 0\le\delta\le dV_i(q)
$$

각 활성 CBF 제약은 hard constraint다.

$$
\nabla_q h(q)\dot q\ge-\alpha h(q)
$$

따라서 다음 조건부 해석은 가능하다.

- QP가 feasible하고 CLF slack이 충분히 작으면 활성 단계 CLF의 감소를 유도한다.
- CBF 모델과 거리장 오차 여유가 타당하고 QP 제약이 만족되면 안전 집합의 전방
  불변성을 기대할 수 있다.
- 목표 pose로의 전역 수렴은 보장하지 않는다. 안전 IK 해 존재, 단계별 QP
  feasibility, 속도 제한, stage transition, 장애물 배치에 의존한다.

따라서 논문이나 보고서에서는 “전역 안정성 보장”보다 **feasible한 조건에서의
단계별 CLF 감소와 hard CBF 안전 제약 만족**을 주장하고, 전체 성공 범위는
randomized experiment로 제시하는 편이 정확하다.

## 9. 권장 실행 순서

1. 현재 72-case paired comparison을 baseline으로 실행한다.
2. 성공/실패와 failure taxonomy를 먼저 표로 만든다.
3. 성공, CBF-boundary stall, QP infeasible 사례를 한 개씩 시각화한다.
4. direct와 sequential의 수렴 시간·최소 barrier·effort를 paired plot으로 비교한다.
5. `clf_rate`, `safe_distance`, `final_yaw_weight` 순으로 민감도 분석한다.
6. 마지막으로 `q*` 직접 CLF baseline을 추가해 task-space 순차화의 고유 기여를
   확인한다.
