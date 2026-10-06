# Panda 순차 제어 몬테카를로 실험 설계

Panda에서 **중간 링크 위치를 거쳐 최종 말단 pose를 제어하는 방법이, 직접 말단
CLF-CBF QP보다 목표 도달률을 높이는지** 검증한다. RDF는 두 방법이 공유하는
자기 충돌 CBF다. 연구의 중심 지표는 목표 도달률이며, 도달 시간과 안전 여유를 함께 보고한다.

작성일은 2026-10-05, 검토한 제어 코드는 `panda`의 `15c6f8b`다.
단계 설정에는 같은 날 결정한 `link2`에서 `link3`로의 변경을 반영했다.
이 문서는 실험 프로토콜이다. 원격 환경과 기구학 확인을 마쳤고,
같은 날 Panda 몬테카를로 실행기와 아래의 공통 판정기를 추가했다.
실험 결과는 이 문서의 설계값과 구분해 실행 디렉토리에 저장한다.
사용자 선택에 따라 주실험 단계는 `link3 → link5 → 최종 말단 pose`로 정한다.

## 전체 흐름

```text
난수 seed와 공통 설정
  → 안전한 시작 자세와 목표 자세, 장애물로 case 생성
  → 목표 pose에 대한 안전 IK 탐색과 RDF 재검증 결과 저장
  → 같은 case를 Direct와 Sequential에 각각 입력
  → 각 상태에서 동일한 목표 도달·안전 기준으로 판정
  → case별 성공 쌍, 실패 원인, 도달 시간과 계산 비용 비교
```

| 단계 | 입력 | 출력 |
|---|---|---|
| 표본 생성 | seed, 관절 한계, 초기 자세, 장애물 분포 | `case_id`, 초기 관절각, 목표 pose, 장애물, 생성에 사용한 목표 관절각 |
| IK 준비 | 목표 pose, 초기 관절각, 장애물, 고정 탐색 예산 | 안전 후보 목록, 선택한 목표 관절각, 실패 사유, 소요 시간 |
| Direct | 초기 관절각, 목표 pose, 공통 CBF와 QP 설정 | 중간 단계 없이 최종 pose를 추종하는 궤적 |
| Sequential | Direct 입력과 선택한 목표 관절각 | `link3`, `link5` 위치를 거쳐 최종 pose를 추종하는 궤적 |
| 공통 판정 | 시간, 실제 관절각, 목표 pose, 새로 계산한 안전값 | 목표 도달 여부, 안전 조건을 포함한 성공 여부, 실패 원인 |
| 비교 분석 | 동일 case의 두 결과 | 성공률 차이와 신뢰구간, 성공 조합, 시간·안전·비용의 차이 |

## 기존 결과와 현재 구현에서 확인한 점

로컬의 `outputs/sequential_comparison/run_20260911_151710/analysis.json`에는
planar 3-DOF 300개 paired case의 결과가 남아 있다.
Direct는 216개 성공해 72.0%, Sequential은 253개 성공해 84.3%였다.
Sequential만 성공한 case는 47개, Direct만 성공한 case는 10개다.
둘 다 성공한 206개에서는 Sequential의 도달 시간이 중앙값 3.98초 더 길었다.
Panda에서는 이러한 **도달률과 시간의 교환 관계가 재현되는지** 확인한다.
이 파일은 로컬 실행 산출물이며 Git에 포함된 결과 파일이 아니다.
로봇, 허용오차, 제한 시간과 안전 판정이 다르므로 두 로봇의 성공률을 직접 비교하지 않는다.

현재 Panda 구현에는 다음 특성이 있다.

- 중간 목표는 선택한 IK 해에서 계산한 **프레임 원점의 위치**다. 관절을 하나씩
  고정하는 방식이 아니며, 이전 단계의 위치를 이후 단계에서 계속 강제하지 않는다.
- `link2` 원점은 모든 관절각에서 $[0,0,0.333]^T$ m에 고정되어 위치 Jacobian이
  0이다. 기존 설정에서는 첫 단계가 즉시 통과했다. 주실험은 관절 움직임에 따라
  위치가 변하는 `link3` 원점으로 바꾸어 **두 개의 중간 위치 목표**를 사용한다.
- `link5`와 `link6`의 원점도 서로 같다. 프레임 이름을 모두 나열하면 독립적인
  중간 목표가 늘어나는 것은 아니다. 추가 단계는 움직이는 점을 확인한 뒤 선택한다.
- 현재 Panda는 중간 목표에 도달해야 다음 단계로 간다. 정지하면 다음 IK 후보로
  바꾸고 첫 단계부터 시작하며, 후보가 없으면 종료한다. QP 실패도 종료 사유다.
  planar 기록의 정지·QP 실패 시 다음 단계로 넘어가는 정책과 다르다.

`link2` 고정과 `link5`/`link6` 중복은 [MJCF](../../assets/robots/panda.xml)의
배치와 [기구학 계산](../../src/safe_ik/adapters/panda_kinematics.py)을 대조하고,
원격에서 초기 자세와 난수 관절 자세 100개로 확인했다.
[제어기](../../src/safe_ik/control/panda_controller.py)가 실제로 사용하는 원점을
기준으로 한 확인이며, 링크 형상 전체가 움직이지 않는다는 뜻은 아니다.

## 주실험의 비교 방법

| 방법 | 제어 목표 | IK 후보 사용 |
|---|---|---|
| Direct | 처음부터 최종 말단 위치와 방향 | 사용하지 않음 |
| Sequential single | `link3 → link5 → 최종 말단 pose` | 안전 후보 중 초기 관절각에 가장 가까운 해 하나로 고정 |

한 case의 초기 상태, 목표 pose, 장애물, CBF, QP, 속도·가속도 제한, 제한 시간은 같다.
IK 후보의 거리는 현재 구현처럼 관절각 벡터의 유클리드 거리로 정하며,
동률이면 seed 인덱스로 순서를 고정한다. 후보와 순서를 저장해 실행 순서에 영향받지 않게 한다.
주실험은 정지 시 다른 IK 해로 바꾸는 기능을 끄고, 후보 변경의 효과는 추가 비교에서 측정한다.

Direct는 IK 정보를 제어에 쓰지 않으므로, 이 비교에는 **IK로 정한 중간 목표의 정보와
순차 제어의 효과가 함께 포함**된다. 순차화 자체의 고유 기여까지 주장하려면
같은 IK 해를 직접 추종하는 관절 CLF 기준선을 추가해야 한다.

## 표본 생성과 공통 모집단

주실험은 seed `2332`부터 `2336`까지 각각
60개씩 총 **300개 독립 case**를 제안한다. 각 case에 두 방법을 적용하므로
최대 600개 궤적이다. 실행기는 YAML의 `main`에 설정한 장면만 바로 실행한다.
300개는 초기 규모이며 검정력 계산으로 확정한 표본 수는 아니다.

| 항목 | 주실험 제안값 |
|---|---|
| 초기 자세 | `configs/panda.yaml`의 고정 `robot.initial_q` |
| 목표 관절각 생성 | 각 관절의 모델 한계에서 양쪽 0.1 rad를 제외한 구간의 독립 균등분포 |
| 목표 pose | 생성한 관절각의 `gripper_tip` FK 위치와 방향 |
| 목표 위치 범위 | 월드 좌표 $x,y\in[-0.7,0.7]$ m, $z\in[0.1,1.0]$ m |
| 최소 과제 거리 | 초기 말단 위치에서 0.15 m 이상 떨어진 목표 |
| 장애물 | 구 2개, 각각 반지름 $r\sim U(0.04,0.10)$ m |
| 장애물 중심 | 위 목표 위치 범위 안에서 독립 균등분포 |
| 장면 채택 | 초기 자세와 생성한 목표 관절각에서 모든 활성 barrier가 0.005 m 이상 |
| 생성 예산 | seed별 제안 장면 최대 10,000개. 60개가 안 모이면 부족분과 사유를 보고 |

생성한 목표 관절각을 $q_{\mathrm{witness}}$라 두고 다음과 같이 목표를 만든다.

$$
(p^*,Q^*)=\operatorname{FK}_{\mathrm{tip}}(q_{\mathrm{witness}}).
$$

이 관절각은 안전한 목표 자세가 적어도 하나 존재한다는 확인에만 쓰며,
IK 초기값이나 Sequential의 목표 관절각으로 전달하지 않는다.
실제 IK는 초기 자세를 첫 seed로 하는 기존 탐색을 별도로 수행한다.
방향은 FK에서 얻으므로 임의의 위치와 방향 조합 때문에 생기는 불가능한 목표를 줄인다.
이 분포는 **관절 공간 표본을 조건부 채택한 분포**이며, Cartesian pose 균등분포가 아니다.
순수 방향 변경 과제와 임의 초기 자세에 대한 결론은 별도 실험이 필요하다.

장면 채택은 제어 실행 전에 끝낸다. 이후 실패한 case를 다른 case로 바꾸지 않는다.
초기·목표 자세가 안전해도 두 자세 사이에 안전한 경로가 존재한다는 보장은 없다.
관절 선형 보간 경로의 barrier를 난이도 참고값으로 저장할 수 있지만,
충돌하는 보간 경로만 선택하거나 두 방법이 실패한 case를 제외하지 않는다.
생성 시 공간 범위·안전 여유 등으로 거절한 횟수도 사유별로 저장한다.

300개 장면을 고정한 뒤 각각 IK 탐색을 한 번 수행하고, capsule로 찾은 후보를
실행에 사용하는 RDF로 재검증한다. IK 난수는 장면 생성 난수와 별도 스트림으로
`case_id`에서 결정한다. 안전 후보를 얻은 case 수를 $K$라 하면 다음 두 분석을 모두 낸다.

1. **주 분석**은 $K$개 동일 case에서 Direct와 Sequential single을 비교한다.
   안전 IK가 확보된 조건에서의 제어 성능으로 해석한다.
2. **전체 과정 분석**은 원래 300개를 분모로 쓴다. Direct는 IK 실패 장면도 실행하고,
   Sequential의 IK 실패는 전체 과정 실패로 포함한다. IK 확보율 $K/300$도 제시한다.

탐색 예산 안에서 해를 못 찾은 결과를 기하학적으로 도달 불가능하다는 증명으로
분류하지 않는다. $K$가 지나치게 작으면 IK 확보율과 함께 보고하고,
표본 분포나 IK 예산을 바꾼 후속 실험은 조건과 seed를 별도로 확정한다.

## 공통 설정과 성공 판정

아래 값은 별도 표시가 없으면 [현재 설정](../../configs/panda.yaml)을 유지한다.

| 항목 | 값과 의미 |
|---|---|
| 제어 시간과 주기 | 60 s, 0.02 s, 최대 3,000 step |
| 중간 위치 허용오차 | 0.05 m |
| 최종 위치 허용오차 | 0.01 m |
| 최종 방향 허용오차 | 오차 quaternion 벡터 norm 0.01 |
| CLF | 감소율 10, slack 가중치 1,000, 최종 방향 가중치 1 |
| 속도 제한 | 관절 1–4는 0.0725 rad/s, 관절 5–7은 0.087 rad/s |
| 정지 판정 | 0.5 s 동안 평균 관절 속도 norm이 0.001 rad/s 이하 |
| 안전 모델 | 자기 충돌 RDF CUDA, 외부 구 장애물은 capsule–sphere |
| 안전 여유 | RDF 물리 여유 0.010 m와 오차 여유 0.010 m, 외부 장애물 여유 0.05 m |
| RDF 근사 설정 | `top_k=2`, `sphere_gate=true`, `softmin_error=0.005` m |
| CBF 이득 | 자기 충돌과 외부 장애물 모두 1 |
| IK 예산 | 20 seeds, seed당 최대 1,000회, CPU worker 5개 |
| QP | OSQP 최대 20,000회, 절대·상대 허용오차 각각 0.001 |

가속도 제한, 충돌 제외 쌍과 나머지 RDF 설정도 같은 설정 파일에서 가져온다.
RDF 파일과 학습 당시 검증 조건은 고정하며, 실험 중 다시 학습하지 않는다.
IK의 `convergence_hold_duration=3.0`은 내부 수치 시간 기준 300회 연속 만족 조건이지,
3초의 실제 실행 시간이나 궤적 목표 유지 시간이 아니다.

현재 방향 오차는 회전각 자체가 아니다. 오차 quaternion의 벡터 부분을
$\epsilon$이라 하면 최소 회전각 $\theta$는 다음 관계를 갖는다.

$$
\lVert\epsilon\rVert=\sin(\theta/2),\qquad
\lVert\epsilon\rVert\le0.01
\Longleftrightarrow
\theta\le2\arcsin(0.01)\approx0.0200\ \mathrm{rad}\approx1.146^\circ.
$$

**실험 실행기의 공통 판정 규칙**은 다음으로 제안한다.

- 초기 상태와 매 적분 후 실제 상태에서 위치·방향 오차를 계산한다. 두 허용오차에
  처음 들어오면 방법이나 현재 stage에 관계없이 목표 도달로 기록하고 종료한다.
  단계 번호와 별도로 최종 과제의 도달 여부를 측정한다.
- 안전 조건을 포함한 성공은 도달 시점까지 모든 기록 상태에서 활성 barrier가
  0 이상이고 관절 한계를 지킨 경우다. 비교와 저장에는 원래 실수값을 그대로 사용한다.
  추가로 $-10^{-6}$ m까지 허용한 수치 민감도 결과를 병기하되 주 성공률과 섞지 않는다.
- 외부 장애물과 자기 충돌의 최소 barrier를 따로 저장한다. 검사 제외 쌍까지
  안전하다고 해석하지 않는다. 도달했으나 안전 조건을 위반한 경우도 따로 집계한다.
- 정지, QP 실패, 시간 초과는 현재 정책을 유지하되 두 방법에 같은 종료 의미를 적용한다.
  주실험은 후보가 하나뿐이므로 정지 시 후보 변경 없이 종료한다.
- 기록 간격 0.02 s의 판정을 주 지표로 쓰고, 저장 궤적을 0.004 s 간격으로
  보간해 barrier를 재평가한 결과도 모든 case에 대해 별도 보고한다.
  이는 수치적 추가 검사이며 연속 시간의 무충돌 증명이나 독립 mesh 검증은 아니다.

일반 [rollout](../../src/safe_ik/application/panda_rollout.py)은 적분 전 관절각을
trace에 저장한다. 비교 실행기는 **최종 상태를 포함한 기록과 새 상태의 안전 재평가**,
stage 완료 여부에 의존하지 않는 목표 도달 판정을 사용하므로 일반 실행의
`summary.json` 성공값과 구별한다.

2026-10-06 제어기 수정 이후에는 낮은 속도에 더해 최근 목표 오차의 개선도 확인한다.
기본값은 `stop_progress_window=2.0` s, `stop_progress_tolerance=0.001`이며,
정규화와 단계별 초기화는 [현재 구현의 정체 판정](../current/README.md#느린-수렴과-정체의-구분)에 설명한다.
이 정책은 Direct와 Sequential 양쪽에 적용된다. 일반 실행의 목표 성공 판정은
`success_barrier_tolerance=1e-6` m을 사용하지만, 비교 실험의 주 성공률은 위의 엄격한
전체 경로 안전 판정을 유지하고 수치 허용오차 결과를 별도로 보고한다.
따라서 실험 YAML에서 `success_barrier_tolerance`를 바꿔도 비교 실험의 `success`와
그 필드로 집계하는 성공률에는 적용되지 않는다. `goal_reached`는 위치·방향 허용오차만
판정하고, 보조 필드 `success_barrier_tolerance_1e6`도 YAML과 무관하게 $10^{-6}$ m로
고정되어 있다. 일반 제어기의 `status.success`와 비교 실험의 저장 필드는 서로 다른 기준이다.
변경된 코드·설정은 manifest에 반영되므로 10월 5일의 기존 결과와 이어 붙이지 않고
새 출력 디렉토리에서 비교한다.

## 보고할 지표와 해석

| 질문 | 지표 |
|---|---|
| 목표에 더 자주 도달하는가 | 목표 도달률, 안전 조건을 포함한 성공률, paired 성공률 차이 |
| 어느 case가 개선되는가 | 둘 다 성공, Sequential만 성공, Direct만 성공, 둘 다 실패 |
| 더 오래 걸리는가 | 전체 분모의 시간별 성공 비율, 둘 다 성공한 case의 도달 시간 차이 |
| 어디에서 막히는가 | IK 예산 내 미발견, RDF 후보 재검증 탈락, 정지, QP infeasible, solver 오류, timeout |
| 안전 여유를 더 쓰는가 | 자기 충돌·외부 장애물별 최소 barrier, 위반 수와 최악값 |
| 순차 단계가 기여하는가 | stage 체류 시간, 전환 원인, 모든 중간 목표 오차와 이전 목표에서 다시 멀어지는 정도 |
| 비용은 얼마인가 | IK 시간, 궤적 실행 시간, QP·RDF 시간, 관절 이동량, 속도·가속도 제한 잔차 |

종료 원인과 안전 위반 flag는 별도 필드로 둬, 정지와 안전 위반이 동시에 발생한
case도 보존한다. CLF slack은 단계마다 다른 목적함수의 완화량이므로,
평균 slack이 작아졌다는 사실만으로 최종 목표 수렴이 개선됐다고 해석하지 않는다.
동일한 최종 pose CLF의 값도 전 구간에 걸쳐 별도로 계산한다.

주 분석 집합에서 case별 성공을 $S_i^{\mathrm{seq}},S_i^{\mathrm{dir}}\in\{0,1\}$라 두면,

$$
\widehat{\Delta p}=\frac{1}{K}\sum_{i=1}^{K}
\left(S_i^{\mathrm{seq}}-S_i^{\mathrm{dir}}\right).
$$

case 쌍을 함께 재표본화하는 bootstrap 10,000회로 95% 신뢰구간을 구하고,
불일치 성공 쌍으로 exact McNemar 검정을 보고한다. $K=0$이면 조건부 분석은 불가능하다고
명시한다. 두 방법이 모두 성공한 case의 시간 차이에는 그 case 수를 반드시 붙인다.
전체 모집단에 대한 시간별 성공 비율도 함께 제시해 성공한 case만의 시간으로 결론내리지 않는다.
여러 seed는 재현성 확인용이며, 유리한 seed만 선택하지 않는다.

현재 속도 제한은 작으므로 60초 timeout이 곧 정지점의 존재를 뜻하지 않는다.
선택한 IK 해까지의 관절 이동 시간 참고값
$\max_j |q_j^*-q_{0,j}|/\dot q_{j,\max}$도 기록한다.
이는 그 관절각에 도달하는 데 필요한 시간의 하한일 뿐, 다른 IK 해로도 가능한
최종 pose 도달 시간의 일반적인 하한은 아니다.

## 추가 비교의 우선순위

| 순서 | 비교 | 확인하려는 내용 |
|---|---|---|
| 1 | 같은 300개 case를 두 방법 모두 120 s로 실행 | 성공률 차이가 단순한 시간 부족 때문인지 |
| 2 | 주실험과 `link5 → link3 → pose`, `link3 → pose`, `link5 → pose` 비교 | 움직이는 중간 목표의 수와 순서가 미치는 영향 |
| 3 | Sequential single과 기존 복수 후보 변경 | 순차 목표 효과와 IK 재시도 효과 분리 |
| 4 | 같은 목표 관절각의 직접 관절 CLF-CBF | IK 목표 정보만으로 설명되는 개선인지 |
| 5 | 장애물 없는 100개 새 case, 안전한 임의 초기 자세의 100개 새 case | 외부 장애물 및 초기 자세에 대한 일반화 |

2번 이후의 추가 비교는 실행 전에 범위를 정하고 탐색적 분석으로 구분한다.
순서 비교에는 모든 방법이 같은 목표 관절각을 사용한다.
정지·QP 실패 시 다음 stage로 넘어가는 planar 방식은 별도 정책 비교로 다룬다.
주실험의 단계나 전환 정책을 결과를 본 뒤 교체하지 않는다.

파라미터 민감도는 우선 중간 허용오차 0.02/0.05/0.10 m와 제한 시간 60/120 s를
각각 비교한다. 이후 CLF 감소율 3/10/30, slack 가중치 100/1,000/10,000을
한 번에 하나씩 바꾼다. RDF 표현·여유·CBF 이득까지 함께 바꾸면 순차화의 효과와
안전 모델의 효과를 분리하기 어려우므로 주실험에서는 고정한다.

## 원격 실행 환경과 필요한 구현

작업 위치는 `lab-server:/home/cdsl/jinoo_ws/safe-inv-clbf-robot-arm-panda-gpu`다.
2026-10-05에 아래를 확인했다.

- 원격 `panda`를 `8ff567a`에서 `15c6f8b`로 fast-forward했다.
  원격 GitHub HTTPS 인증이 없어 `git pull`이 실패했고, 로컬에서 origin과 일치함을
  확인한 커밋을 Git bundle로 전달했다. 원격 GitHub 인증 설정은 변경하지 않았다.
- RTX 4070 약 12 GB, 드라이버 `580.105.08`, PyTorch `2.8.0+cu128`에서 CUDA 사용 가능.
- RDF 모델을 실제로 로딩해 기본 초기 자세의 46개 barrier와 gradient가 모두
  유한함을 확인했다. 최소 barrier는 약 0.002671 m, 제한 쌍은 `obstacle1_link5`였다.
  이는 기본 장면의 확인이며 위 난수 장면의 결과가 아니다.
- RDF 파일 SHA-256은
  `392dd422f0813822a7ac953ec7fe3c2e5211897d6e256bbd37f7c3f90ce08743`이다.
- 위 분포를 seed `20261005`로 200번 제안하는 사전 점검에서 78개 장면이 채택됐다.
  나머지는 작업공간 범위 79개, 초기 안전 여유 17개, 목표 안전 여유 25개,
  최소 과제 거리 1개로 거절됐다. IK와 제어는 실행하지 않았으므로 이 39%는
  장면 채택률이며 IK 확보율이나 제어 성공률이 아니다.

[기존 비교 CLI](../../src/safe_ik/application/compare_sequential.py)는 planar 전용이다.
Panda에는 [전용 실행기](../../src/safe_ik/application/panda_comparison.py)를 추가했다.
[표본과 분석](../../src/safe_ik/experiments/panda_comparison.py),
[공통 판정 rollout](../../src/safe_ik/application/panda_comparison_rollout.py)을 분리했다.
준비한 IK 결과를 case별로 저장하고 선택한 후보 하나를 기존 Panda 제어기에 전달한다.
실행마다 IK를 다시 구해 비교 대상의 목표가 달라지지 않도록 한다.

처음에는 GPU 궤적 worker 1개와 IK CPU worker 5개로 실행한다.
프로세스별 CUDA 문맥과 모델 중복 적재를 피하도록 궤적 worker가 모델을 유지하고,
case마다 초기 상태, 장애물, QP·controller 내부 상태를 초기화한다.
다중 GPU worker는 메모리와 처리량을 측정한 뒤 정한다.
Direct/Sequential 실행 순서는 case별로 균형 있게 교대하며 GUI와 영상 생성은 끈다.
GPU warm-up은 측정 전에 끝내고 CUDA 작업 완료를 보장한 시간과 전체 wall time을 구분한다.

실험 디렉토리는 `outputs/panda_comparison/<run_id>/`이며,
아래 산출물을 이 디렉토리에 바로 저장한다.

| 산출물 | 내용 |
|---|---|
| `manifest.json`, `metadata.json` | 코드·자산·프로토콜·설정 hash, Git 기준점, 환경, seed, 공통 판정과 종료 정책 |
| `source_snapshot.tar.gz`, `working_tree.patch` | 실행 소스 원본과 Git 기준점 대비 변경분 |
| `cases.jsonl`, `sampling.json` | 난수 장면, 생성용 목표 관절각, IK seed, 장면 거절 통계 |
| `<case_id>/ik.json` | IK 후보, RDF 재검증, 선택 해와 IK 탐색 시간 |
| `<case_id>/<method>/` | 관절각·속도·오차·barrier·stage·QP trace, 최종 관절각, 실패 상세 |
| `summary.csv`, `analysis.json` | 전체 300개와 IK 확보 집합의 결과, paired 통계, 시간과 실패 분류 |

완료 여부는 `case_id`와 방법, 설정 hash로 관리한다. 중단 후 같은 입력만 재개하고,
수치 실패를 삭제하거나 성공할 때까지 다시 실행하지 않는다.
인프라 오류에 따른 재실행은 이전 기록과 재시도 사유를 보존한다.
IK는 안전 탐색을 한 번 수행하며, pose-only 진단 탐색은 수행하지 않는다.
IK 준비·탐색·후보 검증 시간을 캐시한 IK를 재사용한 궤적 시간과 별도로 보고한다.

## 실행과 재개

실험 설정은 [configs/experiments/panda_comparison.yaml](../../configs/experiments/panda_comparison.yaml)에서
관리한다. `controller_config`의 공통 로봇 설정에 `controller_overrides`를 덮어쓴다.
표본 수, 난수 seed, 장면 생성 범위, 적분 구간의 추가 검사 수도 이 파일에서 바꾼다.
일반 단일 궤적 실행의 설정 파일을 편집하지 않고 실험 조건을 바꿀 수 있다.

`lab-server`에서는 `/home/cdsl/jinoo_ws/safe-inv-clbf-robot-arm-panda-gpu`에서 실행한다.

```bash
# 파일을 편집한 뒤 실제 적용될 설정을 확인한다. 계산·결과 저장은 하지 않는다.
make panda-compare ARGS="--dry-run"

# YAML의 main 설정으로 바로 비교 실험 실행
make panda-compare

# 다른 조건을 적은 실험 YAML 선택
make panda-compare PANDA_EXPERIMENT_CONFIG=configs/experiments/my_experiment.yaml

# YAML의 output 대신 이 디렉토리에 저장; 같은 명령으로 중단된 실험 재개
make panda-compare OUTPUT=outputs/panda_comparison/gain20_dt001

# 일회성 재정의: 지정한 표본 수와 시간 제한
make panda-compare ARGS="--seeds 2332 --cases-per-seed 10 --duration 90"
```

`my_experiment.yaml`은 전용 YAML을 같은 폴더에 복사해서 만든다. 기본 파일은
300개 장면, gain 20, 계산 주기 0.01초, 제한 시간 90초다.
`controller_overrides.trajectory`에서 이득·주기·단계 순서·정체 조건을,
`controller_overrides.qp`에서 QP 허용오차를 조절한다. `main`의 장면 수는
`seeds` 개수와 `cases_per_seed`의 곱이다. 한 장면에서 두 방법을 실행한다.
예를 들어 시드 5개와 `cases_per_seed: 1`이면 정확히 5개 장면, 최대 10개 궤적이다.

2026-10-06부터 파일럿과 캠페인 실행 경로를 제거했다. 기존 실험 YAML에서는
`command`와 `pilot` 항목을 제거하고, 명령행의 `run`도 생략한다.
사후 보간 안전 검사는 별도 설정인 `validation.subdivisions`로 조절한다.
이 값이 1이면 추가 보간 검사를 생략하고, 제어 중 CBF와 기록 상태의 안전 판정은 수행한다.

`output: null`이면 `output_root` 아래 UTC 시간·`run`·고유 식별자로 새 디렉토리를 만든다.
특정 YAML `output`을 지정하면 같은 폴더에서 재개한다. YAML의 입력·출력 경로는
실험 YAML 기준이며, 명령행 `--output`과 Make `OUTPUT`은 실행 디렉토리 기준이다.
`--config` 또는 명시적 `PANDA_CONFIG`는 공통 제어 파일을 재정의하고,
`--rdf-device` 또는 명시적 `PANDA_RDF_DEVICE`는 계산 장치를 재정의한다.
Make의 기본값이 YAML의 장치를 덮어쓰지는 않는다. CPU 전용 의존성으로 실행하려면
`PANDA_RDF_DEVICE=cpu`도 지정한다. `ARGS="--help"`로 전체 옵션을 확인한다.

### 결과에 저장되는 설정

실험 결과 디렉토리에 다음 파일을 저장한다.

| 파일 | 내용 |
|---|---|
| `experiment.yaml` | 사용자가 선택한 실험 YAML 원본, 주석·줄바꿈 포함 |
| `controller_source.yaml` | 실제로 읽은 공통 제어 YAML 원본 |
| `experiment.resolved.yaml` | 명령행 재정의·기본값까지 적용한 배치, 장면 생성, 검증, 출력 설정 |
| `config.yaml` | 공통 설정과 실험 재정의를 합친 제어 YAML, 모델 경로는 절대 경로 |
| `resolved_config.json` | 검증된 최종 제어 설정과 파생값 |
| `manifest.json` | 입력·소스·자산·프로토콜 식별 정보 및 저장 설정 파일의 SHA-256 |

`experiment.resolved.yaml`은 옆의 `config.yaml`을 참조한다. 저장 조건으로 새 실험을
재현하려면 이 파일과 새 `OUTPUT`을 지정한다. 원본 실험 YAML의 상대 경로는 원래
위치에서 해석되므로, 결과 디렉토리의 원본 사본을 그대로 실행하는 대신 resolved 파일을 사용한다.
명령행으로 변경한 seed·시간 제한·장치도 최종 설정에서 확인할 수 있다.

`progress.json`에 완료한 장면 수를 기록한다. 모든 paired 결과와 최종 상태·안전 판정·
보간 검사 기록이 일치하면 `complete.json`을 저장한다. IK 실패 장면도 실패 결과로 기록한다.
같은 출력과 입력으로 재개하면 완료한 결과를 재사용한다. 원본 YAML·소스·프로토콜·자산이
바뀌거나 저장된 설정·장면·궤적 파일의 hash가 맞지 않으면 재개를 거부한다.
파일럿 제거 전 결과는 보존하고, 변경한 코드로 실행할 때는 새 출력 디렉토리를 사용한다.

2026-10-05의 기준 실험은 파일럿과 본실험 모두 60초를 사용했다. 이후 gain·계산 주기·제한 시간을
바꾼 실험은 새 출력에 저장하고 기준 실험과 조건을 구분한다. 실행 확인용 smoke test는 주 통계에서 제외한다.

## 저장된 비교 실험 재생

`panda-replay`는 결과 폴더의 `config.yaml`, `cases.jsonl`과 선택한 방법의
`trajectory.npz`를 읽어 당시 목표·장애물·초기 자세를 복원한다. 현재 실험 YAML을
사용하거나 IK·RDF·QP를 다시 계산하지 않으며, 기록된 마지막 상태까지 1배속으로
표시한다. 마지막 자세에서 창이 유지되고, 창을 닫으면 종료한다.

```bash
# 저장된 케이스 ID와 두 방법의 종료 원인·성공 여부 조회
make panda-replay RESULTS_DIR=outputs/panda_comparison/my_experiment ARGS="--list"

# 같은 케이스의 순차 제어 또는 직접 제어 재생
make panda-replay RESULTS_DIR=outputs/panda_comparison/my_experiment CASE_ID=seed2333_0000 METHOD=sequential
make panda-replay RESULTS_DIR=outputs/panda_comparison/my_experiment CASE_ID=seed2333_0000 METHOD=direct

# GUI 없이 저장 파일과 로봇 모델만 점검
make panda-replay RESULTS_DIR=outputs/panda_comparison/my_experiment CASE_ID=seed2333_0000 ARGS="--dry-run"
```

`RESULTS_DIR`는 `cases.jsonl`이 있는 폴더를 지정한다. 과거 캠페인 상위 폴더를
지정하면 그 아래 `main/`을 자동 선택한다. IK 실패로 궤적이 저장되지 않은 방법은
실패 원인을 표시하고 종료한다. 재생 시 저장된 성공 여부를 다시 판정하지 않는다.

서버 결과를 로컬로 복사했다면 저장된 로봇 경로가 존재하지 않을 수 있다. 이때는
현재 checkout의 `assets/robots/panda.xml`을 찾고, 기록된 모델·메시 해시와 비교한다.
다른 위치에 동일한 모델을 두었다면 `ARGS="--model-path /absolute/path/panda.xml"`로
지정한다. 설정이나 원본 결과 파일은 수정하지 않는다.

CUDA는 필요 없지만 MuJoCo 창을 표시할 GUI 환경은 필요하다. 디스플레이가 없는 SSH
세션에서는 목록 조회·파일 점검을 사용하고, 재생은 결과를 복사한 로컬 데스크톱이나
그래픽 원격 세션에서 실행한다.
