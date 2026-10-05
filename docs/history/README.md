# 기존 설계와 연구 기록

2026-10-05 문서 분리 이전의 요구사항, 설계 설명, 구현 상세, 실험 결과를 보존한다.
지금 실행되는 기능의 입력·출력은 [현재 구현 지도](../current/README.md)에서 먼저 확인한다.
아래 문서의 제안, 확정 정책, 측정 결과는 각각의 작성 시점에 따라 해석한다.

| 기록 | 포함 내용 |
|---|---|
| [이전 프로젝트 안내](project_readme.md) | 분리 이전 루트 README와 실행 예시 |
| [초기 구현 요청](clarification.md) | Panda pose, IK, 안전 제약, 단계 전환에 관한 요구사항 |
| [아키텍처 변경](architecture.md) | 계층 분리, solver 계약, 실행 기록 구조의 설계 이유 |
| [Make 명령 안내](make_commands.md) | 기존 명령 설명과 예시 |
| [Panda 안전 자세 IK](panda_safe_pose_ik.md) | IK와 궤적 제어의 수식 및 실패 분류 |
| [RDF 초기 설계](panda_rdf_self_collision_design.md) | 자기 충돌 거리장 설계와 당시 미결정 사항 |
| [RDF 명세](panda_rdf_specification.md) | 확정 정책과 이후 결정 기록 |
| [RDF 구현 상세](panda_rdf_implementation.md) | 학습·검증·실행 방법과 구현 설명 |
| [RDF 실험 진행 기록](panda_rdf_implementation_status.md) | 날짜별 후보, 실패, 검증 및 측정 결과 |
| [RDF 실행 최적화](panda_rdf_runtime_optimization.md) | 캐시, 단방향 평가, 구 선별, CUDA와 측정 결과 |
| [hand 추가 학습](panda_rdf_hand_targeted_learning.md) | 취약 영역 탐색, 가중치 변경, 연속 학습 실험 |
| [평면 로봇 순차 제어](sequential_control.md) | 평면 로봇 IK와 단계별 CLF 수식 |
| [평면 로봇 성능 해석](sequential_performance_analysis.md) | 비교 지표, 실패 분류, 분석 계획 |
| [거리장 확장 시각화](rdf_lower_bound_visualization.html) | 과거 2차원 확장 함수 설명용 HTML |

HTML의 파일명에 있는 `lower_bound`는 과거 명칭이다. 현재 확장 RDF가 실제 거리의
전역 하한이라는 뜻으로 사용하지 않는다. 해당 해석은 [RDF 명세](panda_rdf_specification.md)에 기록되어 있다.
