# Collision-Aware Sequential Kinematic Control of Robot Manipulators Using Control Barrier Functions and Supervised Learning-based Task Sequencing

20분 석사 졸업 발표 자료. **청중용 Beamer는 영어, 발표 대본·질의응답·작업 안내는 한국어**로 관리한다. 본문 36장과 보충 자료 21페이지로 청중용 PDF는 57페이지, 발표자용 PDF는 93페이지이다. 보충 자료는 목차 1개와 A1–A17이며 A7 참고문헌은 4페이지이다. 목표 발화 1,140초와 화면 전환·호흡 60초를 합해 20분으로 구성한다.

## 발표 흐름

1. 로봇팔 작업 전반에서 안전한 목표 도달의 필요성; 자율실험실은 이를 보여주는 한 가지 사례(2–3쪽).
2. Kinematic control, CLF constraint, CBF, QP → 기존 연구의 문제 접근 → Direct의 실패(4–8쪽).
3. 인간의 장애물 회피 연구에서 얻은 팔 자세·subgoal의 설계 직관 → 논문 3장의 방법론 구성(9–10쪽).
4. 초기 상태·목표·장애물과 IK 기반 subtask target 구성(11–12쪽).
5. 같은 QP의 활성 CLF constraint를 바꾸는 세 subtask와 함수 개형(13–16쪽).
6. 실행 전 목표·sequence 구성, 제어 중 QP와 전환/종료 판정 및 분기별 반복, Planar 결과와 task transition의 작동 원리(17–19쪽).
7. MuJoCo 그림과 Planar/Spatial 비교로 확장을 소개한 뒤 self-collision, IK, spatial pose CLF와 All-link Sequential 설명(20–24쪽).
8. 동작 시간 증가, link subtask 두 개로 축소, episode별 선택 가능성(25–27쪽).
9. Binary Cross Entropy와 selection loss, cross-attention, 논문 기호의 sequence 선택(28–30쪽).
10. Test 성공률·동작 시간·pair 선택 예시, 결론과 향후 검증(31–35쪽).
11. 감사 인사와 Appendix 목차(36–37쪽).

## 화면과 용어 규칙

슬라이드 형식은 CDSL@UoS의 4:3 Beamer template이다. 본문·부제·그림·도표의 기준은 10pt, 표지·frame 제목은 14.4pt로 관리한다. 수학 첨자와 벡터 그림의 물리적 배율은 본문 typography와 구분한다. 핵심 메시지는 필요한 장의 상단에 검은색으로 표시하며, 모든 장에 요약을 반복하지 않는다. 연구실과 학교 로고를 함께 사용한다. 강조 텍스트는 원색에 가까운 빨강(#FF0000)·파랑(#0000FF)과 채도 높은 초록(#008000)·주황(#E66A00)을 사용한다. 단계별 색상 의미를 유지하며 제목·진행 표시·도형·그래프의 색과 별도로 관리한다.

하단 점은 Motivation, Background, Planar, Spatial, Learning, Results의 본문 진행을 표시한다. Background는 4–8쪽, Planar는 9–19쪽이다. 표지를 제외한 35개 점에서 완료·현재·남은 슬라이드를 구분한다. 보충 자료에는 별도 진행과 목차 복귀 링크를 사용한다. 부제는 논문의 chapter/section/subsection 번호와 제목을 기준으로 하며 발표 구간 이름과 구분한다. 3.2.1은 사용자 요청에 따라 `Decomposition of Regulation Task`를 표시명으로 사용하고, 실제 논문의 `Intermediate Targets and Sequential Tasks`는 메타데이터에 보존한다. 논문 파일 자체는 변경하지 않는다. 3.3.2의 실제 장명 `Intermediate Position Tasks and Final Pose Task`는 부제에 그대로 유지한다. 10쪽은 3장의 TOC short title을 사용한다. Conclusion에는 하위 section이 없어 chapter 제목을 사용하고 참고문헌 부제는 Bibliography이다. 원래 기종명이 들어간 논문 제목은 표시상 Spatial Manipulator로 바꾸되 실제 source heading을 메타데이터에 보존한다.

방법 이름은 Direct / Sequential / All-link Sequential / 10-pair mean / Learned selection이다. Direct는 처음부터 end-effector의 pose regulation만 수행하며 Cartesian 직선 경로를 뜻하지 않는다. Sequential은 IK로 구성한 link subtasks를 순서대로 수행한 뒤 원래 regulation task에 해당하는 final EE subtask를 수행한다. Subtask target은 각 subtask가 도달할 목표 위치·pose이며, subtask 자체와 구분한다. 10-pair mean은 각 episode에서 열 개 후보의 성공 여부를 평균한 값으로, 한 후보를 균등하게 선택할 때의 경험적 기대 성공률이다. 한 pair의 실행 결과나 여러 학습 seed의 평균과 구분한다. A14의 추가 참고 비교인 Train-best fixed pair (3,5)는 Train 최고 성공률로 선정한 뒤 Validation/Test에서 고정한 방법이다. 화면의 Panda 표기는 Spatial Manipulator로 통일하며 원본 데이터의 파일명과 식별자는 유지한다.

논문의 원래 EE regulation task는 `T_{ee}`, 단계별 subtask는 `ST_{ee,s}`로 표시한다. `s`는 실행 단계 번호, `i_s`는 그 단계에서 선택한 link 번호, `L`은 최종 EE 이전의 link subtask 개수이다. 마지막 `ST_{ee,L+1} = T_{ee}`는 원래 목표 pose를 추종하며 IK configuration `q*` 자체로의 수렴을 요구하지 않는다. Planar 예시의 `(1,2,3)`에서는 단계와 link 번호가 일치하지만 일반 규칙은 아니다. All-link는 `L=7`로 마지막 단계가 `ST_{ee,8}`, 두-link 선택은 `L=2`로 마지막 단계가 `ST_{ee,3}`이다. 학습 후보 `S_k`는 선택할 link pair를 나타내므로 subtask 기호로 바꾸지 않는다.

동일성과 공통성은 문맥에 맞게 표현한다. 16쪽은 `Unchanged QP objective, CBFs, and input limits`, 23쪽은 `Identical CLF form`으로 식의 구조를 설명한다. CLF 함수의 값이나 joint-space 지형까지 동일하다는 뜻은 아니다. 29·30쪽의 `Common obstacle encoder`는 모든 장애물에 동일한 학습 파라미터를 적용한다는 의미이다. 31쪽은 `identical success criteria`, 33쪽은 `Common initial state and each method’s final configuration`으로 초기 조건과 서로 다른 종료 상태를 구분한다. A6 제목은 `Paired Test Comparisons`, 시간 표의 집합은 `Success in both`로 표시하며 episode와 episode–pair 단위를 유지한다. A12와 Appendix 목차는 `Common CLF–CBF QP`로 통일한다.

## 피드백 반영과 해석

- 주변 물체와 설비가 있는 환경은 Cluttered Workspaces로 표현. 사람과 로봇의 협업 또는 접촉 안전성을 평가한 자료라는 뜻은 아님. 2쪽에는 Liverpool 대학의 실제 mobile robotic chemist 사진 두 장을 사용하고 Burger et al. (2020)을 BibTeX로 인용. 3쪽은 전체 link의 충돌 회피를 설명하기 위한 별도의 MuJoCo 정적 예시.
- 9쪽 `Human-Inspired Intuition: Posture and Subgoals`는 Menger et al. (2012)의 초기 팔 자세에 따른 회피 동작 차이와 Lukic et al. (2014)의 subgoal 해석을 BibTeX로 인용. 사용자가 제공한 Lukic et al. (2014)의 인간 동작 3패널 사진을 작게 추가하고, 출처를 해당 BibTeX 항목으로 인용. 이 사진은 외부 문헌의 human demonstration이며 본 연구의 실험이나 IK 기반 제어의 검증 자료가 아님. 문헌의 subgoal은 formal link subtask의 검증 결과가 아니라 도달 동작을 구성하는 목표를 뜻함. 사진 아래의 세 자세 그림은 `Conceptual illustration`로 표시하고 `Initial posture → During reaching → Goal reached`로 설명. 어깨를 먼저 움직인다는 보편적 실행 순서의 근거로 사용하지 않으며, IK 해에서 link subtask target을 추출하는 절차는 문헌을 참고한 본 연구의 설계 아이디어로 구분.
- 7쪽에서 기존 연구의 제어 설계 초점(Design focus)과 운동 유도 방식(How motion is guided)을 비교한 뒤, 8쪽에서 Direct를 정의하고 기본 QP의 실패 사례로 정체 문제를 구체화. QP 형식 유지나 잔여 평형점은 우리 방법의 차별점으로 사용하지 않음. 관련 연구 하단은 로봇팔의 kinematic structure가 목표 진행을 어떻게 유도할 수 있는지 묻는 연구 질문. 9쪽 직관을 거쳐 12쪽에서 IK 해의 전체 팔 배치 → link subtask target → 순차 subtask 구성을 제안 방법으로 설명하며, 7·8쪽에서 Sequential 아이디어를 미리 설명하지 않음. Gonçalves도 link 거리·joint limit·목표 configuration을 사용하므로 운동학을 활용하지 않았다고 주장하지 않음. 차별점은 비교한 세 연구에 제시되지 않은 IK 해에서의 link subtask 구성 절차이며, 전체 분야 최초나 직접 성능 우위로 확대하지 않음. 차원 가정은 A17 및 한국어 Q&A에서 구분.
- 10쪽 `Chapter 3: Method Overview`는 3.1 Problem Formulation, 3.2 Sequential Kinematic Control Framework, 3.3 Application to the 3-DOF Planar Manipulator, 3.4 Extension to the 7-DOF Spatial Manipulator, 3.5 Learning-Based Selection of Two Link Subtasks의 다섯 section을 표시. 논문 3장의 연구 구성을 안내하며 전체 논문 목차나 발표 순서의 재현이 아님. 부제는 실제 TOC의 3장 short title이며 전체 chapter heading과 3.4·3.5의 원전 제목은 메타데이터에 보존.
- 11–15쪽 로봇 그림은 약 48 mm의 동일 영역과 위치·시점·축척을 유지. 11–12쪽에서 목표 정보를 추가하고 13–15쪽에서 저장된 실행 상태로 이어가며 애니메이션처럼 단계 변화를 비교. 6쪽은 목적함수·CLF·CBF의 기본 세 줄로 설명. 13–15쪽은 각 `ST_{ee,1}`, `ST_{ee,2}`, `ST_{ee,3}=T_{ee}`와 그 QP를 연결하고, 가중행렬과 joint/input limits를 포함한 실제 QP에서 변하는 CLF constraint만 강조. 16쪽은 같은 subtask 열과 함수 개형을 함께 표시.
- 17쪽 `Overall Sequential Control Framework`는 실행 전 safe IK에서 subtask target과 `ST_{ee,1} → … → ST_{ee,L+1}`을 한 번 구성한 뒤, 활성 `ST_{ee,s}`의 QP → 전환/종료 판정에서 세 갈래로 분기. Continue는 유효한 u를 적용하고 q를 갱신한 뒤 같은 subtask의 QP를 반복. Advance는 s를 증가시키고 입력 적용 없이 같은 q에서 다음 subtask의 QP를 계산. Stop은 실행 종료. QP 실패가 무조건 입력 적용으로 이어지는 구조가 아님. Planar의 boundary stall/QP infeasibility에 따른 예외 task 전환과 Spatial Manipulator의 stall/infeasibility/solver error 종료 정책은 노트·Q&A·A5에 보존.
- 그림 출처·저작자·figure 번호와 페이지 수 표기는 작은 글씨 사용을 허용. 본문·Appendix의 페이지 번호만 7 pt로 축소하며, section 진행 표시와 설명 본문·수식은 기존 크기 유지.
- 20쪽은 기존 MuJoCo 그림을 유지하고 Planar/Spatial 비교표를 배치. Planar의 위치·yaw와 Spatial의 위치·quaternion 기반 task, 양쪽의 capsule 모델과 Spatial self-collision용 RDF를 구분하며, capsule 사용 여부 자체를 모델 간 차이로 설명하지 않음. 17쪽에서는 모델별 비교를 생략하고 제어 흐름에 집중. 21쪽 왼쪽은 복잡한 두 2차원 물체의 개념도로 a의 표면 query point와 b의 거리장 관계를 시각화. 실제 로봇 RDF 학습 결과가 아니라 직접 제작한 도식이며, 우측은 query 좌표변환·거리장 평가·soft-min 및 safety margin으로 CBF를 구성하는 절차. 두 link의 운동은 CBF gradient에 함께 반영.
- 24쪽의 간결한 성공 설명은 기존 평가 기준을 바꾸지 않음. 목표 pose, 기록된 안전 조건, joint limit, solver 판정을 모두 만족한 결과를 사용.
- 26쪽은 특정 pair 대신 열 개 후보의 균등 선택 평균으로 단계 축소 효과를 설명. Validation 성공률 54.73%, 공동 성공 4,545개 episode–pair 결과의 시간 46.99 → 35.05초(−25.41%). 이는 577개 episode의 여러 후보를 포함하며 독립 episode 4,545개라는 뜻이 아님. 27쪽은 후보 평균·Validation 사후 최고 단일 pair·oracle 비교. 31쪽과 결론에서도 10-pair mean을 기준으로 학습 효과를 비교. 31쪽 그래프에는 Test에서 관측된 최고 단일 pair (3,5)의 성공률 58.70%(587/1,000)도 함께 표시하며, 해당 pair 대비 차이의 신뢰구간은 A14에서 별도 설명. 재집계 근거는 `evidence/two_link_average_audit.json`(Validation)과 `evidence/pair_mean_results_audit.json`(Test). Test 재집계와 그림은 `scripts/analyze_pair_mean_results.py`, `scripts/build_pair_mean_results_figures.py`로 재현.
- 32쪽은 All-link 대비 단계 축소의 동작 시간 효과에 집중. A6의 같은 3단계 비교에서는 공동 성공 4,792개 episode–pair 결과(609개 고유 episode)의 평균 시간이 10-pair mean 36.05초, Learned selection 36.20초로 추가 속도 향상이 관찰되지 않음. Learned 시간도 성공 pair 수만큼 반복되는 조건부 평균이며 4,792개의 독립 episode가 아님.
- 33쪽은 열 개 후보 중 (5,6)만 성공하는 설명용 사례. 실패 rollout은 Example pair (3,5), 성공 rollout은 Learned selection (5,6)으로 표시. 이 episode의 후보 평균 성공률은 10%이며 특정 예시를 전체 결과의 대표 표본으로 제시하지 않음.
- 35쪽 `Future Work: Experiments and Sequencing`은 두 향후 연구 방향에 각각 두 하위항목을 제시. 실제 로봇 실험은 폐루프 제어 실행과 성공률·최소 margin·완료 시간 평가, 뉴럴넷 기반 sequencing 개선은 상대기하 특징·network 구조 개선과 `Link subtask selection without a fixed count`로 구체화. 기존 모델도 로봇·목표·장애물 기하 입력과 attention을 사용하며 이를 개선하려는 계획. 현재 두 개로 고정한 link subtask 수를 episode에 따라 선택할 link와 개수를 정하는 방식으로 확장 검토. Final EE subtask와 QP의 충돌·joint-limit 안전 제약을 유지하며, 수행 완료 결과나 개선 성능을 주장하지 않음. 기존 온라인 실행·형상·센싱 관련 세부 한계와 현재 학습 loss 설명은 한국어 Q&A에 보존.
- Appendix도 같은 설명에 동기화. A3은 clipping 전 보조항의 비음수 1차 변화, A4는 고정 query 집합의 gradient와 soft-min 가중치, A5는 모델별 stall 조건을 구분. A9·A11은 충돌과 joint limit를 함께 다루며 A9에 연속시간 궤적·적용 속도·유한 전환 가정을 명시. A12는 논문의 공통 기본 QP, A13은 17쪽의 판정·분기 흐름. A14는 Train-best fixed pair (3,5)와의 추가 비교에서 CI가 0을 포함한다는 점과 기록된 평가 한계를 설명. 평균 대비 개선을 가장 좋은 고정 pair 대비 우월성으로 일반화하지 않으며, 35쪽의 실제 로봇 실험과 뉴럴넷 기반 sequencing 개선으로 연결. A15는 episode별 Learned 성공 여부와 후보 평균의 차이 및 bootstrap 집계 단위를 설명.

저장된 성공 label, 궤적, episode 집합은 변경하지 않고 비교 기준을 재집계한다. Test 성공률은 Direct 45.50%, All-link Sequential 55.20%, 10-pair mean 55.20%, Test 최고 단일 pair (3,5) 58.70%, Learned selection 60.90%이다. Learned–10-pair mean의 차이는 +5.70%p, episode 단위 paired bootstrap 95% CI는 [3.74, 7.65]%p이다. 한 episode의 열 개 후보를 함께 묶어 1,000개 episode를 10,000회 복원 추출하며 seed는 20261023이다. 추가 참고인 Train-best fixed pair의 Test 성공률은 58.70%, Learned와의 차이는 +2.20%p / CI [−0.30, 4.70]%p로 우월성이 확립되지 않음. Test는 개발 중 확인한 분할이다. Oracle 82.00%(Validation)·80.20%(Test)는 사후 상한이며 실행 가능한 정책 성능이 아니다.

## 현재 논문과 기호 동기화

2026-11-08 현재 Method의 명시적 정의를 기준으로 T_d, delta_p/delta_o, 활성 subtask V_i와 c_i, 입력 가중행렬 W_u(u는 upright 표기), joint-limit gain alpha_j 및 velocity bounds를 맞춘다. 13–15쪽의 실제 QP와 A12에는 joint-limit bounds와 velocity bounds를 논문처럼 두 행의 부등식으로 명시한다. 6쪽은 배경 설명을 위해 이 구현 세부사항을 생략하고, 입력 크기와 slack의 비용 및 CLF·CBF 두 제약만 제시한다. 배경 이론의 c_V와 Spatial의 relative unit quaternion epsilon_i·w_R는 역할에 맞게 유지한다. 발표의 실행 흐름은 `ST_{ee,1} → … → ST_{ee,L+1}=T_{ee}`로 표시한다. Link stage `s`의 활성 CLF는 `V_{i_s}`, 최종 단계는 `V_{ee}`로 구분한다. 논문 일반 정의의 mathcal S와 Spatial link-frame sequence는 최종 EE 포함 범위가 달라 그대로 통합하지 않는다. 학습 후보 집합 Omega의 원소 S_k는 선택할 link pair를 가리킨다. 목적함수에는 N_tr와 lambda_sel = 3을 명시한다.

Planar의 현재 함수는 V_i = 0.5 ||e_p,i||², V_e = 0.5 ||p_e − p_d||² + 0.5 w_o [1 − cos(psi_e − psi_d)], V3 = V_e이다. 그림에는 같은 저장 궤적을 이용해 이 함수를 다시 평가한 값을 사용한다. w_o = 0.5에서는 원래 저장된 함수 값의 1/20이다. 원시 궤적과 성공률·완료 시간은 그대로이며 새 QP의 실행 결과가 아니다. CLF scaling을 바꾸면서 slack penalty를 고정하면 일반적으로 같은 제어 입력이 보장되지 않는다.

논문에 남은 이전 기호나 정의 간 혼재는 `기호_동기화_기록.txt`와 `reviews/notation_sync_revision.json`에 기록한다. 일반 sequence와 Spatial의 link-subtask frame sequence 구분, Planar 도함수의 이전 가중치, IK/전환/최종 평가 tolerance를 임의로 합치지 않는다. 논문 파일은 수정하지 않는다.

## 파일

- `main.tex`: 영어 audience 슬라이드. `references.bib`: BibTeX/IEEEtran 참고문헌.
- `speaker-notes.tex`: 36개 한국어 대본과 시간의 원본. `발표대본.txt`: label 순서로 내보낸 대본.
- `발표구성.txt`: 실제 슬라이드 제목·시간·section·Appendix 페이지 대응.
- `질의응답_메모.txt`: 화면에서 덜어낸 수식 조건, 데이터 판정, 출처와 해석.
- `build/main.pdf`: 영어 57페이지. `build/presenter.pdf`: 영어 슬라이드+한국어 노트 93페이지.
- `evidence/story_plan.json`, `evidence/chapter_mapping.json`, `evidence/terminology.json`: 구조·논문 section·표기 원본.
- `Fig/`, `scripts/`, `evidence/`: 그림과 재생성 코드, 저장된 원본·해시·검증.
- `feedback.md`: 이전 수정의 사용자 피드백. `기호_동기화_기록.txt`: 현재 논문과의 기호 대응·혼재 항목. `reviews/`: 수정 및 검토 기록.
- `masters_defense_source.zip`: build 결과를 제외한 전체 소스 묶음.

## 빌드

필수 구성 요소: TeX Live, Beamer, ko.TeX, TikZ, Latin Modern, appendixnumberbeamer, latexmk, BibTeX, IEEEtran.bst. latexmk가 참고 문헌 생성과 반복 컴파일을 관리한다.

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error presenter.tex
python3 scripts/export_notes.py
python3 scripts/verify_artifacts.py
```

LaTeX Workshop은 설정된 latexmk recipe와 `-outdir=%OUTDIR%`를 사용한다. 해당 recipe를 우회할 수 있는 `% !TEX program`은 `main.tex`에 두지 않는다. 터미널 빌드의 기본 출력 위치는 `build/`이며 Workshop은 설정된 출력 디렉터리를 사용한다.

발표자 노트는 `\SpeakerNote{frame label}{seconds}{spoken text}` 형식으로 작성하며 고유한 frame label에 연결한다. 순서나 시간이 바뀌면 `evidence/story_plan.json`도 갱신한다. `main.tex`만 빌드할 때는 노트 파일을 불러오지 않는다.

그림을 다시 생성하려면 NumPy, SciPy, Matplotlib, pdfLaTeX, pdftoppm이 추가로 필요하다. PGF backend는 ko.TeX와 Latin Modern을 포함한 pdfLaTeX를 사용하며 PNG 미리보기는 생성된 PDF에서 렌더링한다.

```bash
python3 scripts/build_figures.py
python3 scripts/build_boundary_case.py
python3 scripts/build_story_figures.py
python3 scripts/build_concept_figures.py
python3 scripts/build_planar_setup.py
python3 scripts/build_progressive_planar.py
python3 scripts/build_rdf_query_figure.py
```

정량 결과와 방법 설명 그림은 포함된 보관 데이터와 명시한 함수를 사용하므로 제어 rollout이나 학습을 새로 수행할 필요가 없다. 연구 동기의 2쪽은 외부 실제 사진이며, 3쪽은 별도로 구성한 정적 기하 모델로 평가 데이터셋에 포함되지 않는다. 9쪽의 인간 동작 사진은 Lukic et al. (2014)의 문헌 자료이며 아래의 로봇팔 개념도와 구분한다. 2·3쪽이 같은 실험실이나 같은 제어 실행을 보여주는 것은 아니다.

2쪽 사진의 출처는 University of Liverpool, Cooper Group의 [2020-07-08 공식 기사](https://www.liverpool.ac.uk/cooper-group/news/stories/title,1215738,en.php)이다. 실험실 전경 [원본](https://www.liverpool.ac.uk/media/livacuk/coopergroup-new/images/IMG_4459_web.jpg)은 `Fig/motivation_liverpool_lab.jpg`, 시료 취급 [원본](https://www.liverpool.ac.uk/media/livacuk/coopergroup-new/images/robot%2Cscientist.jpg)은 `Fig/motivation_liverpool_handling.jpg`에 보관한다. 화면에는 `Photos: University of Liverpool`을 표시한다. 공식 페이지에 개별 사진가와 별도 이미지 이용허락 조건이 명시되지 않아 특정 사진가나 CC 라이선스를 부여하지 않는다. 아래 렌더링 명령은 사진을 생성하지 않으며 3쪽의 개념 그림을 재생성한다. 이전 `Fig/motivation_task.png`는 보관하되 현재 발표에서는 사용하지 않는다.

```bash
/home/jinoo/prj/safe-inv-clbf-robot-arm/.venv/bin/python scripts/render_motivation.py --repo-root /home/jinoo/prj/safe-inv-clbf-robot-arm --azimuth 110
/home/jinoo/prj/safe-inv-clbf-robot-arm/.venv/bin/python scripts/render_panda_slide.py --repo-root /home/jinoo/prj/safe-inv-clbf-robot-arm --azimuth 45 --elevation -25 --distance 3.2
/home/jinoo/prj/safe-inv-clbf-robot-arm/.venv/bin/python scripts/build_recovery_comparison.py --repo-root /home/jinoo/prj/safe-inv-clbf-robot-arm --azimuth 45 --elevation -20 --distance 3.5
```

MuJoCo 렌더링에는 MuJoCo, PyOpenGL, Pillow와 저장소의 Franka Emika 모델·mesh가 필요하다. 20쪽은 저장된 configuration을 물리 시뮬레이션 step 없이 렌더링한다. 33쪽도 저장된 초기·최종 joint vector를 동일한 카메라로 렌더링하며 중간 궤적을 재구성하지 않는다. 렌더링 설정, 원본 해시, configuration, 목표, 장애물은 해당 검증 기록에 저장되어 있다. 이전 `panda_scene.pdf`는 보관하되 20쪽에서는 사용하지 않는다. 기존 그림으로 빌드하면 Python이나 MuJoCo 없이 LaTeX만으로 컴파일할 수 있다.

## 근거와 검증

논문 원본은 `/home/jinoo/latex-ws/LaTeX-env/ws/masters_tethis`이며 저자 Jinwoo Lee, 지도교수 Gyunghoon Park이다. 발표 제목은 사용자의 feedback.md에 따라 갱신했으며 논문 원본 제목 자체는 변경하지 않는다. 발표 날짜는 지정하지 않는다.

외부 자율실험실의 실제 사진, 개념적 MuJoCo 동기 그림, 저장된 초기·최종 configuration의 정적 렌더링, 기존 실험의 수치 그래프를 구분한다. 이번 편집에서는 새 제어 rollout이나 학습을 수행하지 않는다. 선택한 IK는 subtask target의 출처이며 실행 경로가 아니다. Planar와 Spatial Manipulator의 CLF 구성 차이, all-joint velocity 사용, 수치 solver와 안전성 평가의 제한은 한국어 Q&A에 남긴다.

주요 검증 기록은 `evidence/sources.json`, `progressive_planar_audit.json`, `rdf_query_figure_audit.json`, `rdf_original_figure_audit.json`, `rdf_complex_query_figure_audit.json`, `two_link_average_audit.json`, `two_link_average_figure_audit.json`, `recovery_comparison_audit.json`, `refinement_claims_audit.json`, `verification.json`이다. `scripts/verify_artifacts.py`는 source 해시, PDF 페이지·배치·폰트·언어, note label/시간, 논문 section 제목, 참고문헌을 확인한다.

## 수정 이력

이번 feedback.md 반영 전 파일은 `build/feedback_revision/before/`에 보존한다. 이전 검토 기록은 아래와 같다.

관련 연구와 거리장 query 수정은 `reviews/literature_rdf_revision.json`에 기록되어 있으며 수정 전 백업은 `build/literature_rdf_revision/before/`에 있다.

영어 슬라이드 전환, 점 진행 표시, 17쪽 그래프 축소의 수정 전 백업은 `build/english_dots_revision/before/`에 있다. 이후 언어 방침을 영어 슬라이드와 한국어 발표 대본·질의응답 설명으로 구분했다.

21쪽 도식은 `scripts/build_rdf_complex_query_figure.py`로 생성하며 `evidence/rdf_complex_query_figure_audit.json`에 기하 검증과 파일 해시를 기록. 이전 capsule 도식과 [원전 Fig. 1(a)](https://arxiv.org/html/2307.00533v3/fig/robot_sdf_compressed.png)는 재현용으로 보존. 원전 파일의 출처·동일성 기록은 `evidence/rdf_original_figure_audit.json`에 유지.

과거 기록은 출처 추적을 위해 유지한다.

- Background 확장, 필요한 takeaway만 유지, 연속 progress bar: `evidence/background_progress_audit.json`; 백업 `build/background_progress_revision/before/`.
- 선배 발표 자료 비교와 단계별 시각 자료: `reviews/visual_story_revision.json`, `reviews/reference_comparison.json`; 백업 `build/visual_story_revision/before/`.
- 연구 질문, 전환·평가 조건, 학습과 제어의 연결, 실제 최종 configuration: `reviews/refinement_revision.json`; 백업 `build/refinement_revision/before/`.
- Planar 입력 구성과 실행의 구분: `reviews/planar_setup_revision.json`; 백업 `build/planar_setup_revision/before/`.
- 공식 제목, 용어 통일, 첫 등장 시 정의, Panda mesh 렌더링: `reviews/wording_mujoco_revision.json`; 백업 `build/wording_mujoco_revision/before/`.
- 개념 중심 도입부, 인용, 단계별 QP, 챕터 부제: `reviews/concept_revision.json`; 백업 `build/concept_revision/before/`.
- 전체 발표 흐름 개편: `reviews/story_revision.json`; 백업 `build/story_revision/before/`. 이전 질의응답 메모: `reviews/qanda_before_story.txt`.
- 최초 두 차례 비평·수정: `reviews/비평_수정_기록.txt`, `cycle_1.json`, `cycle_2.json`.
- 후속 요청: `style_revision.json`, `case30_revision.json`, `boundary_revision.json`, `flow_revision.json`, `notes_revision.json`, `workshop_revision.json`, `terminology_revision.json`.

과거 기록의 페이지 번호는 각 수정 당시의 발표 자료를 기준으로 한다. 기존 링크를 위해 한국어 파일명은 유지한다. 과거 검토 기록, evidence, 프로그램 코드는 번역 대상에 포함하지 않는다.

## 부제 번호와 입력 설명 — 2026-11-09

부제가 있는 본문·보충 슬라이드 50개의 제목 앞에 논문의 실제 장·절 번호를 표시. 예: 4.2.2 Effect of All-Link Sequential Control, 3.5.3 Network Architecture. 논문 source의 input 순서, TOC, AUX를 대조했으며 근거는 `evidence/thesis_section_numbers.json`에 보관. Bibliography는 원문처럼 번호 없음.

25쪽 takeaway는 All-link Sequential의 완료시간 증가로 수정. 29·30쪽의 `z_r`는 논문에 정의되지 않은 발표용 묶음 표기이므로 제거하고, 입력 상자를 Start, IK, and goal inputs로 명시. q0는 초기 관절 상태, q*는 검증된 IK 해, p_d와 R_d는 목표 EE pose이며, encoder 출력 및 attention query와 구분.

### 사진 출처와 공통 알고리즘 도식 — 2026-11-09

2쪽 사진 출처는 7 pt로 표시. 공통 제어 흐름은 17쪽에서 실행 전 준비와 제어 중 반복·분기로 구분. 논문의 capsule-shaped 링크 정의에 따른 Planar collision model의 `Capsule distances` 표기는 20쪽 모델 비교표에 유지. 17·20쪽은 각각 25초이며 전체 발화 계획 1,140초 유지.
