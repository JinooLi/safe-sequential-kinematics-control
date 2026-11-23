# CICS Planar 발표자료

제출한 HWP 원고를 기준으로 별도 편집한 **10분 발표** 자료입니다. 한국어 슬라이드와 발표 대본을 분리했으며, 기존 석사 졸업 발표는 수정하지 않았습니다.

제목·설명·그래프·부록은 한국어로 구성하고, CLF·CBF·QP·IK·planar·end-effector 등 전문용어와 Direct·Sequential 제어 방식 이름은 필요에 따라 영어로 유지합니다. 상단 진행 표시의 섹션 이름은 영어입니다. 인용 논문의 제목·서지정보와 부록의 원고 그림은 원문을 보존합니다.

- `main.tex`, `main.pdf`: 청중용, 본문 18장 + 질의응답용 부록 8쪽 = 총 26쪽
- `speaker-notes.tex`, `발표대본.txt`: 한국어 대본. 발화 9분 20초 + 전환·호흡 여유 40초
- `presenter.tex`, `presenter.pdf`: 슬라이드와 대본을 번갈아 배치한 리허설용
- `발표구성.txt`: 슬라이드별 제목과 시간 배분
- `질의응답_메모.txt`: 원고의 주장 범위와 질문 대응
- `source/submitted_manuscript.hwp`: 제공된 원고의 변경 없는 복사본
- `source/manuscript.txt`, `source/equations.txt`: HWP에서 추출한 본문과 수식 스크립트
- `references.bib`: 원고의 세 참고문헌, motivation 사진 예시, task 분해·계층적 제어 관련 연구 두 편
- `reference-footnotes.tex`: 본문 인용과 같은 번호로 슬라이드 하단에 간략 서지정보 표시
- `slide-spacing.tex`: 본문 줄 간격, 설명 묶음 간격, 그림과 설명 사이 간격의 공통 설정

## 구성

1. 안전한 목표 도달의 필요성 및 planar 문제 정의
2. Capsule 거리 기반 CBF, 최종 pose CLF 및 Direct QP
3. 기존 task 분해·우선순위 제어와 goal IK 활용에 관한 연구 질문
4. 하나의 IK에서 link 목표 생성, link 1 → link 2 → end-effector 순차 제어
5. 이전 link task 해제, 활성 CLF 전환과 낮은 task 차원의 운동학적 의미
6. 동일한 300 episodes의 결과 및 원고 Figure 1의 시간 변화
7. 효과와 한계, spatial manipulator 확장 방향

3번·8–11번 그림은 동일한 좌표계와 크기를 사용합니다. 3번 그림에는 현재 end-effector 위치·방향 `p_3`, `psi_3`를 검정색으로, 목표 위치·방향 `p_d`, `psi_d`를 녹색으로 표시한 전용 그림을 사용합니다. 9–11번을 차례로 넘기면 실제 저장 자세의 진행을 설명할 수 있습니다. 본문의 부제는 **CICS 원고 절 번호**에 맞췄습니다. 추가한 관련 연구 슬라이드는 서론에 대응합니다. 상단 진행 표시의 점 하나가 본문 슬라이드 한 장이며 부록은 별도 번호입니다.

인용이 있는 슬라이드의 하단에는 `[번호] 저자, 논문 제목. 학술지, 연도.` 형식의 각주를 표시합니다. `\slidecite{key}`가 해당 frame의 인용을 수집하며, 제목과 연도 및 저자·학술지 약칭(`shortauthor`, `shortvenue`)은 `references.bib`에서 읽습니다. 인용 frame에는 고유 label이 필요합니다. 각주는 6.5pt이며 본문 글자는 10pt를 유지합니다. 전체 서지정보는 부록 참고문헌에 있습니다.

본문 간격은 `slide-spacing.tex`에서 조절합니다. 기본 글자 크기는 10pt, 기준선 간격은 13.2pt입니다. 같은 설명 묶음은 `\detailgap`으로 1mm, 다른 묶음은 `\groupgap`으로 3mm, 그림과 설명은 `\figuregap`으로 2mm를 추가합니다. 수식 행의 추가 간격은 1.5mm, 표 행 높이 계수는 1.25입니다. 남는 공간은 본문 전체의 위아래에 배분하고, 텍스트 사이를 자동으로 늘리지 않습니다. 부록 목차의 메뉴 항목은 균등 배치를 유지합니다. 상단 영역 높이 2.5mm는 `progress-style.tex`의 headline에서 별도로 조절합니다.

5쪽은 위첨자·분수 등 수식의 높이를 고려해 설명과 수식 사이에 `\groupgap`을 사용합니다. 대신 해당 QP 내부 행의 추가 간격만 0.5mm로 설정해 CLF와 QP 설명 묶음 사이의 여유를 확보합니다.

## 관련 연구 반영

7쪽은 각 연구의 설계 초점을 비교표로 보여줍니다. Mansard·Chaumette(2007)는 task 활성화·해제, Basso·Pettersen(2020)는 상위 task의 성능 보존, 본 연구는 목표 IK에서 링크별 위치 목표 구성에 초점을 둡니다. 하단에는 목표 IK의 팔 배치를 활용해 목표 도달을 돕는 링크 subtask를 구성할 수 있는지 묻는 연구 질문을 배치합니다. Task 분해, IK 활용 또는 QP 유지 자체를 최초 기여라고 주장하지 않습니다. 현재 planar 실험은 고정된 link 1 → link 2 → end-effector 순서이며 학습 기반 subtask 선택을 포함하지 않습니다.

12쪽과 대본에는 전환 시 이전 link task를 해제한다는 설명을 보강했습니다. 17쪽 결론에는 Direct만 성공한 10개 episode와 전역 수렴 미보장을 포함합니다. 관련 문헌은 발표의 배경 설명을 위한 추가이며 제출 원고와 실험 수치는 변경하지 않았습니다.

## 빌드

필요 환경: TeX Live, pdfLaTeX, latexmk, BibTeX, usebib, ko.TeX, Latin Modern, Python 3.

```sh
make all
```

LaTeX Workshop에서는 `main.tex`를 root로 지정하고 `latexmk -pdf` 레시피를 사용합니다. 기존 workspace의 `-outdir` 설정도 지원합니다. 리허설 PDF는 `presenter.tex`를 별도로 빌드합니다. 대본은 청중용 TeX 본문에 삽입하지 않습니다.

그래프 재생성에는 NumPy, Matplotlib와 pdfLaTeX가 필요합니다.

```sh
python3 scripts/build_cics_figures.py
python3 scripts/build_cics_direct_figure.py
python3 scripts/build_goal_pose_figure.py
```

## 원고·데이터 동기화

- 원고 국문 제목, 공동저자 5명, 소속 및 연구비 지원 정보 반영.
- CLF: `V_i = w_v ||p_i-p_i*||²` (i=1,2), `V_3 = w_v ||p_3-p_d||² + w_psi(1-cos(psi_3-psi_d))`.
- 발표 도입부 5쪽에서는 첨자 없는 `V(q)`와 `c`로 최종 pose CLF를 소개합니다. 9–10쪽에서 link별 `V_1`, `V_2`를 도입하고, 11쪽에서 `V_3(q) ≡ V(q)`, `c_3=c`로 원고 표기에 연결합니다.
- QP 기호는 원고의 `W`, `c_i`, `u_j,min/max`. `h_r`에는 충돌뿐 아니라 관절각 한계 CBF도 포함됩니다.
- 원고 본문의 `p_i*=p_i(q)`는 바로 앞의 IK 목표 정의에 따라 `p_i*=p_i(q*)`로 표기했습니다.
- 원고 Figure 1은 저장된 episode 30입니다. 원래 구현은 `10[||e_p||² + 0.5(1-cos e_psi)]`이므로 **원고의 계수 체계에서는 w_v=10, w_psi=5**입니다. 최신 학위논문용으로 1/20 정규화한 그래프는 사용하지 않았습니다.
- `cics_clf_time.pdf`는 저장 궤적과 CLF를 확인하여 다시 그린 벡터 그림입니다. 새 제어 실험이나 학습을 수행한 결과가 아닙니다.
- Table 1: Direct 216/300=72.00%, Sequential 253/300=84.33%. 차이는 상대 증가율이 아닌 **12.33 percentage points**입니다.
- Timeout 65/36, QP infeasibility 17/10, safety violation 2/1. Paired 결과 206/47/10/37은 동일 데이터의 추가 진단이며 원고 Table 1과 별도로 구분합니다.
- 사례 30의 Direct는 기록상 `qp_infeasible`입니다. 그림만으로 수학적으로 입증된 non-goal equilibrium 또는 timeout으로 단정하지 않습니다.
- 위치·방향 목표와 충돌 barrier 허용 기준은 원고대로 0.1 m, 0.1 rad, −0.01 m입니다. 기록된 collision barrier와 관절각 한계 CBF는 단위·집계를 구분합니다.
- Link-position task의 Jacobian rank는 일반적인 configuration에서 1, 2이며 최종 pose task는 3입니다. CLF 자체는 scalar 부등식이고, Jacobian nullity를 QP의 실제 가용 운동 방향 수와 동일시하지 않습니다.
- 초기 충돌 없는 상태, 연속 운동, 정확한 모델과 실행 가능한 CBF 제약 등의 조건 없이 전역 수렴·완전한 무충돌을 주장하지 않습니다. 구체적인 해석은 한국어 Q&A에 정리했습니다.

기존 학위 발표의 Panda, RDF, quaternion, 학습 기반 pair 선택 및 Train/Validation/Test 비교는 포함하지 않습니다. Motivation의 robotic chemist 사진은 안전한 목표 도달의 응용 예시이며 이번 실험 장비를 뜻하지 않습니다.

## 기록

`evidence/source_manifest.json`에 제출 원고와 기존 발표 파일의 SHA-256을 기록했습니다. `evidence/cics_figures.json`에 수치·CLF 재구성 검증을, `evidence/verification.json`에 최종 빌드·페이지·대본 검사를 기록합니다. `reviews/`에는 두 차례의 검토와 수정 내용을 남깁니다.

2026-11-10 관련 연구 보완과 검증 결과는 `reviews/research_gap_revision.json`에 기록했습니다.

2026-11-11에는 16쪽 대본을 subtask 전환 → 팔의 configuration 변화 → 동일한 최종 CLF로 목표 도달이라는 설명으로 보완했습니다. 16쪽에 40초를 배정하고, 전체 발화 9분 20초와 전환 여유 40초로 조정했습니다.
