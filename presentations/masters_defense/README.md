# Collision-Aware Sequential Kinematic Control of Robot Manipulators Using Control Barrier Functions and Supervised Learning-based Subtask Sequencing

20분 석사 졸업 발표 자료. **청중용 Beamer는 영어, 발표 대본·질의응답·작업 안내는 한국어**로 관리한다. 본문 36장과 보충 자료 21페이지로 청중용 PDF는 57페이지, 발표자용 PDF는 93페이지이다. 보충 자료는 목차 1개와 A1–A17이며 A7 참고문헌은 4페이지이다. 목표 발화 1,140초와 화면 전환·호흡 60초를 합해 20분으로 구성한다.

## 발표 흐름

1. 자율실험실·제조 환경에서 안전한 목표 도달의 필요성(2–3쪽).
2. Kinematic control, CLF constraint, CBF, QP와 기존 연구의 문제 접근(4–7쪽).
3. Direct의 실패 → 사람의 팔 배치에서 얻은 직관 → 논문 목차(8–10쪽).
4. 초기 상태·목표·장애물과 IK 기반 중간 목표 구성(11–12쪽).
5. 같은 QP의 활성 CLF constraint를 바꾸는 세 subtask와 함수 개형(13–16쪽).
6. 공통 Sequential 알고리즘, Planar 결과와 task transition의 작동 원리(17–19쪽).
7. Spatial Manipulator의 self-collision, IK, spatial pose CLF와 All-link Sequential(20–24쪽).
8. 동작 시간 증가, 중간 task 두 개로 축소, episode별 선택 가능성(25–27쪽).
9. Binary Cross Entropy와 selection loss, cross-attention, 논문 기호의 sequence 선택(28–30쪽).
10. Test 성공률·동작 시간·회복 예시, 결론과 향후 검증(31–35쪽).
11. 감사 인사와 Appendix 목차(36–37쪽).

## 화면과 용어 규칙

슬라이드 형식은 CDSL@UoS의 4:3 Beamer template이다. 본문·부제·그림·도표의 기준은 10pt, 표지·frame 제목은 14.4pt로 관리한다. 수학 첨자와 벡터 그림의 물리적 배율은 본문 typography와 구분한다. 핵심 메시지는 필요한 장의 상단에 검은색으로 표시하며, 모든 장에 요약을 반복하지 않는다. 연구실과 학교 로고를 함께 사용한다.

하단 점은 Motivation, Background, Planar, Spatial, Learning, Results의 본문 진행을 표시한다. 표지를 제외한 35개 점에서 완료·현재·남은 슬라이드를 구분한다. 보충 자료에는 별도 진행과 목차 복귀 링크를 사용한다. 부제는 논문의 실제 section/subsection 제목이며 발표 구간 이름과 구분한다. Conclusion에는 하위 section이 없어 chapter 제목을 사용하고 참고문헌 부제는 Bibliography이다. 원래 기종명이 들어간 논문 제목은 표시상 Spatial Manipulator로 바꾸되 실제 source heading을 메타데이터에 보존한다.

방법 이름은 Direct / Sequential / All-link Sequential / Preset pair (3,5) / Learned selection / Uniform selection이다. Direct는 중간 task 없이 end-effector 목표만 추종하며 Cartesian 직선 경로를 뜻하지 않는다. Sequential은 중간 link task 이후 EE task를 수행한다. Preset pair (3,5)는 Train 최고 평균 성공률로 선정하고 Validation/Test에서 고정한 baseline이다. A2는 Validation 결과이므로 최고 pair가 (2,5)인 것과 모순되지 않는다. 화면의 Panda 표기는 Spatial Manipulator로 통일하며 원본 데이터의 파일명과 식별자는 유지한다.

## 피드백 반영과 해석

- Shared workspace는 ISO의 사람–로봇 협업 문맥과 관련된 용어지만 현재 그림은 설비가 있는 환경이므로 Cluttered Workspaces로 명확화. 자율실험실의 실제 사례는 Burger et al. (2020)을 BibTeX로 인용.
- 사람의 어깨·팔꿈치 배치는 설계 직관이며 일반적인 인간 운동 법칙이나 별도 실험 결과로 주장하지 않음.
- 기존 연구는 일반 제어 구조를 수정하는 접근이며 Gonçalves는 4-DOF manipulator 실험도 포함. 기본 이론의 짝수 차원 조건과 별도의 홀수 차원 확장·15차원 실험을 A17에서 구분.
- 13–15쪽은 CLF 개형 대신 6쪽 QP를 반복하고 변하는 CLF constraint만 강조. 함수 개형은 16쪽에 유지.
- 17쪽은 공통 IK 목표 구성·CLF-CBF QP·Sequential 실행 구조를 설명. Planar의 예외 task 전환과 Spatial Manipulator의 stall/infeasibility 종료 정책은 다르며 A5에 보존.
- 21쪽은 RDF를 CBF candidate로 사용하기 위한 좌표변환, 표면 표본, soft-min, safety margins, 양쪽 link 운동 미분을 설명. 그림은 해석적 capsule 거리장 단면이며 실제 학습 RDF의 정확도 검증 결과가 아님.
- 24쪽의 간결한 성공 설명은 기존 평가 기준을 바꾸지 않음. 목표 pose, 기록된 안전 조건, joint limit, solver 판정을 모두 만족한 결과를 사용.
- 32쪽은 All-link 대비 단계 축소의 동작 시간 효과에 집중. 같은 3단계인 Preset pair 대비 시간 비교는 A6에 유지하며 학습 자체의 속도 향상으로 해석하지 않음.
- 35쪽은 수정 QP·고정 모델의 새 episode 평가, 온라인 계산/추종 검증, 실제 로봇의 형상 오차·센싱·충돌 여유 검증을 구분.

수치 결과, 저장된 궤적, episode 집합은 변경하지 않는다. Test 성공률은 Direct 45.50%, All-link Sequential 55.20%, Preset pair 58.70%, Learned selection 60.90%이다. Learned–Preset의 +2.20%p에 대한 paired bootstrap 95% CI는 [−0.30, 4.70]%p로 추가 개선의 불확실성이 남는다. Oracle 82.00%(Validation)·80.20%(Test)는 사후 상한이며 실행 가능한 정책 성능이 아니다.

## 파일

- `main.tex`: 영어 audience 슬라이드. `references.bib`: BibTeX/IEEEtran 참고문헌.
- `speaker-notes.tex`: 36개 한국어 대본과 시간의 원본. `발표대본.txt`: label 순서로 내보낸 대본.
- `발표구성.txt`: 실제 슬라이드 제목·시간·section·Appendix 페이지 대응.
- `질의응답_메모.txt`: 화면에서 덜어낸 수식 조건, 데이터 판정, 출처와 해석.
- `build/main.pdf`: 영어 57페이지. `build/presenter.pdf`: 영어 슬라이드+한국어 노트 93페이지.
- `evidence/story_plan.json`, `evidence/chapter_mapping.json`, `evidence/terminology.json`: 구조·논문 section·표기 원본.
- `Fig/`, `scripts/`, `evidence/`: 그림과 재생성 코드, 저장된 원본·해시·검증.
- `feedback.md`: 이번 수정의 사용자 피드백. `reviews/`: 수정 및 검토 기록.
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

정량 결과와 방법 설명 그림은 포함된 보관 데이터와 명시한 함수를 사용하므로 제어 rollout이나 학습을 새로 수행할 필요가 없다. 연구 동기의 작업 공간은 별도로 구성한 정적 기하 모델이며 평가 데이터셋에 포함되지 않는다.

```bash
/home/jinoo/prj/safe-inv-clbf-robot-arm/.venv/bin/python scripts/render_motivation.py --repo-root /home/jinoo/prj/safe-inv-clbf-robot-arm --azimuth 110
/home/jinoo/prj/safe-inv-clbf-robot-arm/.venv/bin/python scripts/render_panda_slide.py --repo-root /home/jinoo/prj/safe-inv-clbf-robot-arm --azimuth 45 --elevation -25 --distance 3.2
/home/jinoo/prj/safe-inv-clbf-robot-arm/.venv/bin/python scripts/build_recovery_comparison.py --repo-root /home/jinoo/prj/safe-inv-clbf-robot-arm --azimuth 45 --elevation -20 --distance 3.5
```

MuJoCo 렌더링에는 MuJoCo, PyOpenGL, Pillow와 저장소의 Franka Emika 모델·mesh가 필요하다. 20쪽은 저장된 configuration을 물리 시뮬레이션 step 없이 렌더링한다. 33쪽도 저장된 초기·최종 joint vector를 동일한 카메라로 렌더링하며 중간 궤적을 재구성하지 않는다. 렌더링 설정, 원본 해시, configuration, 목표, 장애물은 해당 검증 기록에 저장되어 있다. 이전 `panda_scene.pdf`는 보관하되 20쪽에서는 사용하지 않는다. 기존 그림으로 빌드하면 Python이나 MuJoCo 없이 LaTeX만으로 컴파일할 수 있다.

## 근거와 검증

논문 원본은 `/home/jinoo/latex-ws/LaTeX-env/ws/masters_tethis`이며 저자 Jinwoo Lee, 지도교수 Gyunghoon Park이다. 발표 제목은 사용자의 feedback.md에 따라 갱신했으며 논문 원본 제목 자체는 변경하지 않는다. 발표 날짜는 지정하지 않는다.

개념적 MuJoCo 동기 그림, 저장된 초기·최종 configuration의 정적 렌더링, 기존 실험의 수치 그래프를 구분한다. 이번 편집에서는 새 제어 rollout이나 학습을 수행하지 않는다. 선택한 IK는 중간 목표의 출처이며 실행 경로가 아니다. Planar와 Spatial Manipulator의 CLF 구성 차이, all-joint velocity 사용, 수치 solver와 안전성 평가의 제한은 한국어 Q&A에 남긴다.

주요 검증 기록은 `evidence/sources.json`, `progressive_planar_audit.json`, `rdf_query_figure_audit.json`, `recovery_comparison_audit.json`, `refinement_claims_audit.json`, `verification.json`이다. `scripts/verify_artifacts.py`는 source 해시, PDF 페이지·배치·폰트·언어, note label/시간, 논문 section 제목, 참고문헌을 확인한다.

## 수정 이력

이번 feedback.md 반영 전 파일은 `build/feedback_revision/before/`에 보존한다. 이전 검토 기록은 아래와 같다.

관련 연구와 거리장 query 수정은 `reviews/literature_rdf_revision.json`에 기록되어 있으며 수정 전 백업은 `build/literature_rdf_revision/before/`에 있다.

영어 슬라이드 전환, 점 진행 표시, 17쪽 그래프 축소의 수정 전 백업은 `build/english_dots_revision/before/`에 있다. 이후 언어 방침을 영어 슬라이드와 한국어 발표 대본·질의응답 설명으로 구분했다.

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
