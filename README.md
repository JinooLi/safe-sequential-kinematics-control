# 안전 역기구학 기반 CLF 로봇 팔

Panda 7자유도 팔의 안전 자세 IK, CLF-CBF 관절 속도 제어, RDF 자기 충돌 평가를
연구하는 저장소다. 평면 3자유도 로봇의 비교 실험도 포함한다.

**구현을 처음 읽을 때는 [현재 구현 지도](docs/current/README.md)부터 시작한다.**
전체 실행 흐름을 먼저 설명하고, 핵심 기능이 받는 입력과 반환하는 출력을 연결한다.

| 문서 | 용도 |
|---|---|
| [현재 구현 지도](docs/current/README.md) | 현재 코드의 실행 흐름, 기능별 입력·출력, 결과 파일 |
| [코드 의존성 구조](docs/current/dependencies.md) | 계층별 import 방향, 공통 타입, 파일 이동과 호출 변경 |
| [기존 연구 기록](docs/history/README.md) | 설계 이유, 이전 요구사항, 상세 수식, 실험·성능 측정 기록 |
| [문서 관리 기준](docs/README.md) | 현재 설명과 과거 기록의 구분 및 갱신 방법 |
| [설정 파일 안내](configs/README.md) | 실행·학습·재현용 설정의 위치 |

## 실행 시작

```bash
uv sync --locked
# CPU에서 direct 경로를 짧게 확인한다. 수렴 성능 검증용 실행은 아니다.
make panda-direct PANDA_RDF_DEVICE=cpu ARGS="--duration 0.1"
make help
```

기본 입력은 [Panda 설정](configs/panda.yaml)이다. 위 명령의 `direct`는 IK를 생략하며,
RDF를 사용하는 경우 설정과 호환되고 검증을 통과한 학습 산출물이 필요하다.
실행 결과는 새 `outputs/panda/<run-id>/` 디렉토리에 저장한다.

`make panda-view`는 현재 YAML의 모드로 계산·저장한 뒤 기록을 재생한다.
Panda 실행 Make 명령은 기본적으로 CUDA를 선택하므로, CPU를 쓰려면
`PANDA_RDF_DEVICE=cpu`를 명시한다. GUI에는 display가 필요하다.
세부 옵션의 기준은 [Makefile](Makefile)과 각 명령의 `--help`다.

기존 긴 README는 [이전 프로젝트 안내](docs/history/project_readme.md)에 보존했다.
