# Panda 실물 궤적 재생

저장한 `trajectory.npz`의 `time`, `q`를 실물 Panda에서 재생하기 위한 독립 도구다.
`prepare.py`는 오프라인 경로 생성·검증을 맡고, C++ `panda_replay`는 로봇과
유선 연결된 호스트에서 libfranka의 1 kHz 관절 위치 제어를 실행한다.
원래 실험의 CLF-CBF 제어기를 다시 실행하는 것은 아니다.

## 경로와 가정

- 현재 `q_d`에서 기록의 첫 자세까지 joint-space 경로를 자동 생성한다.
  직접 경로가 검사를 통과하면 사용한다. 실패하면 고정 시드 RRT로 우회점을 찾는다.
  각 우회점에서 정지하는 5차 구간을 써서, 보간으로 검증 경로의 모서리를 자르지 않는다.
- 기록에는 **5차 interpolating B-spline**을 적용한다. 각 기록 자세를 지나고,
  시작/끝의 속도와 가속도는 0이다. 접근·0.5초 정지·기록 재생·0.5초 정지를 연결한다.
  기록 내부는 C4, 경로 연결부는 C2이며 jerk에는 유한한 단차가 있을 수 있다.
- Bernstein 계수의 convex-hull 성질로 구간 **전체**의 위치·속도·가속도·jerk를
  제한한다. 시간만 늘리므로 경로는 바뀌지 않는다. 기본 상한은 0.15 rad/s,
  2 rad/s², 100 rad/s³이며 실제 최대치는 이보다 작을 수 있다.
  관절 위치에는 0.02 rad 여유를 둔다. 이 수치는 제조사 한계보다 낮은 재생기 정책값이다.
- 환경 제약은 수평 바닥 하나다. 기본 `--floor-z 0`은 **로봇 베이스 장착면**을
  바닥으로 간주한다. 베이스가 수평으로 고정되어 있어야 한다. 실제 바닥이 장착면보다
  아래이면 보수적인 설정이다. 더 높은 바닥/단차나 기울어진 베이스에는 이 가정이 맞지 않는다.
- 바닥에는 visual/collision mesh 전체의 convex hull을 사용한다. 고정된 장착 베이스
  `link0`만 바닥 여유에서 제외한다. 표준 Panda Hand의 0–80 mm 개방 범위를 포함하며,
  그리퍼를 움직이는 명령은 보내지 않는다. 별도 도구·물체·케이블은 모델에 포함되지 않는다.
- 자체 충돌에는 collision mesh와 그리퍼 개방 envelope를 사용한다. 인접 링크 및 같은
  고정 그리퍼 조립체 내부만 제외한다. 과거 RDF 실험의 비인접 링크 제외 목록은 적용하지 않는다.
  MuJoCo 거리 함수는 분리 방향의 후보만 제공한다. 메시 정점을 투영한 분리 평면에서
  실제 간격의 하한을 따로 계산하여 거리 함수의 수치 실패를 안전한 분리로 취급하지 않는다.
- 10 ms 이하 간격으로 형상을 검사하고, 구간 사이 이동량은 관절 속도의 연속 상한과
  링크별 회전 반경으로 보수적으로 차감한다. 허용 추종 오차 0.005 rad/관절과
  시작점 보정 0.0001 rad/관절도 차감한다. 그 뒤에도 바닥 20 mm, 자체 충돌 3 mm를 요구한다.
  이 계산은 모델과 명시한 오차 안에서의 경로 검사이며, 실제 제동 거리나 안전 인증은 아니다.

## 빌드

로컬에서는 프로젝트 `.venv`에 설치된 NumPy/SciPy/MuJoCo를 사용한다. 새 Python 의존성은 없다.
실물 실행 호스트에는 C++14, CMake, 호환되는 libfranka가 필요하다.

```bash
# 로봇 없이 오프라인 빌드
cmake -S hardware/panda_replay -B outputs/panda_replay_build -DCMAKE_BUILD_TYPE=Release
cmake --build outputs/panda_replay_build -j2
(cd outputs/panda_replay_build && ctest --output-on-failure)

# lab-server: 설치된 libfranka 0.8.0으로 빌드
cmake -S /home/cdsl/jinoo_ws/panda-hardware-replay/panda_replay \
  -B /home/cdsl/jinoo_ws/panda-hardware-replay/build \
  -DPANDA_WITH_FRANKA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build /home/cdsl/jinoo_ws/panda-hardware-replay/build -j2
(cd /home/cdsl/jinoo_ws/panda-hardware-replay/build && ctest --output-on-failure)
```

CMake가 없는 로컬에서는 `g++ -std=c++14 -O2 trajectory.cpp main.cpp -o panda_replay`로
오프라인 실행 파일을 빌드할 수 있다. 하드웨어 옵션은 명시적으로 켜야 한다.

## 생성과 읽기 전용 확인

1. lab-server에서 실제 관절 상태를 읽는다. `--snapshot`은 모션을 시작하지 않는다.

   ```bash
   /home/cdsl/jinoo_ws/panda-hardware-replay/build/panda_replay \
     --snapshot 172.16.0.2 > state.json
   ```

2. `state.json`과 선택한 `trajectory.npz`를 계획용 컴퓨터에 둔다.
   저장 결과가 정확히 어떤 실험인지 확인한 후 다음을 실행한다.
   출력 디렉터리는 매번 새 이름을 사용한다. 실패한 생성으로 이전 계획을 덮어쓰지 않는다.

   ```bash
   .venv/bin/python hardware/panda_replay/prepare.py \
     --trajectory /absolute/path/trajectory.npz \
     --state /absolute/path/state.json \
     --output outputs/hardware_checks/my_new_replay \
     --floor-z 0
   ```

   결과는 `replay.plan`, `report.json`, `preview.npz`다. 계획에는 spline과 바닥 검사 형상이
   함께 들어간다. 보고서는 원본/계획 SHA-256, 시작 자세, 접근점, 제한 및 여유를 기록한다.
   `.plan` 내부의 FNV-1a 체크섬은 실수로 생긴 파일 손상을 검출한다. 서명이나 신뢰 증명은
   아니므로 이 도구로 생성·검증한 파일만 사용한다. C++는 자체 충돌 검사 결과를 포함한
   오프라인 준비 과정을 신뢰하고, 바닥·관절·연속성 검사는 별도로 다시 수행한다.

3. `replay.plan`을 lab-server로 복사하고 읽기 전용 preflight를 실행한다.

   ```bash
   /home/cdsl/jinoo_ws/panda-hardware-replay/build/panda_replay \
     --preflight /absolute/path/replay.plan 172.16.0.2
   ```

   `--check PLAN`은 로봇 연결도 하지 않는다. `--preflight`는 전체 오프라인 검사 후
   현재 상태만 읽는다. 로봇이 snapshot 이후 0.0001 rad보다 이동하면 접근 경로를 다시 만들어야 한다.
   실제 관절 속도, 목표 속도/가속도, 추종 오차, Idle 상태, 오류, 기본 hand/payload 설정도 확인한다.

## 실제 실행

`--execute`는 실제 궤적 움직임을 시작한다. `--hold-test`도 위치 제어를 활성화해 현재
자세를 유지하므로 읽기 전용 검사가 아니다. 로봇 근처 운영자가 모델의 바닥 기준·표준 빈 그리퍼·
비어 있는 작업 공간을 확인하고 정지 장치를 사용할 수 있을 때 lab-server의 로컬 터미널에서 실행한다.
현재 작업은 외부 장애물을 검사하지 않으므로 사람이 로봇 경로에 있는 상태를 허용하는 설정은 아니다.

```bash
/home/cdsl/jinoo_ws/panda-hardware-replay/build/panda_replay \
  --execute /absolute/path/replay.plan 172.16.0.2
```

PREEMPT_RT와 메모리 잠금 권한을 요구한다. callback에서는 사전 적재한 spline을 실제
`Duration`으로 평가하고, 파일 읽기·출력·정상 경로의 동적 메모리 할당을 하지 않는다.
첫 callback에서도 현재 자세를 다시 확인한다. 허용 범위의 작은 시작 오차만 경로 전체에
상수로 더하며, 그 오차는 오프라인 형상 검사에 이미 포함되어 있다.

기본 통신 정책 `strict`에서 `--execute`는 먼저 시작 자세에서 0.5초 동안 위치를 유지하며 제어 명령 수신률을 확인한다.
최근 100개 명령 통계가 채워지는 첫 100 ms 이후에는 수신 성공률 **99% 이상**을 요구한다.
이 조건이 만족되어야 접근 경로를 시작하며, 재생 중에도 같은 조건을 계속 검사한다.
99%는 이 재생기의 정책값이며 제조사 안전 기능을 대체하지 않는다.

기존 충돌 임계값·게인·하중·안전 설정을 변경하지 않는다. 관절 임피던스 모드와
libfranka rate limiter를 사용하고, 저역통과 필터는 `kMaxCutoffFrequency`로 끈다.
기본 정책에서는 2 ms 초과 callback 간격, 낮은 명령 수신률, 관절 추종 오차, 목표 상태 편차, 바닥 여유 위반 또는 로봇 오류가
발생하면 callback 예외로 libfranka가 모션을 취소하도록 한다. 자동 오류 복구·재시작은 없다.
Ctrl-C/SIGTERM도 같은 취소 경로를 사용한다. 이것은 물리적인 비상정지 장치를 대체하지 않는다.
취소 중 실제 정지 경로는 계획의 형상 여유로 보장되지 않는다.

### 기존 제어기와 비교하는 통신 경고 모드

실제 제어 명령에 `--communication-policy=warn`을 명시하면 재생기 자체의
**2 ms 초과 간격과 99% 미만 수신률**만 중단 사유에서 제외한다. 시작 자세의
0.5초 유지 시간은 그대로다. 낮은 수신률이어도 이 시간이 지나면 경로가 진행된다.
경고 횟수와 최저 수신률은 제어 종료 후 출력하며 콜백 안에서는 출력하지 않는다.
기존 QPC 코드의 지연 경고 방식을 비교하기 위한 옵션이며, 토크 제어기로 바꾸는 옵션은 아니다.

```bash
./build/panda_replay --hold-test ./panda_cpp_replay_after_eee/replay.plan 172.16.0.2 --communication-policy=warn
# 위 10초 시험이 종료 코드 0으로 완료되고 시작 상태가 계속 일치할 때만 실행:
./build/panda_replay --execute ./panda_cpp_replay_after_eee/replay.plan 172.16.0.2 --communication-policy=warn
```

로봇 자체의 통신 오류와 충돌 검출, 시작 상태 검사, 관절 추종 오차 0.005 rad,
목표 상태 편차 0.001 rad, 바닥·경로 제한 및 잘못된 피드백 검사는 유지한다.
자동 오류 복구·재시작은 하지 않는다. `warn`에서 자세 유지 시험이 완료됐다는 것은
99% 통신 기준을 만족했다는 뜻이 아니다. 명시하지 않은 다음 실행은 다시 `strict`다.
이 옵션은 `--hold-test`와 `--execute`에서만 허용한다.

목표 상태 편차도 비교하려면 별도로 `--desired-state-policy=warn`을 추가한다.
기존 연구실의 QPC/CLBF 토크 제어기와 기본 `MotionGenerator` 관절 위치 예제에는
직전 요청과 로봇의 `q_d` 차이에 대한 0.001 rad 취소 검사가 없다. 이 옵션은
재생기에 추가한 해당 검사만 경고로 바꾸며, 종료 후 초과 횟수·최대 편차·관절 번호를
출력한다. NaN/무한대 피드백은 여전히 거부하고, 실측 `q`의 추종 오차 0.005 rad,
시작 상태와 바닥·경로 제한은 유지한다. 관절 위치 제어와 rate limiter도 유지한다.

```bash
./build/panda_replay --hold-test /absolute/path/to/fresh/replay.plan 172.16.0.2 \
  --communication-policy=warn --desired-state-policy=warn
# 자세 유지 완료 후, 같은 새 계획으로:
./build/panda_replay --execute /absolute/path/to/fresh/replay.plan 172.16.0.2 \
  --communication-policy=warn --desired-state-policy=warn
```

두 옵션은 독립적이며, 명시하지 않은 검사는 `strict`다. 알 수 없거나 중복된 옵션,
읽기 전용 모드에 붙인 완화 옵션은 로봇에 연결하기 전에 거부한다.

2026-11-02 첫 `warn` 비교 시험에서 10초 자세 유지는 완료됐다(9979 callback,
최저 수신률 49%, 최대 로봇 시각 간격 5 ms). 이어 실행한 저장 계획은 0.5초 시작
유지 뒤 접근 경로 약 0.3초에서 7번 관절의 목표 상태 편차 0.00103166 rad로 취소됐다.
이때 최저 수신률은 57%였다. 종료 후 상태 읽기에서는 `Reflex`와
`last_motion_errors: communication_constraints_violation`이 확인됐다. Reflex가
콜백 취소 전후 어느 시점에 발생했는지는 이 로그만으로 확정하지 않는다.
자동 복구·재시도는 실행하지 않았다. 자세가 바뀌었으므로 다음 실행에는 복구 후
새 snapshot으로 접근 경로를 다시 만들어야 한다. 기존 계획의 시작 허용값을 늘리지 않는다.
로그는 lab-server의 `warn_trial_20261102T040340Z/`에 저장했다.

목표 편차도 `warn`으로 바꾼 후 새 접근 경로로 한 번 더 시험했다.
10초 자세 유지는 완료됐지만 재생 중 실측 추종 편차가 0.0050661 rad에 도달해
유지한 0.005 rad 한계에서 취소됐다. 목표 편차는 최대 0.00509411 rad,
최저 명령 수신률은 51%였다. 로그는 `desired_warn_trial_20261102T040830Z/`에 있다.

### 원래 매뉴얼 예제와 비교한 결과

이후 사용자는 기존 매뉴얼의 기본 모션 예제 재현을 선택했다. 서버에 원래 있던
`/home/cdsl/libfranka/build/examples/generate_cartesian_pose_motion 172.16.0.2`를
수정 없이 한 번 실행했다. 실행 전 기존 `MotionGenerator`를 오프라인으로 평가하고
Cartesian 경로의 수치 IK 표본에서 바닥 여유를 확인했다. 이는 로봇 내부 IK 또는
통신 오류 발생 시 실제 경로의 보증은 아니다.

초기 관절 자세로 이동하는 단계는 완료됐지만 Cartesian 단계는 로봇 Reflex
`cartesian_motion_generator_joint_acceleration_discontinuity`로 중단됐다.
libfranka가 오류와 함께 보고한 `control_command_success_rate`는 0.478261이었다.
종료 후에도 같은 오류 기록과 Reflex 상태를 확인했고, 복구·재시도하지 않았다.
원래 예제에는 새 재생기의 통신/목표 편차/실측 추종 검사 코드가 포함되지 않는다.
현재 환경에서는 해당 검사 없이도 실패가 발생한다는 비교 결과이며, NIC·커널·로봇
중 어느 것이 근본 원인인지는 이 결과로 확정하지 않는다.

원래 예제는 자체의 기본 충돌 감도·임피던스 설정과 libfranka 기본 100 Hz 필터를
적용했다. 임시 NIC 드라이버와 performance governor는 유지했으므로 과거의 모든
시스템 설정을 복원한 시험은 아니다. 기록은 `manual_baseline_20261102/`에 있다.

## `FCI callback gap exceeds 2 ms` 진단

이 문구는 기본 `strict` 정책의 주기 감시에서 취소했다는 뜻이다.
Franka의 통신·안전 설정은 바꾸지 않는다. 먼저 로봇이 정지한 상태에서 다음을 실행한다.
`--diagnose`는 10초 동안 `Robot::read`만 사용하며 모션/제어 명령을 보내지 않는다.
실시간 우선순위와 메모리 잠금을 적용하고 spline 평가·바닥 계산의 비용도 측정한다.

```bash
cd /home/cdsl/jinoo_ws/panda-hardware-replay
./build/panda_replay --diagnose ./panda_cpp_replay_precise/replay.plan 172.16.0.2
```

종료 후 총 콜백 수, 최대 수신 간격, 1/2 ms 초과 횟수, 최대 콜백 계산 시간을 출력한다.
`--execute`도 정상 종료/취소 시 같은 통계를 출력한다. 마지막 1024개 콜백은 현재 디렉터리의
`panda_replay_timing_<timestamp>.csv`에 저장된다. 파일 기록과 출력은 콜백 종료 후에만 수행한다.
읽기 진단은 로봇 상태 수신과 호스트 계산만 검사하므로 실물 제어 명령 전달 품질을 입증하지 않는다.

CPU governor가 `powersave`이면 제조사 권장에 따라 `performance`로 전환해 비교할 수 있다.
아래 명령에는 서버의 sudo 암호가 필요하다. 부팅 설정을 변경하지 않는 런타임 설정이다.

```bash
cat /sys/devices/system/cpu/cpufreq/policy*/scaling_governor
sudo sh -c 'for f in /sys/devices/system/cpu/cpufreq/policy*/scaling_governor; do echo performance > "$f" || exit 1; done'
```

2026-11-02 lab-server의 동일한 10초 읽기 진단에서는 `powersave`에서 2 ms 초과 23회,
최대 4 ms가 관측됐고, `performance`에서는 2 ms 초과 0회, 최대 2 ms였다.
콜백 계산 최대 시간은 각각 70.777 µs와 15.601 µs였다. 이 결과만으로 장시간 제어 중
지연이 다시 발생하지 않는다고 보장하지는 않는다. 기본 정책의 2 ms 제한은 그대로 유지한다.
재부팅 또는 전원 관리 서비스의 설정 변경 후에는 governor를 다시 확인한다.

실험을 마친 후 기존 절전 설정으로 되돌리려면 다음을 사용한다.

```bash
sudo sh -c 'for f in /sys/devices/system/cpu/cpufreq/policy*/scaling_governor; do echo powersave > "$f" || exit 1; done'
```

## 상태 수신은 정상인데 제어 명령 전달이 불안정한 경우

`period=1 ms`는 연속 콜백에 전달된 **로봇 시각**의 차이다. 서버에서 실제로 콜백이
호출된 벽시계 간격이나 로봇의 제어 명령 수신 지연을 직접 측정한 값은 아니다.
2026-11-02 재생 중에는 로봇 시각 차이가 모두 1 ms였지만, 저장된 1024개 표본의
`control_command_success_rate`는 55–83%, 마지막 값은 57%였다. 로봇에도
`communication_constraints_violation`이 남았다. 이때 0.001 rad 편차 제한을 늘리거나
rate limiter를 끄는 것으로 해결하지 않는다.

명령 편차 로그는 `reference_1..7`(직전 요청), `measured_1..7`(실측),
`desired_1..7`(로봇 목표값)을 함께 저장한다. 차이의 최대치와 관절 번호도 종료 후 출력한다.

NIC EEE 절전은 다음 표준 ethtool 명령으로 확인/해제할 수 있다. 현재 서버는 시스템에
설치하지 않고 Ubuntu ethtool 패키지를 `nic_tools/unpacked`에 풀어 사용한다.
EEE 해제가 모든 지연의 해결을 보장하지는 않는다. NIC/드라이버 설정은 재연결·재부팅 후 재확인한다.

```bash
cd /home/cdsl/jinoo_ws/panda-hardware-replay
./nic_tools/unpacked/sbin/ethtool --show-eee enp4s0
sudo ./nic_tools/unpacked/sbin/ethtool --set-eee enp4s0 eee off
```

통신 Reflex만 남아 있을 때 운영자가 명시적으로 복구하려면 아래 명령을 사용한다.
활성 오류가 없고 마지막 오류가 `communication_constraints_violation` 하나인 경우만 받는다.
복구 후 상태를 출력하고 종료한다. 자동 재생이나 재시작을 하지 않는다.

```bash
./build/panda_replay --recover-communication 172.16.0.2
```

복구 후 snapshot으로 접근 경로를 다시 생성한다. 이번 장애 후 새로 만든 계획 경로는
`/home/cdsl/jinoo_ws/panda-hardware-replay/panda_cpp_replay_after_eee/replay.plan`이다.
운영자가 주변을 확인한 뒤 아래 **활성 위치 유지 테스트**를 직접 실행한다. 10초 동안 시작
자세를 유지하며 실제 제어 명령을 보낸다. 궤적은 진행하지 않지만 로봇 제어는 활성화된다.

```bash
./build/panda_replay --preflight ./panda_cpp_replay_after_eee/replay.plan 172.16.0.2
./build/panda_replay --hold-test ./panda_cpp_replay_after_eee/replay.plan 172.16.0.2
```

EEE 해제 후 자세 유지 시험에서도 100 ms 시점의 명령 수신 성공률이 62%여서
중단되었다. 콜백에 전달된 로봇 시각 차이는 1 ms, 최대 콜백 계산 시간은 18.825 µs,
목표 관절값 편차는 0이었다. 따라서 이 단계에서는 궤적을 실행하지 않고
명령 전달 지연을 조사한다. 호스트에서 관찰한 수신/송신 시점은 물리 선로의
시점을 직접 측정한 값이 아니므로 NIC나 로봇 측 문제를 단독으로 확정하지 못한다.

운영자는 아래 스크립트로 자세 유지 시험과 5초 패킷 캡처를 함께 실행할 수 있다.
**위치 제어를 활성화하므로 로봇 주변과 정지 장치를 확인한 뒤 직접 실행한다.**
sudo 암호는 서버 터미널에 입력한다. sudo는 패킷 캡처에만 쓰며 재생기는 일반
사용자 권한으로 실행된다. 오류 복구나 네트워크 설정 변경은 수행하지 않는다.

```bash
cd /home/cdsl/jinoo_ws/panda-hardware-replay
bash ./panda_replay/capture_hold.sh ./panda_cpp_replay_after_eee/replay.plan 172.16.0.2
```

출력의 `Capture directory`에 `control.pcap`, `capture.log`, `hold.log`가 저장된다.
`capture exit status: 124`는 5초 캡처가 시간 제한으로 종료되었다는 뜻이다.

2026-11-02 03:03 UTC 캡처에서는 명령 100개에 대응하는 상태 패킷 중 34개의
IP 조각 간격이 500 µs를 넘었으며 최대 1136 µs였다. 완성된 상태 패킷 수신 후
명령 송신까지의 호스트 캡처 시간은 중앙값 20 µs, 최대 46 µs였고,
명령 수신 성공률은 66%였다. 캡처 자체의 커널 드롭은 0이었다.
GRO가 켜진 상태였다. 임시 해제 후에도 100개 중 41개의 조각 간격이 500 µs를
넘었고 수신 성공률은 59%였다. GRO 해제로 해결되지 않았다.

```bash
sudo ./nic_tools/unpacked/sbin/ethtool -K enp4s0 gro off
# 원래 설정으로 복원하려면: sudo ./nic_tools/unpacked/sbin/ethtool -K enp4s0 gro on
```

현재 `5.9.1-rt20`의 r8169 드라이버에는 이후 upstream에서 수정한 RT 인터럽트
처리 코드가 남아 있다. 공식 5.9.1 소스를 수정 없이 다시 빌드한 `srcversion`이
설치된 모듈의 `4910F4E779F70AFA2802017`과 일치했다. `napi_schedule_irqoff`를
`napi_schedule`로 바꾼 첫 비교 모듈은 빌드·임시 로드됐지만 수신률 56%로 실패했다.
따라서 이 한 줄 변경은 해결책으로 검증되지 않았다.

MSI/MSI-X일 때 `IRQF_NO_THREAD`로 인터럽트를 등록하는 공식 후속 수정까지 포함한
두 번째 모듈은 `nic_tools/r8169_rt_fix/source_no_thread/r8169.ko`에 별도 빌드했다.
`srcversion`은 `25C40327344ACA546A12AC5`, SHA-256은
`e293a2c7e3e35d0d92853908460f4a1c052056db215efcdd035af7837f56ddd6`이다.
현재 NIC가 MSI-X를 사용함을 확인했다. 이 모듈을 운영자가 임시 로드한 후 읽기
전용으로 측정했으나, 상태 패킷 10,013개 중 870개의 IP 조각 간격이 500 µs를
넘었다. 제어 UDP 송신은 0개였고 캡처 드롭도 0이었다. **수신 지연이 남아 있으며
실제 제어 명령 전달 품질은 이 모듈에서 검증하지 않았다.**
적용 도구 `nic_tools/load_r8169_temporary.sh`는 해당 서버 전용이며 모듈 체크섬,
커널 버전, Idle/시작 자세를 검사한다. 기존 시스템 모듈 파일은 바꾸지 않고
임시 로드하며 실패 시 기존 모듈 복원을 시도한다. `--restore`로 원본 복원이 가능하다.

2026-11-02 사용자는 한때 에이전트에게 **로봇을 움직이지 말라**고 지시했다.
그 기간에는 `--hold-test`, `--execute`, 활성 `communication_test`, 오류 복구를 실행하지 않았다.
이후 통신 경고 모드의 **자세 유지 시험 통과 후 저장 궤적까지 재생**을 명시적으로 허용했다.
오류 복구와 실패한 시험의 자동 재시도는 이 절차에 포함하지 않는다. 아래 캡처는 상태 읽기만
수행하며 실제 명령 전달 품질을 검증하지 않는다. sudo 권한은 패킷 캡처에 필요하다.

```bash
bash ./panda_replay/capture_readonly.sh ./panda_cpp_replay_after_eee/replay.plan 172.16.0.2
```

공식 수정 근거: [RT에서의 NAPI 호출 수정](https://lists.openwall.net/netdev/2020/10/29/237),
[MSI/MSI-X 인터럽트 처리 수정](https://lists.openwall.net/netdev/2020/11/01/108).

진단 로그에는 `max_host_gap_ms`, `host_gaps_over_1_5ms`, CSV의
`host_interval_ms`도 추가했다. 이는 서버의 monotonic clock으로 잰 콜백 시작
간격이며 기존 `max_gap_ms`/`period_ms`의 로봇 시각 차이와 구분한다.
이 읽기 시험 당시에는 제어 중단 기준을 바꾸지 않았다. 후속 10초 읽기 시험 결과는 다음과 같다.

| 읽기 프로세스 배치 | 최대 호스트 간격 | 1.5 ms 초과 횟수 |
| --- | ---: | ---: |
| 기본 배치 | 5.481 ms | 1743 |
| CPU 3 | 6.161 ms | 1366 |
| CPU 21 | 5.182 ms | 454 |
| CPU 21 + 일시적 낮은 우선순위 계산 작업 | 7.370 ms | 108 |

비교용 계산 작업은 종료됐고 CPU affinity는 해당 진단 프로세스에만 적용했다.
CPU 유휴 드라이버는 `none`으로 표시됐다. 현재 5.9.1 RT 커널/NIC 조합에서
안정적인 수신 타이밍이 확인되지 않았으므로 이 결과를 실물 재생 통과로 취급하지 않는다.
이후 운영자가 `--restore`로 원본 NIC 모듈을 복원했으며, 로드된 `srcversion`이
`4910F4E779F70AFA2802017`임을 읽기 전용으로 확인했다. 운영자가 원본
`generate_cartesian_pose_motion` 예제를 실행했으나 종료 처리 중 관절 속도·가속도
불연속 오류가 발생했고 명령 수신 성공률은 0.58로 보고됐다. 이후 상태는 Reflex였고
목표 관절 속도는 모두 0이었다. libfranka의 `Motion finished commanded, but the robot
is still moving!` 문구는 종료 응답이 Reflex일 때 출력되므로 현재도 움직인다는 증거가
아니다. 원본 드라이버와 원본 예제에서도 실패했으며 통신·시간 지연의 물리적 원인은
아직 특정하지 못했다. 이 후속 확인에서는 제어·오류 복구·모션을 실행하지 않았다.

후속 읽기 전용 진단은 `outputs/hardware_checks/readonly_latency_20261102/`에 기록했다.
FIFO 99·메모리 잠금 상태의 1 kHz 타이머 80,000회에서 기상 지연은 최대 6.995 µs,
로봇 상태 170,014회 읽기에서 안정 구간의 커널 수신→프로그램 수신 완료 지연은
최대 37.693 µs였다. 그러나 커널의 상태 패킷 수신 간격은 최대 6.748 ms였다.
상태 메시지 번호 누락이나 NIC 오류·드롭 증가 없이 패킷이 늦게 몰려 들어왔다.

제어 기능이 없는 별도 ICMP 도구로 1 kHz·10,000회씩 비교했다. payload 1200,
1472, 1473바이트의 커널 수신 기준 왕복 지연은 각각 최대 0.231, 0.239, 0.274 ms였다.
로봇 상태 UDP payload와 같은 2341바이트의 기본 조건 두 시험에서는 중앙값이
약 1.020 ms, 1 ms 초과 비율이 67.3%, 69.1%였으며 두 시험 중 최대값은 61.396 ms였다.
모든 응답은 수신됐다. 작은 두 번째 IP 조각만 생기는 1473바이트 시험은 정상이므로
IP 분할 자체만으로 원인을 확정하지 않는다. 큰 패킷의 유선/NIC/드라이버/상대측 통신
경로에서 지연이 재현됐고, 일반적인 사용자 스레드 스케줄링 지연은 작았다.

NIC IRQ CPU에 임시 낮은 우선순위 작업을 두면 상태 수신 공백 횟수는 줄었지만,
큰 ICMP 패킷 지연은 해결되지 않았다. 이는 CPU 활동에 민감한 현상이라는 근거이며
특정 절전 상태나 드라이버 결함을 증명하지 않는다. 소프트웨어 타임스탬프만으로
NIC/IRQ/PCIe 지연과 선로·로봇 측 지연을 완전히 분리할 수는 없다. 이 진단에서는
로봇 제어·복구와 시스템 설정 변경을 하지 않았고 모든 임시 작업은 종료했다.
최종 로봇 상태는 기존 오류가 남은 Reflex, 목표 관절 속도는 0이었다.

후속 관리자 진단 결과는
`outputs/hardware_checks/network_isolation_20261102/privileged_20261102T052202Z_TTeOIW/`
에 저장했다. 6개 조건 모두 완료됐고 제어 UDP 송신은 0개였다. IRQ affinity/우선순위,
체크섬, NAPI 설정과 irqbalance의 원복을 직접 확인했다. 이 시험 전후 로봇은 Idle이었다.
**현재 작업 범위는 로봇 상태 읽기와 통신 진단이며, 모션·자세 유지·오류 복구는 금지다.**

인터럽트 스레드 기상→실행은 기본 조건 최대 3 µs, IRQ를 CPU 3/우선순위 80에 둔
조건 최대 5 µs였다. CPU 3 조건의 추적에는 첫 1500바이트 IP 조각을 처리하고 poll이
끝난 뒤, 다음 ICMP 송신 인터럽트 때 이전 응답의 889바이트 마지막 조각을 처리하는
사례가 반복됐다. 대표 시각은 13429.599405초(첫 조각), 13429.600237초(다음 요청 송신),
13429.600254초(이전 응답의 마지막 조각)다. IRQ 스레드가 장시간 기다리는 현상과
구분되며, NIC의 인터럽트 발생/수신 처리 경로에 문제가 있다는 강한 근거다.
소프트웨어 추적만으로 NIC 내부 상태나 실제 선로 도착 시각을 확정한 것은 아니다.

추적 없이 ICMP 송신 주기를 0.5/2/10 ms로 바꾸면 지연 중앙값도 각각 약
0.520/2.020/8.820 ms로 바뀌었다. 패킷 크기 1600~2100바이트에서는 재현되지 않았다.
추적을 켠 기본 조건은 오히려 좋아졌으므로, 이 조건과 기존 무추적 측정값을 단순
비교해 설정 효과로 해석하지 않는다. IRQ CPU 3 조건의 trace buffer는 앞부분이
일부 덮어써졌으며 분석은 남은 구간에 한정된다. 체크섬 변경과 기존 NAPI 타이머
시험에서도 상태 수신의 수 ms 공백은 남았다.

5.9.1 r8169의 poll은 `napi_complete_done()` 반환값을 무시하여, 재확인 타이머가
요청돼도 IRQ를 다시 켠다. [upstream 수정](https://lists.openwall.net/netdev/2020/08/19/202)
및 [NAPI 문서](https://docs.kernel.org/networking/napi.html)의 동작과 비교해 확인했다.
이를 고친 변경과 기존 RT NAPI 호출 수정만 포함한 별도 모듈을
`/home/cdsl/jinoo_ws/panda-hardware-replay/nic_tools/r8169_napi_check_20261102/source/r8169.ko`
에 빌드했다. `srcversion=683CD37F350CC777FB8D9F0`, SHA-256은
`0520ee206532228fa77aa30b5fc783985ded35901b201e9dd9db7a4d21a3fa18`이다.
빌드 및 vermagic 일치는 확인했으나 **아직 해결 검증 전**이다.

`nic_tools/check_napi_readonly.sh`는 원본 드라이버, 수정 드라이버의 기본값과
20/100 µs 재확인 설정을 추적 없이 비교하고 원본 드라이버·NIC 설정을 복원한다.
ICMP와 `Robot::read`만 호출하며, 모듈 파일이나 부팅 설정을 설치·수정하지 않는다.
2026-11-02 SSH에서 직접 실행을 시도했으나 `sudo: a password is required`로 중단돼
서버 터미널에서 이 고정 작업을 시작하도록 요청했다. 승인 부족이 아닌 관리자 인증
제약이며, 이 시도에서는 드라이버를 로드하거나 로봇 제어를 시작하지 않았다.

운영자가 실행한 `run_20261102T053545Z_hUdcbE`에서는 원본 드라이버의 5초 측정만
완료됐다. ICMP 왕복 지연 중앙값은 1.020 ms, 상태 패킷 수신 간격은 최대 5.127 ms로
기존 현상이 재현됐다. 수정 모듈 로드 후 설정 단계에서 ethtool 5.4가 지원하지 않는
표시용 기능 이름을 `-K` 인자로 전달한 스크립트 오류로 중단됐다. 원본 모듈은 다시
로드됐으나 같은 오류로 링크 대기를 건너뛰고 GRO도 켜진 채 남아 복원 검증에 실패했다.
이 결과는 수정 드라이버의 통신 성능을 평가한 결과가 아니다.

후속 직접 확인에서 원본 srcversion, IP/링크, EEE off, IRQ CPU 21/FIFO 50,
NAPI 설정 0/0, irqbalance active, 로봇 Idle/error-free를 확인했다. 기능 차이는
GRO on 한 가지였다. 스크립트를 지원되는 `-K enp4s0 gro off` 명령으로 수정하고,
설정 실패 시에도 링크 대기와 나머지 복원을 계속하게 했다. 상태 읽기 실패는 제한된
읽기 재시도 후 그대로 보고하며 빈 JSON을 파싱하지 않는다. NIC·로봇에 접근하지 않는
실패 처리 회귀 테스트 4개가 통과했다. 실제 GRO 변경은 관리자 권한 부족으로 아직
수행하지 못했으며, 다음 고정 작업은 남은 GRO 차이를 먼저 복원하도록 준비했다.

수정 후 `run_20261102T053953Z_1fafZd`의 6개 조건은 모두 완료됐고 원복도 검증했다.
추적·패킷 캡처 없이 측정했으며, 아래 값은 초기 100개를 제외한 안정 구간이다.
원본·타이머 없는 수정본은 조건별 5초, 재확인 설정은 조건별 10초씩 측정했다.

| 조건 | ICMP 왕복 중앙값 / 최대 | ICMP 1 ms 초과 | 상태 수신 간격 1.5 ms 초과 비율 / 최대 |
| --- | --- | --- | --- |
| 원본, 시험 전 | 1.020 / 9.113 ms | 4006/4900 | 28.46% / 6.315 ms |
| 수정본, 타이머 없음 | 1.020 / 68.043 ms | 4129/4900 | 29.21% / 5.519 ms |
| 수정본, 20 µs·2회 | 0.167 / 3.160 ms | 3/9900 | 0.77% / 5.664 ms |
| 수정본, 100 µs·2회 | 0.234 / 0.470 ms | 0/9900 | 0.97% / 6.405 ms |
| 수정본, 100 µs·20회 | 0.191 / 0.399 ms | 0/9900 | 0.98% / 5.711 ms |
| 원본, 원복 후 | 1.020 / 11.633 ms | 4250/4900 | 16.12% / 4.696 ms |

모든 ICMP 응답이 수신됐고 상태 메시지 번호 누락과 제어 UDP 송신은 0이었다.
기존의 잦은 약 1 ms ICMP 지연은 수정본과 수신 재확인을 함께 사용할 때 개선됐다.
그러나 상태 수신의 드문 수 ms 공백은 남아 있어 완전한 해결이나 실물 제어 검증으로
취급하지 않는다. 현재 드라이버와 GRO/EEE, NAPI, IRQ, irqbalance는 기존 값이며
로봇은 Idle/error-free다.

후속 `check_napi_readonly.sh --residual`은 재확인을 최대 약 100 ms 동안 유지하는
100 µs·1000회 조건에서 CPU 21과 CPU 3을 각각 20초씩 비교하고, 마지막 상태 읽기
8초에만 NAPI·타이머·스케줄링 추적과 패킷 캡처를 붙인다. 추적 자료는 지연 위치를
판별하는 용도로 쓰고 무추적 성능 수치와 혼합하지 않는다. 종료 시 원본 모듈과
설정을 복원하며 여전히 모션·자세 유지·복구는 호출하지 않는다. SSH sudo 인증이
없어 이 고정 작업은 운영자의 서버 터미널 실행을 기다리는 상태다.

후속 `run_20261102T054605Z_Y3qppP`는 완료됐고 원본 모듈·GRO off·NAPI 0/0·
IRQ CPU 21/FIFO 50·irqbalance active 및 로봇 Idle/error-free를 다시 확인했다.
수신 확인을 지속해도 상태 수신 공백은 최대 7.188/6.261 ms로 남았다.
ICMP 누락 983/972개는 각각 재로드 직후 연속된 시작 번호에만 해당했다.
두 경우 첫 응답은 시작 약 1.029/1.018초 후 도착했다. 링크 초기화·이웃 주소 확인
구간과 관련된 것으로 추정하지만 ARP를 캡처하지 않았으므로 원인을 확정하지 않는다.
시작 2초 이후에는 모든 ICMP 응답이 수신됐고 수정본의 최대 왕복 지연은
0.337/0.411 ms였다. 이 안정 구간 수치로 시작 구간 누락을 숨기거나 전체 무손실을
주장하지 않는다.

8초 캡처의 로봇 UDP 8,000개는 모두 두 조각이 112 µs 이내에 처리됐고 제어 UDP
송신은 0개였다. 앞선 큰 조각 간격 문제는 이 캡처에서 없어졌지만, 전체 상태 패킷
간격에는 최대 4.314 ms 공백이 있었다. gzip의 파일 크기 경고는 tracefs의 가상 파일
크기 때문이며 gzip 전체 읽기 검증은 통과했다. 저장된 trace는 35,509개 이벤트를
모두 보존했으나 NAPI 이벤트가 없었다. CPU 3만 추적한 설정이 NIC 수신 처리를
놓쳤기 때문이다. 실행 중인 NAPI의 pinned timer는 IRQ affinity 변경만으로 옮겨지지
않을 수 있으므로, 이 시험의 `continuous_cpu3` 라벨은 IRQ 배치를 뜻하며 실제
NAPI 처리 CPU 변경을 검증한 것으로 취급하지 않는다.

추적 마스크를 CPU 3+21로 수정하고, 향후 CPU 비교는 수신 타이머를 먼저 끝낸 뒤
IRQ를 옮기게 했다. NIC 재로드 후 읽기 전용 연결 검증을 거쳐 초기화와 성능 측정을
분리하고, tracefs는 파이프로 압축하도록 바꿨다. `--trace-only`는 8초 상태 수신 추적만
실행한 후 원복하는 고정 작업이다. 문법 및 실패 처리 회귀 테스트 4개가 통과했고,
SSH sudo 인증이 없어 이 짧은 작업의 서버 터미널 실행을 요청했다.

`run_20261102T055339Z_O3Z2fi`에서는 두 CPU 추적이 정상 완료됐다. 679,581개 이벤트가
모두 보존됐고 NAPI 확인과 `napi_watchdog` 타이머가 각각 78,088회 기록됐다.
실제 NAPI CPU는 21이었다. 안정 구간의 확인 주기는 중앙값 102 µs, p99 108 µs,
최대 129 µs였다. 가장 긴 상태 수신 공백 5.032 ms 안에서 NAPI가 48번 실행됐지만
모두 `work 0`이었고, 이후 Ethernet 프레임 8개(상태 4개)가 함께 처리됐다.
3 ms를 넘는 공백 24개 중 22개는 내부 확인 구간이 전부 빈 poll이었다.
나머지 공백에는 다른 프레임이나 경계의 일부 조각도 있어 모든 공백을 동일한
빈 구간으로 일반화하지 않는다.

이 관측은 남은 공백이 사용자 스레드 또는 NAPI 실행이 수 ms 지연돼 발생했다는
설명을 배제하는 근거다. 그러나 빈 RX descriptor 관측만으로 물리 선로에 패킷이
없었다고 증명할 수는 없다. 남은 범위는 NIC 내부 수신/DMA 처리부터 선로와 로봇
송신 측까지다. 현재 r8169에는 하드웨어 수신 타임스탬프와 PTP clock이 없고
lab-server의 유선 NIC도 RTL8125B 하나뿐이다. 운영자는 현재 다른 비교 장비가 없다고
답했다. 상세 결과와 최대 공백 그림은 해당 실행 폴더의
`residual_trace_analysis.json`, `largest_gap_trace_excerpt.txt`, `receive_gap_evidence.png`다.
원본 srcversion, GRO off, NAPI 0/0, IRQ CPU 21/FIFO 50, irqbalance active 및 로봇
Idle/error-free를 다시 확인했다. 제어 UDP 송신은 0개다.

추가 장비 없이 드라이버 구현을 비교하기 위해 Realtek 제조사 `r8125` 9.016.01을
[Debian 공식 원본 소스](https://deb.debian.org/debian/pool/non-free/r/r8125/)
에서 받아 `.dsc`의 SHA-256과 일치함을 검증하고 별도 빌드했다. 소스는 변경하지 않았고
빌드 옵션 `DISABLE_MULTI_MSIX_VECTOR=y`로 기존과 같은 단일 IRQ 비교를 준비했다.
모듈은 `nic_tools/r8125_vendor_20261102/r8125-9.016.01/src/r8125.ko`이며
`srcversion=26D0F0DB2E88580E19B3327`, SHA-256은
`8e04c1dda8436c2708dc9277d2e8ece00dc4bd894ecfeea180abdb9a2ed8432b`다.
실행 중인 5.9.1-rt20과 vermagic 일치를 확인했다. 아직 로드·해결 검증 전이다.

`check_napi_readonly.sh --vendor`는 제조사 드라이버를 절전·하드웨어 타이머 인터럽트
해제 조건(`aspm=0 eee_enable=0 eee_giga_lite=0 timer_count=0`)으로 임시 로드한다.
수신 인터럽트 방식과 100 µs 재확인 방식을 비교하고 원본 r8169와 설정을 복원한다.
모듈 설치, DKMS 등록, 부팅 설정 변경은 하지 않는다. NIC·로봇에 접근하지 않는
실패 처리 회귀 테스트는 제조사 모듈 제거/원복까지 포함해 6개 통과했다.
실행 시도는 SSH의 sudo 암호 요구로 중단돼 운영자의 고정 작업 실행을 기다리고 있다.

제조사 비교 `run_20261102T060715Z_aGdDxB`가 완료됐다. 아래 지연 통계는 각 측정의
처음 100개를 제외한다. 기존 드라이버 전/후는 각각 ICMP 5초와 상태 읽기 5초,
제조사 두 조건은 각각 ICMP 20초와 상태 읽기 20초다. 서로 다른 길이의 관측이므로
최댓값만으로 장기 신뢰성을 판단하지 않는다.

| 조건 | ICMP 중앙값 / 최대 (ms) | 상태 간격 >1.5 ms | 최대 상태 간격 (ms) |
| --- | --- | --- | --- |
| 기존 r8169, 전 | 1.020 / 26.053 | 1,316 / 4,903 (26.84%) | 5.689 |
| 제조사 r8125, 인터럽트 | 0.136 / 0.262 | 190 / 19,902 (0.95%) | 5.392 |
| 제조사 r8125, 100 µs 확인 | 0.232 / 0.346 | 179 / 19,902 (0.90%) | 6.300 |
| 기존 r8169, 후 | 1.020 / 26.097 | 1,587 / 4,902 (32.37%) | 4.970 |

네 조건 모두 ICMP 응답을 전부 수신했고 상태 메시지 ID는 연속이었다. 제조사 두 조건은
안정 구간 ICMP 1 ms 초과가 0/19,900개였다. 측정 중 NIC 오류 카운터 증가, 기록 버퍼
초과, 타임스탬프 설정 실패 및 제어 UDP 송신은 없었다. 제조사 드라이버의 일반
인터럽트 방식에서 큰 개선이 반복 확인된 원본 전/후 결과와 대비되므로, 기존 r8169의
수신 구현/하드웨어 초기화 경로가 빈번한 지연에 기여한다는 근거다. 개별 소스 라인이나
NIC 하드웨어 고장을 확정하는 시험은 아니다. 제조사 드라이버에서도 드문 5–6 ms
상태 공백은 남으므로 전체 FCI 문제가 해결됐다고 판단하거나 실물 실행을 허용하지 않는다.
추가 100 µs 확인의 이득도 이 표본에서는 뚜렷하지 않다.

직접 후속 확인에서 원본 srcversion, 기능/EEE/route 파일 일치, GRO off, NAPI 0/0,
IRQ CPU 21/FIFO 50, power on, irqbalance active 및 로봇 Idle/error-free를 확인했다.
제조사 드라이버는 현재 로드돼 있지 않다. 결과는
`outputs/hardware_checks/r8125_vendor_20261102/run_20261102T060715Z_aGdDxB/`에 보관한다.

관리자 설정 변경 없이 추가한 `concurrent_20261102T061118Z_3enc3vae`에서는 원본
r8169에서 상태 읽기 24초(CPU 2)와 ICMP 20초(CPU 3)를 동시에 수행했다. 상태
24,003개와 ICMP 20,000개를 모두 수신했고 제어 UDP 송신은 0개였다. 두 측정이 겹치는
안정 구간 약 18초에서 3 ms 초과 상태 공백 5개 모두 내부에 완결된 ICMP 왕복이
있었다. 그중 3개에는 0.5 ms 미만 왕복도 있었다. 가장 긴 겹침 구간 공백 5.328 ms
안에서 ICMP 4개가 완결됐고 그중 2개의 왕복은 147/138 µs였다. 전체 24초의 최대 상태
간격은 5.766 ms이지만 ICMP와 겹치지 않는 끝 구간이므로 이 동시성 근거에 쓰지 않는다.

이 관측은 해당 상태 공백 동안 NIC/선로/로봇 네트워크 전체가 완전히 정지했다는
설명을 배제한다. 로봇의 상태 UDP 송신 경로와 패킷 종류별 NIC 수신 처리는 아직
구분할 수 없다. ICMP가 IRQ를 유발해 원래 증상을 바꿀 수 있으므로 동시 측정 결과를
자연 상태 성능이나 FCI 제어 품질로 일반화하지 않는다. 분석은
`outputs/hardware_checks/network_isolation_20261102/analyze_concurrent.py`로 재현하며
상세 근거는 실행 폴더의 `concurrent_gap_analysis.json`에 저장했다.

향후 사용자가 실물 제어를 다시 허용하고 `Active hold test passed`를 확인한 후,
상태가 바뀌지 않았다면 같은 계획으로 `--execute`를
실행할 수 있다. 실패하면 출력과 CSV를 확인하고 재생을 진행하지 않는다. 위치 유지 테스트가
통과해도 전체 움직임 동안의 통신 품질이나 추종 성능까지 보장하는 것은 아니다.

## 오프라인 테스트

```bash
PANDA_REPLAY_BIN=outputs/panda_replay_build/panda_replay \
  .venv/bin/python -m unittest discover -s hardware/panda_replay/tests -p 'test_*.py' -v
```

원본 자세 통과, 정지 경계 조건, 연속 구간 상한, 입력 오류, 바닥 위반, 파일 손상,
C2 단절, 오래된 시작 자세 거부, Python/C++의 FK·보간 일치를 검사한다.
하드웨어를 움직이지 않는 테스트는 FCI 1 kHz 통신 품질이나 실물 추종 성능을 증명하지 않는다.
libfranka를 포함해 빌드하면 `test_franka_filter PLAN`으로 동일 라이브러리의 rate limiter에
전체 궤적을 넣어 볼 수 있다. 이 수치 테스트는 완전한 1 kHz 명령 전달을 가정하며,
Robot 객체를 만들거나 네트워크에 연결하지 않는다.

제조사 자료: [Panda 관절 제한과 경계 조건](https://frankarobotics.github.io/docs/robot_specifications.html),
[libfranka 제어 루프](https://frankarobotics.github.io/docs/doc/libfranka/docs/overview.html).
전원 정책과 통신 지연: [Franka 문제 해결 안내](https://frankarobotics.github.io/docs/troubleshooting.html#disabling-cpu-frequency-scaling).
MuJoCo 거리 계산을 독립적으로 확인하는 배경:
[convex mesh 거리 함수 이슈](https://github.com/google-deepmind/mujoco/issues/3383).
