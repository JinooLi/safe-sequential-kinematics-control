# safe inverse kinematics CLBF(?)


## 주의점

### No single source of truth

우선 이 코드에서 settings를 보면 single source of truth 원칙은 작살났다. 따라서, 무엇이 중복되는 정보인지 명시라도 하겠다.

1. 모델 정보는 `mujoco.xml`과 `urdf.xml` 두 파일에 모두 존재하며 그 형식이 상이하다.
2. 장애물 정보는 `mujoco.xml`과 `controller.yaml` 두 파일에 모두 존재한다.

