# DEVEL_FOR_NEW_ROBOT

새 로봇(예: Franka Panda)을 이 코드베이스에 붙일 때의 개발 순서 초안입니다.

목표는 아래 3가지를 분리해서 진행하는 것입니다.
- `model`: URDF, FK, IK, planning
- `real`: 실기 read / execute
- `app`: visualization, calibration, runtime UI

## 1. Asset 준비

먼저 로봇 모델 자산을 고정합니다.

- URDF 또는 xacro -> 최종 URDF
- mesh 경로 정리
- base / ee / camera / tool frame 이름 정리
- 실제 하드웨어와 맞는 joint 순서 확인
- URDF에 ghost link 추가
  - `ee`
  - `camera`
  - `palm` (hand / tool center가 있으면)

권장 결과물:
- `assets/<robot_name>/<robot_name>.urdf`
- `assets/<robot_name>/<robot_name>_spherized.yml`

주의:
- ROS package path에 의존하지 않는 URDF로 정리하는 편이 이후 Python 로딩이 쉽습니다.
- `ee`, `camera`, `palm` 같은 ghost link는 초반에 넣어두는 편이 좋습니다.
- sphere collision 자산은 `bubblify`를 사용해서 만들고, 파일 이름은 반드시 `{urdf_name}_spherized.yml`로 맞춥니다.

### 1-1. Asset 검증

현재 구조에서는 먼저 `RobotModel` 재사용 가능성을 봅니다.

해야 할 일:
- 새 URDF로 `RobotModel.from_urdf(...)`가 동작하는지 확인
- arm joint 이름 추출 확인
- `get_link_pose("ee")`, `get_link_pose("camera")` 확인
- joint limit이 URDF 기준으로 정상인지 확인

확인 항목:
- FK가 맞는지
- `visual_configuration` 길이와 actuated joint 수가 맞는지
- planning 대상 arm joint subset이 명확한지

만약 기존 `RobotModel`로 충분하면 새 클래스를 만들지 않습니다.
실기 연동이 필요할 때만 `RobotReal` 계층을 추가합니다.

## 2. Camera 계층 구현

카메라가 붙는 로봇이면 카메라도 별도 계층으로 분리합니다.

권장 구조:
- 카메라 서버는 별도 프로세스에서 실행
- 메인 앱과 직접 socket RPC로 이미지 전체를 주고받지 않음
- 이미지 / depth / intrinsics / 상태 메타는 shared memory로 공유

현재 코드베이스 기준 권장 방식:
- `uv run <camera_pkg> connect --serial <serial>`
- 서버 프로세스가 카메라에 직접 연결
- 최신 `rgb`, `depth`, `meta`를 shared memory에 계속 write
- 메인 프로세스는 reader만 attach해서 polling

왜 이렇게 하는가:
- 이미지 데이터는 크기가 커서 RPC보다 shared memory가 적합
- 메인 앱이 죽어도 카메라 프로세스는 계속 살릴 수 있음
- camera bring-up / reconnect / 디버깅을 UI와 분리할 수 있음

최소 요구사항:
- `list` 또는 동등 명령으로 serial 확인 가능
- `connect --serial <serial>`로 서버 실행 가능
- reader API에서 최소 아래 정보가 나와야 함
  - `rgb`
  - `depth`
  - `intrinsics`
  - `serial`

주의:
- depth / color optical center가 다를 수 있으므로 frame 정의를 초반에 명확히 해야 합니다.
- image frustum용 frame과 depth pcd용 frame이 같지 않을 수 있습니다.

## 3. Real Robot 패키지 추가

하드웨어 API는 별도 패키지로 분리하는 것이 좋습니다.

예:
- `packages/franka_panda`

권장 책임:
- `connect` CLI: 하드웨어 서버 프로세스 실행
- `client.py`: 메인 프로세스에서 쓰는 얇은 API
- `protocol.py`: request / response dataclass
- `config.yaml`: ip, home, joint limit override, 기타 실기 설정

최소 API:
- `get_joints()`
- `execute_joint_trajectory(...)`

있으면 좋은 API:
- `get_state()`
- `get_limits()`
- `stop()`

원칙:
- 외부 통신은 `multiprocessing.connection` 같은 로컬 IPC
- 하드웨어 세션은 별도 프로세스가 독점
- 메인 앱은 하드웨어 라이브러리를 직접 잡지 않음

## 4. RobotReal 구현

실기 패키지가 준비되면 `src/scdm_realworld/robot_real_<robot>.py` 또는 기존 구조에 맞는 `RobotReal` 확장 클래스를 만듭니다.

최소 책임:
- `get_joints()`
- `sync_from_real()`
- `execute_trajectory(...)`
- 필요하면 `get_joint_limits()` override

주의:
- URDF limit와 실기 limit가 다를 수 있으므로 override 지점을 분리해 두는 것이 좋습니다.

## 5. Main App 연결

그 다음에만 `main run`에 연결합니다.

필요한 것:
- real robot polling
- desired robot visualization
- arm panel
- planning / execute / goto
- health panel

권장 순서:
1. real robot만 보이게
2. desired robot 추가
3. slider 추가
4. plan
5. execute

처음부터 다 넣지 않는 편이 디버깅이 쉽습니다.

## 6. Calibration 연결

카메라가 로봇에 붙는 경우 calibration 체인을 분리합니다.

구분:
- `base_T_cam_ext`
- `cam_wrist_correction` 또는 `ee_T_cam_wrist`

권장 파일:
- `assets/system_calibration.yaml`

권장 작업 순서:
1. 외부 카메라 calibration
2. wrist camera calibration
3. wrist correction 수동 미세조정
4. main runtime에서 같은 config 재사용

## 7. Environment / Collision / Planning 확인

새 로봇이 들어오면 planning 쪽도 다시 확인해야 합니다.

확인 항목:
- joint limit
- self collision sphere set
- environment box collision
- straight-line feasibility
- RRT-Connect 성능

필요한 자산:
- `assets/<robot_name>/<robot_name>_spherized.yml`

## 8. Recommended Bring-up Order

실제 구현 순서는 아래가 안전합니다.

1. URDF를 standalone으로 visualize
2. FK / ee frame 검증
3. 실기 `get_joints()`만 연결
4. main에서 real robot 시각화
5. execute 단일 trajectory
6. planning 연결
7. calibration / camera 연결
8. environment collision 연결

## 9. Common Failure Modes

- URDF joint 순서와 실기 joint 순서가 다름
- URDF limit와 실기 limit가 다름
- camera frame이 optical center가 아니라 하우징 중심임
- ee/tool frame 정의가 실제 하드웨어와 다름
- 실기 API는 degree, 내부 모델은 radian
- execute는 되는데 state polling이 block됨
- visualization은 맞는데 pcd/world alignment는 안 맞음
- camera process는 살아 있는데 shared memory reader attach가 안 됨

## 10. Minimum Success Criteria

새 로봇 추가가 끝났다고 보기 위한 최소 조건:

- `uv run <robot_pkg> connect`가 된다
- `client.get_joints()`가 된다
- `main run`에서 real robot이 움직인다
- 카메라 서버가 별도 프로세스로 뜨고 main이 shared memory로 영상을 읽는다
- desired / real robot을 동시에 볼 수 있다
- 작은 joint trajectory execute가 된다
- `main calibrate` 또는 동등한 calibration 경로가 있다
- environment collision / planning이 최소 수준으로 돈다

## 11. Suggested Deliverables

- `assets/<robot>/<robot>.urdf`
- `assets/<robot>/<robot>_spherized.yml`
- `packages/<robot_pkg>/`
- `packages/<camera_pkg>/`
- `src/scdm_realworld/robot_real_<robot>.py` 또는 동등 구조
- `assets/system_calibration.yaml` 연동
- README 또는 bring-up 문서

이 문서는 초안입니다. 실제 새 로봇을 붙이면서
- 하드웨어 API 방식
- gripper 포함 여부
- camera chain
- collision 자산
에 따라 구체화하면 됩니다.
