# scdm_realworld

현재 워크플로우는 아래 순서입니다.

기본 robot 설정은 [`assets/config.yaml`](assets/config.yaml)에서 관리합니다.
- `robot.urdf`
- `q_preset.arm_home_q`
- `q_preset.hand_home_q`

## 1. Hardware Servers

Kinova:
```bash
uv run kinova_gen3 connect
```

Allegro:
```bash
uv run allegro_v5 connect
```

RS415:
```bash
uv run rs415 list
uv run rs415 connect --serial <serial>
```
카메라가 2대면 `list`로 serial을 확인한 뒤 각 serial마다 별도 프로세스로 띄웁니다.

## 2. System Calibration

```bash
uv run main calibrate
```

Notes:
- `cam_ext`, `cam_wrist` live image/frustum/pcd 확인
- `base_T_cam_ext` 추정
- `cam_wrist_correction` 으로 `cam_ext` 의 pointcloud와 urdf surface가 잘 일치하도록 수동 미세조정
- 결과 `base_T_cam_ext`, `cam_wrist_correction`를 `assets/system_calibration.yaml`에 저장

## 3. Environment Editor

```bash
uv run main env
```

현재 포함 기능:
- box environment 편집
- robot sphere / box collision 시각화
- 결과를 `assets/box_env.yaml`에 저장

## 4. Main Runtime

```bash
uv run main run
```

현재 포함 기능:
- Kinova + Allegro 실시간 시각화
- `cam_ext`, `cam_wrist` frustum / pcd 시각화
- arm planning / execute / goto
- hand goto
- health panel로 각 모듈 상태 확인

## 5. Saved Configs

- runtime config: [`assets/config.yaml`](assets/config.yaml)
- system calibration: [`assets/system_calibration.yaml`](assets/system_calibration.yaml)
- Kinova config: [`packages/kinova_gen3/config.yaml`](packages/kinova_gen3/config.yaml)
- box environment: [`assets/box_env.yaml`](assets/box_env.yaml)
