# Allegro Hand V5 (ROS-free core)

Linux SocketCAN을 사용한 Allegro Hand V5 저수준 드라이버와 ZMQ 기반 런너, Python 시각화/텔레오프 스크립트를 포함합니다.

## 1) Install / 환경 준비
- SocketCAN 설정(1 Mbps, can0 예시):
  ```bash
  sudo apt install can-utils libzmq3-dev
  sudo modprobe can can_raw peak_usb
  ---
  sudo ip link set can0 down
  sudo ip link set can0 type can bitrate 1000000
  sudo ip link set can0 up
  ---
  ip -brief link show can0      # 링크 확인
  # 모니터 (선택)
  candump can0
  ```
- PCAN 커널 모듈이 설치돼 있다면 제거:
  ```bash
  sudo rmmod pcan
  sudo rm /etc/modprobe.d/pcan.conf /etc/modprobe.d/blacklist-peak.conf 2>/dev/null
  sudo modprobe peak_usb
  ```
- Python 의존성 설치(소스에서 실행):
  ```bash
  uv sync
  ```
- C++ 빌드:
  ```bash
  cmake -S . -B build
  cmake --build build
  ```

## 2) 예제 실행
- 드라이버/컨트롤 핑(읽기 전용):
  ```bash
  ./build/bin/ping_can can0
  ./build/bin/ping_control can0 left B # can_id, left/right, A/B
  ```
- ZMQ 런타임(기본 read-only, `--write`로 토크 입력 활성화):
  ```bash
  ./build/bin/allegro_run --help # help
  ./build/bin/allegro_run --can can0 --hand left --type B --write  # example
  ```
- Python 스크립트(uv run 사용):
  ```bash
  # 텔레메트리만 보기
  uv run telemetry_subscriber --port 5556

  # 시각화
  uv run vis_allegro --port 5556 --urdf left_B

  # GUI 슬라이더로 제어(REQ/REP + PUB)
  uv run control_allegro --rep 5555 --pub 5556 --urdf left_B --write
  ```

### 추가 정보
- C++ 상세 설정/옵션은 각 `cpp/...` 폴더의 README(또는 소스 주석)를 참고하세요.
