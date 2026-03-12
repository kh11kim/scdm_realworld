// Simple CAN ping to verify AllegroHandDrv without ROS.
// Usage: ./ping_can [can_interface] (default: can0)

#include <chrono>
#include <string>
#include <thread>

#include "allegro_hand_driver/AllegroHandDrv.h"
#include "allegro_hand_driver/logging.h"

int main(int argc, char** argv) {
  std::string can_if = argc > 1 ? argv[1] : std::string("can0");

  allegro::AllegroHandDrv drv;
  if (!drv.init(can_if)) {
    AH_LOG_ERROR("Failed to initialize driver on %s", can_if.c_str());
    return 1;
  }

  double position[allegro::DOF_JOINTS] = {0};

  for (int iter = 0; iter < 10; ++iter) {
    int emergency = drv.readCANFrames();
    if (emergency < 0) {
      AH_LOG_ERROR("Emergency stop detected, exiting.");
      return 2;
    }

    if (drv.isJointInfoReady()) {
      drv.getJointInfo(position);
      drv.resetJointInfoReady();
      AH_LOG_INFO("[%02d] joints[0..3]=%.3f %.3f %.3f %.3f", iter,
                  position[0], position[1], position[2], position[3]);
    } else {
      AH_LOG_WARN("[%02d] joint info not ready", iter);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  AH_LOG_INFO("Ping complete on %s", can_if.c_str());
  return 0;
}
