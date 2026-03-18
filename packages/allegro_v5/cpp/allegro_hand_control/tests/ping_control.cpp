// Minimal control loop: read state from driver, run control core, print torques.
// Usage: ./ping_control [can_interface] [left|right] [A|B]

#include <chrono>
#include <thread>

#include "allegro_hand_driver/AllegroHandDrv.h"
#include "allegro_hand_driver/logging.h"
#include "allegro_hand_control/control_core.h"

int main(int argc, char** argv) {
  std::string can_if = argc > 1 ? argv[1] : std::string("can0");
  ah_set_verbose(true);

  eHandType hand = eHandType_Right;
  if (argc > 2) {
    std::string hand_str = argv[2];
    if (hand_str == "left") hand = eHandType_Left;
  }

  eHardwareType hw = eHardwareType_B;
  if (argc > 3) {
    std::string type_str = argv[3];
    if (type_str == "A" || type_str == "a") hw = eHardwareType_A;
  }

  allegro::AllegroHandDrv driver;
  if (!driver.init(can_if)) {
    AH_LOG_ERROR("Failed to init driver on %s", can_if.c_str());
    return 1;
  }

  allegro_control::ControlCore core(hand, hw);

  allegro_control::StateInput state{};
  allegro_control::CommandInput cmd{};

  // Example: keep current pose and run default grasp force
  for (int iter = 0; iter < 10; ++iter) {
    int emergency = driver.readCANFrames();
    if (emergency < 0) {
      AH_LOG_ERROR("Emergency stop detected.");
      return 2;
    }

    if (!driver.isJointInfoReady()) {
      AH_LOG_WARN("[%02d] joint info not ready", iter);
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }

    double pos_buf[allegro::DOF_JOINTS];
    driver.getJointInfo(pos_buf);
    driver.resetJointInfoReady();
    for (int i = 0; i < allegro::DOF_JOINTS; ++i) {
      state.position[i] = pos_buf[i];
      cmd.desired_position[i] = pos_buf[i];
    }

    auto out = core.update(state, cmd, iter);
    AH_LOG_INFO("[%02d] torque[0..3]=%.1f %.1f %.1f %.1f", iter,
                out.torque[0], out.torque[1], out.torque[2], out.torque[3]);

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  AH_LOG_INFO("Control ping complete on %s", can_if.c_str());
  return 0;
}
