// ROS-free control core for Allegro Hand.

#pragma once

#include <array>
#include <cstdint>
#include <string>

#include "allegro_hand_driver/AllegroHandDef.h"
#include "allegro_hand_driver/AllegroHandDrv.h"
#include "allegro_hand_driver/logging.h"
#include "bhand/BHand.h"

namespace allegro_control {

struct StateInput {
  std::array<double, allegro::DOF_JOINTS> position{};
  std::array<double, allegro::DOF_JOINTS> velocity{};
  std::array<int32_t, 4> tactile{};
  std::array<unsigned char, allegro::DOF_JOINTS> temperature{};
  std::array<double, 3> imu_rpy{};
};

struct CommandInput {
  std::array<double, allegro::DOF_JOINTS> desired_position{};
  eMotionType motion_type = eMotionType_JOINT_PD;
  std::array<double, 4> grasp_force{1.0, 1.0, 1.0, 1.0};
  double motion_time = 1.0;      // seconds
};

struct ControlOutput {
  std::array<double, allegro::DOF_JOINTS> torque{};
  std::array<double, 4> fk_x{};
  std::array<double, 4> fk_y{};
  std::array<double, 4> fk_z{};
};

class ControlCore {
 public:
  ControlCore(eHandType hand, eHardwareType hw);
  ~ControlCore();

  // Configure time step (seconds). Default matches original loop 0.002s.
  void set_time_step(double dt);

  // Update controller given latest state and desired command. Returns torque/fk.
  ControlOutput update(const StateInput& state, const CommandInput& cmd, long frame_idx);

  // Access to BHand gains or other tuning can be added later.

 private:
  double dt_ = 0.002;  // control interval seconds
  BHand* bhand_ = nullptr;
};

}  // namespace allegro_control
