#include "allegro_hand_control/control_core.h"

namespace allegro_control {

ControlCore::ControlCore(eHandType hand, eHardwareType hw) {
  bhand_ = new BHand(hand);
  // Hand type (geared/non-geared) selection if supported by BHand
  bhand_->GetType(hw);
  bhand_->SetTimeInterval(dt_);
}

ControlCore::~ControlCore() {
  delete bhand_;
}

void ControlCore::set_time_step(double dt) {
  dt_ = dt;
  if (bhand_) {
    bhand_->SetTimeInterval(dt_);
  }
}

ControlOutput ControlCore::update(const StateInput& state, const CommandInput& cmd, long frame_idx) {
  ControlOutput out{};

  // Update desired values into BHand
  bhand_->SetJointPosition(const_cast<double*>(state.position.data()));
  bhand_->SetJointDesiredPosition(const_cast<double*>(cmd.desired_position.data()));
  bhand_->SetGraspingForce(const_cast<double*>(cmd.grasp_force.data()));
  bhand_->SetMotionType(cmd.motion_type);

  // Some motions depend on motion time
  bhand_->SetMotiontime(cmd.motion_time);

  // Update control
  bhand_->UpdateControl(static_cast<double>(frame_idx) * dt_);

  // Outputs
  bhand_->GetJointTorque(out.torque.data());
  bhand_->GetFKResult(out.fk_x.data(), out.fk_y.data(), out.fk_z.data());

  return out;
}

}  // namespace allegro_control
