// Standalone runner replacing ROS launch: CAN loop + ZMQ command/telemetry.
// Usage: ./allegro_run --can can0 --hand left|right --type A|B [--write]
// ZMQ: PUB telemetry on tcp://*:5556, REP commands on tcp://*:5555

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "allegro_hand_control/control_core.h"
#include "allegro_hand_driver/AllegroHandDrv.h"

#include <zmq.h>
#include <iostream>

using namespace std::chrono_literals;

namespace {

std::atomic<bool> g_running{true};

void handle_sigint(int) { g_running.store(false); }

struct SharedCommand {
  allegro_control::CommandInput cmd;
  eMotionType motion_type = eMotionType_JOINT_PD;
};

std::optional<double> parse_number(const std::string &msg, const std::string &key) {
  auto pos = msg.find(key);
  if (pos == std::string::npos) return std::nullopt;
  pos = msg.find(':', pos);
  if (pos == std::string::npos) return std::nullopt;
  try {
    size_t endpos = 0;
    double val = std::stod(msg.substr(pos + 1), &endpos);
    (void)endpos;
    return val;
  } catch (...) {
    return std::nullopt;
  }
}

eMotionType motion_from_string(const std::string &s) {
  if (s == "home") return eMotionType_HOME;
  if (s == "grasp_3") return eMotionType_GRASP_3;
  if (s == "grasp_4") return eMotionType_GRASP_4;
  if (s == "pinch_it") return eMotionType_PINCH_IT;
  if (s == "pinch_mt") return eMotionType_PINCH_MT;
  if (s == "envelop") return eMotionType_ENVELOP;
  if (s == "gravcomp") return eMotionType_GRAVITY_COMP;
  if (s == "save") return eMotionType_SAVE;
  return eMotionType_JOINT_PD;
}

std::string motion_to_string(eMotionType m) {
  switch (m) {
    case eMotionType_HOME: return "home";
    case eMotionType_GRASP_3: return "grasp_3";
    case eMotionType_GRASP_4: return "grasp_4";
    case eMotionType_PINCH_IT: return "pinch_it";
    case eMotionType_PINCH_MT: return "pinch_mt";
    case eMotionType_ENVELOP: return "envelop";
    case eMotionType_JOINT_PD: return "joint_pd";
    case eMotionType_POSE_PD: return "pose_pd";
    case eMotionType_GRAVITY_COMP: return "gravcomp";
    case eMotionType_SAVE: return "save";
    default: return "unknown";
  }
}

}  // namespace

int main(int argc, char **argv) {
  std::string can_if = "can0";
  eHandType hand = eHandType_Right;
  eHardwareType hw = eHardwareType_B;
  bool do_write = false;
  bool verbose = false;
  std::string pub_addr = "tcp://*:5556";
  std::string rep_addr = "tcp://*:5555";
  bool show_help = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      show_help = true;
    } else if (arg == "--can" && i + 1 < argc) can_if = argv[++i];
    else if (arg == "--hand" && i + 1 < argc) {
      std::string h = argv[++i];
      if (h == "left") hand = eHandType_Left; else hand = eHandType_Right;
    } else if (arg == "--type" && i + 1 < argc) {
      std::string t = argv[++i];
      if (t == "A" || t == "a") hw = eHardwareType_A; else hw = eHardwareType_B;
    } else if (arg == "--write") do_write = true;
    else if (arg == "--verbose") verbose = true;
    else if (arg == "--pub-port" && i + 1 < argc) {
      pub_addr = "tcp://*:" + std::string(argv[++i]);
    }
    else if (arg == "--rep-port" && i + 1 < argc) {
      rep_addr = "tcp://*:" + std::string(argv[++i]);
    }
    else if (arg == "--pub" && i + 1 < argc) pub_addr = argv[++i];
    else if (arg == "--rep" && i + 1 < argc) rep_addr = argv[++i];
  }

  if (show_help) {
    std::cout << "allegro_run --can <if> --hand <left|right> --type <A|B> [--write]\n";
    std::cout << "            [--rep-port N] [--pub-port N] [--rep addr] [--pub addr]\n";
    std::cout << "  --can       CAN interface (default can0)\n";
    std::cout << "  --hand      left|right (default right)\n";
    std::cout << "  --type      A|B hardware type (default B)\n";
    std::cout << "  --write     enable torque write (off by default)\n";
    std::cout << "  --verbose   print CAN/control debug logs\n";
    std::cout << "  --rep-port  REP port (default 5555)\n";
    std::cout << "  --pub-port  PUB port (default 5556)\n";
    std::cout << "  --rep       full REP endpoint (e.g., tcp://*:5555)\n";
    std::cout << "  --pub       full PUB endpoint (e.g., tcp://*:5556)\n";
    return 0;
  }

  ah_set_verbose(verbose);
  std::signal(SIGINT, handle_sigint);

  allegro::AllegroHandDrv driver;
  if (!driver.init(can_if)) {
    AH_LOG_ERROR("Failed to init driver on %s", can_if.c_str());
    return 1;
  }

  allegro_control::ControlCore core(hand, hw);
  SharedCommand shared;
  shared.motion_type = eMotionType_JOINT_PD;
  shared.cmd.motion_type = eMotionType_JOINT_PD;
  shared.cmd.grasp_force = {1.0, 1.0, 1.0, 1.0};
  shared.cmd.motion_time = 1.0;
  bool cmd_initialized = false;

  std::mutex cmd_mutex;

  // ZMQ setup
  void *ctx = zmq_ctx_new();
  void *pub = zmq_socket(ctx, ZMQ_PUB);
  void *rep = zmq_socket(ctx, ZMQ_REP);
  zmq_bind(pub, pub_addr.c_str());
  zmq_bind(rep, rep_addr.c_str());

  // Command thread (REP)
  std::thread cmd_thread([&]() {
    while (g_running.load()) {
      char buf[2048];
      int n = zmq_recv(rep, buf, sizeof(buf) - 1, ZMQ_DONTWAIT);
      if (n <= 0) {
        std::this_thread::sleep_for(5ms);
        continue;
      }
      buf[n] = '\0';
      std::string msg(buf);
      bool ok = true;
      std::string error;

      std::lock_guard<std::mutex> lock(cmd_mutex);
      if (msg.find("set_grasp") != std::string::npos || msg.find("motion") != std::string::npos) {
        auto pos = msg.find("motion");
        if (pos != std::string::npos) {
          auto start = msg.find('"', pos + 6);
          auto end = msg.find('"', start + 1);
          if (start != std::string::npos && end != std::string::npos) {
            std::string m = msg.substr(start + 1, end - start - 1);
            shared.motion_type = motion_from_string(m);
            shared.cmd.motion_type = shared.motion_type;
          }
        }
      } else if (msg.find("set_force") != std::string::npos) {
        auto f = parse_number(msg, "force");
        if (f) {
          shared.cmd.grasp_force = {f.value(), f.value(), f.value(), f.value()};
        } else {
          ok = false; error = "force parse failed";
        }
      } else if (msg.find("set_motion_time") != std::string::npos || msg.find("motion_time") != std::string::npos) {
        auto t = parse_number(msg, "motion_time");
        if (t) {
          shared.cmd.motion_time = t.value();
        } else {
          ok = false; error = "motion_time parse failed";
        }
      } else if (msg.find("set_joint_command") != std::string::npos || msg.find("desired") != std::string::npos) {
        // Expect a comma-separated list after "desired": [x,y,...]
        auto pos = msg.find("desired");
        auto lb = msg.find('[', pos);
        auto rb = msg.find(']', lb);
        if (lb != std::string::npos && rb != std::string::npos) {
          std::string list = msg.substr(lb + 1, rb - lb - 1);
          std::stringstream ss(list);
          std::string item;
          int idx = 0;
          while (std::getline(ss, item, ',') && idx < allegro::DOF_JOINTS) {
            try {
              shared.cmd.desired_position[idx] = std::stod(item);
            } catch (...) {}
            ++idx;
          }
          AH_LOG_INFO("cmd set_joint_command desired[0:4]=%.3f %.3f %.3f %.3f",
                      shared.cmd.desired_position[0], shared.cmd.desired_position[1],
                      shared.cmd.desired_position[2], shared.cmd.desired_position[3]);
          shared.cmd.motion_type = eMotionType_JOINT_PD;
          shared.motion_type = shared.cmd.motion_type;
        } else {
          ok = false; error = "desired parse failed";
        }
      } else {
        ok = false; error = "unknown cmd";
      }

      std::string reply = ok ? "{\"ok\":true}" : ("{\"ok\":false,\"error\":\"" + error + "\"}");
      zmq_send(rep, reply.c_str(), reply.size(), 0);
    }
  });

  allegro_control::StateInput state{};
  double prev_position[allegro::DOF_JOINTS] = {0.0};
  bool have_prev = false;
  long frame = 0;
  bool warmup = true;

  while (g_running.load()) {
    int emergency = driver.readCANFrames();
    if (emergency < 0) {
      AH_LOG_ERROR("Emergency stop detected.");
      break;
    }

    if (!driver.isJointInfoReady()) {
      std::this_thread::sleep_for(2ms);
      continue;
    }

    double pos_buf[allegro::DOF_JOINTS];
    int tactile_buf[4] = {0};
    unsigned char temp_buf[allegro::DOF_JOINTS] = {0};
    double imu_buf[3] = {0.0, 0.0, 0.0};
    driver.getJointInfo(pos_buf);
    driver.getTactileSensors(tactile_buf);
    driver.getTemperature(temp_buf);
    driver.getImu(imu_buf);

    for (int i = 0; i < allegro::DOF_JOINTS; ++i) {
      state.position[i] = pos_buf[i];
      if (have_prev) {
        state.velocity[i] = (pos_buf[i] - prev_position[i]) / 0.002; // crude velocity
      } else {
        state.velocity[i] = 0.0;
      }
      state.temperature[i] = temp_buf[i];
      prev_position[i] = pos_buf[i];
    }
    have_prev = true;
    for (int i = 0; i < 4; ++i) state.tactile[i] = tactile_buf[i];
    for (int i = 0; i < 3; ++i) state.imu_rpy[i] = imu_buf[i];

    allegro_control::CommandInput cmd_copy;
    {
      std::lock_guard<std::mutex> lock(cmd_mutex);
      cmd_copy = shared.cmd;
      cmd_copy.motion_type = shared.motion_type;
      if (!cmd_initialized) {
        for (int i = 0; i < allegro::DOF_JOINTS; ++i) {
          cmd_copy.desired_position[i] = state.position[i];
        }
        shared.cmd.desired_position = cmd_copy.desired_position;
        cmd_initialized = true;
      }
    }

    auto out = core.update(state, cmd_copy, frame);

    if (do_write && !warmup) {
      if (frame % 200 == 0) {
        AH_LOG_INFO("ctrl frame %ld motion %s des[0:4]=%.3f %.3f %.3f %.3f pos[0:4]=%.3f %.3f %.3f %.3f tau[0:4]=%.3f %.3f %.3f %.3f",
                    frame,
                    motion_to_string(cmd_copy.motion_type).c_str(),
                    cmd_copy.desired_position[0], cmd_copy.desired_position[1],
                    cmd_copy.desired_position[2], cmd_copy.desired_position[3],
                    state.position[0], state.position[1],
                    state.position[2], state.position[3],
                    out.torque[0], out.torque[1], out.torque[2], out.torque[3]);
      }
      driver.setTorque(out.torque.data());
      driver.writeJointTorque();
    }

    driver.resetJointInfoReady();

    // Publish telemetry JSON
    std::ostringstream os;
    os << "{\"frame\":" << frame
       << ",\"hand\":\"" << (hand == eHandType_Left ? "left" : "right") << "\""
       << ",\"type\":\"" << (hw == eHardwareType_A ? "A" : "B") << "\""
       << ",\"motion\":\"" << motion_to_string(cmd_copy.motion_type) << "\"";
    os << ",\"position\":[";
    for (int i = 0; i < allegro::DOF_JOINTS; ++i) {
      if (i) os << ','; os << state.position[i];
    }
    os << "]";
    os << ",\"torque\":[";
    for (int i = 0; i < allegro::DOF_JOINTS; ++i) {
      if (i) os << ','; os << out.torque[i];
    }
    os << "]";
    os << ",\"tactile\":[" << state.tactile[0] << ',' << state.tactile[1] << ',' << state.tactile[2] << ',' << state.tactile[3] << "]";
    os << ",\"temperature\":[";
    for (int i = 0; i < allegro::DOF_JOINTS; ++i) { if (i) os << ','; os << static_cast<int>(state.temperature[i]); }
    os << "]";
    os << ",\"imu_rpy\":[" << state.imu_rpy[0] << ',' << state.imu_rpy[1] << ',' << state.imu_rpy[2] << "]}";
    auto msg = os.str();
    zmq_send(pub, msg.c_str(), msg.size(), ZMQ_DONTWAIT);

    warmup = false;
    frame++;
    std::this_thread::sleep_for(2ms);
  }

  g_running.store(false);
  cmd_thread.join();
  zmq_close(pub);
  zmq_close(rep);
  zmq_ctx_term(ctx);

  return 0;
}
