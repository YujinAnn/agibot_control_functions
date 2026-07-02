// run_deploy_policy.cpp — C++ port of Yujin/agibot_control_functions/deploy_policy.py
// =====================================================================================
// The Python deploy_policy runs the ROM+FOM tracking stack at 50 Hz but spends ~20 ms
// per tick (mostly ROM rollout via casadi python bindings), which breaks the 50 Hz
// budget and destabilises ARM mode. This binary reuses the SAME C++ controller that
// run_standalone_sim uses (StandaloneTrackerController), so a tick costs a few ms and
// the policy is bit-identical to the working sim.
//
// It is a thin wrapper — NO MuJoCo plant, NO viewer, NO sim loop. Just text-file IPC
// with the ROS bridge (identical format to deploy_policy.py):
//   read  sensor_txt : t + 31*(pos vel) + torso quat(x y z w) + torso angvel(x y z)
//   write policy_txt : t + 31 pos + 31 kp + 31 kd        (all in SENSOR joint order)
// Joystick: A = ARM (drive policy), X = STAND_HOLD (stiff standing gains).
//
// Pipeline unchanged: [run_deploy_policy] --(policy_txt)--> robot_ros_bridge.py --(ROS)-->
//                     mujoco_x2.py --(ROS)--> robot_ros_bridge.py --(sensor_txt)--> [run_deploy_policy]

#include <unistd.h>  // getpid

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <boost/program_options.hpp>

#include "rom/deploy/config.hpp"
#include "rom/deploy/joystick_input.hpp"
#include "rom/deploy/paths.hpp"
#include "rom/deploy/types.hpp"
#include "rom/joystick.hpp"
#include "rom/tracker/abi.hpp"
#include "rom/tracker/standalone_controller.hpp"

namespace po = boost::program_options;

#ifndef ROM_REPO_ROOT
#define ROM_REPO_ROOT "."
#endif

namespace {

// SENSOR joint order — MUST match robot_ros_bridge.py / deploy_policy.py JOINT_ORDER
// (head 2 + waist 3 + arm 14 + leg 12 = 31). sensor_txt/policy_txt use this order.
const std::vector<std::string> kSensorOrder = {
    "head_yaw_joint", "head_pitch_joint",
    "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint",
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
};
constexpr int kNJoint = 31;
constexpr double kBasePosZ = 0.68;              // dummy base height (No-State-Estimation)
constexpr double kStandRomSyncPeriodS = 0.5;    // re-sync ROM from robot while disarmed

struct CliArgs {
  std::filesystem::path config;
  std::filesystem::path sensor_txt;
  std::filesystem::path policy_txt;
  bool joystick = true;
  double auto_arm_after = 0.0;  // debug: auto-ARM after N seconds (joystick A substitute)
  bool print_status = false;    // --print: 상태줄을 1Hz 로 출력
};

CliArgs parse_args(int argc, char** argv) {
  CliArgs args;
  const std::filesystem::path repo_root(ROM_REPO_ROOT);
  const std::filesystem::path io_dir = repo_root / "Yujin" / "agibot_control_functions";
  args.config = repo_root / "rom_deploy" / "configs" / "x2.yaml";
  args.sensor_txt = io_dir / "sensor_txt";
  args.policy_txt = io_dir / "policy_txt";

  po::options_description desc("C++ deploy policy: sensor_txt -> tracking controller -> policy_txt (50 Hz).");
  desc.add_options()("help,h", "Show help")(
      "config", po::value<std::filesystem::path>(&args.config)->default_value(args.config),
      "Deploy config (default: <repo>/rom_deploy/configs/x2.yaml)")(
      "sensor", po::value<std::filesystem::path>(&args.sensor_txt)->default_value(args.sensor_txt),
      "sensor_txt path (state input from bridge)")(
      "policy", po::value<std::filesystem::path>(&args.policy_txt)->default_value(args.policy_txt),
      "policy_txt path (command output to bridge)")(
      "no-joystick", "Disable joystick (stay in STAND_HOLD unless --auto-arm-after)")(
      "auto-arm-after", po::value<double>(&args.auto_arm_after)->default_value(0.0),
      "Debug: auto-ARM after N seconds (0=off)")(
      "print", "Print the status line at 1 Hz (default: only on button press)");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  if (vm.count("help")) {
    std::cout << desc << '\n';
    std::exit(0);
  }
  if (vm.count("no-joystick")) {
    args.joystick = false;
  }
  if (vm.count("print")) {
    args.print_status = true;
  }
  po::notify(vm);

  const std::filesystem::path repo(ROM_REPO_ROOT);
  args.config = rom::deploy::resolve_deploy_config_path(repo, args.config);
  return args;
}

// Atomic write: tmp (PID-unique) + rename, so the reader never sees a partial line and
// concurrent writers on the same dir don't collide on the tmp name.
void atomic_write(const std::filesystem::path& path, const std::string& text) {
  const std::filesystem::path tmp = path.string() + ".tmp." + std::to_string(::getpid());
  {
    std::ofstream f(tmp, std::ios::out | std::ios::trunc);
    f << text;
  }
  std::filesystem::rename(tmp, path);
}

// Parsed sensor frame (last line of sensor_txt).
struct SensorFrame {
  double t_sensor = 0.0;
  std::unordered_map<std::string, float> pos;
  std::unordered_map<std::string, float> vel;
  Eigen::Vector4f quat_wxyz = Eigen::Vector4f(1.f, 0.f, 0.f, 0.f);
  Eigen::Vector3f ang_vel = Eigen::Vector3f::Zero();
};

std::optional<SensorFrame> read_sensor(const std::filesystem::path& path) {
  std::ifstream f(path);
  if (!f) return std::nullopt;
  std::string line, last;
  while (std::getline(f, line)) {
    if (!line.empty()) last = line;
  }
  if (last.empty()) return std::nullopt;

  std::istringstream ss(last);
  std::vector<double> v;
  double x;
  while (ss >> x) v.push_back(x);
  const int need = 1 + 2 * kNJoint;  // t + 31*(pos,vel)
  if (static_cast<int>(v.size()) < need) return std::nullopt;

  SensorFrame fr;
  fr.t_sensor = v[0];
  for (int k = 0; k < kNJoint; ++k) {
    fr.pos[kSensorOrder[k]] = static_cast<float>(v[1 + 2 * k]);
    fr.vel[kSensorOrder[k]] = static_cast<float>(v[1 + 2 * k + 1]);
  }
  const auto* rest = v.data() + need;
  const int nrest = static_cast<int>(v.size()) - need;
  if (nrest >= 4) {  // quat x y z w -> wxyz
    fr.quat_wxyz = Eigen::Vector4f(static_cast<float>(rest[3]), static_cast<float>(rest[0]),
                                   static_cast<float>(rest[1]), static_cast<float>(rest[2]));
  }
  if (nrest >= 7) {  // ang vel x y z
    fr.ang_vel = Eigen::Vector3f(static_cast<float>(rest[4]), static_cast<float>(rest[5]),
                                 static_cast<float>(rest[6]));
  }
  return fr;
}

rom::deploy::RobotState make_robot_state(const SensorFrame& fr,
                                         const std::vector<std::string>& deploy_names) {
  rom::deploy::RobotState st;
  const int n = static_cast<int>(deploy_names.size());
  st.joint_pos.resize(n);
  st.joint_vel.resize(n);
  for (int i = 0; i < n; ++i) {
    const auto p = fr.pos.find(deploy_names[i]);
    const auto d = fr.vel.find(deploy_names[i]);
    st.joint_pos[i] = (p != fr.pos.end()) ? p->second : 0.f;
    st.joint_vel[i] = (d != fr.vel.end()) ? d->second : 0.f;
  }
  st.base_quat_wxyz = fr.quat_wxyz;
  st.base_ang_vel_b = fr.ang_vel;
  st.base_pos_w = Eigen::Vector3f(0.f, 0.f, static_cast<float>(kBasePosZ));  // dummy (ROM init)
  return st;
}

// Write a JointCommand (in deploy order) to policy_txt in SENSOR order.
void write_policy(const std::filesystem::path& path, const std::vector<std::string>& deploy_names,
                  const rom::deploy::JointCommand& jc) {
  std::unordered_map<std::string, float> q, kp, kd;
  for (int i = 0; i < static_cast<int>(deploy_names.size()); ++i) {
    q[deploy_names[i]] = jc.q[i];
    kp[deploy_names[i]] = jc.kp[i];
    kd[deploy_names[i]] = jc.kd[i];
  }
  const double t = std::chrono::duration<double>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count();
  std::ostringstream out;
  out << std::fixed << std::setprecision(6) << t;
  for (const auto& nm : kSensorOrder) out << ' ' << std::setprecision(6) << q[nm];
  for (const auto& nm : kSensorOrder) out << ' ' << std::setprecision(4) << kp[nm];
  for (const auto& nm : kSensorOrder) out << ' ' << std::setprecision(4) << kd[nm];
  out << '\n';
  atomic_write(path, out.str());
}

double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Wall clock (Unix epoch) — comparable with the bridge's Python time.time() stamp in sensor_txt.
double wall_now_s() {
  return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

}  // namespace

int main(int argc, char** argv) {
  // 작은 정책망은 단일 스레드가 더 빠르고, mujoco_x2/bridge 와의 코어 경합(스레드 오버서브)을
  // 막아 tick 벽시계 시간이 튀는 걸 방지. (env 로 이미 지정돼 있으면 그 값을 존중: overwrite=0)
  ::setenv("OMP_NUM_THREADS", "1", 0);
  ::setenv("MKL_NUM_THREADS", "1", 0);
  ::setenv("OPENBLAS_NUM_THREADS", "1", 0);
  try {
    const CliArgs args = parse_args(argc, argv);
    const std::filesystem::path repo_root =
        std::filesystem::weakly_canonical(std::filesystem::path(ROM_REPO_ROOT));

    const auto deploy_cfg = rom::deploy::load_deploy_config(args.config);
    if (!std::filesystem::exists(deploy_cfg.checkpoint_path)) {
      throw std::runtime_error("checkpoint_path not found: " + deploy_cfg.checkpoint_path.string());
    }
    // ARM(정책) ABI 를 policy/fom 에서 로드 — kp/kd, tracking 정책, ROM 정책 모두 이 기준.
    const std::filesystem::path abi_path =
        repo_root / "Yujin" / "agibot_control_functions" / "policy" / "fom" / "tracker_abi.yaml";
    const std::filesystem::path tracking_policy_path = abi_path.parent_path() / "tracking_policy.pt";
    if (!std::filesystem::exists(abi_path) || !std::filesystem::exists(tracking_policy_path)) {
      throw std::runtime_error("Missing " + abi_path.string() + " or " + tracking_policy_path.string());
    }
    auto abi = rom::tracker::TrackerAbi::load_yaml(abi_path);
    // abi 의 rom_policy_path/rom_config_path 는 policy/ 기준 상대경로 -> 절대경로로 변환
    // (C++ 컨트롤러는 이 경로를 cwd 기준 그대로 쓰므로 절대경로여야 함).
    const std::filesystem::path policy_dir = abi_path.parent_path().parent_path();  // .../policy
    auto make_abs = [&](std::optional<std::string>& p) {
      if (p && std::filesystem::path(*p).is_relative())
        *p = (policy_dir / *p).string();
    };
    make_abs(abi.rom_policy_path);
    make_abs(abi.rom_config_path);
    const std::vector<std::string> deploy_names = abi.deploy_joint_names;

    // Standing pose from robots/<robot>/actuator.yaml; STAND_HOLD gains from deploy YAML.
    const auto actuator_yaml = deploy_cfg.repo_root / "robots" / deploy_cfg.robot / "actuator.yaml";
    const std::vector<float> standing_pose =
        rom::deploy::load_standing_pose(actuator_yaml, deploy_names);
    // Stand command (built inline to avoid pulling in AsyncControlLoop -> MuJoCo plant):
    // standing_pose / stand_kp / stand_kd are all aligned to deploy_names order.
    if (deploy_cfg.stand_kp.size() != deploy_names.size() ||
        deploy_cfg.stand_kd.size() != deploy_names.size()) {
      throw std::runtime_error("stand_kp/stand_kd size mismatch with deploy joints "
                               "(set joint_command.stand_kp/stand_kd in deploy YAML).");
    }
    rom::deploy::JointCommand stand_cmd;
    stand_cmd.joint_names = deploy_names;
    stand_cmd.q = Eigen::Map<const Eigen::VectorXf>(
        standing_pose.data(), static_cast<Eigen::Index>(standing_pose.size()));
    stand_cmd.dq = Eigen::VectorXf::Zero(stand_cmd.q.size());
    stand_cmd.tau = Eigen::VectorXf::Zero(stand_cmd.q.size());
    stand_cmd.kp = Eigen::Map<const Eigen::VectorXf>(
        deploy_cfg.stand_kp.data(), static_cast<Eigen::Index>(deploy_cfg.stand_kp.size()));
    stand_cmd.kd = Eigen::Map<const Eigen::VectorXf>(
        deploy_cfg.stand_kd.data(), static_cast<Eigen::Index>(deploy_cfg.stand_kd.size()));

    // 조이스틱 B 버튼 -> LIMP: 자세는 stand 와 같되 kp=kd=0 이라 모든 관절에 힘 0(무력).
    rom::deploy::JointCommand limp_cmd = stand_cmd;
    limp_cmd.kp.setZero();
    limp_cmd.kd.setZero();

    rom::tracker::StandaloneTrackerConfig tracker_cfg;
    tracker_cfg.tracker_abi_path = abi_path;
    tracker_cfg.tracking_policy_path = tracking_policy_path;
    rom::tracker::StandaloneTrackerController controller(tracker_cfg, abi);
    controller.set_joystick_limits(deploy_cfg.joystick);

    // Init IPC files (deploy_policy starts first; bridge waits on these).
    atomic_write(args.sensor_txt, "");
    atomic_write(args.policy_txt, "");
    std::cout << "[policy] config=" << args.config << "\n";
    std::cout << "[policy] sensor_txt=" << args.sensor_txt << "  policy_txt=" << args.policy_txt << "\n";
    std::cout << "[policy] controller ready. deploy joints=" << deploy_names.size()
              << " step_dt=" << controller.step_dt() << "\n";

    std::optional<rom::Joystick> joystick;
    if (args.joystick) {
      joystick.emplace();
      if (!joystick->is_connected()) {
        std::cout << "[policy] joystick not detected -> STAND_HOLD only.\n";
        joystick.reset();
      } else {
        std::cout << "[policy] joystick connected (A=ARM, X=STAND_HOLD)\n";
      }
    }

    // Wait for first sensor frame, then reset ROM.
    std::cout << "[policy] waiting for first sensor_txt state...\n";
    std::optional<SensorFrame> fr;
    while (!(fr = read_sensor(args.sensor_txt)).has_value()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    rom::deploy::RobotState state = make_robot_state(*fr, deploy_names);
    controller.reset(&state);
    std::cout << "[policy] reset done. running at 50 Hz "
                 "(STAND=stand gains, ARM=policy gains)\n";

    const double period = 1.0 / 50.0;
    const double loop_t0 = now_s();
    double last_sync = now_s();
    bool armed = false, prev_a = false, prev_x = false;
    bool limp = false, prev_b = false;   // B 버튼 -> LIMP(모든 kp=kd=0)
    // status-line body-velocity EMA (base lin vel is 0 in No-State-Estimation; wz from IMU)
    double fwz = 0.0;
    constexpr double kBeta = 0.1;

    double next_t = now_s();
    double last_print = now_s();
    while (true) {
      fr = read_sensor(args.sensor_txt);
      if (!fr.has_value()) {
        write_policy(args.policy_txt, deploy_names, limp ? limp_cmd : stand_cmd);
        next_t += period;
        const double s = next_t - now_s();
        if (s > 0) std::this_thread::sleep_for(std::chrono::duration<double>(s));
        else next_t = now_s();
        continue;
      }
      state = make_robot_state(*fr, deploy_names);

      rom::deploy::UserCommand cmd;
      bool announce = false;   // 버튼(A/X/B)을 누른 순간에만 정보 한 줄 출력
      if (joystick.has_value()) {
        joystick->update();
        if (joystick->b && !prev_b) { limp = true; armed = false; announce = true; }                          // B -> LIMP
        if (joystick->a && !prev_a) { limp = false; armed = true; controller.prepare_for_arm(); announce = true; }  // A -> ARM
        if (joystick->x && !prev_x) { limp = false; armed = false; announce = true; }                         // X -> STAND
        prev_a = joystick->a;
        prev_x = joystick->x;
        prev_b = joystick->b;
        if (armed) {
          cmd = rom::deploy::user_command_from_joystick(*joystick, deploy_cfg.joystick, false);
        }
      }
      if (args.auto_arm_after > 0.0 && !armed && (now_s() - loop_t0) >= args.auto_arm_after) {
        armed = true;
        controller.prepare_for_arm();
        announce = true;
      }

      // Always tick (warms obs history / prev_action), like the Python version.
      const double t0 = now_s();
      const rom::deploy::JointCommand jc = controller.tick(state, cmd);
      const double policy_ms = (now_s() - t0) * 1e3;

      if (limp) {
        write_policy(args.policy_txt, deploy_names, limp_cmd);  // all kp=kd=0 (joints limp)
        if (kStandRomSyncPeriodS > 0.0 && (now_s() - last_sync) >= kStandRomSyncPeriodS) {
          controller.sync_from_robot(state);
          last_sync = now_s();
        }
      } else if (armed) {
        write_policy(args.policy_txt, deploy_names, jc);
      } else {
        write_policy(args.policy_txt, deploy_names, stand_cmd);  // stiff standing hold
        if (kStandRomSyncPeriodS > 0.0 && (now_s() - last_sync) >= kStandRomSyncPeriodS) {
          controller.sync_from_robot(state);
          last_sync = now_s();
        }
      }

      // sensor staleness = now - (bridge's write timestamp). fwz = torso yaw-rate EMA.
      const double stale_ms = (wall_now_s() - fr->t_sensor) * 1e3;
      fwz = (1.0 - kBeta) * fwz + kBeta * state.base_ang_vel_b.z();
      // 버튼을 누른 순간에는 항상, --print 면 1Hz 로도 상태 한 줄 출력.
      if (announce || (args.print_status && now_s() - last_print >= 1.0)) {
        last_print = now_s();
        std::cout << "[" << (limp ? "LIMP " : (armed ? "ARM " : "STAND")) << "] twist cmd vx=" << std::fixed
                  << std::setprecision(2) << cmd.lin_vel_x << " vy=" << cmd.lin_vel_y
                  << " wz=" << cmd.ang_vel_z << "  body wz=" << fwz
                  << "  policy_ms=" << std::setprecision(1) << policy_ms
                  << "  stale=" << stale_ms << "ms" << std::endl;
      }

      next_t += period;
      const double s = next_t - now_s();
      if (s > 0) std::this_thread::sleep_for(std::chrono::duration<double>(s));
      else next_t = now_s();
    }
  } catch (const std::exception& exc) {
    std::cerr << "\nrun_deploy_policy failed: " << exc.what() << '\n';
    return 1;
  }
}
