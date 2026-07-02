// robot_ros_bridge.cpp  —  C++ port of robot_ros_bridge.py (self-contained, 500 Hz)
// =====================================================================================
// SECTION ORDER MIRRORS robot_ros_bridge.py:
//   CONFIG: paths -> file IO -> CONFIG: constants -> joint model -> topics -> ROS nodes -> main
// (C++ needs names before use, so a few "data types" / "forward declarations" appear early;
//  the real definitions stay in the same section order as the .py.)
//
// A user normally edits ONLY record_states() and policy_joint_command() (in the file IO
// section). get_robot_states() and publish() (ROS nodes) are fixed infrastructure.
//
// Main loop (same as the .py):
//   auto [imus, head, waist, arm, leg] = client->get_robot_states();  // (1) read state
//   record_states(imus, head, waist, arm, leg);                       // (2) write sensor_txt
//   auto cmds = policy_joint_command();                               // (3) read policy_txt
//   for (auto& [group, cmd] : cmds) commander->publish(group, cmd);   // (4) publish commands
//
// Text-file formats (space-separated, single line, atomic update = always latest):
//   sensor_txt : t + 31*(pos vel) + base quat(x y z w) + base angvel(x y z)   (= 70 values)
//   policy_txt : t + 31 pos + 31 kp + 31 kd                                   (= 94 values)
// Joint order = head(2) + waist(3) + arm(14) + leg(12) = 31.
//
// Flags:  --torso (base IMU = torso/imu_1 instead of chest/pelvis imu_0)   --print (1 Hz rate)
// Build:  source ~/ros_aimdk_env.sh && cmake -B build -S . && cmake --build build -j4

#include <unistd.h>   // getpid

#include <array>
#include <chrono>
#include <cstdlib>    // getenv
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/executors/single_threaded_executor.hpp>
#include <aimdk_msgs/msg/joint_state_array.hpp>
#include <aimdk_msgs/msg/joint_command_array.hpp>
#include <sensor_msgs/msg/imu.hpp>


// ============================= CONFIG: paths =============================
// sensor_txt / policy_txt are shared with run_deploy_policy; both MUST use the same paths.
static std::string io_dir() {
  const char* home = std::getenv("HOME");
  return std::string(home ? home : ".") + "/ROM-RL/Yujin/agibot_control_functions";
}
static const std::string IO_DIR = io_dir();
static const std::string SENSOR_TXT = IO_DIR + "/sensor_txt";
static const std::string POLICY_TXT = IO_DIR + "/policy_txt";
// Base IMU for sensor_txt: "chest" = imu_0 (pelvis, default), "torso" = imu_1 (--torso).
static std::string BASE_IMU = "chest";

// data types + forward declarations needed by the file-IO functions below
// (real definitions are further down, in the same order as the .py).
struct ImuReading {
  std::string source;
  std::array<double, 4> quat{{0, 0, 0, 1}};   // x, y, z, w
  std::array<double, 3> ang_vel{{0, 0, 0}};   // x, y, z
  std::array<double, 3> lin_acc{{0, 0, 0}};   // x, y, z
  std::string frame_id;
  double stamp = 0.0;
};
struct JointReading { std::string name; double position = 0, velocity = 0, effort = 0; };
using ImuMap = std::map<std::string, ImuReading>;
using JointList = std::vector<JointReading>;
using AreaCommand = std::pair<std::string, aimdk_msgs::msg::JointCommandArray>;
static const int NJ = 31;
extern const std::vector<std::string> AREAS;                                 // joint model section
extern const std::map<std::string, std::vector<std::string>> JOINT_NAMES;    // joint model section
extern const std::vector<std::string> JOINT_ORDER;                           // joint model section


// =============================== file IO ===============================
static void atomic_write(const std::string& path, const std::string& text) {
  const std::string tmp = path + ".tmp." + std::to_string(::getpid());   // PID-unique temp
  { std::ofstream f(tmp, std::ios::trunc); f << text; }
  std::rename(tmp.c_str(), path.c_str());
}
static double wall_s() {
  return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

// persisted command: positions update every tick; kp/kd update ONLY when the policy sends
// them (otherwise previous gains are kept). Initial stage (no gains yet): kp/kd = 0.
static std::map<std::string, double> g_last_pos, g_last_kp, g_last_kd;
static bool g_have_pos = false, g_have_kp = false, g_have_kd = false;

// (2) [EDIT ME] Write the current robot state to sensor_txt as one line.
void record_states(const ImuMap& imus, const JointList& head, const JointList& waist,
                   const JointList& arm, const JointList& leg) {
  std::map<std::string, const JointReading*> by_name;
  for (const JointList* lst : {&head, &waist, &arm, &leg})
    for (const auto& jr : *lst) by_name[jr.name] = &jr;

  std::ostringstream out;
  out << std::fixed << std::setprecision(6) << wall_s();
  for (const auto& nm : JOINT_ORDER) {
    auto it = by_name.find(nm);
    if (it == by_name.end()) out << " 0.0 0.0";
    else out << ' ' << it->second->position << ' ' << it->second->velocity;
  }
  // Base orientation / angular velocity from the selected IMU (BASE_IMU: chest=pelvis / torso).
  auto it = imus.find(BASE_IMU);
  if (it != imus.end()) {
    const auto& q = it->second.quat; const auto& w = it->second.ang_vel;
    out << ' ' << q[0] << ' ' << q[1] << ' ' << q[2] << ' ' << q[3];   // x y z w
    out << ' ' << w[0] << ' ' << w[1] << ' ' << w[2];                  // x y z
  } else {
    out << " 0 0 0 1 0 0 0";
  }
  out << '\n';
  atomic_write(SENSOR_TXT, out.str());
}

// Parse the newest policy_txt line -> (pos, kp, kd). has_gains=false for positions-only frames.
struct PolicyFrame { std::map<std::string, double> pos, kp, kd; bool has_gains = false; };
static std::optional<PolicyFrame> read_latest_policy() {
  std::ifstream f(POLICY_TXT);
  if (!f) return std::nullopt;
  std::string line, last;
  while (std::getline(f, line)) if (!line.empty()) last = line;
  if (last.empty()) return std::nullopt;
  std::istringstream ss(last);
  std::vector<double> v; double x;
  while (ss >> x) v.push_back(x);
  const int sz = static_cast<int>(v.size());
  PolicyFrame fr;
  if (sz == 1 + 3 * NJ) {              // t + 31 pos + 31 kp + 31 kd
    for (int i = 0; i < NJ; ++i) {
      fr.pos[JOINT_ORDER[i]] = v[1 + i];
      fr.kp[JOINT_ORDER[i]] = v[1 + NJ + i];
      fr.kd[JOINT_ORDER[i]] = v[1 + 2 * NJ + i];
    }
    fr.has_gains = true;
    return fr;
  }
  if (sz == NJ + 1 || sz == NJ) {      // positions only
    const int off = (sz == NJ + 1) ? 1 : 0;
    for (int i = 0; i < NJ; ++i) fr.pos[JOINT_ORDER[i]] = v[off + i];
    return fr;
  }
  return std::nullopt;
}

// (3) [EDIT ME] Read the newest policy_txt frame and build per-area commands.
std::vector<AreaCommand> policy_joint_command() {
  auto fr = read_latest_policy();
  if (fr) {
    g_last_pos = fr->pos; g_have_pos = true;         // positions: always take the newest
    if (fr->has_gains) {                             // gains: keep previous if this frame has none
      g_last_kp = fr->kp; g_last_kd = fr->kd; g_have_kp = g_have_kd = true;
    }
  }
  std::vector<AreaCommand> out;
  if (!g_have_pos) return out;   // no positions yet -> publish nothing

  for (const auto& area : AREAS) {
    aimdk_msgs::msg::JointCommandArray arr;
    for (const auto& nm : JOINT_NAMES.at(area)) {
      aimdk_msgs::msg::JointCommand jc;
      jc.name = nm;
      jc.position = g_last_pos[nm];   // updated every tick
      jc.velocity = 0.0;
      jc.effort = 0.0;
      jc.stiffness = g_have_kp ? g_last_kp[nm] : 0.0;   // 0 at the initial stage
      jc.damping = g_have_kd ? g_last_kd[nm] : 0.0;
      arr.joints.push_back(jc);
    }
    out.emplace_back(area, arr);
  }
  return out;
}


// =========================== CONFIG: constants ===========================
static constexpr double CONTROL_RATE_HZ = 500.0;   // bridge loop rate (Hz)


// ============================== joint model ==============================
// Joint names/order per area (head + waist + arm + leg = 31). No gains here; gains come
// from the policy (policy_txt kp/kd).
const std::vector<std::string> AREAS = {"head", "waist", "arm", "leg"};
const std::map<std::string, std::vector<std::string>> JOINT_NAMES = {
    {"head", {"head_yaw_joint", "head_pitch_joint"}},
    {"waist", {"waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"}},
    {"arm", {"left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
             "left_elbow_joint", "left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint",
             "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
             "right_elbow_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint"}},
    {"leg", {"left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
             "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
             "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
             "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"}},
};
static std::vector<std::string> make_joint_order() {
  std::vector<std::string> o;
  for (const auto& a : AREAS)
    for (const auto& n : JOINT_NAMES.at(a)) o.push_back(n);
  return o;
}
const std::vector<std::string> JOINT_ORDER = make_joint_order();   // head+waist+arm+leg = 31


// ================================ topics ================================
static const std::map<std::string, std::string> IMU_TOPICS = {
    {"chest", "/aima/hal/imu/chest/state"},   // imu_0 = pelvis
    {"torso", "/aima/hal/imu/torso/state"},   // imu_1 = torso
};


// ======================= ROS nodes (no need to change) =======================
class RobotStateClient : public rclcpp::Node {
 public:
  RobotStateClient() : Node("robot_state_client") {
    auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();   // newest sample only
    for (const auto& [src, topic] : IMU_TOPICS)
      imu_sub_[src] = create_subscription<sensor_msgs::msg::Imu>(
          topic, qos, [this, s = src](sensor_msgs::msg::Imu::SharedPtr m) {
            std::lock_guard<std::mutex> g(lock_); imu_msg_[s] = *m;
          });
    for (const auto& a : AREAS)
      state_sub_[a] = create_subscription<aimdk_msgs::msg::JointStateArray>(
          "/aima/hal/joint/" + a + "/state", qos,
          [this, a](aimdk_msgs::msg::JointStateArray::SharedPtr m) {
            std::lock_guard<std::mutex> g(lock_); joint_msg_[a] = *m;
          });
  }
  bool ready() {
    std::lock_guard<std::mutex> g(lock_);
    return imu_msg_.size() == IMU_TOPICS.size() && joint_msg_.size() == AREAS.size();
  }
  // Return (imus, head, waist, arm, leg) from the newest cached messages.
  std::tuple<ImuMap, JointList, JointList, JointList, JointList> get_robot_states() {
    std::lock_guard<std::mutex> g(lock_);
    ImuMap imus;
    for (const auto& [src, msg] : imu_msg_) imus[src] = to_imu(src, msg);
    return {imus, to_joints("head"), to_joints("waist"), to_joints("arm"), to_joints("leg")};
  }

 private:
  static ImuReading to_imu(const std::string& src, const sensor_msgs::msg::Imu& m) {
    ImuReading r;
    r.source = src;
    r.quat = {m.orientation.x, m.orientation.y, m.orientation.z, m.orientation.w};
    r.ang_vel = {m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z};
    r.lin_acc = {m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z};
    r.frame_id = m.header.frame_id;
    r.stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9;
    return r;
  }
  JointList to_joints(const std::string& area) {
    JointList out;
    auto it = joint_msg_.find(area);
    if (it == joint_msg_.end()) return out;
    const auto& names = JOINT_NAMES.at(area);
    for (size_t i = 0; i < it->second.joints.size(); ++i) {
      const auto& js = it->second.joints[i];
      std::string nm = i < names.size() ? names[i] : ("joint_" + std::to_string(i));
      out.push_back(JointReading{nm, js.position, js.velocity, js.effort});
    }
    return out;
  }
  std::mutex lock_;
  std::map<std::string, sensor_msgs::msg::Imu> imu_msg_;
  std::map<std::string, aimdk_msgs::msg::JointStateArray> joint_msg_;
  std::map<std::string, rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr> imu_sub_;
  std::map<std::string, rclcpp::Subscription<aimdk_msgs::msg::JointStateArray>::SharedPtr> state_sub_;
};

class WholeBodyCommander : public rclcpp::Node {
 public:
  WholeBodyCommander() : Node("whole_body_commander") {
    auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();
    for (const auto& a : AREAS)
      pub_[a] = create_publisher<aimdk_msgs::msg::JointCommandArray>(
          "/aima/hal/joint/" + a + "/command", qos);
  }
  void publish(const std::string& area, const aimdk_msgs::msg::JointCommandArray& cmd) {
    pub_.at(area)->publish(cmd);
  }
 private:
  std::map<std::string, rclcpp::Publisher<aimdk_msgs::msg::JointCommandArray>::SharedPtr> pub_;
};


// =============================== main loop ===============================
int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  bool print_status = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--print") print_status = true;
    if (a == "--torso") BASE_IMU = "torso";
  }

  auto client = std::make_shared<RobotStateClient>();       // reads state topics
  auto commander = std::make_shared<WholeBodyCommander>();  // publishes commands

  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(client);
  exec.add_node(commander);
  std::thread spin_thread([&]() { exec.spin(); });

  RCLCPP_INFO(client->get_logger(), "sensor_txt=%s", SENSOR_TXT.c_str());
  RCLCPP_INFO(client->get_logger(), "policy_txt=%s", POLICY_TXT.c_str());
  RCLCPP_INFO(client->get_logger(), "base IMU = %s. waiting for robot state topics...", BASE_IMU.c_str());
  while (rclcpp::ok() && !client->ready()) std::this_thread::sleep_for(std::chrono::milliseconds(100));
  if (!rclcpp::ok()) { rclcpp::shutdown(); if (spin_thread.joinable()) spin_thread.join(); return 0; }
  RCLCPP_INFO(client->get_logger(), "state ready. running at %.0f Hz", CONTROL_RATE_HZ);

  const double period = 1.0 / CONTROL_RATE_HZ;
  double next_t = wall_s(), rate_t0 = wall_s();
  long n = 0;
  while (rclcpp::ok()) {
    auto [imus, head, waist, arm, leg] = client->get_robot_states();   // (1) read state
    record_states(imus, head, waist, arm, leg);                        // (2) write sensor_txt
    auto cmds = policy_joint_command();                                // (3) read policy_txt
    for (auto& [group, cmd] : cmds)                                    // (4) publish commands
      commander->publish(group, cmd);

    if (print_status && ++n % 500 == 0) {
      double w = wall_s();
      RCLCPP_INFO(client->get_logger(), "loop rate = %.0f Hz", n / (w - rate_t0));
      rate_t0 = w; n = 0;
    }
    next_t += period;
    double sleep = next_t - wall_s();
    if (sleep > 0) std::this_thread::sleep_for(std::chrono::duration<double>(sleep));
    else next_t = wall_s();
  }
  exec.cancel();
  rclcpp::shutdown();
  if (spin_thread.joinable()) spin_thread.join();
  return 0;
}
