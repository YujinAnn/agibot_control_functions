// ============================================================================
// Robot state reader + whole-body command sequencer  (C++ / rclcpp port)
// Multi-IMU.  Faithful port of robot_states.py.
// ----------------------------------------------------------------------------
// main() loop (every 2 ms):
//   auto [imus, head, waist, arm, leg] = client->get_robot_states();   // read
//   record_states_csv(imus, head, waist, arm, leg);                    // csv
//   print_state(imus, head, waist, arm, leg);                          // print/sec
//   auto [joint_group, cmd] = seq.policy_joint_command(head, waist, arm, leg);// policy
//   commander->publish(joint_group, cmd);                              // publish
//
// IMUs are a std::map keyed by source: imus["chest"], imus["torso"].
// Requires C++17. Build deps: rclcpp, sensor_msgs, aimdk_msgs, ruckig.
// ============================================================================

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <aimdk_msgs/msg/joint_state_array.hpp>
#include <aimdk_msgs/msg/joint_command_array.hpp>
#include <aimdk_msgs/msg/joint_command.hpp>
#include <ruckig/ruckig.hpp>

#include <array>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

using sensor_msgs::msg::Imu;
using aimdk_msgs::msg::JointStateArray;
using aimdk_msgs::msg::JointCommandArray;
using aimdk_msgs::msg::JointCommand;

// =============================== constants & configuration ===============================

// ---- (2) joint areas & types ----
enum class JointArea { HEAD, WAIST, ARM, LEG };   // order = head/waist/arm/leg

static const char *area_name(JointArea a) {
  switch (a) {
    case JointArea::HEAD:  return "HEAD";
    case JointArea::WAIST: return "WAIST";
    case JointArea::ARM:   return "ARM";
    case JointArea::LEG:   return "LEG";
  }
  return "?";
}

struct JointInfo {
  std::string name;
  double lower_limit, upper_limit, kp, kd;
};

struct JointReading {
  std::string name;
  double position{};      // rad
  double velocity{};      // rad/s
  double effort{};        // N*m
  int coil_temp{};
  int motor_temp{};
  int motor_vol{};
  std::string msg_name;   // raw JointState.name (often empty)
};

struct ImuReading {
  std::string source;               // "chest" / "torso" / ...
  std::array<double, 4> quat{};     // orientation (x, y, z, w)
  std::array<double, 3> ang_vel{};  // angular_velocity (x, y, z) rad/s
  std::array<double, 3> lin_acc{};  // linear_acceleration (x, y, z) m/s^2
  std::string frame_id;
  double stamp{};
};

using JointVec = std::vector<JointReading>;
using ImuMap = std::map<std::string, ImuReading>;
using RobotState = std::tuple<ImuMap, JointVec, JointVec, JointVec, JointVec>;

// ---- (3) topics ----
static const std::vector<std::pair<std::string, std::string>> IMU_TOPICS = {
    {"chest", "/aima/hal/imu/chest/state"},
    {"torso", "/aima/hal/imu/torso/state"},
};

// state topics, keyed by area (names/order come from robot_model -> JOINT_NAMES)
static const std::vector<std::pair<JointArea, std::string>> JOINT_TOPICS = {
    {JointArea::HEAD,  "/aima/hal/joint/head/state"},
    {JointArea::WAIST, "/aima/hal/joint/waist/state"},
    {JointArea::ARM,   "/aima/hal/joint/arm/state"},
    {JointArea::LEG,   "/aima/hal/joint/leg/state"},
};

// command topics per area, ordered head/waist/arm/leg.  *** verify head/waist:
//   ros2 topic list | grep -E 'head|waist'
// (HEAD is included for completeness but SEQUENCE never targets it.)
static const std::vector<std::pair<JointArea, std::string>> CMD_TOPICS = {
    {JointArea::HEAD,  "/aima/hal/joint/head/command"},
    {JointArea::WAIST, "/aima/hal/joint/waist/command"},
    {JointArea::ARM,   "/aima/hal/joint/arm/command"},
    {JointArea::LEG,   "/aima/hal/joint/leg/command"},
};

// which element of (head, waist, arm, leg) corresponds to each area
static int reading_index(JointArea a) {
  switch (a) {
    case JointArea::HEAD:  return 0;
    case JointArea::WAIST: return 1;
    case JointArea::ARM:   return 2;
    case JointArea::LEG:   return 3;
  }
  return 0;
}

// ---- (4) robot_model (joint order MUST match the state/command arrays) ----
static const std::map<JointArea, std::vector<JointInfo>> robot_model = {
    {JointArea::HEAD, {
        {"head_yaw_joint", -0.366, 0.366, 20.0, 2.0},
        {"head_pitch_joint", -0.3838, 0.3838, 20.0, 2.0},
    }},
    {JointArea::WAIST, {
        {"waist_yaw_joint", -3.43, 2.382, 20.0, 4.0},
        {"waist_pitch_joint", -0.314, 0.314, 20.0, 4.0},
        {"waist_roll_joint", -0.488, 0.488, 20.0, 4.0},
    }},
    {JointArea::ARM, {
        {"left_shoulder_pitch_joint", -3.08, 2.04, 20.0, 2.0},
        {"left_shoulder_roll_joint", -0.061, 2.993, 20.0, 2.0},
        {"left_shoulder_yaw_joint", -2.556, 2.556, 20.0, 2.0},
        {"left_elbow_joint", -2.3556, 0.0, 20.0, 2.0},
        {"left_wrist_yaw_joint", -2.556, 2.556, 20.0, 2.0},
        {"left_wrist_pitch_joint", -0.558, 0.558, 20.0, 2.0},
        {"left_wrist_roll_joint", -1.571, 0.724, 20.0, 2.0},
        {"right_shoulder_pitch_joint", -3.08, 2.04, 20.0, 2.0},
        {"right_shoulder_roll_joint", -2.993, 0.061, 20.0, 2.0},
        {"right_shoulder_yaw_joint", -2.556, 2.556, 20.0, 2.0},
        {"right_elbow_joint", -2.3556, 0.0000, 20.0, 2.0},
        {"right_wrist_yaw_joint", -2.556, 2.556, 20.0, 2.0},
        {"right_wrist_pitch_joint", -0.558, 0.558, 20.0, 2.0},
        {"right_wrist_roll_joint", -0.724, 1.571, 20.0, 2.0},
    }},
    {JointArea::LEG, {
        {"left_hip_pitch_joint", -2.704, 2.556, 180.0, 5.0},
        {"left_hip_roll_joint", -0.235, 2.906, 100.0, 5.0},
        {"left_hip_yaw_joint", -1.684, 3.430, 100.0, 5.0},
        {"left_knee_joint", 0.0000, 2.4073, 100.0, 5.0},
        {"left_ankle_pitch_joint", -0.803, 0.453, 100.0, 5.0},
        {"left_ankle_roll_joint", -0.2625, 0.2625, 100.0, 5.0},
        {"right_hip_pitch_joint", -2.704, 2.556, 180.0, 5.0},
        {"right_hip_roll_joint", -2.906, 0.235, 100.0, 5.0},
        {"right_hip_yaw_joint", -3.430, 1.684, 100.0, 5.0},
        {"right_knee_joint", 0.0000, 2.4073, 100.0, 5.0},
        {"right_ankle_pitch_joint", -0.803, 0.453, 100.0, 5.0},
        {"right_ankle_roll_joint", -0.2625, 0.2625, 100.0, 5.0},
    }},
};

// joint names/order per area, derived from robot_model (single source of truth)
static std::vector<std::string> names_of(JointArea a) {
  std::vector<std::string> out;
  for (const auto &ji : robot_model.at(a)) out.push_back(ji.name);
  return out;
}

// ---- (5) per-part targets / sequence / area map ----
// *** PLACEHOLDERS -- set safe values for YOUR robot ***
static const std::map<std::string, std::vector<std::pair<std::string, double>>> PART_TARGETS = {
    {"right_arm", {{"right_shoulder_pitch_joint", -1.00139}, {"right_shoulder_roll_joint", -1.01251},
                   {"right_shoulder_yaw_joint", 0.34543}, {"right_elbow_joint", -1.19850},
                   {"right_wrist_yaw_joint", 0.58990}, {"right_wrist_pitch_joint", 0.00973},
                   {"right_wrist_roll_joint", -0.00134}}},
    {"left_arm", {{"left_shoulder_pitch_joint", -1.17722}, {"left_shoulder_roll_joint", 0.84224},
                  {"left_shoulder_yaw_joint", -0.46623}, {"left_elbow_joint", -1.27328},
                  {"left_wrist_yaw_joint", -0.04170}, {"left_wrist_pitch_joint", 0.10433},
                  {"left_wrist_roll_joint", -0.00172}}},
    {"right_leg", {{"right_hip_pitch_joint", -1.50280}, {"right_hip_roll_joint", -0.07833},
                   {"right_hip_yaw_joint", 0.02752}, {"right_knee_joint", -0.05781},
                   {"right_ankle_pitch_joint", 0.50592}, {"right_ankle_roll_joint", 0.08945}}},
    {"left_leg", {{"left_hip_pitch_joint", -1.48382}, {"left_hip_roll_joint", 0.20430},
                  {"left_hip_yaw_joint", 0.12396}, {"left_knee_joint", -0.06702},
                  {"left_ankle_pitch_joint", 0.50669}, {"left_ankle_roll_joint", -0.09060}}},
    {"waist", {{"waist_yaw_joint", 0.81597}, {"waist_pitch_joint", 0.34910},
               {"waist_roll_joint", 0.07626}}},
};

// run order + which AREA each part lives in  (arm -> leg -> waist; no head)
static const std::vector<std::string> SEQUENCE = {
    "right_arm", "left_arm", "right_leg", "left_leg", "waist"};
static const std::map<std::string, JointArea> PART_AREA = {
    {"waist", JointArea::WAIST},
    {"right_arm", JointArea::ARM}, {"left_arm", JointArea::ARM},
    {"right_leg", JointArea::LEG}, {"left_leg", JointArea::LEG}};

// ---- (6) conservative motion limits ----
static constexpr double CONTROL_PERIOD = 0.002;   // 2 ms (500 Hz)
static constexpr double HOLD_SECONDS = 5.0;       // hold at target / gap between parts
static constexpr double MAX_VELOCITY = 0.4;       // rad/s
static constexpr double MAX_ACCELERATION = 0.4;   // rad/s^2
static constexpr double MAX_JERK = 4.0;           // rad/s^3

// ---- (7) CSV logging ----
static const char *CSV_PATH = "robot_state_full.csv";

static double now_sec() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// ----------------------------- the client node -----------------------------
class RobotStateClient : public rclcpp::Node {
 public:
  RobotStateClient() : rclcpp::Node("robot_state_client") {
    rclcpp::QoS qos(rclcpp::KeepLast(1));   // depth=1: newest only, no backlog
    qos.best_effort().durability_volatile();

    for (const auto &kv : IMU_TOPICS) {
      const std::string src = kv.first;
      imu_msg_[src] = nullptr;
      imu_subs_.push_back(create_subscription<Imu>(
          kv.second, qos, [this, src](Imu::SharedPtr m) { imu_cb(src, m); }));
    }
    for (const auto &kv : JOINT_TOPICS) {
      const JointArea area = kv.first;
      joint_msg_[area] = nullptr;
      joint_subs_.push_back(create_subscription<JointStateArray>(
          kv.second, qos, [this, area](JointStateArray::SharedPtr m) { joint_cb(area, m); }));
    }
  }

  bool wait_ready(double timeout_sec = 10.0) {
    const double t0 = now_sec();
    while (rclcpp::ok()) {
      {
        std::lock_guard<std::mutex> lk(lock_);
        bool ready = true;
        for (const auto &kv : IMU_TOPICS) if (!imu_msg_[kv.first]) ready = false;
        for (const auto &kv : JOINT_TOPICS) if (!joint_msg_[kv.first]) ready = false;
        if (ready) return true;
      }
      if (now_sec() - t0 > timeout_sec) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    RCLCPP_ERROR(get_logger(), "Timed out waiting for state topics.");
    return false;
  }

  // Return {imus, head, waist, arm, leg} from newest cached messages.
  //
  // HOW TO READ (collaborator):
  //   auto [imus, head, waist, arm, leg] = client->get_robot_states();
  //   double wx = imus["torso"].ang_vel[0];          // base gyro
  //   // by name:
  //   std::map<std::string,JointReading> jmap;
  //   for (const auto& v : {head,waist,arm,leg}) for (auto& jr : v) jmap[jr.name]=jr;
  //   double q_knee = jmap["left_knee_joint"].position;
  //   // ordered (MUST match training order):
  //   std::vector<JointReading> order;
  //   for (const auto& v : {head,waist,arm,leg}) order.insert(order.end(),v.begin(),v.end());
  RobotState get_robot_states() {
    std::map<std::string, Imu::SharedPtr> imu_msgs;
    std::map<JointArea, JointStateArray::SharedPtr> joint_msgs;
    {
      std::lock_guard<std::mutex> lk(lock_);
      imu_msgs = imu_msg_;
      joint_msgs = joint_msg_;
    }
    for (const auto &kv : IMU_TOPICS)
      if (!imu_msgs[kv.first]) throw std::runtime_error("State not ready (call wait_ready first).");
    for (const auto &kv : JOINT_TOPICS)
      if (!joint_msgs[kv.first]) throw std::runtime_error("State not ready (call wait_ready first).");

    ImuMap imus;
    for (const auto &kv : IMU_TOPICS) imus[kv.first] = imu_reading(kv.first, *imu_msgs[kv.first]);

    JointVec head  = name_list(joint_msgs[JointArea::HEAD]->joints,  names_of(JointArea::HEAD));
    JointVec waist = name_list(joint_msgs[JointArea::WAIST]->joints, names_of(JointArea::WAIST));
    JointVec arm   = name_list(joint_msgs[JointArea::ARM]->joints,   names_of(JointArea::ARM));
    JointVec leg   = name_list(joint_msgs[JointArea::LEG]->joints,   names_of(JointArea::LEG));
    return {imus, head, waist, arm, leg};
  }

 private:
  template <typename JointArray>
  static JointVec name_list(const JointArray &joints, const std::vector<std::string> &names) {
    JointVec out;
    out.reserve(joints.size());
    for (size_t i = 0; i < joints.size(); ++i) {
      const auto &js = joints[i];
      std::string nm = (i < names.size()) ? names[i] : ("joint_" + std::to_string(i));
      out.push_back(JointReading{nm, js.position, js.velocity, js.effort,
                                 js.coil_temp, js.motor_temp, js.motor_vol, js.name});
    }
    return out;
  }

  static ImuReading imu_reading(const std::string &source, const Imu &m) {
    ImuReading ir;
    ir.source = source;
    ir.quat = {m.orientation.x, m.orientation.y, m.orientation.z, m.orientation.w};
    ir.ang_vel = {m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z};
    ir.lin_acc = {m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z};
    ir.frame_id = m.header.frame_id;
    ir.stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9;
    return ir;
  }

  void imu_cb(const std::string &source, Imu::SharedPtr m) {
    std::lock_guard<std::mutex> lk(lock_);
    imu_msg_[source] = m;
  }
  void joint_cb(JointArea area, JointStateArray::SharedPtr m) {
    std::lock_guard<std::mutex> lk(lock_);
    joint_msg_[area] = m;
  }

  std::mutex lock_;
  std::map<std::string, Imu::SharedPtr> imu_msg_;
  std::map<JointArea, JointStateArray::SharedPtr> joint_msg_;
  std::vector<rclcpp::SubscriptionBase::SharedPtr> imu_subs_;
  std::vector<rclcpp::SubscriptionBase::SharedPtr> joint_subs_;
};

// ----------------------------- CSV recording -----------------------------
class Recorder {
 public:
  void record(const ImuMap &imus, const JointVec &head, const JointVec &waist,
              const JointVec &arm, const JointVec &leg) {
    std::vector<std::pair<std::string, const JointVec *>> groups = {
        {"head", &head}, {"waist", &waist}, {"arm", &arm}, {"leg", &leg}};
    static const char *IMU_SUF[10] = {"quat_x", "quat_y", "quat_z", "quat_w",
                                      "ang_vel_x", "ang_vel_y", "ang_vel_z",
                                      "lin_acc_x", "lin_acc_y", "lin_acc_z"};
    if (!file_.is_open()) {
      file_.open(CSV_PATH);
      t0_ = now_sec();
      file_ << "t_sec";
      for (const auto &kv : IMU_TOPICS)
        for (const char *s : IMU_SUF) file_ << "," << kv.first << "." << s;
      for (const auto &gp : groups)
        for (const auto &jr : *gp.second)
          file_ << "," << jr.name << ".position"
                << "," << jr.name << ".velocity"
                << "," << jr.name << ".effort";
      file_ << "\n";
    }
    double t = now_sec() - t0_;
    file_ << std::fixed << std::setprecision(4) << t << std::setprecision(6);
    for (const auto &kv : IMU_TOPICS) {
      const ImuReading &im = imus.at(kv.first);
      for (double v : im.quat) file_ << "," << v;
      for (double v : im.ang_vel) file_ << "," << v;
      for (double v : im.lin_acc) file_ << "," << v;
    }
    for (const auto &gp : groups)
      for (const auto &jr : *gp.second)
        file_ << "," << jr.position << "," << jr.velocity << "," << jr.effort;
    file_ << "\n";
    ++rows_;
  }

  void close() {
    if (file_.is_open()) {
      file_.flush();
      file_.close();
      std::printf("[recorder] saved %s (%d rows)\n", CSV_PATH, rows_);
    }
  }

 private:
  std::ofstream file_;
  double t0_{0.0};
  int rows_{0};
};

static Recorder &recorder() { static Recorder r; return r; }
void record_states_csv(const ImuMap &imus, const JointVec &head, const JointVec &waist,
                       const JointVec &arm, const JointVec &leg) {
  recorder().record(imus, head, waist, arm, leg);
}
void close_csv() { recorder().close(); }

// ----------------------------- console printing -----------------------------
void print_state(const ImuMap &imus, const JointVec &head, const JointVec &waist,
                 const JointVec &arm, const JointVec &leg) {
  static double last = 0.0;
  double now = now_sec();
  if (now - last < 1.0) return;   // at most once per second
  last = now;

  std::printf("\n========== robot state ==========\n");
  for (const auto &kv : IMU_TOPICS) {
    auto it = imus.find(kv.first);
    if (it == imus.end()) continue;
    const ImuReading &im = it->second;
    std::printf("IMU[%s] frame_id='%s' quat=(%.3f,%.3f,%.3f,%.3f) "
                "ang_vel=(%.4f,%.4f,%.4f) lin_acc=(%.3f,%.3f,%.3f)\n",
                kv.first.c_str(), im.frame_id.c_str(),
                im.quat[0], im.quat[1], im.quat[2], im.quat[3],
                im.ang_vel[0], im.ang_vel[1], im.ang_vel[2],
                im.lin_acc[0], im.lin_acc[1], im.lin_acc[2]);
  }
  std::vector<std::pair<std::string, const JointVec *>> groups = {
      {"head", &head}, {"waist", &waist}, {"arm", &arm}, {"leg", &leg}};
  for (const auto &gp : groups) {
    const JointVec &v = *gp.second;
    if (v.empty()) { std::printf("[%s] (none)\n", gp.first.c_str()); continue; }
    std::printf("[%s] %zu joints\n", gp.first.c_str(), v.size());
    std::printf("  %3s  %-24s %-24s %-5s %10s %11s\n", "idx", "expected (doc order)",
                "msg.name", "match", "pos(rad)", "vel(rad/s)");
    for (size_t i = 0; i < v.size(); ++i) {
      const JointReading &jr = v[i];
      std::string mn = jr.msg_name.empty() ? "(empty)" : jr.msg_name;
      const char *match = (jr.msg_name == jr.name) ? "O" : "X";
      std::printf("  %3zu  %-24s %-24s %-5s %+10.4f %+11.4f\n",
                  i, jr.name.c_str(), mn.c_str(), match, jr.position, jr.velocity);
    }
  }
}

// ======================================================================
//  COMMAND CONTROL SEQUENCE
//  Drives one body part at a time: home -> target -> hold -> home -> gap.
//  Sequence: right arm, left arm, right leg, left leg, waist.
//  SAFETY: stop MC (aima em stop-app mc on .40), robot FULLY SUSPENDED,
//          TARGETS are PLACEHOLDERS, only the ACTIVE part is commanded.
// ======================================================================
constexpr size_t RDOF = ruckig::DynamicDOFs;

class JointSequencer {
 public:
  JointSequencer()
      : steps_(SEQUENCE), active_(PART_AREA.at(SEQUENCE[0])),
        log_(rclcpp::get_logger("joint_sequencer")) {
    phase_ = steps_[0] + "_to_target";
    std::printf("Sequence: ");
    for (size_t i = 0; i < steps_.size(); ++i)
      std::printf("%s%s", steps_[i].c_str(), i + 1 < steps_.size() ? " -> " : "");
    std::printf("  (each: target, hold 5s, back to home)\n");
    std::printf("TARGETS are PLACEHOLDERS. Robot MUST be fully suspended. MC stopped on .40.\n");
  }

  const std::string &phase() const { return phase_; }

  // Return {area, JointCommandArray} for this 2 ms tick.
  std::pair<JointArea, JointCommandArray>
  policy_joint_command(const JointVec &head, const JointVec &waist,
                       const JointVec &arm, const JointVec &leg) {
    if (!captured_) {
      home_[JointArea::ARM]   = positions_of(arm);
      home_[JointArea::LEG]   = positions_of(leg);
      home_[JointArea::WAIST] = positions_of(waist);
      captured_ = true;
    }

    JointArea area = active_;
    const JointVec &readings = readings_of(area, head, waist, arm, leg);
    std::vector<double> cur_pos = positions_of(readings);
    std::vector<double> cur_vel = velocities_of(readings);
    phase_ = (sub_ == "done") ? "done" : (steps_[idx_] + "_" + sub_);

    if (sub_ == "done")
      return {area, build_cmd(area, home_[area], zeros(home_[area].size()))};

    const std::string &step = steps_[idx_];

    if (sub_ == "to_target" || sub_ == "to_home") {
      if (need_init_) {
        std::vector<double> goal = (sub_ == "to_target")
            ? clamp_target(area, home_[area], PART_TARGETS.at(step))
            : home_[area];
        if (sub_ == "to_target") target_ = goal;
        new_ruckig(area, cur_pos, cur_vel, goal);
        need_init_ = false;
      }
      ruckig::Result res = rk_->update(*rin_, *rout_);
      rout_->pass_to_input(*rin_);
      JointCommandArray cmd = build_cmd(area, rout_->new_position, rout_->new_velocity);
      if (res == ruckig::Result::Finished) {
        sub_ = (sub_ == "to_target") ? "hold" : "gap";
        hold_until_ = now_sec() + HOLD_SECONDS;
      }
      return {area, cmd};
    }

    // hold (at target) or gap (at home)
    const std::vector<double> &hold_pos = (sub_ == "hold") ? target_ : home_[area];
    JointCommandArray cmd = build_cmd(area, hold_pos, zeros(hold_pos.size()));
    if (now_sec() >= hold_until_) {
      if (sub_ == "hold") {
        sub_ = "to_home"; need_init_ = true;
      } else {                              // gap done -> advance
        ++idx_;
        if (idx_ >= static_cast<int>(steps_.size())) {
          sub_ = "done";
        } else {
          active_ = PART_AREA.at(steps_[idx_]);
          sub_ = "to_target"; need_init_ = true;
        }
      }
    }
    return {area, cmd};
  }

 private:
  static std::vector<double> zeros(size_t n) { return std::vector<double>(n, 0.0); }
  static std::vector<double> positions_of(const JointVec &v) {
    std::vector<double> o; o.reserve(v.size());
    for (const auto &jr : v) o.push_back(jr.position);
    return o;
  }
  static std::vector<double> velocities_of(const JointVec &v) {
    std::vector<double> o; o.reserve(v.size());
    for (const auto &jr : v) o.push_back(jr.velocity);
    return o;
  }
  static const JointVec &readings_of(JointArea a, const JointVec &head, const JointVec &waist,
                                     const JointVec &arm, const JointVec &leg) {
    switch (reading_index(a)) {
      case 0: return head;
      case 1: return waist;
      case 2: return arm;
      default: return leg;
    }
  }

  std::vector<double> clamp_target(JointArea area, const std::vector<double> &home_pos,
                                   const std::vector<std::pair<std::string, double>> &part_target) {
    const auto &info = robot_model.at(area);
    std::vector<double> target = home_pos;
    for (const auto &nv : part_target) {
      const std::string &nm = nv.first;
      double val = nv.second;
      int idx = -1;
      for (size_t i = 0; i < info.size(); ++i)
        if (info[i].name == nm) { idx = static_cast<int>(i); break; }
      if (idx < 0) continue;
      double lo = info[idx].lower_limit, hi = info[idx].upper_limit;
      double clamped = std::max(lo, std::min(hi, val));
      if (std::abs(clamped - val) > 1e-9)
        RCLCPP_WARN(log_, "%s: target %.6f -> clamped to %.6f (limit [%g, %g])",
                    nm.c_str(), val, clamped, lo, hi);
      target[idx] = clamped;
    }
    return target;
  }

  void new_ruckig(JointArea area, const std::vector<double> &cur_pos,
                  const std::vector<double> &cur_vel, const std::vector<double> &goal) {
    size_t dofs = robot_model.at(area).size();
    rk_ = std::make_unique<ruckig::Ruckig<RDOF>>(dofs, CONTROL_PERIOD);
    rin_ = std::make_unique<ruckig::InputParameter<RDOF>>(dofs);
    rout_ = std::make_unique<ruckig::OutputParameter<RDOF>>(dofs);
    rin_->max_velocity = std::vector<double>(dofs, MAX_VELOCITY);
    rin_->max_acceleration = std::vector<double>(dofs, MAX_ACCELERATION);
    rin_->max_jerk = std::vector<double>(dofs, MAX_JERK);
    rin_->current_position = cur_pos;
    rin_->current_velocity = cur_vel;
    rin_->current_acceleration = std::vector<double>(dofs, 0.0);
    rin_->target_position = goal;
    rin_->target_velocity = std::vector<double>(dofs, 0.0);
    rin_->target_acceleration = std::vector<double>(dofs, 0.0);
  }

  JointCommandArray build_cmd(JointArea area, const std::vector<double> &positions,
                              const std::vector<double> &velocities) {
    JointCommandArray cmd;
    const auto &info = robot_model.at(area);
    for (size_t i = 0; i < info.size(); ++i) {
      JointCommand jc;
      jc.name = info[i].name;
      jc.position = positions[i];
      jc.velocity = velocities[i];
      jc.effort = 0.0;
      jc.stiffness = info[i].kp;
      jc.damping = info[i].kd;
      cmd.joints.push_back(jc);
    }
    return cmd;
  }

  std::vector<std::string> steps_;
  int idx_ = 0;
  std::string sub_ = "to_target";        // to_target / hold / to_home / gap / done
  JointArea active_;
  std::map<JointArea, std::vector<double>> home_;
  bool captured_ = false;
  bool need_init_ = true;
  double hold_until_ = 0.0;
  std::vector<double> target_;
  std::string phase_;
  rclcpp::Logger log_;
  std::unique_ptr<ruckig::Ruckig<RDOF>> rk_;
  std::unique_ptr<ruckig::InputParameter<RDOF>> rin_;
  std::unique_ptr<ruckig::OutputParameter<RDOF>> rout_;
};

// ----------------------------- command publisher node -----------------------------
class WholeBodyCommander : public rclcpp::Node {
 public:
  WholeBodyCommander() : rclcpp::Node("whole_body_commander") {
    rclcpp::QoS qos(rclcpp::KeepLast(10));
    qos.reliable().durability_volatile();
    for (const auto &kv : CMD_TOPICS)
      pub_[kv.first] = create_publisher<JointCommandArray>(kv.second, qos);
  }
  void publish(JointArea area, const JointCommandArray &cmd) { pub_.at(area)->publish(cmd); }

 private:
  std::map<JointArea, rclcpp::Publisher<JointCommandArray>::SharedPtr> pub_;
};

// ----------------------------- main loop -----------------------------
int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto client = std::make_shared<RobotStateClient>();      // node 1: reads state
  auto commander = std::make_shared<WholeBodyCommander>(); // node 2: publishes commands

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(client);
  executor.add_node(commander);
  std::thread spin_thread([&executor]() { executor.spin(); });

  if (!client->wait_ready(10.0)) {
    rclcpp::shutdown();
    if (spin_thread.joinable()) spin_thread.join();
    return 1;
  }

  JointSequencer seq;

  std::cout << ">>> Ready? Press Enter to start (Ctrl+C to cancel) <<<" << std::flush;
  { std::string line; std::getline(std::cin, line); }

  const auto period = std::chrono::microseconds(2000);   // 2 ms = 500 Hz
  auto next = std::chrono::steady_clock::now();
  try {
    while (rclcpp::ok()) {
      auto [imus, head, waist, arm, leg] = client->get_robot_states();          // (1) read
      record_states_csv(imus, head, waist, arm, leg);                           // (1-1) csv
      print_state(imus, head, waist, arm, leg);                                 // (1-2) print/sec
      auto [joint_group, cmd] = seq.policy_joint_command(head, waist, arm, leg);  // (2) policy
      commander->publish(joint_group, cmd);                                       // (3) publish

      next += period;
      std::this_thread::sleep_until(next);
    }
  } catch (const std::exception &e) {
    std::printf("loop stopped: %s\n", e.what());
  }

  close_csv();
  rclcpp::shutdown();
  if (spin_thread.joinable()) spin_thread.join();
  return 0;
}
