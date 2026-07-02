// run_mujoco_x2.cpp — C++ port of Yujin/agibot_control_functions/mujoco_x2.py
// =====================================================================================
// The "real robot" of the ROS deploy pipeline: a MuJoCo sim of Agibot X2 with all
// 31 joints FREE (no weld / no equality — fully realistic), driven over the ROS HAL.
// Python + the glfw passive viewer runs slow on Wayland (software GL -> RTF ~0.35);
// this native C++ node renders on the GPU and hits real-time (RTF ~1.0).
//
// Identical ROS interface to mujoco_x2.py (so robot_ros_bridge.py + run_deploy_policy
// are unchanged):
//   subscribes  /aima/hal/joint/{head,waist,arm,leg}/command  (aimdk_msgs/JointCommandArray)
//   publishes   /aima/hal/joint/{head,waist,arm,leg}/state    (aimdk_msgs/JointStateArray)
//               /aima/hal/imu/{chest,torso}/state             (sensor_msgs/Imu)
//   PD:  tau = kp*(pos - q) - kd*qd + kd*vel_ff + effort   via position actuators
//   QoS: state = BEST_EFFORT depth1, command = RELIABLE depth10
//
// Physics matches mujoco_x2.py: robot_full + baked parent-child contact excludes
// (robot_full_flat_ground_excl.xml, fixes hip self-collision jam), implicitfast
// integrator, actuator physics (armature/damping/forcerange from robots/x2/actuator.yaml).

#include <cmath>
#include <cstring>
#include <chrono>
#include <map>
#include <mutex>
#include <regex>
#include <string>
#include <thread>
#include <vector>

#include <climits>
#include <cstdlib>
#include <sys/stat.h>
#include <unistd.h>

#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <yaml-cpp/yaml.h>

#include <rclcpp/rclcpp.hpp>
#include <aimdk_msgs/msg/joint_command_array.hpp>
#include <aimdk_msgs/msg/joint_state_array.hpp>
#include <sensor_msgs/msg/imu.hpp>

namespace {

// Locate the repo root at runtime so the binary is relocatable (no hard-coded
// absolute path baked in at compile time). Walk upward from the executable's
// directory, then from the current working directory, looking for the marker
// data file. Override with the MX2_REPO_ROOT environment variable if needed.
inline bool path_exists(const std::string& p) {
  struct stat st;
  return ::stat(p.c_str(), &st) == 0;
}
std::string find_repo_root() {
  const std::string marker = "/robots/x2/robot_full_flat_ground_excl.xml";
  if (const char* env = std::getenv("MX2_REPO_ROOT"))
    if (*env && path_exists(std::string(env) + marker)) return env;
  std::vector<std::string> starts;
  char buf[PATH_MAX];
  ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (n > 0) { buf[n] = '\0'; std::string exe(buf); starts.push_back(exe.substr(0, exe.find_last_of('/'))); }
  if (::getcwd(buf, sizeof(buf))) starts.push_back(buf);
  for (std::string dir : starts) {
    for (int up = 0; up < 16 && !dir.empty(); ++up) {
      if (path_exists(dir + marker)) return dir;
      auto slash = dir.find_last_of('/');
      if (slash == std::string::npos) break;
      dir = dir.substr(0, slash);  // strip last path component; "" once we pass root
    }
  }
  return ".";  // fallback: resolve data paths relative to the current directory
}

const std::string REPO = find_repo_root();
const std::string MODEL_XML = REPO + "/robots/x2/robot_full_flat_ground_excl.xml";
const std::string ACTUATOR_YAML = REPO + "/robots/x2/actuator.yaml";
const std::string DEPLOY_CFG = REPO + "/robots/x2/x2.yaml";
constexpr double GRAVITY = 9.81;

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

// Standing pose (matches mujoco_x2.py STANDING_POSE); joints not listed = 0.
const std::map<std::string, double> STANDING_POSE = {
    {"left_hip_pitch_joint", -0.1}, {"right_hip_pitch_joint", -0.1},
    {"left_knee_joint", 0.2}, {"right_knee_joint", 0.2},
    {"left_ankle_pitch_joint", -0.1}, {"right_ankle_pitch_joint", -0.1},
    {"left_shoulder_pitch_joint", 0.2}, {"right_shoulder_pitch_joint", 0.2},
    {"left_shoulder_roll_joint", 0.2}, {"right_shoulder_roll_joint", -0.2},
    {"left_elbow_joint", -0.6}, {"right_elbow_joint", -0.6},
};
// Pre-command hold gains per area (used until first ROS command arrives).
const std::map<std::string, double> DEFAULT_KP = {{"head", 20}, {"waist", 20}, {"arm", 20}, {"leg", 100}};
const std::map<std::string, double> DEFAULT_KD = {{"head", 2}, {"waist", 4}, {"arm", 2}, {"leg", 5}};

struct Desired { double pos = 0, vel = 0, kp = 0, kd = 0, effort = 0; };
struct JMap { int qadr, dadr, act; };

double wall_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

}  // namespace

class MujocoX2 : public rclcpp::Node {
 public:
  MujocoX2() : Node("mujoco_x2_hal_bridge") {
    char err[1000] = "";
    model_ = mj_loadXML(MODEL_XML.c_str(), nullptr, err, sizeof(err));
    if (!model_) throw std::runtime_error(std::string("mj_loadXML failed: ") + err);
    model_->opt.integrator = mjINT_IMPLICITFAST;
    apply_actuator_physics();
    data_ = mj_makeData(model_);

    // joint index map + init standing pose + pre-command gains
    std::map<int, int> jid2act;
    for (int a = 0; a < model_->nu; ++a)
      if (model_->actuator_trntype[a] == mjTRN_JOINT)
        jid2act[model_->actuator_trnid[a * 2]] = a;
    load_stand_gains();
    for (const auto& area : AREAS) {
      for (const auto& nm : JOINT_NAMES.at(area)) {
        int jid = mj_name2id(model_, mjOBJ_JOINT, nm.c_str());
        if (jid < 0) throw std::runtime_error("joint not in model: " + nm);
        int act = jid2act.count(jid) ? jid2act[jid] : -1;
        if (act < 0) throw std::runtime_error("no actuator for joint: " + nm);
        JMap jm{model_->jnt_qposadr[jid], model_->jnt_dofadr[jid], act};
        jmap_[nm] = jm;
        double init = STANDING_POSE.count(nm) ? STANDING_POSE.at(nm) : 0.0;
        data_->qpos[jm.qadr] = init;
        double kp = stand_kp_.count(nm) ? stand_kp_[nm] : DEFAULT_KP.at(area);
        double kd = stand_kd_.count(nm) ? stand_kd_[nm] : DEFAULT_KD.at(area);
        desired_[nm] = Desired{init, 0, kp, kd, 0};
      }
    }
    mj_forward(model_, data_);
    imu_sid_["chest"] = mj_name2id(model_, mjOBJ_SITE, "imu_0");  // imu_0 = pelvis
    imu_sid_["torso"] = mj_name2id(model_, mjOBJ_SITE, "imu_1");  // imu_1 = torso

    // QoS: state BEST_EFFORT depth1, command RELIABLE depth10
    auto state_qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
    auto cmd_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();
    for (const auto& a : AREAS) {
      state_pub_[a] = create_publisher<aimdk_msgs::msg::JointStateArray>(
          "/aima/hal/joint/" + a + "/state", state_qos);
      cmd_sub_[a] = create_subscription<aimdk_msgs::msg::JointCommandArray>(
          "/aima/hal/joint/" + a + "/command", cmd_qos,
          [this](aimdk_msgs::msg::JointCommandArray::SharedPtr msg) { cmd_cb(*msg); });
    }
    for (const auto& s : {"chest", "torso"})
      imu_pub_[s] = create_publisher<sensor_msgs::msg::Imu>(std::string("/aima/hal/imu/") + s + "/state", state_qos);

    RCLCPP_INFO(get_logger(), "x2 C++ HAL sim. model=%s dt=%.4f njnt=%d", MODEL_XML.c_str(),
                model_->opt.timestep, (int)model_->njnt);
  }

  ~MujocoX2() override {
    if (data_) mj_deleteData(data_);
    if (model_) mj_deleteModel(model_);
  }

  mjModel* model() { return model_; }
  mjData* data() { return data_; }
  std::mutex& lock() { return lock_; }

  // one physics step: PD via position actuators -> mj_step (matches mujoco_x2.py step())
  void step() {
    std::lock_guard<std::mutex> g(lock_);
    for (const auto& [nm, jm] : jmap_) {
      const Desired& d = desired_[nm];
      model_->actuator_gainprm[jm.act * mjNGAIN + 0] = d.kp;
      model_->actuator_biasprm[jm.act * mjNBIAS + 0] = 0.0;
      model_->actuator_biasprm[jm.act * mjNBIAS + 1] = -d.kp;
      model_->actuator_biasprm[jm.act * mjNBIAS + 2] = -d.kd;
      data_->ctrl[jm.act] = d.pos;
      data_->qfrc_applied[jm.dadr] = d.kd * d.vel + d.effort;  // ff (vel/effort = 0 from policy)
    }
    mj_step(model_, data_);
  }

  void publish_state() {
    auto stamp = now();
    std::lock_guard<std::mutex> g(lock_);
    for (const auto& area : AREAS) {
      aimdk_msgs::msg::JointStateArray arr;
      arr.header.stamp = stamp;
      for (const auto& nm : JOINT_NAMES.at(area)) {
        const JMap& jm = jmap_[nm];
        aimdk_msgs::msg::JointState js;
        js.name = nm;
        js.position = data_->qpos[jm.qadr];
        js.velocity = data_->qvel[jm.dadr];
        js.effort = data_->qfrc_actuator[jm.dadr] + data_->qfrc_applied[jm.dadr];
        arr.joints.push_back(js);
      }
      state_pub_[area]->publish(arr);
    }
    for (const auto& [name, sid] : imu_sid_) {
      sensor_msgs::msg::Imu m;
      m.header.stamp = stamp;
      double quat[4];
      mju_mat2Quat(quat, data_->site_xmat + sid * 9);  // wxyz
      m.orientation.w = quat[0]; m.orientation.x = quat[1];
      m.orientation.y = quat[2]; m.orientation.z = quat[3];
      mjtNum vel6[6];
      mj_objectVelocity(model_, data_, mjOBJ_SITE, sid, vel6, 1);  // local (body) frame
      m.angular_velocity.x = vel6[0]; m.angular_velocity.y = vel6[1]; m.angular_velocity.z = vel6[2];
      const mjtNum* R = data_->site_xmat + sid * 9;  // g_body = R^T @ [0,0,g]
      m.linear_acceleration.x = R[2] * GRAVITY;
      m.linear_acceleration.y = R[5] * GRAVITY;
      m.linear_acceleration.z = R[8] * GRAVITY;
      imu_pub_[name]->publish(m);
    }
  }

 private:
  void cmd_cb(const aimdk_msgs::msg::JointCommandArray& msg) {
    std::lock_guard<std::mutex> g(lock_);
    for (const auto& jc : msg.joints) {
      auto it = desired_.find(jc.name);
      if (it == desired_.end()) continue;  // joint not in model
      it->second = Desired{jc.position, jc.velocity, jc.stiffness, jc.damping, jc.effort};
    }
  }

  void load_stand_gains() {
    try {
      YAML::Node jc = YAML::LoadFile(DEPLOY_CFG)["joint_command"];
      auto jn = jc["joint_names"], skp = jc["stand_kp"], skd = jc["stand_kd"];
      for (size_t i = 0; i < jn.size() && i < skp.size(); ++i)
        stand_kp_[jn[i].as<std::string>()] = skp[i].as<double>();
      for (size_t i = 0; i < jn.size() && i < skd.size(); ++i)
        stand_kd_[jn[i].as<std::string>()] = skd[i].as<double>();
    } catch (const std::exception& e) {
      RCLCPP_WARN(get_logger(), "stand gains load failed (%s); using area defaults", e.what());
    }
  }

  // port of mujoco_x2.py _apply_actuator_physics: armature/damping/forcerange from actuator.yaml
  void apply_actuator_physics() {
    YAML::Node root;
    try { root = YAML::LoadFile(ACTUATOR_YAML); }
    catch (const std::exception& e) {
      RCLCPP_WARN(get_logger(), "actuator.yaml load failed (%s); physics not applied", e.what());
      return;
    }
    std::vector<std::pair<std::regex, double>> arm_r, damp_r, eff_r;
    for (const auto& kv : root["actuators"]) {
      const YAML::Node& g = kv.second;
      if (!g.IsMap() || !g["joint_names_expr"]) continue;
      for (const auto& e : g["joint_names_expr"]) {
        std::string pat = e.as<std::string>();
        if (g["armature"]) arm_r.emplace_back(std::regex(pat), g["armature"].as<double>());
        if (g["damping"]) damp_r.emplace_back(std::regex(pat), g["damping"].as<double>());
        if (g["effort_limit_sim"]) eff_r.emplace_back(std::regex(pat), g["effort_limit_sim"].as<double>());
      }
    }
    std::map<int, int> jid2act;
    for (int a = 0; a < model_->nu; ++a)
      if (model_->actuator_trntype[a] == mjTRN_JOINT)
        jid2act[model_->actuator_trnid[a * 2]] = a;
    auto resolve = [](const std::vector<std::pair<std::regex, double>>& rules, const std::string& n,
                      double& out) {
      for (const auto& [re, v] : rules) if (std::regex_match(n, re)) { out = v; return true; }
      return false;
    };
    int n_set = 0;
    for (int jid = 0; jid < model_->njnt; ++jid) {
      const char* nm = mj_id2name(model_, mjOBJ_JOINT, jid);
      int dof = model_->jnt_dofadr[jid];
      if (!nm || dof < 0) continue;
      std::string name(nm);
      double v;
      bool set = false;
      if (resolve(arm_r, name, v)) { model_->dof_armature[dof] = v; set = true; }
      if (resolve(damp_r, name, v)) { model_->dof_damping[dof] = v; set = true; }
      if (resolve(eff_r, name, v)) {
        auto it = jid2act.find(jid);
        if (it != jid2act.end()) {
          model_->actuator_forcerange[it->second * 2 + 0] = -v;
          model_->actuator_forcerange[it->second * 2 + 1] = v;
        }
      }
      if (set) ++n_set;
    }
    RCLCPP_INFO(get_logger(), "actuator physics applied: %d joints", n_set);
  }

  mjModel* model_ = nullptr;
  mjData* data_ = nullptr;
  std::mutex lock_;
  std::map<std::string, JMap> jmap_;
  std::map<std::string, Desired> desired_;
  std::map<std::string, double> stand_kp_, stand_kd_;
  std::map<std::string, int> imu_sid_;
  std::map<std::string, rclcpp::Publisher<aimdk_msgs::msg::JointStateArray>::SharedPtr> state_pub_;
  std::map<std::string, rclcpp::Subscription<aimdk_msgs::msg::JointCommandArray>::SharedPtr> cmd_sub_;
  std::map<std::string, rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr> imu_pub_;
};

// ------------------------------- minimal GLFW MuJoCo viewer -------------------------------
struct Viewer {
  GLFWwindow* win = nullptr;
  mjvCamera cam;
  mjvOption opt;
  mjvScene scn;
  mjrContext con;
  bool ok = false;

  bool init(mjModel* m) {
    if (!glfwInit()) return false;
    win = glfwCreateWindow(1200, 900, "mujoco_x2 (C++)", nullptr, nullptr);
    if (!win) { glfwTerminate(); return false; }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(0);  // no vsync -> don't cap the loop
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);
    mjv_makeScene(m, &scn, 2000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);
    cam.distance = 3.0; cam.elevation = -20; cam.azimuth = 90;
    cam.lookat[2] = 0.5;
    ok = true;
    return true;
  }
  bool running() { return ok && !glfwWindowShouldClose(win); }
  void sync(mjModel* m, mjData* d, std::mutex& lk) {
    mjrRect vp{0, 0, 0, 0};
    glfwGetFramebufferSize(win, &vp.width, &vp.height);
    { std::lock_guard<std::mutex> g(lk); mjv_updateScene(m, d, &opt, nullptr, &cam, mjCAT_ALL, &scn); }
    mjr_render(vp, &scn, &con);
    glfwSwapBuffers(win);
    glfwPollEvents();
  }
  void close() {
    if (!ok) return;
    mjr_freeContext(&con); mjv_freeScene(&scn);
    if (win) glfwDestroyWindow(win);
    glfwTerminate();
    ok = false;
  }
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  bool use_viewer = true;
  bool print_status = false;   // --print: pelvis z / RTF 를 1Hz 로 출력
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--no-viewer") use_viewer = false;
    if (std::string(argv[i]) == "--print") print_status = true;
  }

  auto node = std::make_shared<MujocoX2>();
  std::thread spin_thread([&]() { rclcpp::spin(node); });  // ROS callbacks

  const double dt = node->model()->opt.timestep;               // 0.002 = 500 Hz
  const int steps_per_batch = std::max(1, (int)std::lround(0.004 / dt));  // pace every ~4 ms
  const double batch_dt = steps_per_batch * dt;
  const int pelvis_bid = mj_name2id(node->model(), mjOBJ_BODY, "pelvis");

  // PHYSICS THREAD: real-time 500 Hz, decoupled from rendering so a slow (software-GL)
  // viewer can NOT drag the sim below real-time. RTF stays ~1.0 -> control timing stays
  // in sync with the wall-clock 50 Hz policy.
  std::atomic<bool> run{true};
  std::thread phys_thread([&]() {
    double next_t = wall_s(), rtf_t0 = wall_s();
    long steps = 0, rtf_steps = 0;
    RCLCPP_INFO(node->get_logger(), "physics %.0f Hz (render decoupled)", 1.0 / dt);
    while (run.load() && rclcpp::ok()) {
      for (int i = 0; i < steps_per_batch; ++i) { node->step(); node->publish_state(); }
      steps += steps_per_batch;
      rtf_steps += steps_per_batch;
      if (print_status && steps % 500 == 0) {  // ~1 s
        double w = wall_s();
        double rtf = (rtf_steps * dt) / std::max(1e-6, w - rtf_t0);
        rtf_t0 = w; rtf_steps = 0;
        double z = node->data()->xpos[pelvis_bid * 3 + 2];
        RCLCPP_INFO(node->get_logger(), "pelvis z=%.3f  RTF=%.2f (1.0=real-time)", z, rtf);
      }
      next_t += batch_dt;
      double sleep = next_t - wall_s();
      if (sleep > 0) std::this_thread::sleep_for(std::chrono::duration<double>(sleep));
      else next_t = wall_s();  // fell behind -> reset (avoid slow-motion)
    }
  });

  // RENDER on main thread (GL context must be main-thread). Best-effort: renders as fast as
  // GL allows; slow software GL just lowers FPS, not sim rate.
  Viewer viewer;
  if (use_viewer && !viewer.init(node->model()))
    RCLCPP_WARN(node->get_logger(), "viewer init failed; running headless");
  if (viewer.ok) {
    while (rclcpp::ok() && run.load() && viewer.running()) {
      viewer.sync(node->model(), node->data(), node->lock());
      std::this_thread::sleep_for(std::chrono::milliseconds(16));  // cap ~60 FPS
    }
    run.store(false);   // window closed -> stop sim
    viewer.close();
  } else {
    if (phys_thread.joinable()) phys_thread.join();  // headless: wait for SIGINT
  }

  run.store(false);
  if (phys_thread.joinable()) phys_thread.join();
  rclcpp::shutdown();
  if (spin_thread.joinable()) spin_thread.join();
  return 0;
}
