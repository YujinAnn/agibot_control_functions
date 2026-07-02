#!/usr/bin/env python3
"""
robot_ros_bridge.py  —  self-contained ROS2 <-> text-file bridge (500 Hz)
=========================================================================
Pipeline:
    run_deploy_policy --(policy_txt, 50Hz)--> [robot_ros_bridge] --(ROS cmd)--> mujoco_x2 (robot)
    mujoco_x2 (robot)  --(ROS state)--> [robot_ros_bridge] --(sensor_txt, 500Hz)--> run_deploy_policy

Every loop (500 Hz):
    imus, head, waist, arm, leg = client.get_robot_states()   # (1) read state via ROS
    record_states(imus, head, waist, arm, leg)           # (2) write sensor_txt
    cmds = policy_joint_command()                             # (3) read policy_txt
    for joint_group, cmd in cmds:                             # (4) publish commands
        commander.publish(joint_group, cmd)

Text-file formats (space-separated, single line, atomic update = always latest):
    sensor_txt : t  [31*(pos vel)]  [base quat x y z w]  [base angvel x y z]   (= 70 values)
    policy_txt : t  [31 pos]  [31 kp]  [31 kd]                                 (= 94 values)
Joint order = head(2) + waist(3) + arm(14) + leg(12) = 31.

This file is SELF-CONTAINED: it does NOT import robot_states_control.py. The small pieces
it needs (JointArea, robot_model, ImuReading, RobotStateClient, WholeBodyCommander) are
inlined below.

Run (conda off, source ~/ros_aimdk_env.sh):
    python3 robot_ros_bridge.py
"""

import os
import sys
import time
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.signals import SignalHandlerOptions
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Imu
from aimdk_msgs.msg import JointStateArray, JointCommandArray, JointCommand


# ============================= CONFIG: paths =============================
# sensor_txt / policy_txt are shared with run_deploy_policy; both MUST use the same paths.
# NOTE: Python open() does NOT expand '~', so expand it with os.path.expanduser.
_IO_DIR = os.path.expanduser("~/ROM-RL/Yujin/agibot_control_functions")
SENSOR_TXT = os.path.join(_IO_DIR, "sensor_txt")
POLICY_TXT = os.path.join(_IO_DIR, "policy_txt")

# Base IMU for sensor_txt orientation/angular-velocity. Default "chest" (= imu_0, pelvis =
# the policy base). Pass --torso to use "torso" (= imu_1) instead (A/B testing).
BASE_IMU = "torso" if "--torso" in sys.argv else "chest"


# =============================== file IO ===============================
# NOTE: these functions read module globals (JOINT_ORDER, robot_model, _last_pos/kp/kd)
# that are defined further below; that's fine since they are only referenced at call time.
def _atomic_write(path: str, text: str) -> None:
    """Write to a temp file then rename -> race-free 'latest' read (no partial reads).
    The temp name includes the PID so concurrent writers never collide."""
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def record_states(imus, head, waist, arm, leg) -> None:
    """(2) Write the current robot state to sensor_txt as one line (always latest)."""
    by_name = {jr.name: jr for jr in (list(head) + list(waist) + list(arm) + list(leg))}
    out = [f"{time.time():.6f}"]
    for nm in JOINT_ORDER:
        jr = by_name.get(nm)
        if jr is None:
            out += ["0.0", "0.0"]
        else:
            out += [f"{jr.position:.6f}", f"{jr.velocity:.6f}"]
    # Base (floating-base) orientation / angular velocity. The policy's base is the PELVIS,
    # where the root free joint lives. IMU mapping: "chest" = imu_0 (pelvis), "torso" = imu_1.
    # Default uses "chest" (pelvis, matches run_standalone_sim); --torso switches to "torso"
    # (imu_1) for A/B testing (torso base makes ARM diverge while walking).
    base_imu = imus.get(BASE_IMU) if isinstance(imus, dict) else None   # BASE_IMU: chest / torso
    if base_imu is not None:
        out += [f"{v:.6f}" for v in base_imu.quat]      # x, y, z, w
        out += [f"{v:.6f}" for v in base_imu.ang_vel]   # x, y, z
    else:
        out += ["0.0"] * 7
    _atomic_write(SENSOR_TXT, " ".join(out) + "\n")


def read_latest_policy():
    """Read the latest action (joint targets) from policy_txt -> dicts by name.
    Returns None if the file is missing or unparsable."""
    try:
        with open(POLICY_TXT) as f:
            data = f.read().strip()
        if not data:
            return None
        toks = data.splitlines()[-1].split()   # last (newest) line
        vals = [float(x) for x in toks]
    except (FileNotFoundError, ValueError, IndexError):
        return None
    n = len(JOINT_ORDER)
    if len(vals) == 1 + 3 * n:   # t + 31 pos + 31 kp + 31 kd (policy sent its own gains)
        pos = dict(zip(JOINT_ORDER, vals[1:1 + n]))
        kp = dict(zip(JOINT_ORDER, vals[1 + n:1 + 2 * n]))
        kd = dict(zip(JOINT_ORDER, vals[1 + 2 * n:1 + 3 * n]))
        return pos, kp, kd
    if len(vals) == n + 1:       # t + 31 positions (no gains)
        return dict(zip(JOINT_ORDER, vals[1:])), None, None
    if len(vals) == n:           # 31 positions
        return dict(zip(JOINT_ORDER, vals)), None, None
    return None


def policy_joint_command():
    """(3) Read the latest policy and build a per-area JointCommandArray.
    Positions are taken from the newest policy_txt frame every tick. kp/kd are updated ONLY
    when the frame carries them; otherwise the previously received gains are kept (so a
    positions-only frame does NOT overwrite kp/kd with 0). Initial stage (before ANY gains
    arrive): kp/kd = 0.
    Returns [(JointArea, JointCommandArray), ...] (empty until positions arrive)."""
    global _last_pos, _last_kp, _last_kd
    p = read_latest_policy()
    if p is not None:
        pos, kp, kd = p
        _last_pos = pos               # positions: always take the newest
        if kp is not None:            # gains: keep previous if this frame has none
            _last_kp = kp
        if kd is not None:
            _last_kd = kd
    if _last_pos is None:
        return []   # no positions yet -> nothing to publish

    out = []
    for area in _AREA_ORDER:
        cmd = JointCommandArray()
        for ji in robot_model[area]:
            jc = JointCommand()
            jc.name = ji.name
            jc.position = float(_last_pos.get(ji.name, 0.0))   # updated every tick
            jc.velocity = 0.0
            jc.effort = 0.0
            # gains: last received value; 0 at the initial stage (no gains yet).
            jc.stiffness = float(_last_kp[ji.name]) if _last_kp is not None else 0.0
            jc.damping = float(_last_kd[ji.name]) if _last_kd is not None else 0.0
            cmd.joints.append(jc)
        out.append((area, cmd))
    return out


# =========================== CONFIG: constants ===========================
CONTROL_RATE_HZ = 500.0   # bridge loop rate (Hz)


# =============================== QoS ===============================
# State subscriptions: BEST_EFFORT, depth=1 -> always keep only the newest sample.
STATE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST, depth=1,
    durability=DurabilityPolicy.VOLATILE,
)
# Command publishers: RELIABLE, depth=10 (must match the robot/sim subscriber QoS).
PUB_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST, depth=10,
    durability=DurabilityPolicy.VOLATILE,
)


# =========================== joint model ===========================
class JointArea(Enum):
    HEAD = 'HEAD'
    WAIST = 'WAIST'
    ARM = 'ARM'
    LEG = 'LEG'


@dataclass
class JointInfo:
    name: str
    lower_limit: float
    upper_limit: float


@dataclass
class JointReading:
    name: str
    position: float       # rad
    velocity: float       # rad/s
    effort: float         # N*m
    coil_temp: int
    motor_temp: int
    motor_vol: int
    msg_name: str = ""


@dataclass
class ImuReading:
    source: str                               # which IMU ("chest" / "torso" / ...)
    quat: Tuple[float, float, float, float]   # orientation (x, y, z, w)
    ang_vel: Tuple[float, float, float]       # angular velocity (x, y, z) rad/s
    lin_acc: Tuple[float, float, float]       # linear acceleration (x, y, z) m/s^2
    frame_id: str
    stamp: float                              # sensor stamp (s)


# robot_model: joint order MUST match the state/command arrays. It defines only the joint
# names/order (+ limits); gains come from the policy (policy_txt carries its own kp/kd).
robot_model: Dict[JointArea, List[JointInfo]] = {
    JointArea.HEAD: [
        JointInfo("head_yaw_joint", -0.366, 0.366),
        JointInfo("head_pitch_joint", -0.3838, 0.3838),
    ],
    JointArea.WAIST: [
        JointInfo("waist_yaw_joint", -3.43, 2.382),
        JointInfo("waist_pitch_joint", -0.314, 0.314),
        JointInfo("waist_roll_joint", -0.488, 0.488),
    ],
    JointArea.ARM: [
        JointInfo("left_shoulder_pitch_joint", -3.08, 2.04),
        JointInfo("left_shoulder_roll_joint", -0.061, 2.993),
        JointInfo("left_shoulder_yaw_joint", -2.556, 2.556),
        JointInfo("left_elbow_joint", -2.3556, 0.0),
        JointInfo("left_wrist_yaw_joint", -2.556, 2.556),
        JointInfo("left_wrist_pitch_joint", -0.558, 0.558),
        JointInfo("left_wrist_roll_joint", -1.571, 0.724),
        JointInfo("right_shoulder_pitch_joint", -3.08, 2.04),
        JointInfo("right_shoulder_roll_joint", -2.993, 0.061),
        JointInfo("right_shoulder_yaw_joint", -2.556, 2.556),
        JointInfo("right_elbow_joint", -2.3556, 0.0000),
        JointInfo("right_wrist_yaw_joint", -2.556, 2.556),
        JointInfo("right_wrist_pitch_joint", -0.558, 0.558),
        JointInfo("right_wrist_roll_joint", -0.724, 1.571),
    ],
    JointArea.LEG: [
        JointInfo("left_hip_pitch_joint", -2.704, 2.556),
        JointInfo("left_hip_roll_joint", -0.235, 2.906),
        JointInfo("left_hip_yaw_joint", -1.684, 3.430),
        JointInfo("left_knee_joint", 0.0000, 2.4073),
        JointInfo("left_ankle_pitch_joint", -0.803, 0.453),
        JointInfo("left_ankle_roll_joint", -0.2625, 0.2625),
        JointInfo("right_hip_pitch_joint", -2.704, 2.556),
        JointInfo("right_hip_roll_joint", -2.906, 0.235),
        JointInfo("right_hip_yaw_joint", -3.430, 1.684),
        JointInfo("right_knee_joint", 0.0000, 2.4073),
        JointInfo("right_ankle_pitch_joint", -0.803, 0.453),
        JointInfo("right_ankle_roll_joint", -0.2625, 0.2625),
    ],
}

# joint NAMES/ORDER per area, derived from robot_model (single source of truth).
JOINT_NAMES = {area: [ji.name for ji in infos] for area, infos in robot_model.items()}


# =============================== topics ===============================
IMU_TOPICS = {
    "chest": "/aima/hal/imu/chest/state",   # imu_0 = pelvis (the policy base)
    "torso": "/aima/hal/imu/torso/state",   # imu_1 = torso
}
JOINT_TOPICS = {
    JointArea.HEAD:  "/aima/hal/joint/head/state",
    JointArea.WAIST: "/aima/hal/joint/waist/state",
    JointArea.ARM:   "/aima/hal/joint/arm/state",
    JointArea.LEG:   "/aima/hal/joint/leg/state",
}
CMD_TOPICS = {
    JointArea.HEAD:  "/aima/hal/joint/head/command",
    JointArea.WAIST: "/aima/hal/joint/waist/command",
    JointArea.ARM:   "/aima/hal/joint/arm/command",
    JointArea.LEG:   "/aima/hal/joint/leg/command",
}


# ============================ ROS nodes ============================
class RobotStateClient(Node):
    """Subscribes to N IMUs + 4 joint-group topics and caches the latest of each."""

    def __init__(self):
        super().__init__("robot_state_client")
        self._lock = threading.Lock()
        self._imu_msg = {src: None for src in IMU_TOPICS}      # source    -> Imu msg
        self._joint_msg = {a: None for a in JOINT_TOPICS}      # JointArea -> JointStateArray

        for src, topic in IMU_TOPICS.items():
            self.create_subscription(
                Imu, topic, lambda msg, s=src: self._imu_cb(s, msg), STATE_QOS)
        for area, topic in JOINT_TOPICS.items():
            self.create_subscription(
                JointStateArray, topic,
                lambda msg, a=area: self._joint_cb(a, msg), STATE_QOS)

    @staticmethod
    def _name_list(msg_joints, names) -> List[JointReading]:
        out = []
        for i, js in enumerate(msg_joints):
            nm = names[i] if i < len(names) else f"joint_{i}"
            out.append(JointReading(nm, js.position, js.velocity, js.effort,
                                    js.coil_temp, js.motor_temp, js.motor_vol,
                                    msg_name=js.name))
        return out

    @staticmethod
    def _imu_reading(source: str, msg: Imu) -> ImuReading:
        o, w, a = msg.orientation, msg.angular_velocity, msg.linear_acceleration
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        return ImuReading(source, (o.x, o.y, o.z, o.w), (w.x, w.y, w.z),
                          (a.x, a.y, a.z), msg.header.frame_id, stamp)

    def _imu_cb(self, source: str, msg: Imu):
        with self._lock:
            self._imu_msg[source] = msg

    def _joint_cb(self, area: JointArea, msg: JointStateArray):
        with self._lock:
            self._joint_msg[area] = msg

    def wait_ready(self, timeout_sec: float = 10.0) -> bool:
        """Block until every topic (all IMUs + all joint groups) has delivered once."""
        t0 = time.time()
        while time.time() - t0 < timeout_sec:
            with self._lock:
                if (all(self._imu_msg[s] is not None for s in IMU_TOPICS) and
                        all(self._joint_msg[g] is not None for g in JOINT_TOPICS)):
                    return True
            time.sleep(0.02)
        self.get_logger().error("Timed out waiting for state topics.")
        return False

    def get_robot_states(self):
        """Return (imus, head, waist, arm, leg) from the newest cached messages.
        imus is a Dict[str, ImuReading] keyed by source ("chest", "torso", ...).
        Non-blocking; raises RuntimeError if any topic has no data yet."""
        with self._lock:
            imu_msgs = {s: self._imu_msg[s] for s in IMU_TOPICS}
            joint_msgs = {g: self._joint_msg[g] for g in JOINT_TOPICS}
        if (any(imu_msgs[s] is None for s in IMU_TOPICS) or
                any(joint_msgs[g] is None for g in JOINT_TOPICS)):
            raise RuntimeError("State not ready (call wait_ready first).")
        imus = {s: self._imu_reading(s, imu_msgs[s]) for s in IMU_TOPICS}
        head  = self._name_list(joint_msgs[JointArea.HEAD].joints,  JOINT_NAMES[JointArea.HEAD])
        waist = self._name_list(joint_msgs[JointArea.WAIST].joints, JOINT_NAMES[JointArea.WAIST])
        arm   = self._name_list(joint_msgs[JointArea.ARM].joints,   JOINT_NAMES[JointArea.ARM])
        leg   = self._name_list(joint_msgs[JointArea.LEG].joints,   JOINT_NAMES[JointArea.LEG])
        return imus, head, waist, arm, leg


class WholeBodyCommander(Node):
    """Owns one command publisher per area. Usage: commander.publish(area, cmd)."""

    def __init__(self):
        super().__init__("whole_body_commander")
        self._pub = {area: self.create_publisher(JointCommandArray, topic, PUB_QOS)
                     for area, topic in CMD_TOPICS.items()}

    def publish(self, area, cmd):
        self._pub[area].publish(cmd)


# ==================== derived joint order + runtime state ====================
# Joint order: head -> waist -> arm -> leg (same order as robot_model). 31 joints.
_AREA_ORDER = [JointArea.HEAD, JointArea.WAIST, JointArea.ARM, JointArea.LEG]
JOINT_ORDER = [ji.name for area in _AREA_ORDER for ji in robot_model[area]]

# Persisted latest command. Positions update every tick; kp/kd update ONLY when the policy
# actually sends them, otherwise the previous gains are kept (never overwritten with 0).
_last_pos = None
_last_kp = None
_last_kd = None


# =============================== main loop ===============================
def main(args=None):
    # Create the IPC files if they do not exist yet.
    if not os.path.exists(SENSOR_TXT):
        _atomic_write(SENSOR_TXT, "")
    if not os.path.exists(POLICY_TXT):
        _atomic_write(POLICY_TXT, "")

    # Disable rclpy's SIGINT handler so Ctrl+C raises KeyboardInterrupt (clean shutdown).
    # (With the default handler, SIGINT invalidates the context mid-loop and publish() throws.)
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    client = RobotStateClient()        # subscribes to state topics
    commander = WholeBodyCommander()   # publishes command topics

    executor = SingleThreadedExecutor()
    executor.add_node(client)
    executor.add_node(commander)

    def _spin():
        try:
            executor.spin()
        except Exception:
            pass   # ignore exceptions caused by context shutdown on exit
    spin_thread = threading.Thread(target=_spin, daemon=True)
    spin_thread.start()

    print(f"[bridge] sensor_txt={SENSOR_TXT}")
    print(f"[bridge] policy_txt={POLICY_TXT}")
    # The bridge may start before mujoco_x2 -> wait until state topics appear.
    while rclpy.ok() and not client.wait_ready(timeout_sec=60.0):
        print("[bridge] still waiting for robot state topics (is mujoco_x2 running?)...")
    if not rclpy.ok():
        client.destroy_node(); commander.destroy_node(); rclpy.shutdown()
        return
    print(f"[bridge] state ready. running at {CONTROL_RATE_HZ:.0f} Hz")

    period = 1.0 / CONTROL_RATE_HZ
    next_t = time.perf_counter()
    # Measure the actual loop rate (print once per second). Should be ~500 Hz;
    # a lower rate means stale sensor_txt (a source of control latency).
    _rate_t0 = time.perf_counter()
    _rate_n = 0
    try:
        while rclpy.ok():
            imus, head, waist, arm, leg = client.get_robot_states()   # (1) read state
            record_states(imus, head, waist, arm, leg)           # (2) write sensor_txt
            cmds = policy_joint_command()                             # (3) read policy_txt
            for joint_group, cmd in cmds:                             # (4) publish commands
                commander.publish(joint_group, cmd)

            _rate_n += 1
            _now = time.perf_counter()
            if _now - _rate_t0 >= 1.0:
                print(f"[bridge] loop rate = {_rate_n / (_now - _rate_t0):.0f} Hz", flush=True)
                _rate_t0 = _now
                _rate_n = 0

            next_t += period
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.perf_counter()
    except KeyboardInterrupt:
        pass
    finally:
        # Stop the spin thread first, then destroy -> avoids a shutdown-race crash.
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        client.destroy_node()
        commander.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
