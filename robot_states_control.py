#!/usr/bin/env python3
"""
Robot state reader for collaboration — clean API (TOPIC-based), multi-IMU
========================================================================
Main-loop usage:

    client = RobotStateClient()            # node + subscriptions (4 joint groups + N IMUs)
    # spin the node in a background thread (see main())
    client.wait_ready()                    # waits until all topics have data

    imus, head, waist, arm, leg = client.get_robot_states()
    record_states_csv(imus, head, waist, arm, leg)
    print_state(imus, head, waist, arm, leg)   # at most once per second
    close_csv()                                # on shutdown

    # IMUs are a dict keyed by source:
    imus["chest"].ang_vel      imus["torso"].quat   ...

Freshness / no delay (important for a 500 Hz loop):
  - Every subscription is depth=1 KEEP_LAST: only the newest sample is held.
  - Callbacks (background executor) overwrite a cached message; get_robot_states()
    just reads the cache -> non-blocking.
  - joints ~<=1 ms old (1 kHz), IMUs ~<=2 ms old (500 Hz), + small jitter.
  - Topics are independent -> individually fresh, not sampled at the same instant
    (sub-ms..~1 ms skew). Compare header.stamp if you need strict alignment.
"""

import csv
import time
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Imu
from aimdk_msgs.msg import JointStateArray, JointCommandArray, JointCommand
import ruckig


# =============================== constants & configuration ===============================

# ---- (1) QoS: state (subscriptions) + publisher (commands) ----
# depth=1 -> always hold only the NEWEST sample (no stale backlog).
STATE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST, depth=1,
    durability=DurabilityPolicy.VOLATILE,
)

# publisher QoS (commands are reliable)
PUB_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST, depth=10,
    durability=DurabilityPolicy.VOLATILE,
)


# ---- (2) joint areas & types  (CMD_TOPICS / robot_model / PART_AREA use JointArea) ----
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
    kp: float
    kd: float


@dataclass
class JointReading:
    name: str
    position: float       # rad
    velocity: float       # rad/s
    effort: float         # N·m
    coil_temp: int
    motor_temp: int
    motor_vol: int
    msg_name: str = ""


@dataclass
class ImuReading:
    source: str                               # which IMU ("chest" / "torso" / ...)
    quat: Tuple[float, float, float, float]   # orientation (x, y, z, w)
    ang_vel: Tuple[float, float, float]       # angular_velocity (x, y, z) rad/s
    lin_acc: Tuple[float, float, float]       # linear_acceleration (x, y, z) m/s^2
    frame_id: str
    stamp: float                              # sensor stamp (s)


# ---- (3) topics: state (read) + command (write) ----
# IMUs to read, keyed by a short source name. Add more here if needed
# (e.g. "lidar": "/aima/hal/sensor/lidar_chest_front/imu").
IMU_TOPICS = {
    "chest": "/aima/hal/imu/chest/state",
    "torso": "/aima/hal/imu/torso/state",
}

# Joint state topics, keyed by area. Joint NAMES/ORDER come from robot_model
# (single source of truth) -> see JOINT_NAMES below.
JOINT_TOPICS = {
    JointArea.HEAD:  "/aima/hal/joint/head/state",
    JointArea.WAIST: "/aima/hal/joint/waist/state",
    JointArea.ARM:   "/aima/hal/joint/arm/state",
    JointArea.LEG:   "/aima/hal/joint/leg/state",
}

# command topics per area, ordered head/waist/arm/leg.  *** verify head/waist:
#   ros2 topic list | grep -E 'head|waist'
# (HEAD is included for completeness but the SEQUENCE below never targets it.)
CMD_TOPICS = {
    JointArea.HEAD:  "/aima/hal/joint/head/command",
    JointArea.WAIST: "/aima/hal/joint/waist/command",
    JointArea.ARM:   "/aima/hal/joint/arm/command",
    JointArea.LEG:   "/aima/hal/joint/leg/command",
}

# which element of (head, waist, arm, leg) corresponds to each area
_READING_INDEX = {JointArea.HEAD: 0, JointArea.WAIST: 1, JointArea.ARM: 2, JointArea.LEG: 3}


# ---- (4) robot_model (joint order MUST match the state/command arrays) ----
robot_model: Dict[JointArea, List[JointInfo]] = {
    JointArea.HEAD: [
        JointInfo("head_yaw_joint", -0.366, 0.366, 20.0, 2.0),
        JointInfo("head_pitch_joint", -0.3838, 0.3838, 20.0, 2.0),
    ],
    JointArea.WAIST: [
        JointInfo("waist_yaw_joint", -3.43, 2.382, 20.0, 4.0),
        JointInfo("waist_pitch_joint", -0.314, 0.314, 20.0, 4.0),
        JointInfo("waist_roll_joint", -0.488, 0.488, 20.0, 4.0),
    ],
    JointArea.ARM: [
        JointInfo("left_shoulder_pitch_joint", -3.08, 2.04, 20.0, 2.0),
        JointInfo("left_shoulder_roll_joint", -0.061, 2.993, 20.0, 2.0),
        JointInfo("left_shoulder_yaw_joint", -2.556, 2.556, 20.0, 2.0),
        JointInfo("left_elbow_joint", -2.3556, 0.0, 20.0, 2.0),
        JointInfo("left_wrist_yaw_joint", -2.556, 2.556, 20.0, 2.0),
        JointInfo("left_wrist_pitch_joint", -0.558, 0.558, 20.0, 2.0),
        JointInfo("left_wrist_roll_joint", -1.571, 0.724, 20.0, 2.0),
        JointInfo("right_shoulder_pitch_joint", -3.08, 2.04, 20.0, 2.0),
        JointInfo("right_shoulder_roll_joint", -2.993, 0.061, 20.0, 2.0),
        JointInfo("right_shoulder_yaw_joint", -2.556, 2.556, 20.0, 2.0),
        JointInfo("right_elbow_joint", -2.3556, 0.0000, 20.0, 2.0),
        JointInfo("right_wrist_yaw_joint", -2.556, 2.556, 20.0, 2.0),
        JointInfo("right_wrist_pitch_joint", -0.558, 0.558, 20.0, 2.0),
        JointInfo("right_wrist_roll_joint", -0.724, 1.571, 20.0, 2.0),
    ],
    JointArea.LEG: [
        JointInfo("left_hip_pitch_joint", -2.704, 2.556, 180.0, 5.0),
        JointInfo("left_hip_roll_joint", -0.235, 2.906, 100.0, 5.0),
        JointInfo("left_hip_yaw_joint", -1.684, 3.430, 100.0, 5.0),
        JointInfo("left_knee_joint", 0.0000, 2.4073, 100.0, 5.0),
        JointInfo("left_ankle_pitch_joint", -0.803, 0.453, 100.0, 5.0),
        JointInfo("left_ankle_roll_joint", -0.2625, 0.2625, 100.0, 5.0),
        JointInfo("right_hip_pitch_joint", -2.704, 2.556, 180.0, 5.0),
        JointInfo("right_hip_roll_joint", -2.906, 0.235, 100.0, 5.0),
        JointInfo("right_hip_yaw_joint", -3.430, 1.684, 100.0, 5.0),
        JointInfo("right_knee_joint", 0.0000, 2.4073, 100.0, 5.0),
        JointInfo("right_ankle_pitch_joint", -0.803, 0.453, 100.0, 5.0),
        JointInfo("right_ankle_roll_joint", -0.2625, 0.2625, 100.0, 5.0),
    ],
}

# joint NAMES/ORDER per area, derived from robot_model (single source of truth).
# Used to label the state arrays when reading -> see get_robot_states().
JOINT_NAMES = {area: [ji.name for ji in infos] for area, infos in robot_model.items()}


# ---- (5) per-part targets / sequence / area map ----
# per-part targets (override ONLY that part's joints; rest holds home).
# *** PLACEHOLDERS -- set safe values for YOUR robot ***
PART_TARGETS = {
    "right_arm": {"right_shoulder_pitch_joint": -1.00139, "right_shoulder_roll_joint": -1.01251,
                  "right_shoulder_yaw_joint": 0.34543, "right_elbow_joint": -1.19850,
                  "right_wrist_yaw_joint": 0.58990, "right_wrist_pitch_joint": 0.00973,
                  "right_wrist_roll_joint": -0.00134},
    "left_arm":  {"left_shoulder_pitch_joint": -1.17722, "left_shoulder_roll_joint": 0.84224,
                  "left_shoulder_yaw_joint": -0.46623, "left_elbow_joint": -1.27328,
                  "left_wrist_yaw_joint": -0.04170, "left_wrist_pitch_joint": 0.10433,
                  "left_wrist_roll_joint": -0.00172},
    "right_leg": {"right_hip_pitch_joint": -1.50280, "right_hip_roll_joint": -0.07833,
                  "right_hip_yaw_joint": 0.02752, "right_knee_joint": -0.05781,
                  "right_ankle_pitch_joint": 0.50592, "right_ankle_roll_joint": 0.08945},
    "left_leg":  {"left_hip_pitch_joint": -1.48382, "left_hip_roll_joint": 0.20430,
                  "left_hip_yaw_joint": 0.12396, "left_knee_joint": -0.06702,
                  "left_ankle_pitch_joint": 0.50669, "left_ankle_roll_joint": -0.09060},
    "waist":     {"waist_yaw_joint": 0.81597, "waist_pitch_joint": 0.34910,
                  "waist_roll_joint": 0.07626},
}

# run order + which AREA each part lives in  (arm -> leg -> waist; no head)
SEQUENCE = ["right_arm", "left_arm", "right_leg", "left_leg", "waist"]
PART_AREA = {"waist": JointArea.WAIST,
             "right_arm": JointArea.ARM, "left_arm": JointArea.ARM,
             "right_leg": JointArea.LEG, "left_leg": JointArea.LEG}


# ---- (6) conservative motion limits ----
CONTROL_PERIOD = 0.002      # 2 ms (500 Hz)
HOLD_SECONDS = 5.0          # hold at target / gap between parts
MAX_VELOCITY = 0.4          # rad/s   (slow; tune per area if needed)
MAX_ACCELERATION = 0.4      # rad/s^2
MAX_JERK = 4.0              # rad/s^3


# ---- (7) CSV logging ----
CSV_PATH = "robot_state_full.csv"
JOINT_FIELDS = ["position", "velocity", "effort"]


# ----------------------------- the client node -----------------------------
class RobotStateClient(Node):
    """Subscribes to N IMUs + 4 joint-group topics, caches the latest of each."""

    def __init__(self):
        super().__init__("robot_state_client")
        self._lock = threading.Lock()
        self._imu_msg = {src: None for src in IMU_TOPICS}        # source    -> Imu msg
        self._joint_msg = {a: None for a in JOINT_TOPICS}        # JointArea -> JointStateArray

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
        Non-blocking. Raises RuntimeError if any topic has not produced data yet.

        ------------------------------------------------------------------
        HOW TO READ EACH VALUE  (for the policy developer / collaborator)
        ------------------------------------------------------------------
            imus, head, waist, arm, leg = client.get_robot_states()

        # --- IMUs (a dict of ImuReading, keyed by source) ---
        #   imus["torso"].ang_vel -> (wx, wy, wz)      base angular velocity (rad/s)
        #   imus["torso"].quat    -> (qx, qy, qz, qw)  base orientation (use for gravity)
        #   imus["chest"].lin_acc -> (ax, ay, az)      linear acceleration (m/s^2)
        #   each ImuReading also has .source, .frame_id, .stamp
        #   example:  wx, wy, wz = imus["torso"].ang_vel

        # --- Joints: head/waist/arm/leg are each a List[JointReading] ---
        #   each jr has:  jr.name (e.g. "left_knee_joint"), jr.position (rad),
        #                 jr.velocity (rad/s), jr.effort (N*m)
        #                 (jr.coil_temp / motor_temp / motor_vol also available)

        # (A) read ONE joint BY NAME (order-independent):
        #   joints = {jr.name: jr for jr in (head + waist + arm + leg)}
        #   q_knee = joints["left_knee_joint"].position

        # (B) read ALL joints as ORDERED VECTORS (for the policy observation):
        #   order = head + waist + arm + leg   # <-- MUST match YOUR policy's training order
        #   q   = [jr.position for jr in order]
        #   dq  = [jr.velocity for jr in order]
        #   tau = [jr.effort   for jr in order]
        #
        # NOTE: pick ONE IMU as the policy base (e.g. imus["torso"]) and compute
        #       projected_gravity from its quat. Keep that choice consistent with
        #       how the policy was trained.
        ------------------------------------------------------------------
        """
        with self._lock:
            imu_msgs = {s: self._imu_msg[s] for s in IMU_TOPICS}
            joint_msgs = {g: self._joint_msg[g] for g in JOINT_TOPICS}

        if (any(imu_msgs[s] is None for s in IMU_TOPICS) or
                any(joint_msgs[g] is None for g in JOINT_TOPICS)):
            raise RuntimeError("State not ready (call wait_ready first).")

        # imus: source -> ImuReading (newest of each)
        imus = {s: self._imu_reading(s, imu_msgs[s]) for s in IMU_TOPICS}

        head  = self._name_list(joint_msgs[JointArea.HEAD].joints,  JOINT_NAMES[JointArea.HEAD])
        waist = self._name_list(joint_msgs[JointArea.WAIST].joints, JOINT_NAMES[JointArea.WAIST])
        arm   = self._name_list(joint_msgs[JointArea.ARM].joints,   JOINT_NAMES[JointArea.ARM])
        leg   = self._name_list(joint_msgs[JointArea.LEG].joints,   JOINT_NAMES[JointArea.LEG])
        return imus, head, waist, arm, leg


# ----------------------------- CSV recording -----------------------------
class _Recorder:
    def __init__(self):
        self._file = None
        self._writer = None
        self._t0 = None
        self._rows = 0

    def record(self, imus, head, waist, arm, leg, path=CSV_PATH):
        groups = [("head", head), ("waist", waist), ("arm", arm), ("leg", leg)]
        # stable IMU column order = IMU_TOPICS order
        imu_sources = list(IMU_TOPICS.keys())
        if self._file is None:
            self._file = open(path, "w", newline="")
            self._writer = csv.writer(self._file)
            self._t0 = time.time()
            header = ["t_sec"]
            for src in imu_sources:                       # prefix each IMU by source
                header += [f"{src}.quat_x", f"{src}.quat_y", f"{src}.quat_z", f"{src}.quat_w",
                           f"{src}.ang_vel_x", f"{src}.ang_vel_y", f"{src}.ang_vel_z",
                           f"{src}.lin_acc_x", f"{src}.lin_acc_y", f"{src}.lin_acc_z"]
            for _label, readings in groups:
                for jr in readings:
                    for fld in JOINT_FIELDS:
                        header.append(f"{jr.name}.{fld}")
            self._writer.writerow(header)

        t = time.time() - self._t0
        row = [f"{t:.4f}"]
        for src in imu_sources:
            im = imus[src]
            row += [f"{v:.6f}" for v in im.quat]
            row += [f"{v:.6f}" for v in im.ang_vel]
            row += [f"{v:.6f}" for v in im.lin_acc]
        for _label, readings in groups:
            for jr in readings:
                row += [f"{jr.position:.6f}", f"{jr.velocity:.6f}", f"{jr.effort:.6f}"]
        self._writer.writerow(row)
        self._rows += 1

    def close(self):
        if self._file is not None:
            self._file.flush()
            self._file.close()
            print(f"[recorder] saved {CSV_PATH} ({self._rows} rows)")
            self._file = None


_RECORDER = _Recorder()


def record_states_csv(imus, head, waist, arm, leg):
    """Append one snapshot row (opens file + writes header on first call)."""
    _RECORDER.record(imus, head, waist, arm, leg)


def close_csv():
    _RECORDER.close()


# ----------------------------- console printing -----------------------------
_last_print = [0.0]
PRINT_PERIOD = 1.0   # seconds


def print_state(imus, head, waist, arm, leg):
    """Detailed view (all IMUs + per-joint name check), at most once per second."""
    now = time.time()
    if now - _last_print[0] < PRINT_PERIOD:
        return
    _last_print[0] = now

    print(f"\n========== robot state @ {time.strftime('%H:%M:%S')} ==========")
    for src, im in imus.items():                       # print every IMU
        print(f"IMU[{src}] frame_id='{im.frame_id}'  "
              f"quat={tuple(round(v,3) for v in im.quat)}  "
              f"ang_vel={tuple(round(v,4) for v in im.ang_vel)}  "
              f"lin_acc={tuple(round(v,3) for v in im.lin_acc)}")

    for label, readings in [("head", head), ("waist", waist), ("arm", arm), ("leg", leg)]:
        if not readings:
            print(f"[{label}] (none)")
            continue
        print(f"[{label}] {len(readings)} joints")
        print(f"  {'idx':>3}  {'expected (doc order)':<22} {'msg.name':<22} "
              f"{'match':<5} {'pos(rad)':>10} {'vel(rad/s)':>11}")
        for i, jr in enumerate(readings):
            msg_name = jr.msg_name if jr.msg_name else "(empty)"
            match = "O" if jr.msg_name == jr.name else "X"
            print(f"  {i:>3}  {jr.name:<22} {msg_name:<22} {match:<5} "
                  f"{jr.position:>+10.4f} {jr.velocity:>+11.4f}")


# ======================================================================
#  COMMAND CONTROL SEQUENCE
# ----------------------------------------------------------------------
#  Drives one body part at a time through: home -> target -> hold -> home.
#  Sequence: right arm, left arm, right leg, left leg, waist.
#  main() calls seq.policy_joint_command(...) every 2 ms and publishes the
#  returned command to the matching area's publisher.
#
#  #####################  SAFETY  #####################
#  - Stop MC on 10.0.1.40:  aima em stop-app mc   (no balance after this)
#  - Robot MUST be FULLY SUSPENDED (feet off ground) for the WHOLE run.
#  - TARGETS below are PLACEHOLDERS -> set safe values for YOUR robot.
#  - Only the ACTIVE part is commanded each tick (fine when suspended).
#  - When done, restart MC:  aima em start-app mc
#  ####################################################
# ======================================================================


class JointSequencer:
    """Tiny state machine. policy_joint_command() returns (area, JointCommandArray)
    for the active part/phase each tick, so main needs no big if-chain.

    Phases per part: to_target -> hold -> to_home -> gap -> (next part).
    The returned area is a JointArea; pass it to commander.publish(area, cmd).
    """

    def __init__(self):
        self.steps = SEQUENCE
        self.idx = 0
        self.sub = "to_target"            # to_target / hold / to_home / gap / done
        self.active = PART_AREA[self.steps[0]]    # JointArea
        self.home = {}                    # JointArea -> [home positions]
        self._captured = False
        self._need_init = True
        self._hold_until = 0.0
        self.target = None
        self.rk = self.rin = self.rout = None
        self._phase = f"{self.steps[0]}_to_target"
        self._log = rclpy.logging.get_logger("joint_sequencer")   # for clamp warnings
        print("Sequence:", " -> ".join(SEQUENCE), " (each: target, hold 5s, back to home)")
        print("TARGETS are PLACEHOLDERS. Robot MUST be fully suspended. MC stopped on .40.")

    @property
    def phase(self) -> str:
        return self._phase

    @staticmethod
    def _readings_of(area, head, waist, arm, leg):
        return (head, waist, arm, leg)[_READING_INDEX[area]]

    def _clamp_target(self, area, home_pos, part_target):
        """Override the part's joints in home_pos, clamped to limits (warns on clamp)."""
        info = robot_model[area]
        names = [j.name for j in info]
        target = list(home_pos)
        for nm, val in part_target.items():
            i = names.index(nm)
            lo, hi = info[i].lower_limit, info[i].upper_limit
            clamped = max(lo, min(hi, val))
            if abs(clamped - val) > 1e-9:
                self._log.warn(
                    f"{nm}: target {val:.6f} -> clamped to {clamped:.6f} (limit [{lo}, {hi}])")
            target[i] = clamped
        return target

    def _new_ruckig(self, area, cur_pos, cur_vel, goal):
        dofs = len(robot_model[area])
        self.rk = ruckig.Ruckig(dofs, CONTROL_PERIOD)
        self.rin = ruckig.InputParameter(dofs)
        self.rout = ruckig.OutputParameter(dofs)
        self.rin.max_velocity = [MAX_VELOCITY] * dofs
        self.rin.max_acceleration = [MAX_ACCELERATION] * dofs
        self.rin.max_jerk = [MAX_JERK] * dofs
        self.rin.current_position = list(cur_pos)
        self.rin.current_velocity = list(cur_vel)
        self.rin.current_acceleration = [0.0] * dofs
        self.rin.target_position = list(goal)
        self.rin.target_velocity = [0.0] * dofs
        self.rin.target_acceleration = [0.0] * dofs

    def _build_cmd(self, area, positions, velocities):
        cmd = JointCommandArray()
        for i, j in enumerate(robot_model[area]):
            jc = JointCommand()
            jc.name = j.name
            jc.position = positions[i]
            jc.velocity = velocities[i]
            jc.effort = 0.0
            jc.stiffness = j.kp
            jc.damping = j.kd
            cmd.joints.append(jc)
        return cmd

    def policy_joint_command(self, head, waist, arm, leg):
        """Return (area, JointCommandArray) for this 2 ms tick. area is a JointArea."""
        # capture home (initial pose) on the first call
        if not self._captured:
            self.home = {JointArea.ARM:   [jr.position for jr in arm],
                         JointArea.LEG:   [jr.position for jr in leg],
                         JointArea.WAIST: [jr.position for jr in waist]}
            self._captured = True

        area = self.active
        readings = self._readings_of(area, head, waist, arm, leg)
        cur_pos = [jr.position for jr in readings]
        cur_vel = [jr.velocity for jr in readings]
        # label for THIS tick's command (set before any phase transition below)
        self._phase = "done" if self.sub == "done" else f"{self.steps[self.idx]}_{self.sub}"

        # sequence finished -> hold the last area at home
        if self.sub == "done":
            return area, self._build_cmd(area, self.home[area], [0.0] * len(self.home[area]))

        step = self.steps[self.idx]

        if self.sub in ("to_target", "to_home"):
            if self._need_init:
                goal = (self._clamp_target(area, self.home[area], PART_TARGETS[step])
                        if self.sub == "to_target" else self.home[area])
                if self.sub == "to_target":
                    self.target = goal
                self._new_ruckig(area, cur_pos, cur_vel, goal)
                self._need_init = False
            res = self.rk.update(self.rin, self.rout)
            self.rin.current_position = self.rout.new_position
            self.rin.current_velocity = self.rout.new_velocity
            self.rin.current_acceleration = self.rout.new_acceleration
            cmd = self._build_cmd(area, self.rout.new_position, self.rout.new_velocity)
            if res == ruckig.Result.Finished:
                if self.sub == "to_target":
                    self.sub = "hold"; self._hold_until = time.monotonic() + HOLD_SECONDS
                else:
                    self.sub = "gap"; self._hold_until = time.monotonic() + HOLD_SECONDS
            return area, cmd

        # hold (at target) or gap (at home)
        hold_pos = self.target if self.sub == "hold" else self.home[area]
        cmd = self._build_cmd(area, hold_pos, [0.0] * len(hold_pos))
        if time.monotonic() >= self._hold_until:
            if self.sub == "hold":
                self.sub = "to_home"; self._need_init = True
            else:                                  # gap done -> advance to next part
                self.idx += 1
                if self.idx >= len(self.steps):
                    self.sub = "done"
                else:
                    self.active = PART_AREA[self.steps[self.idx]]
                    self.sub = "to_target"; self._need_init = True
        return area, cmd


# ----------------------------- command publisher node -----------------------------
class WholeBodyCommander(Node):
    """Dedicated node that owns one command publisher per area.

    Usage:  commander = WholeBodyCommander()
            commander.publish(area, cmd)     # area is a JointArea
    """

    def __init__(self):
        super().__init__("whole_body_commander")
        self._pub = {area: self.create_publisher(JointCommandArray, topic, PUB_QOS)
                     for area, topic in CMD_TOPICS.items()}

    def publish(self, area, cmd):
        self._pub[area].publish(cmd)


# ----------------------------- demo main loop -----------------------------
def main(args=None):
    rclpy.init(args=args)
    client = RobotStateClient()        # node 1: reads IMU + joint state topics
    commander = WholeBodyCommander()   # node 2: publishes joint commands

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(client)
    executor.add_node(commander)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    if not client.wait_ready(timeout_sec=10.0):
        client.destroy_node(); commander.destroy_node(); rclpy.shutdown(); return

    seq = JointSequencer()

    input(">>> Ready? Press Enter to start (Ctrl+C to cancel) <<<")

    rate_hz = 500.0
    period = 1.0 / rate_hz
    next_t = time.perf_counter()
    try:
        while rclpy.ok():
            imus, head, waist, arm, leg = client.get_robot_states()              # (1) read
            record_states_csv(imus, head, waist, arm, leg)                       # (1-1) save on csv
            print_state(imus, head, waist, arm, leg)                             # (1-2) print/sec
            joint_group, cmd = seq.policy_joint_command(head, waist, arm, leg)   # (2) Policy
            commander.publish(joint_group, cmd)                                  # (3) Command publish

            next_t += period
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.perf_counter()
    except KeyboardInterrupt:
        pass
    finally:
        close_csv()
        client.destroy_node()
        commander.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
