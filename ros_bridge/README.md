# robot_ros_bridge

A ROS 2 &harr; text-file bridge (500 Hz) between the tracking policy (`run_deploy_policy`) and
the robot / simulator. It reads the robot state over ROS and writes it to `sensor_txt`, and
reads the policy's joint commands from `policy_txt` and publishes them over ROS.

```
 deploy_policy    -->(policy_txt)-->  ros_bridge       -->(ROS command)-->  Physics
 (the policy, 50Hz) <--(sensor_txt)<--  (bridge, 500Hz) <--(ROS state)<---  (robot or MuJoCo sim)
```

There are **two implementations with the same structure and behavior** — pick either:

| File | Language |
|------|----------|
| `robot_ros_bridge.py`  | Python (rclpy) |
| `robot_ros_bridge.cpp` | C++ (rclcpp)   |

Both files use the **same section order** so you can move between them easily:

```
CONFIG: paths  ->  file IO  ->  CONFIG: constants  ->  joint model  ->  topics  ->  ROS nodes  ->  main
```


## Prerequisites

- **ROS 2 Humble** and the **`aimdk_msgs`** message package, built and sourced.
- Message deps: `aimdk_msgs`, `sensor_msgs`, and `rclpy` (Python) / `rclcpp` (C++).
- A robot or simulator publishing the HAL topics (e.g. `mujoco_x2`), otherwise the bridge
  just waits at "waiting for robot state topics".
- **Source the ROS environment in every terminal** (conda OFF):
  ```bash
  conda deactivate
  source ~/ros_aimdk_env.sh
  ```
- `sensor_txt` / `policy_txt` are shared with `run_deploy_policy` — both MUST point to the
  same folder. Default: `~/ROM-RL/Yujin/agibot_control_functions/{sensor_txt,policy_txt}`
  (see `CONFIG: paths`).


## What to change in the code

You normally edit only these **two functions** (in the `file IO` section, same in `.py` and `.cpp`):

| Function | Step | What it does |
|----------|------|--------------|
| `record_states(imus, head, waist, arm, leg)` | (2) | Build one `sensor_txt` line from the robot state |
| `policy_joint_command()`                      | (3) | Read `policy_txt` and return per-area commands |

Do **not** need to change:
- `get_robot_states()` — the ROS state reader (`RobotStateClient`).
- `commander.publish(group, cmd)` — the ROS command publisher (`WholeBodyCommander`).

Common config lives in `CONFIG: paths` (IO folder, base IMU) and the `joint model` /
`topics` sections (joint names/order, topic names).

The main loop is identical in both files:
```
imus, head, waist, arm, leg = client.get_robot_states()   # (1) read state
record_states(imus, head, waist, arm, leg)                # (2) write sensor_txt
cmds = policy_joint_command()                             # (3) read policy_txt
for group, cmd in cmds:                                   # (4) publish commands
    commander.publish(group, cmd)
```


## Build (C++)

```bash
source ~/ros_aimdk_env.sh
cd ~/policy_rosBridge_sim/ros_bridge
cmake -B build -S .          # once (or when CMakeLists.txt changes)
cmake --build build -j4      # compile
```
Binary: `./build/robot_ros_bridge`.
(After editing only the `.cpp`, re-run just `cmake --build build -j4`.)

The Python version needs no build — just run it.


## Run

```bash
source ~/ros_aimdk_env.sh          # required (RMW + aimdk_msgs)

# C++
./build/robot_ros_bridge

# Python
python3 robot_ros_bridge.py
```

### Flags / conditions

| Flag | Effect |
|------|--------|
| *(none)* | Base IMU = `chest` (imu_0 = **pelvis**, the policy's base). Default. |
| `--torso` | Base IMU = `torso` (imu_1). For A/B testing only — the policy expects the pelvis base, so `--torso` makes ARM diverge while walking. |
| `--print` | Print the loop rate (~1 Hz). Otherwise quiet. |

Examples:
```bash
./build/robot_ros_bridge --print          # C++, print loop rate
./build/robot_ros_bridge --torso          # C++, torso IMU
python3 robot_ros_bridge.py --torso --print
```


## Notes

- **kp/kd handling:** the policy (`policy_txt`) sends its own kp/kd. Positions update every
  tick; kp/kd update only when the policy sends them (otherwise the previous gains are kept
  and are never overwritten with 0). At the initial stage (before any gains arrive) kp/kd = 0.
- **Paths:** everything except the shared `sensor_txt`/`policy_txt` folder is relative /
  found via `find_package`, so the folder can be cloned anywhere and built (with the ROS env
  sourced).
- **`aimdk_msgs` not found:** you forgot `source ~/ros_aimdk_env.sh` in that terminal.
