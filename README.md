# policy_rosBridge_sim

A ROS 2 test setup for deploying an Agibot X2 tracking policy against a MuJoCo simulation,
communicating over the Agibot HAL topics (the same interface as the real robot).

```
 deploy_policy    -->(policy_txt)-->  ros_bridge       -->(ROS command)-->  Physics
 (the policy, 50Hz) <--(sensor_txt)<--  (bridge, 500Hz) <--(ROS state)<---  (robot or MuJoCo sim)
```

Three processes talk over **ROS 2 topics** (state / command) and two **text files**
(`sensor_txt`, `policy_txt`) that carry the 50 Hz policy I/O.

## Contents

| Folder | What it is |
|--------|-----------|
| `aimdk/`        | The Agibot HAL **message package** (`aimdk_msgs`) as a colcon workspace. Everything below depends on it. |
| `ros_bridge/`   | ROS 2 &harr; text-file **bridge** (Python + C++). See `ros_bridge/README.md`. |
| `mujoco_x2/`    | C++ **MuJoCo simulator** of the X2 robot (the "robot" side). See `mujoco_x2/README.md`. |
| `deploy_policy/`| The **tracking policy** runner (C++). ⚠️ source only — see note below. |
| `robots/x2/`    | X2 robot **assets/config** (`x2.yaml`, `actuator.yaml`, MuJoCo XML, meshes). |


## What is `aimdk`?

`aimdk` is the Agibot robot's ROS 2 middleware. For this project we only need **`aimdk_msgs`**,
the HAL **message definitions** used by every process:
- `JointStateArray` / `JointState` — robot joint state (position, velocity, effort, ...).
- `JointCommandArray` / `JointCommand` — joint command (position, velocity, effort, stiffness=kp, damping=kd).
- plus standard `sensor_msgs/Imu` for the IMUs.

It is **not** a pip package — it is a ROS 2 (colcon) package you must build and source.

### Install `aimdk_msgs`

Requires **ROS 2 Humble** (Ubuntu 22.04) and `colcon`.

```bash
source /opt/ros/humble/setup.bash
cd policy_rosBridge_sim/aimdk
colcon build --packages-select aimdk_msgs
source install/setup.bash            # makes aimdk_msgs importable (Python) + linkable (C++)
```

After this, `python3 -c "import aimdk_msgs.msg"` should succeed, and C++ `find_package(aimdk_msgs)` works.


## Environment (every new terminal)

Each process needs ROS 2 + `aimdk_msgs` sourced (and conda OFF):
```bash
conda deactivate
source /opt/ros/humble/setup.bash
source ~/policy_rosBridge_sim/aimdk/install/setup.bash
```
Tip: put those lines in a small script (e.g. `~/ros_aimdk_env.sh`) and `source` it.

> If you see `ModuleNotFoundError: aimdk_msgs` (Python) or
> `libaimdk_msgs__...so: cannot open shared object file` (C++), the environment was not sourced.


## Components

### (1) `ros_bridge/`
Bridges ROS &harr; text files at 500 Hz: reads the robot state and writes `sensor_txt`; reads
`policy_txt` and publishes joint commands. Python (`robot_ros_bridge.py`) and C++
(`robot_ros_bridge.cpp`) versions share the **same structure**. Flags: `--torso`, `--print`.
**See [`ros_bridge/README.md`](ros_bridge/README.md)** for build / run / what-to-edit.

### (2) `mujoco_x2/`
The "robot": a C++ MuJoCo simulation of the X2 that subscribes to the command topics and
publishes state + IMU topics (the HAL interface). Flags: `--no-viewer`, `--print`.
**See [`mujoco_x2/README.md`](mujoco_x2/README.md)** for build / run.

### (3) `deploy_policy/` — ⚠️ not runnable yet
`deploy_policy/run_deploy_policy.cpp` is the tracking-policy runner (reads `sensor_txt`,
runs the ROM + tracking policy, writes `policy_txt`).

**Only the C++ source is included here.** It links against a large C++ library
(ROM runtime + tracking controller + kinematics) and depends on **libtorch, casadi/cusadi,
the ROM/tracking policy weights, and the robot kinematics/config data** — these are **not yet
included in this folder**, so `deploy_policy` **cannot be built or run standalone here yet**
(work in progress). For now, run the prebuilt `run_deploy_policy` binary from the full source
tree, pointing it at the same `sensor_txt` / `policy_txt` as the bridge:
```bash
run_deploy_policy --sensor <shared>/sensor_txt --policy <shared>/policy_txt
```

### (4) `robots/x2/`
Robot assets and config shared by the components: `x2.yaml` (deploy config), `actuator.yaml`
(standing pose / actuator physics), `robot_full_flat_ground_excl.xml` (MuJoCo model), `meshes/`.


## Running the pipeline (3 terminals)

Once `aimdk_msgs` is built and sourced in each terminal:

```bash
# Terminal 1 — bridge
cd ros_bridge && cmake -B build -S . && cmake --build build -j4 && ./build/robot_ros_bridge
#   (or: python3 robot_ros_bridge.py)

# Terminal 2 — MuJoCo robot
cd mujoco_x2 && cmake -B build -S . && cmake --build build -j4 && ./build/run_mujoco_x2 --no-viewer

# Terminal 3 — policy   (see deploy_policy note above; not runnable from this folder yet)
run_deploy_policy
```

The bridge and `deploy_policy` MUST use the **same** `sensor_txt` / `policy_txt` paths
(default `~/ROM-RL/Yujin/agibot_control_functions/{sensor_txt,policy_txt}`; configurable).


## Prerequisites summary

- ROS 2 **Humble** + `colcon` (Ubuntu 22.04).
- Build `aimdk_msgs` (above) and source it in every terminal.
- MuJoCo (for `mujoco_x2`) — see its README.
- `deploy_policy`: additional heavy deps (libtorch / casadi / policy data) — not yet packaged here.
