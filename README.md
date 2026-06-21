# Robot State I/O — API for Collaborators

Two calls are all you need to integrate a policy with the robot:

```cpp
auto [imus, head, waist, arm, leg] = client->get_robot_states();   // (1) READ  state
commander->publish(joint_group, cmd);                              // (3) WRITE command
```

`client` is a `RobotStateClient` (reads IMU + joint-state topics) and `commander`
is a `WholeBodyCommander` (owns one publisher per joint area). Both run on a
background executor; your loop just calls these two functions.

## Results (demo video + graphs)

For the result of the running, you can see the video and graph here
(**6/20/2026 — Full Sensor and Control Function Development**):
https://docs.google.com/presentation/d/1htcKBK6phcArd9LEKUdJSYXiPlvqzxycCNV6OPvfAMI/edit?usp=sharing

It includes two runs of `robot_state_control.cpp`:
- **Read-only** — the policy + publish lines below are commented out, so the robot
  does not move; only state is read/logged.
- **Full control** — run as-is; the robot moves `right_arm → left_arm → right_leg →
  left_leg → waist` toward the target values.

## Files

| File | Description |
|------|-------------|
| `robot_states_control.cpp` | **Main file** — the C++ implementation described in this README. |
| `robot_states_control.py` | Python version — all code works the same as `robot_states_control.cpp`. |
| `robot_state_full.csv` | Logged state from **Part 1** (read-only run) of the 6/20/2026 slides. |
| `robot_state_full.csv` | Logged state from **Part 2** (full control run) of the 6/20/2026 slides. |

> **Recommended:** check the value ranges and scale using the two CSVs (3) and (4)
> before wiring in your policy.

> Note: files (3) and (4) share the name `robot_state_full.csv` (one per run).
> Keep them in separate folders, or rename (e.g. `robot_state_full_part1.csv` /
> `robot_state_full_part2.csv`), so one run does not overwrite the other.

---

## (1) `client->get_robot_states()` — read the latest state

```cpp
RobotState get_robot_states();
// RobotState = std::tuple<ImuMap, JointVec, JointVec, JointVec, JointVec>
//            = { imus, head, waist, arm, leg }
```

**Non-blocking.** It copies the newest cached message of each topic under a short
mutex — no DDS round-trip, no spin. Safe to call every control tick.

**Throws** `std::runtime_error` if any topic has not delivered data yet, so call
`client->wait_ready()` once before the loop.

### Return values

| Name | Type | Meaning |
|------|------|---------|
| `imus` | `std::map<std::string, ImuReading>` | IMUs keyed by source: `imus["chest"]`, `imus["torso"]` |
| `head` | `std::vector<JointReading>` | 2 joints |
| `waist` | `std::vector<JointReading>` | 3 joints |
| `arm` | `std::vector<JointReading>` | 14 joints (left 7, then right 7) |
| `leg` | `std::vector<JointReading>` | 12 joints (left 6, then right 6) |

Joint order inside each vector follows `robot_model` (the single source of truth).

### `ImuReading`

| Field | Type | Meaning |
|-------|------|---------|
| `source` | `std::string` | `"chest"` / `"torso"` |
| `quat` | `std::array<double,4>` | orientation `(x, y, z, w)` — use for projected gravity |
| `ang_vel` | `std::array<double,3>` | angular velocity `(x, y, z)` rad/s |
| `lin_acc` | `std::array<double,3>` | linear acceleration `(x, y, z)` m/s² |
| `frame_id` | `std::string` | sensor frame |
| `stamp` | `double` | sensor timestamp (s) |

### `JointReading`

| Field | Type | Meaning |
|-------|------|---------|
| `name` | `std::string` | expected name, e.g. `"left_knee_joint"` |
| `position` | `double` | rad |
| `velocity` | `double` | rad/s |
| `effort` | `double` | N·m |
| `coil_temp` / `motor_temp` / `motor_vol` | `int` | diagnostics |
| `msg_name` | `std::string` | raw `JointState.name` from the message (often empty) |

### Freshness (important for a 500 Hz loop)

- Every subscription is `depth=1 KEEP_LAST` → only the newest sample is held.
- joints (1 kHz) ≤ ~1 ms old, IMU (500 Hz) ≤ ~2 ms old.
- Topics are **independent** → each is individually fresh but not sampled at the
  exact same instant (sub-ms…~1 ms skew). Compare `stamp` if you need strict alignment.

### Usage examples

```cpp
auto [imus, head, waist, arm, leg] = client->get_robot_states();

// base angular velocity
double wz = imus["torso"].ang_vel[2];

// (A) read ONE joint BY NAME (order-independent)
std::map<std::string, JointReading> jmap;
for (const auto& v : {head, waist, arm, leg})
    for (const auto& jr : v) jmap[jr.name] = jr;
double q_knee = jmap["left_knee_joint"].position;

// (B) read ALL joints as ORDERED VECTORS (must match your policy's training order)
std::vector<JointReading> order;
for (const auto& v : {head, waist, arm, leg})
    order.insert(order.end(), v.begin(), v.end());
std::vector<double> q, dq, tau;
for (const auto& jr : order) { q.push_back(jr.position);
                              dq.push_back(jr.velocity);
                              tau.push_back(jr.effort); }
```

> The policy observation usually also needs `projected_gravity` (from `imu.quat`),
> the joystick command, and the previous action. Pick **one** IMU as the base
> (e.g. `imus["torso"]`) and keep that choice consistent with training.

---

## (3) `commander->publish(joint_group, cmd)` — send a command

```cpp
void publish(JointArea joint_group, const JointCommandArray& cmd);
```

Publishes one command array to the command topic of the given area.

| Param | Type | Meaning |
|-------|------|---------|
| `joint_group` | `JointArea` | `HEAD` / `WAIST` / `ARM` / `LEG` — selects the topic/publisher. Returned by `policy_joint_command()`, which reports the area it drives this tick. |
| `cmd` | `JointCommandArray` | one `JointCommand` per joint of that area, in `robot_model` order |

### `JointCommand` fields

| Field | Meaning |
|-------|---------|
| `name` | joint name |
| `position` | target position (rad) |
| `velocity` | target velocity (rad/s) |
| `effort` | feed-forward torque (N·m) |
| `stiffness` | position gain `kp` |
| `damping` | velocity gain `kd` |

### Behaviour

- O(1) lookup of the area's publisher, then `publish()`.
- Publisher QoS = **RELIABLE / KeepLast(10)** (commands must not be dropped).
- Only the **active** area is published each tick (the sequencer drives one body
  part at a time) — fine while the robot is fully suspended.

### Usage example

```cpp
auto [joint_group, cmd] = seq.policy_joint_command(head, waist, arm, leg);
commander->publish(joint_group, cmd);
```

---

## Minimal loop

```cpp
client->wait_ready();                       // block until all topics have data
JointSequencer seq;
while (rclcpp::ok()) {
    auto [imus, head, waist, arm, leg] = client->get_robot_states();   // (1) read
    auto [joint_group, cmd] = seq.policy_joint_command(head, waist, arm, leg);
    commander->publish(joint_group, cmd);                              // (3) publish
}
```

| Area | enum | joints |
|------|------|--------|
| head | `JointArea::HEAD` | 2 |
| waist | `JointArea::WAIST` | 3 |
| arm | `JointArea::ARM` | 14 (L7 + R7) |
| leg | `JointArea::LEG` | 12 (L6 + R6) |
