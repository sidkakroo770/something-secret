# Drone Corridor FSM

ROS 2 / Gazebo Classic autonomous corridor-navigation finite-state machine developed for the [redacted] rotorcraft autonomous mission.

The corridor stack uses a 2-D LiDAR for corridor alignment, centering, obstacle detection / bypass, recovery, and exit handling.

## FSM

```text
PRE_ENTRY_GEOMETRY_LOCK
        ↓
ENTER_CORRIDOR
        ↓
CORRIDOR_CRUISE
        ↓
OBSTACLE_DECISION
        ↓
AVOID_LEFT / AVOID_RIGHT
        ↓
CORRIDOR_CRUISE
        ↓
... repeat for additional obstacles ...
        ↓
EXIT_DETECTION
        ↓
CORRIDOR_EXITED
```

`HOVER_AND_REASSESS` is the recovery state. It can be entered from normal mission states when perception or geometry becomes unreliable. The loop-safe mission bridge limits repeated recovery loops and ultimately transitions to `ABORT_CORRIDOR` if recovery cannot establish a safe mission state.

## Repository layout

```text
SAE-Drone-Corridor-FSM/
├── scripts/
│   ├── pre_entry_geometry_lock_v3_hybrid.py
│   ├── corridor_cruise_v2_integrated.py
│   ├── obstacle_avoidance_v1.py
│   ├── exit_detection_v2_commit.py
│   ├── hover_and_reassess_v2_loop_safe.py
│   └── corridor_mission_bridge_v4_loop_safe.py
├── worlds/
│   └── corridor_obstacles_2m_fsm_test.sdf
├── models/
│   └── corridor_drone/
│       └── model.sdf
├── config/
│   └── corridor_fsm_gazebo_test_params_v2.yaml
└── README.md
```

## Gazebo test environment

The supplied test world contains:

- a 3.5 m wide corridor;
- a 2.0 m static obstacle extending from the left wall;
- a second 2.0 m static obstacle extending from the right wall;
- a 1.5 m open passage beside each obstacle;
- large spacing between both obstacles and the entrance / exit;
- an open corridor end for `EXIT_DETECTION` testing.

The expected obstacle sequence is:

```text
left-side obstacle  → AVOID_RIGHT
right-side obstacle → AVOID_LEFT
```

# Full Gazebo test

Start each command in a separate terminal and leave it running unless stated otherwise.

## Terminal 1 — Start Gazebo

```bash
killall gzserver gzclient 2>/dev/null
source /opt/ros/humble/setup.bash

ros2 launch gazebo_ros gazebo.launch.py \
world:=$HOME/SAE_Drone_Corridor_FSM/worlds/corridor_obstacles_2m_fsm_test.sdf
```

Wait until Gazebo fully opens.

## Terminal 2 — Spawn the test drone

```bash
source /opt/ros/humble/setup.bash

ros2 run gazebo_ros spawn_entity.py \
-entity corridor_drone \
-file $HOME/SAE_Drone_Corridor_FSM/models/corridor_drone/model.sdf \
-x -1.0 \
-y 0.65 \
-z 1.4 \
-Y 0.30
```

Optional LiDAR check:

```bash
ros2 topic hz /scan
```

A healthy simulation should publish the scan at approximately 10 Hz.

## Terminal 3 — PRE_ENTRY_GEOMETRY_LOCK

```bash
source /opt/ros/humble/setup.bash
python3 $HOME/SAE_Drone_Corridor_FSM/scripts/pre_entry_geometry_lock_v3_hybrid.py
```

## Terminal 4 — CORRIDOR_CRUISE

```bash
source /opt/ros/humble/setup.bash

python3 $HOME/SAE_Drone_Corridor_FSM/scripts/corridor_cruise_v2_integrated.py \
--ros-args \
--params-file $HOME/SAE_Drone_Corridor_FSM/config/corridor_fsm_gazebo_test_params_v2.yaml
```

Useful parameter checks:

```bash
ros2 param get /corridor_cruise_v2 front_cone_deg
ros2 param get /corridor_cruise_v2 correction_extreme_lateral_m
```

Expected values for this world:

```text
front_cone_deg = 35.0
correction_extreme_lateral_m = 1.1
```

## Terminal 5 — OBSTACLE_DECISION / AVOID_LEFT / AVOID_RIGHT

```bash
source /opt/ros/humble/setup.bash

python3 $HOME/SAE_Drone_Corridor_FSM/scripts/obstacle_avoidance_v1.py \
--ros-args \
--params-file $HOME/SAE_Drone_Corridor_FSM/config/corridor_fsm_gazebo_test_params_v2.yaml
```

Useful parameter checks:

```bash
ros2 param get /obstacle_avoidance_v1 avoid_timeout_s
ros2 param get /obstacle_avoidance_v1 shift_timeout_s
```

Expected values:

```text
avoid_timeout_s = 28.0
shift_timeout_s = 8.0
```

## Terminal 6 — EXIT_DETECTION

```bash
source /opt/ros/humble/setup.bash
python3 $HOME/SAE_Drone_Corridor_FSM/scripts/exit_detection_v2_commit.py
```

Once cruise has already confirmed an exit candidate, this state commits forward through the known-clear corridor end and then reports `CORRIDOR_EXITED`.

## Terminal 7 — HOVER_AND_REASSESS

```bash
source /opt/ros/humble/setup.bash
python3 $HOME/SAE_Drone_Corridor_FSM/scripts/hover_and_reassess_v2_loop_safe.py
```

## Terminal 8 — Watch the mission state

Start this before the mission bridge.

```bash
source /opt/ros/humble/setup.bash
ros2 topic echo /corridor/mission/state
```

## Terminal 9 — Watch cruise mode

```bash
source /opt/ros/humble/setup.bash
ros2 topic echo /corridor/cruise/mode
```

After an obstacle bypass the drone can temporarily enter internal `CORRECTING` mode while recentering, then return to `NOMINAL` without changing the global mission state.

## Terminal 10 — Start the mission bridge LAST

Starting the bridge starts the mission.

```bash
source /opt/ros/humble/setup.bash

python3 $HOME/SAE_Drone_Corridor_FSM/scripts/corridor_mission_bridge_v4_loop_safe.py \
--ros-args \
-p enter_corridor_timeout_s:=12.0
```

# Expected state sequence

A nominal two-obstacle run should look like:

```text
PRE_ENTRY_GEOMETRY_LOCK
ENTER_CORRIDOR
CORRIDOR_CRUISE
OBSTACLE_DECISION
AVOID_RIGHT
CORRIDOR_CRUISE
OBSTACLE_DECISION
AVOID_LEFT
CORRIDOR_CRUISE
EXIT_DETECTION
CORRIDOR_EXITED
```

`HOVER_AND_REASSESS` may legitimately appear between these states if a controller temporarily loses trustworthy geometry.

# Diagnostics

## Mission diagnostics

```bash
ros2 topic echo /corridor/mission/diagnostics
```

## Obstacle diagnostics

```bash
ros2 topic echo /corridor/obstacle/diagnostics
```

## Reassessment diagnostics

```bash
ros2 topic echo /corridor/reassess/diagnostics
```

## Exit diagnostics

```bash
ros2 topic echo /corridor/exit/diagnostics
```

## Mission result

```bash
ros2 topic echo /corridor/mission/result
```

A successful corridor run should finish with:

```text
SUCCESS:CORRIDOR_EXITED
```

# Important ROS topics

```text
/scan

/corridor/pre_entry/*
/corridor/cruise/*
/corridor/obstacle/*
/corridor/reassess/*
/corridor/exit/*

/corridor/mission/state
/corridor/mission/result
/corridor/mission/diagnostics
```

# Notes

- ROS body-command convention is FLU: `+x` forward, `+y` left, `+z` up, positive yaw counter-clockwise.
- The test bridge is a Gazebo mission arbiter. Vehicle/autopilot integration must convert these high-level commands into the frame and control interface expected by the flight controller.
- The 2 m-obstacle Gazebo parameters are test-specific and intentionally permit the large lateral displacement created during bypass before normal corridor recentering resumes.
