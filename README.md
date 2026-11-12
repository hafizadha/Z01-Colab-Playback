# Z01 Colab Playback

Play a Zlab Z01 walking or BallSprint policy in MuJoCo and watch the rollout inline at 30 FPS.

[Open the playback notebook in Google Colab](https://colab.research.google.com/github/hafizadha/Z01-Colab-Playback/blob/main/Z01_Colab_Playback.ipynb)

1. Select a GPU runtime in Colab.
2. Run the setup cell. It fetches this public repository and installs the pinned simulator dependencies.
3. Select **POLICY** in the playback cell, run it, and upload the matching Zlab PPO checkpoint (`.pt`). Use `RobustWalk2` for Stage 2 walking checkpoints or `BallSprint` for checkpoints trained with `train_walking.sh --ball-sprint`.

Walking rollouts start standing, accelerate to **TARGET_SPEED**, cruise, brake, and stand again. BallSprint spawns a ball 4–11 m away and computes steering and braking commands from its simulated position. It runs for up to 16 seconds and ends early if the robot stops near the ball or falls.

The repository contains the playback notebook, inference script, Z01 meshes, and environment configurations. The Z01 assets and environments are based on [Zvalley Robotics' Zlab Gym](https://github.com/Zvalley-Robotics/Zlab_Gym).
