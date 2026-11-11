# Z01 Colab Playback

Play a Zlab Z01 walking policy in MuJoCo and watch the rollout inline at 30 FPS.

[Open the playback notebook in Google Colab](https://colab.research.google.com/github/hafizadha/Z01-Colab-Playback/blob/main/Z01_Colab_Playback.ipynb)

1. Select a GPU runtime in Colab.
2. Run the setup cell. It fetches this public repository and installs the pinned simulator dependencies.
3. Run the playback cell and upload one Zlab PPO checkpoint (`.pt`). Leave **POLICY** on `RobustWalk2` for Stage 2 checkpoints such as `model_8000.pt`.

The rollout starts standing, accelerates to the selected speed, cruises, brakes, and stands again. It uses the walking task's simulated motor delays and disturbance settings. The notebook displays a recorded video with pause and seek controls.

The repository contains the playback notebook, inference script, Z01 meshes, and walking environment configuration. The Z01 assets and environment are based on [Zvalley Robotics' Zlab Gym](https://github.com/Zvalley-Robotics/Zlab_Gym).
