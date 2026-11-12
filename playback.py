import argparse
from functools import partial
import math
from pathlib import Path

import imageio_ffmpeg
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends

from rsl_rl.models import MLPModel
from src.tasks.ball_sprint import ball_sprint_env_cfg
from src.tasks.velocity.config.z01_std_add_head.running import (
    balance_recovery_env_cfg,
    robust_stride_env_cfg,
    stride_env_cfg,
    walking_env_cfg,
)

PLAYBACK_CONFIGS = {
    "Walk": walking_env_cfg,
    "FastWalk": partial(walking_env_cfg, max_speed=1.6),
    "StrideWalk": stride_env_cfg,
    "LongStrideWalk": partial(stride_env_cfg, longer=True),
    "RobustWalk1": partial(robust_stride_env_cfg, stage=1),
    "RobustWalk2": partial(robust_stride_env_cfg, stage=2),
    "RobustWalk3": partial(robust_stride_env_cfg, stage=3),
    "BalanceRecovery": balance_recovery_env_cfg,
    "BallSprint": ball_sprint_env_cfg,
}


@torch.inference_mode()
def record(checkpoint, task, speed, output):
    ball_sprint = task == "BallSprint"
    if not ball_sprint and (not math.isfinite(speed) or not 0.0 <= speed <= 1.8):
        raise ValueError("Target speed must be between 0 and 1.8 m/s.")
    configure_torch_backends()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading {checkpoint.name} on {device}.", flush=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "actor_state_dict" not in state:
        raise ValueError("Upload a Zlab Z01 PPO checkpoint with an actor_state_dict.")

    env_cfg = PLAYBACK_CONFIGS[task](play=True)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = 2026
    env_cfg.viewer.width = 640
    env_cfg.viewer.height = 480
    env_cfg.viewer.enable_shadows = False
    if not ball_sprint:
        command_cfg = env_cfg.commands["twist"]
        command_cfg.resampling_time_range = (100.0, 100.0)
        command_cfg.rel_standing_envs = 0.0
        command_cfg.ranges.lin_vel_x = (0.0, 0.0)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    try:
        wrapped = RslRlVecEnvWrapper(env)
        policy = MLPModel(
            wrapped.get_observations(), {"actor": ["actor"]}, "actor", wrapped.num_actions,
            hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True,
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.15, "std_type": "scalar"},
        ).to(device).eval()
        policy.load_state_dict(state["actor_state_dict"], strict=True)
        command = None if ball_sprint else env.command_manager.get_command("twist")
        dt = env.step_dt
        duration = env_cfg.episode_length_s if ball_sprint else 10.0

        fps = 30
        frame_index = 0
        writer = imageio_ffmpeg.write_frames(
            str(output), (env_cfg.viewer.width, env_cfg.viewer.height),
            fps=fps, codec="libx264", quality=8,
            output_params=["-movflags", "+faststart"],
        )
        writer.send(None)
        try:
            if ball_sprint:
                print("Recording the approach to the ball.", flush=True)
            else:
                print("Recording stand, accelerate, cruise, brake, and stand.", flush=True)
            for step in range(round(duration / dt)):
                t = step * dt
                if command is not None:
                    command[:, 0] = speed * min(1.0, max(0.0, (t - 1) / 2), max(0.0, (8 - t) / 2))
                    command[:, 1:] = 0.0
                if t >= frame_index / fps - 1e-9:
                    writer.send(env.render().tobytes())
                    frame_index += 1
                _, _, done, _ = wrapped.step(policy(wrapped.get_observations()))
                if done.item():
                    outcome = "The simulation terminated"
                    if ball_sprint:
                        termination = env.termination_manager
                        if termination.get_term("fell_over").item():
                            outcome = "The robot fell"
                        elif termination.get_term("ball_reached").item():
                            outcome = "Reached the ball and stopped"
                        elif termination.time_outs.item():
                            outcome = "The approach timed out"
                    print(f"{outcome} at {t + dt:.2f} seconds.", flush=True)
                    break
        finally:
            writer.close()
        print(f"Recorded {frame_index} frames at {fps} FPS.", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("task", choices=PLAYBACK_CONFIGS)
    parser.add_argument(
        "speed", type=float, help="Walking target speed in m/s. BallSprint sets its own speed."
    )
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    record(args.checkpoint, args.task, args.speed, args.output)
