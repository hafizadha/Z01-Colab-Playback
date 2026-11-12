"""Playback environment for the Zlab Gym BallSprint policy."""

from dataclasses import dataclass
import math

import mujoco
import torch
from mjlab.entity import EntityCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.utils.lab_api.math import wrap_to_pi

from src.tasks.velocity.config.z01_std_add_head.running import robust_stride_env_cfg


BALL_RADIUS = 0.11
STOP_DISTANCE = 0.65
MAX_SPRINT_SPEED = 3.0


def ball_spec():
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="soccer_ball")
    body.add_freejoint(name="ball_free_joint")
    body.add_geom(
        name="ball_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=(BALL_RADIUS, 0.0, 0.0),
        mass=0.4,
        rgba=(0.2, 0.6, 0.9, 1.0),
        condim=6,
        friction=(0.8, 0.005, 0.004),
        solref=(0.02, 1.0),
        solimp=(0.9, 0.95, 0.001, 0.5, 2.0),
        priority=1,
    )
    return spec


def ball_offset(env):
    ball = env.scene["ball"].data
    robot = env.scene["robot"].data
    return ball.root_link_pos_w[:, :2] - robot.root_link_pos_w[:, :2]


class BallApproachCommand(CommandTerm):
    """Use the existing forward and yaw command inputs to approach the ball."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._command = torch.zeros((self.num_envs, 3), device=self.device)
        self._speed = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self):
        return self._command

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids):
        self._command[env_ids] = 0.0
        self._speed[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(
            self.cfg.min_speed, self.cfg.max_speed
        )

    def _update_command(self):
        delta = ball_offset(self._env)
        distance = delta.norm(dim=-1)
        robot = self._env.scene["robot"].data
        error = wrap_to_pi(torch.atan2(delta[:, 1], delta[:, 0]) - robot.heading_w)

        speed = (1.2 * (distance - STOP_DISTANCE)).clamp_min(0.0).minimum(self._speed)
        self._command[:, 0] = speed * error.cos().clamp_min(0.0).square()
        self._command[:, 2] = (1.5 * error).clamp(-0.85, 0.85) * (distance > STOP_DISTANCE)


@dataclass(kw_only=True)
class BallApproachCommandCfg(CommandTermCfg):
    min_speed: float = 2.1
    max_speed: float = MAX_SPRINT_SPEED

    def build(self, env):
        return BallApproachCommand(self, env)


def reset_sprint_ball(env, env_ids):
    ball = env.scene["ball"]
    state = ball.data.default_root_state[env_ids].clone()
    random = torch.rand((len(env_ids), 3), device=env.device)
    distance = 4.0 + 7.0 * random[:, 0]
    # Sample most targets ahead and the rest from all directions.
    angle_range = torch.where(random[:, 1] < 2 / 3, math.pi / 3, math.pi)
    angle = (2.0 * random[:, 2] - 1.0) * angle_range
    state[:, :2] = torch.stack((distance * angle.cos(), distance * angle.sin()), dim=-1)
    state[:, :3] += env.scene.env_origins[env_ids]
    state[:, 7:] = 0.0
    ball.write_root_state_to_sim(state, env_ids)


def ball_reached(env):
    distance = ball_offset(env).norm(dim=-1)
    speed = env.scene["robot"].data.root_link_lin_vel_w[:, :2].norm(dim=-1)
    return (distance >= 0.45) & (distance <= 0.90) & (speed < 0.25)


def ball_sprint_env_cfg(play=False):
    cfg = robust_stride_env_cfg(2, play=play)
    cfg.episode_length_s = 16.0
    cfg.viewer.distance = 5.0
    cfg.commands["twist"] = BallApproachCommandCfg(
        resampling_time_range=(100.0, 100.0),
        min_speed=MAX_SPRINT_SPEED if play else 2.1,
    )
    cfg.scene.entities["ball"] = EntityCfg(
        spec_fn=ball_spec,
        init_state=EntityCfg.InitialStateCfg(pos=(5.0, 0.0, BALL_RADIUS)),
    )
    cfg.events["reset_ball"] = EventTermCfg(func=reset_sprint_ball, mode="reset")
    cfg.terminations["ball_reached"] = TerminationTermCfg(func=ball_reached)
    cfg.rewards.pop("walking_stride")
    return cfg
