"""Periodic deterministic-episode video capture, logged to wandb."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback

from owm.envs.factory import iss_config, make_iss_env


class VideoEvalCallback(BaseCallback):
    def __init__(self, env_conf: dict, every_steps: int, max_frames: int, seed: int):
        super().__init__()
        self._env_conf = env_conf
        self._every = every_steps
        self._max_frames = max_frames
        self._seed = seed
        self._next_at = every_steps
        self._env: gym.Env | None = None

    def _on_training_start(self) -> None:
        # train.py's wandb.init(sync_tensorboard=True) already owns the
        # implicit step axis, which silently drops any step= we pass to
        # wandb.log; eval/* gets its own step metric instead.
        wandb.define_metric("eval/global_step")
        wandb.define_metric("eval/*", step_metric="eval/global_step")

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_at:
            # Schedule from the step actually reached, not just += every: on
            # a resumed run num_timesteps starts far past every_steps, and
            # += would fire on every subsequent step until it caught up.
            self._next_at = self.num_timesteps + self._every
            self._record()
        return True

    def _on_training_end(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def _record(self) -> None:
        if self._env is None:
            # Lazy: rendering pulls in pygfx/GL, and only if videos are on.
            self._env = make_iss_env(iss_config(self._env_conf), seed=self._seed,
                                     render=True)
        vecnorm = self.model.get_vec_normalize_env()
        frames: list[np.ndarray] = []
        ep_return, success = 0.0, False
        obs, _ = self._env.reset(seed=self._seed)
        done = False
        while not done and len(frames) < self._max_frames:
            norm = vecnorm.normalize_obs(obs) if vecnorm is not None else obs
            action, _ = self.model.predict(norm, deterministic=True)
            obs, reward, term, trunc, info = self._env.step(action)
            ep_return += float(reward)
            frames.append(self._env.render())
            success = success or bool(info.get("success"))
            done = term or trunc
        video = np.stack(frames).transpose(0, 3, 1, 2)  # (T, C, H, W)
        fps = self._env.metadata.get("render_fps", 20)
        wandb.log({
            "eval/video": wandb.Video(video, fps=fps, format="mp4"),
            "eval/return": ep_return,
            "eval/success": float(success),
            "eval/global_step": self.num_timesteps,
        })
