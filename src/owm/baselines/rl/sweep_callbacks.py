"""Callbacks a sweep trial needs: objective reports and a wall-clock bound."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import wandb
from owm_envs.envs.iss.config import ISSConfig
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

from owm.envs.factory import make_vec_env

# The sweep's objective. Bayes reads its last value, hyperband reads the
# series, so the same key carries both the periodic and the final report.
OBJECTIVE = "sweep/eval_mean_return"
STEP_METRIC = "sweep/global_step"

# Widest the eval env gets. The periodic report is 5 episodes, so 5 envs run it
# in one round and the 20-episode final one in four, while the training workers
# are idle anyway.
EVAL_ENVS = 5


class EvalReportCallback(BaseCallback):
    """Report deterministic eval return on a cadence, then once at the end.

    The sweep is scored on how the policy actually docks, which is what a
    deterministic rollout measures — SB3's rollout statistics are collected
    under exploration noise and its losses say nothing comparable across
    hyperparameters. Reporting periodically rather than only at the end is
    what gives hyperband something to band on.
    """

    def __init__(
        self,
        run_dir: Path,
        every_steps: int,
        episodes: int,
        final_episodes: int,
        seed: int,
        max_episode_steps: int | None = None,
        vec: str = "subproc",
        obs_mode: str = "vector",
        resnet: dict | None = None,
    ):
        super().__init__()
        # Caught at registration, i.e. at launch: an eval of no episodes only
        # fails hours later, in np.mean of an empty list.
        if episodes < 1 or final_episodes < 1:
            raise ValueError(
                f"eval episodes must be >= 1, got {episodes} periodic and "
                f"{final_episodes} final"
            )
        if every_steps < 1:
            raise ValueError(f"every_steps must be >= 1, got {every_steps}")
        # The env is taken from the run's own env_config.yaml rather than
        # re-resolved from the hydra config: training writes that file as the
        # record of what it actually trained on, and an environments=
        # from_dataset ref can move between two resolutions — which would score
        # the trial on dynamics it never saw.
        self._env_record = Path(run_dir) / "env_config.yaml"
        self._every = every_steps
        self._episodes = episodes
        self._final_episodes = final_episodes
        self._seed = seed
        self._max_episode_steps = max_episode_steps
        self._vec = vec
        # The eval env must observe the way training does or the policy is fed
        # an observation of the wrong width, and VecNormalize's statistics do
        # not describe it either.
        self._obs_mode = obs_mode
        self._resnet = resnet
        self._next_at = every_steps
        self._env: VecEnv | None = None

    def _on_training_start(self) -> None:
        # A resumed run boots at num_timesteps far past every_steps, which the
        # first _on_step would read as an overdue report; the cadence starts
        # from wherever training actually resumes.
        self._next_at = self.num_timesteps + self._every
        # wandb.init(sync_tensorboard=True) already owns the implicit step
        # axis, which silently drops any step= we pass to wandb.log; sweep/*
        # gets its own step metric instead.
        wandb.define_metric(STEP_METRIC)
        wandb.define_metric("sweep/*", step_metric=STEP_METRIC)

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_at:
            # Schedule from the step actually reached, not just += every: on a
            # resumed run num_timesteps starts far past every_steps, and += would
            # fire on every subsequent step until it caught up.
            self._next_at = self.num_timesteps + self._every
            self._report(self._episodes, final=False)
        return True

    def _on_training_end(self) -> None:
        # Runs even when another callback ended training early, so a trial that
        # hit its time bound still reports an objective instead of dropping out
        # of the sweep's ranking entirely. The env has to be closed even when
        # that last report raises, or its worker outlives the trial.
        try:
            self._report(self._final_episodes, final=True)
        finally:
            if self._env is not None:
                self._env.close()
                self._env = None

    def _report(self, episodes: int, final: bool) -> None:
        try:
            mean_return, success_rate = self._evaluate(episodes)
        finally:
            # In a finally so a failed eval does not strand the pool either:
            # under vector_resnet those are GPU contexts, and the trial has
            # hours left to run without them.
            self._release_env()
        payload = {
            OBJECTIVE: mean_return,
            "sweep/eval_success": success_rate,
            STEP_METRIC: self.num_timesteps,
        }
        if final:
            # Same number under a name no intermediate report ever writes, so
            # the finished-trial value can be read back without guessing which
            # history row was the last one.
            payload["sweep/final_mean_return"] = mean_return
            payload["sweep/final_success"] = success_rate
        wandb.log(payload)

    def _evaluate(self, episodes: int) -> tuple[float, float]:
        venv = self._eval_env()
        # The policy sees normalized observations in training, so evaluating it
        # on raw ones would measure a transform mismatch, not the policy.
        # normalize_obs is a pure transform: eval never moves the statistics.
        vecnorm = self.model.get_vec_normalize_env()
        returns: list[float] = []
        successes: list[bool] = []
        # Episodes run a vec-width at a time. One at a time left the training
        # workers idle and made a report cost more than the training it was
        # reporting on.
        for first in range(0, episodes, venv.num_envs):
            batch = min(venv.num_envs, episodes - first)
            batch_returns, batch_successes = self._rollout(venv, vecnorm, first)
            returns.extend(batch_returns[:batch])
            successes.extend(batch_successes[:batch])
        return float(np.mean(returns)), sum(successes) / episodes

    def _rollout(self, venv, vecnorm, first_episode: int) -> tuple[list[float], list[bool]]:
        """Run one episode per env, seeded as episodes first_episode + i."""
        # VecEnv.seed hands env i seed+i at the next reset, which is exactly the
        # numbering an episode-at-a-time loop produced, so a trial's score does
        # not depend on how wide its eval env happens to be.
        venv.seed(self._seed + first_episode)
        obs = venv.reset()
        width = venv.num_envs
        ep_return = np.zeros(width, dtype=np.float64)
        success = np.zeros(width, dtype=bool)
        # A vec env auto-resets a finished env, so anything it reports after
        # that belongs to an episode nobody asked for.
        live = np.ones(width, dtype=bool)
        steps = 0
        while live.any():
            norm = vecnorm.normalize_obs(obs) if vecnorm is not None else obs
            actions, _ = self.model.predict(norm, deterministic=True)
            obs, rewards, dones, infos = venv.step(actions)
            steps += 1
            ep_return += rewards * live
            for index in np.flatnonzero(live & dones):
                success[index] = bool(infos[index].get("success"))
            live &= ~dones
            if self._max_episode_steps is not None and steps >= self._max_episode_steps:
                break
        return ep_return.tolist(), success.tolist()

    def _release_env(self) -> None:
        """Drop the eval pool between reports, for the modes it costs to keep.

        A vector_resnet eval env renders, so each of its workers holds a
        Vulkan device worth ~1.9 GB for as long as the process lives -- five of
        them, for a whole trial, to run five episodes every cadence. Rebuilding
        costs ~15 s a report, which is the cheaper side of that trade on a
        shared GPU. A vector pool costs nothing to hold, so it is held, exactly
        as it always has been.
        """
        if self._obs_mode == "vector_resnet" and self._env is not None:
            self._env.close()
            self._env = None

    def _eval_env(self) -> VecEnv:
        if self._env is None:
            # One env for the whole run: rebuilding it per report would re-pay
            # the simulator's setup cost every cadence. Built here rather than
            # at registration because training writes the record this reads.
            env_conf = ISSConfig.from_yaml(self._env_record).model_dump(mode="json")
            self._env = make_vec_env(
                env_conf,
                n_envs=min(self._episodes, EVAL_ENVS),
                seed=self._seed,
                vec=self._vec,
                obs_mode=self._obs_mode,
                resnet=self._resnet,
            )
        return self._env


class TrialTimeoutCallback(BaseCallback):
    """End training gracefully once a trial has used its wall-clock budget.

    total_timesteps bounds a trial in env steps, but not in time: a sweep can
    draw hyperparameters (large batches, many gradient steps per env step)
    that make those steps slow enough to eat the whole sweep's budget. Ending
    training rather than killing the process leaves the final eval, the final
    artifacts and the objective report intact.
    """

    def __init__(self, max_seconds: float, clock: Callable[[], float] = time.monotonic):
        super().__init__()
        if max_seconds <= 0:
            raise ValueError(f"max_seconds must be > 0, got {max_seconds}")
        self._max_seconds = max_seconds
        self._clock = clock
        # Runs from construction, not from training start: building 8 subproc
        # envs and restoring a replay buffer is trial time like any other, and
        # a trial that spent its budget there has none left to train with.
        self._deadline = clock() + max_seconds

    def _on_step(self) -> bool:
        if self._clock() < self._deadline:
            return True
        print(
            f"[sweep] trial exceeded {self._max_seconds:.0f}s at "
            f"{self.num_timesteps} steps; ending training"
        )
        wandb.log({"sweep/timed_out": 1, STEP_METRIC: self.num_timesteps})
        return False
