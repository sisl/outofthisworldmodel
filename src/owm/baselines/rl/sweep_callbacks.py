"""Callbacks a sweep trial needs: objective reports and a wall-clock bound."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

from owm.baselines.rl.metrics import GOAL_ERROR_KEYS
from owm.envs.factory import (
    DEFAULT_ENV_NAME,
    env_conf_dict,
    make_vec_env,
    task_config_from_yaml,
)

# The sweep's objective. Bayes reads its last value, hyperband reads the
# series, so the same key carries both the periodic and the final report.
OBJECTIVE = "sweep/eval_mean_return"
STEP_METRIC = "sweep/global_step"

# The alternative objective, for a sweep whose search space includes the
# reward weights themselves. Mean return is measured in the trial's own reward
# units, so a trial that was handed a bigger dock bonus or a smaller collision
# penalty scores higher without behaving any better -- the sweep would rank
# reward configs rather than policies. This one is in metres and reads the
# same for every trial.
#
# It is closure, not raw minimum range: an episode that ends in a collision or
# an escape contributes the distance it STARTED at, so it earns no credit for
# how close it got. Raw minimum range would be gamed by flying into the hull,
# since the port sits on it -- and given the last sweep crowned the policy
# that most reliably does nothing, an objective with a cheap degenerate
# maximum is the specific failure worth designing out.
CLOSURE_OBJECTIVE = "sweep/eval_safe_min_pos_m"
OBJECTIVES = {OBJECTIVE: "maximize", CLOSURE_OBJECTIVE: "minimize"}

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
        env_name: str = DEFAULT_ENV_NAME,
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
        # The record is the task config alone; which env of the suite validates
        # and flies it comes from the trial's own composed config, the same
        # place training reads it from.
        self._env_name = env_name
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
        # Warned once, like DockingMetricsCallback: an env with no
        # goal_error_true still reports the objective, just without the
        # diagnostics that key feeds.
        self._goal_error_missing_warned = False

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
            mean_return, success_rate, final_errors, min_errors, closure = self._evaluate(
                episodes
            )
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
        # Additive diagnostics on top of the objective: how close the policy's
        # deterministic eval actually got to the goal, not just whether it
        # counted as a dock. None when the env has no goal_error_true.
        if final_errors is not None:
            payload["sweep/eval_final_pos_m"] = final_errors["pos_m"]
            payload["sweep/eval_final_vel_mps"] = final_errors["vel_mps"]
            payload["sweep/eval_final_att_rad"] = final_errors["att_rad"]
            payload["sweep/eval_final_rate_radps"] = final_errors["rate_radps"]
        if min_errors is not None:
            payload["sweep/eval_min_pos_m"] = min_errors["pos_m"]
            payload["sweep/eval_min_vel_mps"] = min_errors["vel_mps"]
            payload["sweep/eval_min_att_rad"] = min_errors["att_rad"]
            payload["sweep/eval_min_rate_radps"] = min_errors["rate_radps"]
        # Logged on every report whatever the spec is optimising, so the two
        # objectives can be read against each other after the fact.
        if closure is not None:
            payload[CLOSURE_OBJECTIVE] = closure
        if final:
            # Same number under a name no intermediate report ever writes, so
            # the finished-trial value can be read back without guessing which
            # history row was the last one.
            payload["sweep/final_mean_return"] = mean_return
            payload["sweep/final_success"] = success_rate
            if closure is not None:
                payload["sweep/final_safe_min_pos_m"] = closure
        wandb.log(payload)

    def _evaluate(
        self, episodes: int
    ) -> tuple[float, float, dict[str, float] | None, dict[str, float] | None, float | None]:
        venv = self._eval_env()
        # The policy sees normalized observations in training, so evaluating it
        # on raw ones would measure a transform mismatch, not the policy.
        # normalize_obs is a pure transform: eval never moves the statistics.
        vecnorm = self.model.get_vec_normalize_env()
        returns: list[float] = []
        successes: list[bool] = []
        finals: list[dict[str, float]] = []
        mins: list[dict[str, float]] = []
        closures: list[float] = []
        # Episodes run a vec-width at a time. One at a time left the training
        # workers idle and made a report cost more than the training it was
        # reporting on.
        for first in range(0, episodes, venv.num_envs):
            batch = min(venv.num_envs, episodes - first)
            batch_returns, batch_successes, batch_finals, batch_mins, batch_closures = (
                self._rollout(venv, vecnorm, first)
            )
            returns.extend(batch_returns[:batch])
            successes.extend(batch_successes[:batch])
            finals.extend(error for error in batch_finals[:batch] if error is not None)
            mins.extend(error for error in batch_mins[:batch] if error is not None)
            closures.extend(value for value in batch_closures[:batch] if value is not None)
        mean_return = float(np.mean(returns))
        success_rate = sum(successes) / episodes
        closure = float(np.mean(closures)) if closures else None
        return (
            mean_return,
            success_rate,
            self._aggregate(finals),
            self._aggregate(mins),
            closure,
        )

    @staticmethod
    def _aggregate(records: list[dict[str, float]]) -> dict[str, float] | None:
        # None rather than a dict of NaNs: an env with no goal_error_true has
        # no records at all, and that should read as "not available", not as
        # a numeric zero/NaN a dashboard would plot.
        if not records:
            return None
        return {key: float(np.mean([record[key] for record in records])) for key in GOAL_ERROR_KEYS}

    def _rollout(
        self, venv, vecnorm, first_episode: int
    ) -> tuple[list[float], list[bool], list[dict[str, float] | None], list[dict[str, float] | None]]:
        """Run one episode per env, seeded as episodes first_episode + i."""
        # VecEnv.seed hands env i seed+i at the next reset, which is exactly the
        # numbering an episode-at-a-time loop produced, so a trial's score does
        # not depend on how wide its eval env happens to be.
        venv.seed(self._seed + first_episode)
        obs = venv.reset()
        width = venv.num_envs
        ep_return = np.zeros(width, dtype=np.float64)
        success = np.zeros(width, dtype=bool)
        mins = [self._fresh_minima() for _ in range(width)]
        finals: list[dict[str, float] | None] = [None] * width
        # For the closure objective: where each episode began, and whether it
        # ended somewhere that should void its closure credit.
        starts: list[float | None] = [None] * width
        unsafe = np.zeros(width, dtype=bool)
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
            for index in np.flatnonzero(live):
                error = self._goal_error(infos[index])
                if error is not None:
                    if starts[index] is None:
                        starts[index] = error["pos_m"]
                    for key in GOAL_ERROR_KEYS:
                        if error[key] < mins[index][key]:
                            mins[index][key] = error[key]
            for index in np.flatnonzero(live & dones):
                success[index] = bool(infos[index].get("success"))
                finals[index] = self._goal_error(infos[index])
                unsafe[index] = bool(infos[index].get("collision")) or bool(
                    infos[index].get("escaped")
                )
            live &= ~dones
            if self._max_episode_steps is not None and steps >= self._max_episode_steps:
                break
        # A slot's minimum only means something once we know that slot ever
        # had a goal_error_true to track; otherwise it is still the untouched
        # all-inf sentinel.
        episode_mins = [mins[i] if finals[i] is not None else None for i in range(width)]
        # An unsafe episode scores its start, so closing on the hull and
        # hitting it is worth exactly as much as never having left.
        closures = [
            None
            if finals[i] is None or starts[i] is None
            else (starts[i] if unsafe[i] else mins[i]["pos_m"])
            for i in range(width)
        ]
        return ep_return.tolist(), success.tolist(), finals, episode_mins, closures

    def _goal_error(self, info: dict) -> dict[str, float] | None:
        if "goal_error_true" not in info:
            if not self._goal_error_missing_warned:
                print(
                    "[sweep] WARNING infos have no goal_error_true (not an "
                    "owm-envs docking env?); skipping eval goal-error "
                    "diagnostics"
                )
                self._goal_error_missing_warned = True
            return None
        return info["goal_error_true"]

    @staticmethod
    def _fresh_minima() -> dict[str, float]:
        return {key: float("inf") for key in GOAL_ERROR_KEYS}

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
            env_conf = env_conf_dict(
                task_config_from_yaml(self._env_name, self._env_record)
            )
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
