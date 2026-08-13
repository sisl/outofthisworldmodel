"""Per-episode docking telemetry to wandb: outcome and true-state minima.

Reward alone can't tell a policy that is docking more often from one that is
merely colliding less; this logs, per finished episode, why it ended and how
close it actually got to the goal.
"""

from __future__ import annotations

from collections import Counter, deque

import wandb
from stable_baselines3.common.callbacks import BaseCallback

GOAL_ERROR_KEYS = ("pos_m", "vel_mps", "att_rad", "rate_radps")
OUTCOMES = ("docked", "collision", "escaped", "truncated")


class DockingMetricsCallback(BaseCallback):
    """Log each finished episode's outcome and closest true-state approach.

    Tracks, per vec-env slot, the running minimum of each goal_error_true
    component since that slot's episode began. When an episode ends
    (locals["dones"][i]), the terminal info in locals["infos"][i] — which SB3
    vec envs deliver as the finished episode's info on the auto-reset step —
    is classified into an outcome and logged with that episode's minima.
    """

    def __init__(self, window: int = 100):
        super().__init__()
        self._window = window
        self._episode_count = 0
        self._mins: dict[int, dict[str, float]] = {}
        self._starts: dict[int, float | None] = {}
        self._outcomes: deque[str] = deque(maxlen=window)
        self._counts: Counter[str] = Counter()
        self._disabled = False

    def _on_training_start(self) -> None:
        # wandb.init(sync_tensorboard=True) already owns the implicit step
        # axis, which silently drops any step= we pass to wandb.log;
        # docking/* gets its own step metric instead, mirroring val_episodes.py.
        wandb.define_metric("docking/episodes")
        wandb.define_metric("docking/*", step_metric="docking/episodes")

    def _on_step(self) -> bool:
        if self._disabled:
            return True

        infos = self.locals["infos"]
        dones = self.locals["dones"]

        for i, info in enumerate(infos):
            if "goal_error_true" not in info:
                print(
                    "[docking] WARNING infos have no goal_error_true "
                    "(not an owm-envs docking env?); disabling "
                    "DockingMetricsCallback"
                )
                self._disabled = True
                return True

            mins = self._mins.setdefault(i, self._fresh_minima())
            for key in GOAL_ERROR_KEYS:
                value = info["goal_error_true"][key]
                if value < mins[key]:
                    mins[key] = value

            # The range this slot's episode opened at, kept so closest approach
            # can be reported as a fraction of it. Recorded on the episode's
            # first step rather than at reset, which this callback does not see:
            # one 50 ms step from rest moves the chaser a negligible distance
            # against a 100-500 m start.
            if self._starts.get(i) is None:
                self._starts[i] = info["goal_error_true"]["pos_m"]

            if dones[i]:
                self._episode_count += 1
                outcome = self._classify(info)
                self._outcomes.append(outcome)
                self._counts[outcome] += 1

                start = self._starts[i]
                payload = {
                    "docking/episodes": self._episode_count,
                    "docking/docked_rate": self._rate("docked"),
                    "docking/collision_rate": self._rate("collision"),
                    "docking/escaped_rate": self._rate("escaped"),
                    "docking/truncated_rate": self._rate("truncated"),
                    # Cumulative, so a first dock among hundreds of episodes
                    # reads as 1 instead of rounding to a 0.01 rate.
                    "docking/docked_count": self._counts["docked"],
                    "docking/collision_count": self._counts["collision"],
                    "docking/escaped_count": self._counts["escaped"],
                    "docking/truncated_count": self._counts["truncated"],
                    "docking/ep_min_pos_m": mins["pos_m"],
                    "docking/ep_min_vel_mps": mins["vel_mps"],
                    "docking/ep_min_att_rad": mins["att_rad"],
                    "docking/ep_min_rate_radps": mins["rate_radps"],
                    "docking/ep_start_pos_m": start,
                    # ep_min_pos_m as a fraction of the range the episode
                    # opened at -- the approach signal it cannot carry on its
                    # own. Episodes start uniformly over a 100-500 m shell,
                    # whose 115 m standard deviation is the whole spread seen
                    # in ep_min_pos_m, so that series plots where an episode
                    # BEGAN and a policy learning to close would not move it.
                    # Dividing by the start range takes that out.
                    #
                    # Range REMAINING, not range closed, so it falls as the
                    # policy improves exactly as its ep_min_* siblings do: 1.0
                    # is a policy that never got closer than it started, and 0
                    # is one that reached the port.
                    "docking/ep_min_pos_frac": mins["pos_m"] / start,
                }
                if "dock_port" in info:
                    payload["docking/port_index"] = info["dock_port_index"]
                wandb.log(payload)

                self._mins[i] = self._fresh_minima()
                self._starts[i] = None

        return True

    @staticmethod
    def _fresh_minima() -> dict[str, float]:
        return {key: float("inf") for key in GOAL_ERROR_KEYS}

    @staticmethod
    def _classify(info: dict) -> str:
        if info.get("success"):
            return "docked"
        if info.get("collision"):
            return "collision"
        if info.get("escaped"):
            return "escaped"
        return "truncated"

    def _rate(self, outcome: str) -> float:
        if not self._outcomes:
            return 0.0
        return sum(1 for o in self._outcomes if o == outcome) / len(self._outcomes)
