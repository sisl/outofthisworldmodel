"""Hold each action for k environment steps.

The physics is unchanged: the env still integrates at `cfg.dt`. What changes is
how often the POLICY decides, and therefore what one transition in the replay
buffer or rollout spans. At dt=0.05 s and max_steps=7200 an episode is 7,200
decisions; at k=20 it is 360, over the same six minutes of flight.

TWO THINGS THIS BUYS, both measured problems rather than general tidiness.

CREDIT REACHES THE GOAL. A terminal reward 7,200 steps out is worth
gamma**7200 at the start of the episode -- 2e-16 at gamma 0.995, which is zero
in float32, and 7.4e-4 at 0.999. The dock bonus is not small in that regime,
it is absent, and a policy that converges to holding station is optimising the
reward it can actually see. At k=20 the same bonus is discounted by
gamma**360, or 0.70 at gamma 0.999.

EXPLORATION BECOMES TEMPORALLY CORRELATED. Closing the range pays the velocity
term immediately and earns the position term back only as range falls, so a
1 m/s commitment has to persist ~326 steps before it turns a profit. SAC and
PPO both draw an independent action per decision, so at k=1 the thrust
sequence is a random walk and a direction essentially never survives 326
draws. At k=20 that same commitment is 16 decisions.

WHAT IT COSTS. Control resolution: at k=20 the agent cannot correct faster
than once a second, which is free out at 200 m and is not free inside a 0.1 m
dock gate with a 0.05 m/s velocity limit. This is the trade the knob exposes,
and it is why k is worth sweeping rather than fixing.

Rewards over the held steps are SUMMED, so an episode's return is unchanged by
k and returns stay comparable across settings.
"""

from __future__ import annotations

import gymnasium as gym

# Per-step outcome flags that must survive the collapse. Taking the last
# step's info alone would drop a collision that happened partway through a
# hold and did not end the episode -- which is exactly the case a soft
# keep-out zone creates, and exactly the number the docking metrics count.
_STICKY_FLAGS = ("collision", "success", "escaped")


class ActionRepeat(gym.Wrapper):
    """Apply each action `repeat` times, summing reward over the held steps.

    Stops early when the episode ends, so a hold never runs past a terminal
    step. The observation and the non-flag info are the LAST inner step's --
    the state the policy actually has to act from next.
    """

    def __init__(self, env: gym.Env, repeat: int):
        super().__init__(env)
        if repeat < 1:
            raise ValueError(f"action repeat must be >= 1, got {repeat}")
        self._repeat = int(repeat)

    def step(self, action):
        total = 0.0
        sticky: dict[str, bool] = {}
        obs = reward = terminated = truncated = info = None
        for _ in range(self._repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total += float(reward)
            for flag in _STICKY_FLAGS:
                if info.get(flag):
                    sticky[flag] = True
            if terminated or truncated:
                break
        info = {**info, **sticky}
        return obs, total, terminated, truncated, info
