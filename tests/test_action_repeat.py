import gymnasium as gym
import numpy as np
import pytest

from owm.baselines.rl.action_repeat import ActionRepeat


class _CountingEnv(gym.Env):
    """Rewards 1 per step, ends after `ends_after` steps, echoes the action."""

    def __init__(self, ends_after: int | None = None, terminate: bool = True, flags=()):
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        self._ends_after = ends_after
        self._terminate = terminate
        self._flags = list(flags)
        self.actions: list[float] = []
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        self.steps = 0
        self.actions = []
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        self.actions.append(float(np.asarray(action).ravel()[0]))
        done = self._ends_after is not None and self.steps >= self._ends_after
        # A flag raised on this step, e.g. a soft collision partway through.
        info = {flag: True for flag, at in self._flags if at == self.steps}
        return (
            np.full(1, self.steps, dtype=np.float32),
            1.0,
            done and self._terminate,
            done and not self._terminate,
            info,
        )


def test_a_decision_spans_repeat_env_steps():
    env = ActionRepeat(_CountingEnv(), repeat=20)
    env.reset()

    env.step(np.array([0.5], dtype=np.float32))

    assert env.unwrapped.steps == 20


def test_the_same_action_is_held_for_every_held_step():
    env = ActionRepeat(_CountingEnv(), repeat=5)
    env.reset()

    env.step(np.array([0.25], dtype=np.float32))

    assert env.unwrapped.actions == [0.25] * 5


def test_reward_is_summed_so_the_return_does_not_depend_on_the_cadence():
    # The whole point of summing rather than averaging: an episode is worth the
    # same at every repeat, so returns stay comparable across settings.
    plain = _CountingEnv(ends_after=20)
    plain.reset()
    plain_total = sum(plain.step(np.zeros(1, dtype=np.float32))[1] for _ in range(20))

    env = ActionRepeat(_CountingEnv(ends_after=20), repeat=20)
    env.reset()
    _, repeated_total, _, _, _ = env.step(np.zeros(1, dtype=np.float32))

    assert repeated_total == plain_total == 20.0


@pytest.mark.parametrize("terminate", [True, False])
def test_a_hold_stops_at_the_end_of_the_episode(terminate):
    # Running past a terminal step would collect transitions from an episode
    # nobody asked for and charge them to this one.
    env = ActionRepeat(_CountingEnv(ends_after=3, terminate=terminate), repeat=20)
    env.reset()

    _, reward, terminated, truncated, _ = env.step(np.zeros(1, dtype=np.float32))

    assert env.unwrapped.steps == 3
    assert reward == 3.0
    assert terminated is terminate
    assert truncated is not terminate


def test_an_outcome_raised_partway_through_a_hold_survives_the_collapse():
    # A soft keep-out zone raises `collision` on a step that ends nothing, so
    # the last inner step's info alone would drop it -- and that flag is what
    # the docking metrics count.
    env = ActionRepeat(_CountingEnv(flags=[("collision", 2)]), repeat=5)
    env.reset()

    _, _, _, _, info = env.step(np.zeros(1, dtype=np.float32))

    assert info["collision"] is True


def test_repeat_below_one_is_refused():
    with pytest.raises(ValueError, match="action repeat must be >= 1"):
        ActionRepeat(_CountingEnv(), repeat=0)
