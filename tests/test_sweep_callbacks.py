import numpy as np
import pytest

from owm.baselines.rl import sweep_callbacks
from owm.baselines.rl.sweep_callbacks import EvalReportCallback


class _FakeModel:
    """Stands in for the SB3 model: no normalization, a fixed no-op action."""

    def get_vec_normalize_env(self):
        return None

    def predict(self, obs, deterministic=True):
        return np.zeros(len(obs), dtype=int), None


class _FakeVecEnv:
    """One env; each reset() begins the next scripted episode's info sequence."""

    def __init__(self, episodes: list[list[dict]]):
        self.num_envs = 1
        self._episodes = episodes
        self._episode_index = -1
        self._step_index = 0

    def seed(self, seed):
        pass

    def reset(self):
        self._episode_index += 1
        self._step_index = 0
        return np.zeros((1, 1))

    def step(self, actions):
        info = self._episodes[self._episode_index][self._step_index]
        self._step_index += 1
        done = self._step_index == len(self._episodes[self._episode_index])
        return np.zeros((1, 1)), np.array([1.0]), np.array([done]), [info]


def _info(pos_m, vel_mps=1.0, att_rad=1.0, rate_radps=1.0, success=False):
    return {
        "goal_error_true": {
            "pos_m": pos_m, "vel_mps": vel_mps, "att_rad": att_rad, "rate_radps": rate_radps,
        },
        "success": success,
    }


def _callback(**overrides):
    kwargs = dict(run_dir="unused", every_steps=1, episodes=1, final_episodes=1, seed=0)
    kwargs.update(overrides)
    callback = EvalReportCallback(**kwargs)
    callback.model = _FakeModel()
    return callback


def test_rollout_tracks_episode_minimum_separately_from_final():
    callback = _callback()
    # Closest approach (pos_m=1.0) happens mid-episode; the episode ends
    # farther out (pos_m=5.0), so final and minimum must differ.
    venv = _FakeVecEnv([[_info(pos_m=1.0), _info(pos_m=5.0)]])

    _, _, finals, mins, _ = callback._rollout(venv, vecnorm=None, first_episode=0)

    assert finals[0]["pos_m"] == 5.0
    assert mins[0]["pos_m"] == 1.0


def test_rollout_disables_diagnostics_when_goal_error_true_missing(capsys):
    callback = _callback()
    venv = _FakeVecEnv([[{"success": False}, {"success": True}]])

    _, _, finals, mins, _ = callback._rollout(venv, vecnorm=None, first_episode=0)

    assert finals == [None]
    assert mins == [None]
    assert "WARNING" in capsys.readouterr().out


def test_a_safe_episode_scores_its_closest_approach():
    callback = _callback()
    venv = _FakeVecEnv([[_info(pos_m=100.0), _info(pos_m=4.0), _info(pos_m=9.0)]])

    *_, closures = callback._rollout(venv, vecnorm=None, first_episode=0)

    assert closures[0] == 4.0


@pytest.mark.parametrize("outcome", ["collision", "escaped"])
def test_an_unsafe_episode_scores_its_start_not_its_closest_approach(outcome):
    # The port sits on the hull, so a policy that dives at the station reaches
    # a tiny minimum range and then hits it. Scoring the minimum would make
    # that the best trial in the sweep.
    callback = _callback()
    end = _info(pos_m=0.2)
    end[outcome] = True
    venv = _FakeVecEnv([[_info(pos_m=100.0), _info(pos_m=0.5), end]])

    *_, closures = callback._rollout(venv, vecnorm=None, first_episode=0)

    assert closures[0] == 100.0


def test_rollout_warns_only_once_across_calls(capsys):
    callback = _callback()
    venv = _FakeVecEnv([[{"success": True}], [{"success": True}]])

    callback._rollout(venv, vecnorm=None, first_episode=0)
    callback._rollout(venv, vecnorm=None, first_episode=1)

    assert capsys.readouterr().out.count("WARNING") == 1


def test_report_logs_eval_diagnostics_alongside_existing_metrics(monkeypatch):
    logged: list[dict] = []
    monkeypatch.setattr(sweep_callbacks.wandb, "log", lambda payload: logged.append(payload))

    callback = _callback(episodes=2, final_episodes=2)
    callback.num_timesteps = 42
    episodes = [
        [_info(pos_m=1.0), _info(pos_m=3.0)],
        [_info(pos_m=2.0, success=True)],
    ]
    callback._eval_env = lambda: _FakeVecEnv(episodes)

    callback._report(episodes=2, final=False)

    payload = logged[0]
    # The objective and success rate keep their existing definitions.
    assert payload[sweep_callbacks.OBJECTIVE] == pytest.approx(1.5)
    assert payload["sweep/eval_success"] == pytest.approx(0.5)
    # New diagnostics: mean final and mean minimum pos_m across episodes.
    assert payload["sweep/eval_final_pos_m"] == pytest.approx((3.0 + 2.0) / 2)
    assert payload["sweep/eval_min_pos_m"] == pytest.approx((1.0 + 2.0) / 2)
    assert "sweep/final_mean_return" not in payload


def test_report_omits_diagnostics_when_goal_error_true_missing(monkeypatch):
    logged: list[dict] = []
    monkeypatch.setattr(sweep_callbacks.wandb, "log", lambda payload: logged.append(payload))

    callback = _callback()
    callback.num_timesteps = 0
    callback._eval_env = lambda: _FakeVecEnv([[{"success": True}]])

    callback._report(episodes=1, final=True)

    payload = logged[0]
    assert sweep_callbacks.OBJECTIVE in payload
    assert payload["sweep/eval_success"] == 1.0
    assert not any(key.startswith("sweep/eval_final_") or key.startswith("sweep/eval_min_") for key in payload)
