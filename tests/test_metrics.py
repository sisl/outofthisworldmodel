import pytest

from owm.baselines.rl import metrics
from owm.baselines.rl.metrics import DockingMetricsCallback


@pytest.fixture
def logged(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(metrics.wandb, "log", lambda payload: calls.append(payload))
    monkeypatch.setattr(metrics.wandb, "define_metric", lambda *a, **k: None)
    return calls


def _info(pos_m, vel_mps=1.0, att_rad=1.0, rate_radps=1.0, success=False,
          collision=False, escaped=False, port=None):
    info = {
        "goal_error_true": {
            "pos_m": pos_m, "vel_mps": vel_mps, "att_rad": att_rad, "rate_radps": rate_radps,
        },
        "success": success,
        "collision": collision,
        "escaped": escaped,
    }
    if port is not None:
        info["dock_port"] = f"port_{port}"
        info["dock_port_index"] = port
    return info


def _step(callback, infos, dones):
    callback.locals = {"infos": infos, "dones": dones}
    assert callback._on_step() is True


def test_tracks_running_minimum_across_steps(logged):
    callback = DockingMetricsCallback(window=100)
    callback._on_training_start()

    _step(callback, [_info(pos_m=5.0)], [False])
    _step(callback, [_info(pos_m=2.0)], [False])
    _step(callback, [_info(pos_m=3.0, success=True)], [True])

    assert logged[0]["docking/ep_min_pos_m"] == 2.0


@pytest.mark.parametrize("kwargs,outcome", [
    ({"success": True}, "docked"),
    ({"collision": True}, "collision"),
    ({"escaped": True}, "escaped"),
    ({}, "truncated"),
])
def test_classifies_each_outcome(logged, kwargs, outcome):
    callback = DockingMetricsCallback(window=100)
    callback._on_training_start()

    _step(callback, [_info(pos_m=1.0, **kwargs)], [True])

    assert logged[0][f"docking/{outcome}_rate"] == 1.0


def test_episode_counter_increments_across_envs(logged):
    callback = DockingMetricsCallback(window=100)
    callback._on_training_start()

    _step(callback, [_info(1.0), _info(1.0)], [False, False])
    _step(callback, [_info(1.0, success=True), _info(1.0)], [True, False])
    _step(callback, [_info(1.0), _info(1.0, collision=True)], [False, True])

    assert [call["docking/episodes"] for call in logged] == [1, 2]


def test_window_rates_are_a_recent_mean(logged):
    callback = DockingMetricsCallback(window=2)
    callback._on_training_start()

    _step(callback, [_info(1.0, success=True)], [True])
    _step(callback, [_info(1.0, collision=True)], [True])
    _step(callback, [_info(1.0, escaped=True)], [True])

    # window=2 keeps only the last two episodes: collision, then escaped.
    last = logged[-1]
    assert last["docking/docked_rate"] == 0.0
    assert last["docking/collision_rate"] == 0.5
    assert last["docking/escaped_rate"] == 0.5


def test_minima_reset_after_episode_ends(logged):
    callback = DockingMetricsCallback(window=100)
    callback._on_training_start()

    _step(callback, [_info(pos_m=1.0, success=True)], [True])
    _step(callback, [_info(pos_m=9.0)], [False])
    _step(callback, [_info(pos_m=9.0, success=True)], [True])

    assert logged[0]["docking/ep_min_pos_m"] == 1.0
    assert logged[1]["docking/ep_min_pos_m"] == 9.0


def test_port_index_logged_when_present(logged):
    callback = DockingMetricsCallback(window=100)
    callback._on_training_start()

    _step(callback, [_info(pos_m=1.0, success=True, port=3)], [True])

    assert logged[0]["docking/port_index"] == 3


def test_port_index_absent_when_no_ports(logged):
    callback = DockingMetricsCallback(window=100)
    callback._on_training_start()

    _step(callback, [_info(pos_m=1.0, success=True)], [True])

    assert "docking/port_index" not in logged[0]


def test_disables_itself_when_goal_error_true_is_missing(logged, capsys):
    callback = DockingMetricsCallback(window=100)
    callback._on_training_start()

    _step(callback, [{"success": False}], [False])
    assert "disabling" in capsys.readouterr().out

    logged.clear()
    _step(callback, [_info(pos_m=1.0, success=True)], [True])
    assert logged == []
