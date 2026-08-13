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


def test_min_pos_frac_divides_closest_approach_by_the_start_range(logged):
    callback = DockingMetricsCallback(window=100)
    callback._on_training_start()

    # Opens at 400 m, closes to 100 m: a quarter of the way in.
    _step(callback, [_info(pos_m=400.0)], [False])
    _step(callback, [_info(pos_m=100.0)], [False])
    _step(callback, [_info(pos_m=250.0)], [True])

    assert logged[0]["docking/ep_start_pos_m"] == 400.0
    assert logged[0]["docking/ep_min_pos_m"] == 100.0
    assert logged[0]["docking/ep_min_pos_frac"] == pytest.approx(0.25)


def test_min_pos_frac_is_one_when_the_policy_never_closes(logged):
    callback = DockingMetricsCallback(window=100)
    callback._on_training_start()

    _step(callback, [_info(pos_m=300.0)], [False])
    _step(callback, [_info(pos_m=450.0)], [True])

    assert logged[0]["docking/ep_min_pos_frac"] == pytest.approx(1.0)


def test_start_range_is_retaken_each_episode(logged):
    callback = DockingMetricsCallback(window=100)
    callback._on_training_start()

    _step(callback, [_info(pos_m=400.0)], [False])
    _step(callback, [_info(pos_m=200.0)], [True])
    # Next episode opens somewhere else entirely; the first one's start must
    # not leak into it.
    _step(callback, [_info(pos_m=120.0)], [False])
    _step(callback, [_info(pos_m=60.0)], [True])

    assert logged[0]["docking/ep_start_pos_m"] == 400.0
    assert logged[1]["docking/ep_start_pos_m"] == 120.0
    assert logged[1]["docking/ep_min_pos_frac"] == pytest.approx(0.5)


def test_counts_accumulate_where_rates_round_a_rare_outcome_away(logged):
    callback = DockingMetricsCallback(window=100)
    callback._on_training_start()

    for _ in range(99):
        _step(callback, [_info(pos_m=300.0)], [True])
    _step(callback, [_info(pos_m=1.0, success=True)], [True])

    assert logged[-1]["docking/docked_count"] == 1
    assert logged[-1]["docking/truncated_count"] == 99
    assert logged[-1]["docking/docked_rate"] == pytest.approx(0.01)


def test_counts_are_cumulative_past_the_rate_window(logged):
    callback = DockingMetricsCallback(window=10)
    callback._on_training_start()

    _step(callback, [_info(pos_m=1.0, success=True)], [True])
    for _ in range(20):
        _step(callback, [_info(pos_m=300.0)], [True])

    # The dock has fallen out of the 10-episode rate window, but happened.
    assert logged[-1]["docking/docked_rate"] == 0.0
    assert logged[-1]["docking/docked_count"] == 1
