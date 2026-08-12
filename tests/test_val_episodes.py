from pathlib import Path

import numpy as np
import pytest

from owm.baselines.rl import val_episodes
from owm.baselines.rl.val_episodes import ValEpisodeCallback


@pytest.fixture
def no_wandb(monkeypatch):
    monkeypatch.setattr(val_episodes.wandb, "define_metric", lambda *a, **k: None)


def _callback(**overrides):
    kwargs = dict(
        run_dir=Path("unused"),
        env_name="iss-numerical",
        seed=0,
        episodes=3,
        video_episodes=1,
        every_steps=100,
        max_frames=10,
    )
    kwargs.update(overrides)
    return ValEpisodeCallback(**kwargs)


def _drive(callback, steps):
    """Run the callback over a sequence of num_timesteps, returning round points."""
    recorded = []
    callback._run_round = lambda: recorded.append(callback.num_timesteps)
    for step in steps:
        callback.num_timesteps = step
        assert callback._on_step() is True
    return recorded


def test_rejects_a_frame_budget_it_cannot_record():
    # max_frames=0 with video asked for records nothing and dies in np.stack
    # of an empty list — hours into a run. A config error should fail when the
    # config is read.
    with pytest.raises(ValueError, match="max_frames must be >= 1"):
        _callback(max_frames=0)


def test_no_video_episodes_needs_no_frame_budget():
    _callback(video_episodes=0, max_frames=0)


def test_rejects_more_video_episodes_than_episodes():
    with pytest.raises(ValueError, match="video_episodes must be in"):
        _callback(episodes=2, video_episodes=3)


def test_rejects_an_empty_schedule():
    # A callback nothing ever triggers is a config error, not a quiet no-op.
    with pytest.raises(ValueError, match="needs a schedule"):
        _callback(every_steps=None)


def test_cadence_on_a_fresh_run(no_wandb):
    callback = _callback()
    callback.num_timesteps = 0
    callback._on_training_start()

    assert _drive(callback, range(0, 351, 10)) == [100, 200, 300]


def test_cadence_on_a_resumed_run(no_wandb):
    # A resume boots the callback at the restored step count. Scheduling from
    # every_steps would make that first _on_step look overdue and fire a
    # round immediately, off the run's cadence.
    callback = _callback()
    callback.num_timesteps = 4321
    callback._on_training_start()

    assert _drive(callback, range(4321, 4600, 10)) == [4421, 4521]


def test_at_step_fires_once_when_crossed(no_wandb):
    callback = _callback(every_steps=None, at_steps=(250,))
    callback.num_timesteps = 0
    callback._on_training_start()

    # Fires on the first step at or past the mark, and never again.
    assert _drive(callback, range(0, 500, 20)) == [260]


def test_at_step_equal_to_the_resume_step_still_fires(no_wandb):
    # Whether the previous leg fired a mark it died exactly on cannot be
    # known; re-running a round is minutes, silently dropping one loses the
    # mid-point video for good.
    callback = _callback(every_steps=None, at_steps=(4000,))
    callback.num_timesteps = 4000
    callback._on_training_start()

    assert _drive(callback, range(4000, 4100, 20)) == [4000]


def test_at_steps_already_passed_do_not_fire_on_resume(no_wandb):
    callback = _callback(every_steps=None, at_steps=(250,), final=True)
    callback.num_timesteps = 4000
    callback._on_training_start()

    assert _drive(callback, range(4000, 4200, 20)) == []


def test_coinciding_cadence_and_at_step_fire_one_round(no_wandb):
    callback = _callback(every_steps=100, at_steps=(100,))
    callback.num_timesteps = 0
    callback._on_training_start()

    assert _drive(callback, range(0, 151, 50)) == [100]


def test_final_round_runs_at_training_end(no_wandb):
    callback = _callback(every_steps=None, final=True)
    callback.num_timesteps = 999
    callback._on_training_start()

    recorded = []
    callback._run_round = lambda: recorded.append(callback.num_timesteps)
    callback._on_training_end()
    assert recorded == [999]


def test_no_final_round_without_final(no_wandb):
    callback = _callback()
    callback.num_timesteps = 50
    callback._on_training_start()

    recorded = []
    callback._run_round = lambda: recorded.append(callback.num_timesteps)
    callback._on_training_end()
    assert recorded == []


def test_quat_to_rotmat_matches_the_convention():
    """q_bw columns are the body axes in the world frame (owm-envs q_bw)."""
    # 90 degrees about world z: body x maps to world y.
    q = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
    rotmat = val_episodes.quat_to_rotmat(q)
    np.testing.assert_allclose(rotmat @ np.array([1.0, 0, 0]), [0, 1, 0], atol=1e-12)
    # Orthonormal, right-handed.
    np.testing.assert_allclose(rotmat @ rotmat.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(rotmat), 1.0, atol=1e-12)
