import pytest

from owm.baselines.rl import video
from owm.baselines.rl.video import VideoEvalCallback


@pytest.fixture
def no_wandb(monkeypatch):
    monkeypatch.setattr(video.wandb, "define_metric", lambda *a, **k: None)


def _drive(callback, steps):
    """Run the callback over a sequence of num_timesteps, returning record points."""
    recorded = []
    callback._record = lambda: recorded.append(callback.num_timesteps)
    for step in steps:
        callback.num_timesteps = step
        assert callback._on_step() is True
    return recorded


def test_video_cadence_on_a_fresh_run(no_wandb):
    callback = VideoEvalCallback(env_conf={}, every_steps=100, max_frames=10, seed=0)
    callback.num_timesteps = 0
    callback._on_training_start()

    assert _drive(callback, range(0, 351, 10)) == [100, 200, 300]


def test_video_cadence_on_a_resumed_run(no_wandb):
    # A resume boots the callback at the restored step count. Scheduling from
    # every_steps would make that first _on_step look overdue and fire a
    # recording immediately, off the run's cadence.
    callback = VideoEvalCallback(env_conf={}, every_steps=100, max_frames=10, seed=0)
    callback.num_timesteps = 4321
    callback._on_training_start()

    assert _drive(callback, range(4321, 4600, 10)) == [4421, 4521]
