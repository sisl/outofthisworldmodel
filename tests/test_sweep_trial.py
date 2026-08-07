from pathlib import Path

import numpy as np
import pytest
from owm_envs.envs.iss.config import ISSConfig

from owm.baselines.rl import sweep_callbacks
from owm.baselines.rl.run_state import CHECKPOINT_DIR, FINAL_MODEL, FINAL_REPLAY_BUFFER
from owm.baselines.rl.sweep_callbacks import EvalReportCallback, TrialTimeoutCallback
from owm.baselines.rl.sweep_trial import TOTAL_TIMESTEPS, build_cfg, prune_trial_artifacts

PPO_TRIAL = {
    "algo": "ppo",
    "seed": 3,
    "learning_rate": 1.234e-4,
    "n_steps": 1024,
    "batch_size": 512,
    "clip_range": 0.3,
    "n_epochs": 20,
}


def test_trial_config_carries_every_swept_hyperparameter(tmp_path: Path):
    cfg = build_cfg(PPO_TRIAL, tmp_path / "run")

    assert cfg.rl.algo == "ppo"
    assert cfg.rl.hyperparams.learning_rate == 1.234e-4
    assert cfg.rl.hyperparams.n_steps == 1024
    assert cfg.rl.hyperparams.n_epochs == 20
    # Not in conf/rl/ppo.yaml at all: a sweep may tune arguments the group
    # default leaves at SB3's own.
    assert cfg.rl.hyperparams.clip_range == 0.3
    assert cfg.seed == 3
    # Control keys say how to run the trial; they are not SB3 arguments and
    # would be rejected as such.
    assert "algo" not in cfg.rl.hyperparams
    assert "seed" not in cfg.rl.hyperparams


def test_trial_config_forces_the_settings_a_sweep_cannot_choose(tmp_path: Path):
    run_dir = tmp_path / "run"
    cfg = build_cfg(PPO_TRIAL, run_dir)

    assert cfg.rl.total_timesteps == TOTAL_TIMESTEPS
    assert cfg.run_dir == str(run_dir)
    assert cfg.external_wandb is True
    assert cfg.hub.upload is False
    assert cfg.video.enabled is False
    assert cfg.rl.n_envs == 8
    assert cfg.rl.vec == "subproc"
    assert cfg.rl.device == "cpu"


def test_sac_trials_stay_on_the_gpu_they_are_allowed(tmp_path: Path):
    cfg = build_cfg({"algo": "sac", "tau": 5.0e-3, "train_freq": 4}, tmp_path / "run")

    # GPU 1 belongs to another tenant; a trial must never land on it.
    assert cfg.rl.device == "cuda:0"
    assert cfg.rl.hyperparams.tau == 5.0e-3
    assert cfg.rl.hyperparams.train_freq == 4
    assert cfg.seed == 0  # sweeps that do not sweep the seed still get one


def test_an_unknown_algo_fails_before_anything_is_trained(tmp_path: Path):
    with pytest.raises(SystemExit, match="expected one of"):
        build_cfg({"algo": "dqn"}, tmp_path / "run")


def test_pruning_keeps_the_model_and_drops_what_only_a_resume_wants(tmp_path: Path):
    run_dir = tmp_path / "run"
    (run_dir / CHECKPOINT_DIR).mkdir(parents=True)
    (run_dir / CHECKPOINT_DIR / "model_100_steps.zip").write_bytes(b"")
    (run_dir / FINAL_REPLAY_BUFFER).write_bytes(b"")
    (run_dir / FINAL_MODEL).write_bytes(b"")

    prune_trial_artifacts(run_dir)

    assert (run_dir / FINAL_MODEL).exists()
    assert not (run_dir / CHECKPOINT_DIR).exists()
    assert not (run_dir / FINAL_REPLAY_BUFFER).exists()
    # A trial that died before writing anything must not turn cleanup into a
    # second failure.
    prune_trial_artifacts(tmp_path / "never_started")


@pytest.fixture
def no_wandb(monkeypatch):
    monkeypatch.setattr(sweep_callbacks.wandb, "define_metric", lambda *a, **k: None)
    monkeypatch.setattr(sweep_callbacks.wandb, "log", lambda *a, **k: None)


def _drive(callback, steps):
    """Run the callback over a sequence of num_timesteps, returning report points."""
    reported = []
    callback._report = lambda episodes, final: reported.append(callback.num_timesteps)
    for step in steps:
        callback.num_timesteps = step
        assert callback._on_step() is True
    return reported


def _eval_callback(**kwargs):
    defaults = dict(
        run_dir=Path("unused"), every_steps=100, episodes=1, final_episodes=1, seed=0
    )
    return EvalReportCallback(**{**defaults, **kwargs})


def test_report_cadence_on_a_fresh_run(no_wandb):
    callback = _eval_callback()
    callback.num_timesteps = 0
    callback._on_training_start()

    assert _drive(callback, range(0, 351, 10)) == [100, 200, 300]


def test_report_cadence_on_a_resumed_run(no_wandb):
    # A resume boots the callback at the restored step count. Scheduling from
    # every_steps would make that first _on_step look overdue and fire a report
    # immediately, off the run's cadence.
    callback = _eval_callback()
    callback.num_timesteps = 4321
    callback._on_training_start()

    assert _drive(callback, range(4321, 4600, 10)) == [4421, 4521]


def test_a_report_of_no_episodes_is_refused_at_registration():
    with pytest.raises(ValueError, match="episodes must be >= 1"):
        _eval_callback(episodes=0)
    with pytest.raises(ValueError, match="episodes must be >= 1"):
        _eval_callback(final_episodes=0)


class _ZeroPolicy:
    """Enough of an SB3 model for the callback to roll an episode out."""

    def __init__(self, action_dim: int):
        self._action = np.zeros(action_dim, dtype=np.float32)

    def get_vec_normalize_env(self):
        return None

    def predict(self, obs, deterministic: bool):
        return self._action, None


def test_the_eval_env_is_the_one_the_run_recorded_training_on(tmp_path: Path):
    # The record is what training actually trained on; re-resolving
    # environments=from_dataset here instead could score the trial on dynamics
    # it never saw, because that ref can move between two resolutions.
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ISSConfig(max_steps=4).to_yaml(run_dir / "env_config.yaml")

    callback = _eval_callback(run_dir=run_dir)
    callback.model = _ZeroPolicy(action_dim=6)

    mean_return, success_rate = callback._evaluate(episodes=1)
    try:
        # Truncated by the recorded horizon, not by any default of the sweep's.
        assert callback._env.unwrapped.cfg.max_steps == 4
        assert isinstance(mean_return, float)
        assert success_rate == 0.0
    finally:
        callback._env.close()


def test_a_trial_ends_when_its_wall_clock_budget_runs_out(no_wandb):
    now = [0.0]
    callback = TrialTimeoutCallback(max_seconds=10.0, clock=lambda: now[0])
    callback.num_timesteps = 0
    callback._on_training_start()

    assert callback._on_step() is True
    now[0] = 9.9
    assert callback._on_step() is True
    now[0] = 10.1
    assert callback._on_step() is False


def test_the_wall_clock_budget_covers_setup_too(no_wandb):
    # Building eight subproc envs and restoring a replay buffer is trial time
    # like any other; a deadline started at training start would hand a trial
    # that already spent its budget on setup a fresh one to train with.
    now = [0.0]
    callback = TrialTimeoutCallback(max_seconds=10.0, clock=lambda: now[0])
    callback.num_timesteps = 0
    now[0] = 11.0  # setup ran long; training has not begun
    callback._on_training_start()

    assert callback._on_step() is False
