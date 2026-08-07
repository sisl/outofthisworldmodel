from pathlib import Path

import numpy as np
import pytest
from owm_envs.envs.iss.config import ISSConfig

from owm.baselines.rl import sweep_callbacks, sweep_trial
from owm.baselines.rl.run_state import CHECKPOINT_DIR, FINAL_MODEL, FINAL_REPLAY_BUFFER
from owm.baselines.rl.sweep_callbacks import EvalReportCallback, TrialTimeoutCallback
from owm.baselines.rl.sweep_trial import (
    EVAL_REPORTS,
    RESERVED_KEYS,
    build_cfg,
    eval_cadence,
    prune_trial_artifacts,
)

PPO_TRIAL = {
    "algo": "ppo",
    "seed": 3,
    "trial_timesteps": 500_000,
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
    # Reserved keys say how to run the trial; they are not SB3 arguments and
    # would be rejected as such. Driven off the table so a key added to it
    # cannot start leaking into the hyperparameters unnoticed.
    for reserved in RESERVED_KEYS:
        assert reserved not in cfg.rl.hyperparams


def test_reserved_keys_are_routed_to_the_options_they_name(tmp_path: Path):
    cfg = build_cfg({**PPO_TRIAL, "trial_timesteps": 250_000}, tmp_path / "run")

    assert cfg.seed == 3
    assert cfg.rl.total_timesteps == 250_000
    # A trial is disposable, so its one checkpoint sits at the very end rather
    # than every 100k: SAC's carry a replay buffer of hundreds of MB.
    assert cfg.rl.checkpoint.save_freq == 250_000


def test_a_spec_that_pins_no_horizon_fails_before_anything_is_trained(tmp_path: Path):
    trial = {key: value for key, value in PPO_TRIAL.items() if key != "trial_timesteps"}

    with pytest.raises(SystemExit, match="no trial_timesteps"):
        build_cfg(trial, tmp_path / "run")


@pytest.mark.parametrize("horizon", [500_000, 250_000, 40_000])
def test_the_report_cadence_leaves_room_to_be_banded(horizon):
    # Hyperband only bands a trial that has reported min_iter times, so the
    # cadence has to follow the horizon its spec pinned rather than a constant
    # chosen for one of them.
    assert horizon // eval_cadence(horizon) >= EVAL_REPORTS


def test_a_horizon_too_short_to_report_is_refused(tmp_path: Path):
    # SB3 advances the step count by n_envs at a time, so a horizon under a
    # few multiples of that trains nothing and would still hand the sweep an
    # objective to rank against real trials.
    with pytest.raises(SystemExit, match="too short"):
        build_cfg({**PPO_TRIAL, "trial_timesteps": 8}, tmp_path / "run")


def test_routing_an_option_this_checkout_lacks_fails_the_trial_loudly(
    tmp_path: Path, monkeypatch
):
    # A routed key names an option that must already exist. Unlike a
    # hyperparameter, which SB3 rejects the moment the model is built, a
    # not-yet-landed option would be created here and quietly ignored, and the
    # sweep would report having swept something it never trained.
    monkeypatch.setitem(sweep_trial.ROUTES, "encoder", "rl.obs_encoder")
    with pytest.raises(SystemExit, match=r"rl\.obs_encoder"):
        build_cfg({**PPO_TRIAL, "encoder": "vit"}, tmp_path / "run")


def test_obs_mode_routes_onto_the_rl_config(tmp_path: Path):
    assert build_cfg(PPO_TRIAL, tmp_path / "run").rl.obs == "vector"
    cfg = build_cfg({**PPO_TRIAL, "obs": "vector_resnet"}, tmp_path / "run")
    assert cfg.rl.obs == "vector_resnet"


def test_trial_config_forces_the_settings_a_sweep_cannot_choose(tmp_path: Path):
    run_dir = tmp_path / "run"
    cfg = build_cfg(PPO_TRIAL, run_dir)

    assert cfg.run_dir == str(run_dir)
    assert cfg.external_wandb is True
    assert cfg.hub.upload is False
    assert cfg.video.enabled is False
    assert cfg.rl.n_envs == 8
    assert cfg.rl.vec == "subproc"
    assert cfg.rl.device == "cpu"
    # Sweeps tune hyperparameters for the real training distribution, not the
    # group default single-port one.
    assert cfg.environments.dock.ports == [
        "harmony_fwd_pma2",
        "harmony_nadir_cbm",
        "zvezda_aft",
        "pirs_nadir",
        "rassvet_nadir",
    ]


def test_sac_trials_stay_on_the_gpu_they_are_allowed(tmp_path: Path):
    cfg = build_cfg(
        {"algo": "sac", "trial_timesteps": 500_000, "tau": 5.0e-3, "train_freq": 4},
        tmp_path / "run",
    )

    # GPU 1 belongs to another tenant; a trial must never land on it.
    assert cfg.rl.device == "cuda:0"
    assert cfg.rl.hyperparams.tau == 5.0e-3
    assert cfg.rl.hyperparams.train_freq == 4
    assert cfg.seed == 0  # sweeps that do not sweep the seed still get one


def test_a_gpu_trial_on_a_box_with_no_gpu_fails_instead_of_using_the_cpu(
    tmp_path: Path, monkeypatch
):
    # `just sweep-agent <sac-id> ppo_vector` exports CUDA_VISIBLE_DEVICES="",
    # and SB3's get_device would quietly hand the trial a CPU — a night of SAC
    # results reported as the GPU run somebody asked for.
    monkeypatch.setattr(sweep_trial.torch.cuda, "is_available", lambda: False)

    with pytest.raises(SystemExit, match="no CUDA device"):
        build_cfg({"algo": "sac", "trial_timesteps": 500_000}, tmp_path / "run")


def test_a_cpu_trial_does_not_care_whether_a_gpu_exists(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(sweep_trial.torch.cuda, "is_available", lambda: False)

    assert build_cfg(PPO_TRIAL, tmp_path / "run").rl.device == "cpu"


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
    """Enough of an SB3 model for the callback to roll episodes out."""

    def __init__(self, action_dim: int):
        self._action_dim = action_dim

    def get_vec_normalize_env(self):
        return None

    def predict(self, obs, deterministic: bool):
        return np.zeros((len(obs), self._action_dim), dtype=np.float32), None


def _recorded_run(tmp_path: Path, max_steps: int = 4) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ISSConfig(max_steps=max_steps).to_yaml(run_dir / "env_config.yaml")
    return run_dir


def test_the_eval_env_is_the_one_the_run_recorded_training_on(tmp_path: Path):
    # The record is what training actually trained on; re-resolving
    # environments=from_dataset here instead could score the trial on dynamics
    # it never saw, because that ref can move between two resolutions.
    callback = _eval_callback(run_dir=_recorded_run(tmp_path), vec="dummy")
    callback.model = _ZeroPolicy(action_dim=6)

    try:
        mean_return, success_rate = callback._evaluate(episodes=1)
        # Truncated by the recorded horizon, not by any default of the sweep's.
        assert callback._env.get_attr("unwrapped")[0].cfg.max_steps == 4
        assert isinstance(mean_return, float)
        assert success_rate == 0.0
    finally:
        callback._env.close()


def test_the_score_does_not_depend_on_how_wide_the_eval_env_is(tmp_path: Path):
    # Episodes run a vec-width at a time for throughput, so a trial's score has
    # to come out the same whether they ran side by side or one after another —
    # otherwise widening the eval would silently rescore the sweep.
    run_dir = _recorded_run(tmp_path)
    scores = []
    for episodes in (1, 2):  # eval width is min(episodes, EVAL_ENVS)
        callback = _eval_callback(run_dir=run_dir, episodes=episodes, vec="dummy")
        callback.model = _ZeroPolicy(action_dim=6)
        try:
            assert callback._eval_env().num_envs == episodes
            scores.append(callback._evaluate(episodes=2))
        finally:
            callback._env.close()

    assert scores[0] == scores[1]


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


def test_environments_defaults_to_the_random_port_training_distribution(tmp_path: Path):
    cfg = build_cfg(PPO_TRIAL, tmp_path / "run")
    assert len(cfg.environments.dock.ports) == 5
    # No render block at all, not an empty one: a spec that says nothing about
    # environments composes exactly what it composed before this key existed.
    assert "render" not in cfg.environments


def test_a_spec_can_pin_the_env_config_group_it_trains_on(tmp_path: Path):
    # What a pixel sweep needs: the same port distribution, rendered at the
    # size the frozen extractor reads.
    cfg = build_cfg(
        {**PPO_TRIAL, "environments": "iss_coop_goal_ports_render224"}, tmp_path / "run"
    )
    assert cfg.environments.render == {"image_width": 224, "image_height": 224}
    assert cfg.environments.dock.ports == build_cfg(
        PPO_TRIAL, tmp_path / "run2"
    ).environments.dock.ports
    # Spent on the compose, not written into the config it produced.
    assert "environments" not in cfg.rl.hyperparams


def test_an_env_config_this_checkout_lacks_fails_before_anything_is_trained(tmp_path: Path):
    with pytest.raises(SystemExit, match="conf/environments has no such config"):
        build_cfg({**PPO_TRIAL, "environments": "iss_coop_goal_pixels"}, tmp_path / "run")


def test_environments_is_reserved_rather_than_an_sb3_hyperparameter():
    assert sweep_trial.ENVIRONMENTS_KEY in RESERVED_KEYS
    # Not routable: it names a group to compose, so OmegaConf.update would
    # replace the whole environments node with its own name.
    assert sweep_trial.ENVIRONMENTS_KEY not in sweep_trial.ROUTES
