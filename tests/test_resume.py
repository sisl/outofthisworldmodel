from pathlib import Path

import pytest
from conftest import smoke_cfg
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.save_util import load_from_pkl

from owm.baselines.rl import train
from owm.baselines.rl.run_state import (
    FINAL_MODEL,
    FINAL_VECNORM,
    latest_checkpoint,
    load_wandb_id,
    replay_buffer_for,
    vecnormalize_for,
)
from owm.baselines.rl.train import run_training


def test_resume_continues_same_run(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    cfg = smoke_cfg(tmp_path, "ppo", extra=["rl.checkpoint.save_freq=64"])
    run_dir = run_training(cfg)
    first_id = load_wandb_id(run_dir)
    assert first_id is not None
    seen_before = load_from_pkl(vecnormalize_for(latest_checkpoint(run_dir))).obs_rms.count

    cfg2 = smoke_cfg(tmp_path, "ppo", extra=["resume=true", "rl.total_timesteps=600"])
    run_dir2 = run_training(cfg2)
    assert run_dir2 == run_dir
    assert load_wandb_id(run_dir) == first_id  # SAME wandb run
    # Offline wandb names its run dir offline-run-<timestamp>-<id>; both legs
    # reporting the same id is what proves wandb.init actually reattached.
    assert {p.name.rsplit("-", 1)[-1] for p in (run_dir / "wandb").glob("offline-run-*")} == {
        first_id
    }

    model = PPO.load(run_dir / FINAL_MODEL, device="cpu")
    # 600 is the run's total budget, not 600 more steps: the counter continued
    # (not reset) but stopped at the first rollout boundary past the target
    # rather than training a second full budget on top of the first leg.
    rollout = 64 * 2  # n_steps * n_envs
    assert 600 <= model.num_timesteps < 600 + rollout
    # Normalization kept accumulating over the checkpoint's statistics instead
    # of restarting from an empty running mean.
    assert load_from_pkl(run_dir / FINAL_VECNORM).obs_rms.count >= seen_before


def test_resume_passes_saved_id_to_wandb(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    run_dir = run_training(smoke_cfg(tmp_path, "ppo"))
    saved_id = load_wandb_id(run_dir)

    captured = {}
    monkeypatch.setattr(train.wandb, "init", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(train.wandb, "finish", lambda: None)
    # The first leg already spent the budget, so this resume has nothing left to
    # train and stops before learn() — it exercises the reattach and the
    # already-finished no-op together.
    run_training(smoke_cfg(tmp_path, "ppo", extra=["resume=true"]))

    assert captured["id"] == saved_id
    assert captured["resume"] == "must"


def test_noop_resume_leaves_final_artifacts_alone(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    # The 300-step budget is met at the 384-step rollout boundary, but with
    # save_freq=320 the last checkpoint is at 320. A resume rebuilds the model
    # from that checkpoint, so saving finals would roll them back 64 steps.
    run_dir = run_training(smoke_cfg(tmp_path, "ppo", extra=["rl.checkpoint.save_freq=320"]))
    before = PPO.load(run_dir / FINAL_MODEL, device="cpu").num_timesteps
    assert PPO.load(latest_checkpoint(run_dir), device="cpu").num_timesteps < before
    vecnorm_written = (run_dir / FINAL_VECNORM).stat().st_mtime_ns

    run_training(smoke_cfg(tmp_path, "ppo", extra=["resume=true"]))

    assert PPO.load(run_dir / FINAL_MODEL, device="cpu").num_timesteps == before
    assert (run_dir / FINAL_VECNORM).stat().st_mtime_ns == vecnorm_written


def test_resume_writes_finals_a_crash_never_saved(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    run_dir = run_training(smoke_cfg(tmp_path, "ppo"))
    # A crash between the last checkpoint and the final save leaves the budget
    # met with no final artifacts; that resume has to write them, not skip.
    (run_dir / FINAL_MODEL).unlink()
    (run_dir / FINAL_VECNORM).unlink()
    checkpoint_steps = PPO.load(latest_checkpoint(run_dir), device="cpu").num_timesteps

    run_training(smoke_cfg(tmp_path, "ppo", extra=["resume=true"]))

    assert PPO.load(run_dir / FINAL_MODEL, device="cpu").num_timesteps == checkpoint_steps
    assert (run_dir / FINAL_VECNORM).exists()


def test_resume_recreates_finals_if_vecnormalize_missing(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    run_dir = run_training(smoke_cfg(tmp_path, "ppo"))
    model_steps = PPO.load(run_dir / FINAL_MODEL, device="cpu").num_timesteps
    # A crash between model.save() and venv.save() leaves final_model.zip
    # present but vecnormalize.pkl missing; the skip-gate has to require both
    # finals, not just FINAL_MODEL, or this resume would declare the run done
    # without ever writing vecnormalize.pkl or re-publishing.
    (run_dir / FINAL_VECNORM).unlink()

    run_training(smoke_cfg(tmp_path, "ppo", extra=["resume=true"]))

    assert PPO.load(run_dir / FINAL_MODEL, device="cpu").num_timesteps == model_steps
    assert (run_dir / FINAL_VECNORM).exists()


def test_resume_refuses_checkpoint_without_vecnormalize(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    run_dir = run_training(smoke_cfg(tmp_path, "ppo"))
    vecnormalize_for(latest_checkpoint(run_dir)).unlink()

    with pytest.raises(SystemExit, match="vecnormalize sibling"):
        run_training(smoke_cfg(tmp_path, "ppo", extra=["resume=true"]))


def test_sac_resume_reloads_replay_buffer(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    run_dir = run_training(smoke_cfg(tmp_path, "sac", extra=["rl.checkpoint.save_freq=64"]))
    ckpt = latest_checkpoint(run_dir)
    assert ckpt is not None and replay_buffer_for(ckpt) is not None
    filled_before = load_from_pkl(replay_buffer_for(ckpt)).pos

    run_training(smoke_cfg(tmp_path, "sac", extra=["resume=true", "rl.total_timesteps=400"]))
    model = SAC.load(run_dir / FINAL_MODEL, device="cpu")
    assert model.num_timesteps >= 400
    # A buffer that had been dropped instead of reloaded would hold only the
    # transitions collected after the resume, i.e. fewer than it already had.
    filled_after = load_from_pkl(replay_buffer_for(latest_checkpoint(run_dir))).pos
    assert filled_after > filled_before

    # Resuming SAC from a checkpoint whose buffer went missing would silently
    # restart from an empty one, so it has to fail instead.
    replay_buffer_for(latest_checkpoint(run_dir)).unlink()
    with pytest.raises(SystemExit, match="replay_buffer sibling"):
        run_training(smoke_cfg(tmp_path, "sac", extra=["resume=true", "rl.total_timesteps=800"]))


def test_resume_before_first_checkpoint(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    cfg = smoke_cfg(tmp_path, "ppo", extra=["rl.checkpoint.save_freq=1000000"])
    run_dir = run_training(cfg)
    first_id = load_wandb_id(run_dir)
    cfg2 = smoke_cfg(tmp_path, "ppo", extra=["resume=true"])
    run_training(cfg2)
    assert load_wandb_id(run_dir) == first_id


def test_fresh_run_refuses_existing_run_dir(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    run_dir = run_training(smoke_cfg(tmp_path, "ppo"))
    first_id = load_wandb_id(run_dir)
    with pytest.raises(SystemExit, match="already contains a run"):
        run_training(smoke_cfg(tmp_path, "ppo"))  # same run dir, resume not set
    assert load_wandb_id(run_dir) == first_id  # the existing run is untouched
