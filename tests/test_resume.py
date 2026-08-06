from pathlib import Path

import pytest
from conftest import smoke_cfg
from stable_baselines3 import PPO, SAC

from owm.baselines.rl.run_state import (
    FINAL_MODEL,
    latest_checkpoint,
    load_wandb_id,
    replay_buffer_for,
)
from owm.baselines.rl.train import run_training


def test_resume_continues_same_run(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    cfg = smoke_cfg(tmp_path, "ppo", extra=["rl.checkpoint.save_freq=64"])
    run_dir = run_training(cfg)
    first_id = load_wandb_id(run_dir)
    assert first_id is not None

    cfg2 = smoke_cfg(tmp_path, "ppo", extra=["resume=true", "rl.total_timesteps=600"])
    run_dir2 = run_training(cfg2)
    assert run_dir2 == run_dir
    assert load_wandb_id(run_dir) == first_id  # SAME wandb run

    model = PPO.load(run_dir / FINAL_MODEL, device="cpu")
    assert model.num_timesteps >= 600  # counter continued, not reset


def test_sac_resume_reloads_replay_buffer(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    run_dir = run_training(smoke_cfg(tmp_path, "sac", extra=["rl.checkpoint.save_freq=64"]))
    ckpt = latest_checkpoint(run_dir)
    assert ckpt is not None and replay_buffer_for(ckpt) is not None

    run_training(smoke_cfg(tmp_path, "sac", extra=["resume=true", "rl.total_timesteps=400"]))
    model = SAC.load(run_dir / FINAL_MODEL, device="cpu")
    assert model.num_timesteps >= 400


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
    with pytest.raises(SystemExit):
        run_training(smoke_cfg(tmp_path, "ppo"))  # same run dir, resume not set
    assert load_wandb_id(run_dir) == first_id  # the existing run is untouched
