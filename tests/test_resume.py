from pathlib import Path

import pytest
from conftest import smoke_cfg
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.save_util import load_from_pkl

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
