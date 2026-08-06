from pathlib import Path

from omegaconf import OmegaConf

from owm.baselines.rl.evaluate import resolve_checkpoint, run_eval, stats_for_checkpoint
from owm.baselines.rl.run_state import FINAL_MODEL
from owm.baselines.rl.train import run_training
from conftest import smoke_cfg


def test_resolve_checkpoint_local(tmp_path: Path):
    f = tmp_path / "m.zip"
    f.touch()
    assert resolve_checkpoint(str(f)) == f


def test_eval_smoke_model_on_modified_env(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    run_dir = run_training(smoke_cfg(tmp_path, "ppo"))
    ckpt = run_dir / FINAL_MODEL
    assert stats_for_checkpoint(ckpt) is not None

    cfg = smoke_cfg(tmp_path, "ppo")
    cfg.eval = OmegaConf.create(
        {"checkpoint": str(ckpt), "episodes": 2, "deterministic": True,
         "video_path": None}
    )
    # "New environment": same task, sensor noise off — a different ISSConfig.
    cfg.environments.sensor_noise.enabled = False
    # Eval correctness doesn't depend on horizon; cap it so a smoke policy
    # that fails to trigger early termination can't stall the test.
    cfg.environments.max_steps = 500
    results = run_eval(cfg)
    assert results["episodes"] == 2
    assert 0.0 <= results["success_rate"] <= 1.0
    assert results["mean_length"] > 0
