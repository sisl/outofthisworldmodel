from pathlib import Path

import pytest
from conftest import smoke_cfg

from owm.baselines.rl.run_state import CHECKPOINT_DIR, FINAL_MODEL, FINAL_VECNORM, load_wandb_id
from owm.baselines.rl.train import run_training


@pytest.mark.parametrize("algo", ["ppo", "sac"])
def test_train_smoke(tmp_path: Path, algo: str, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    run_dir = run_training(smoke_cfg(tmp_path, algo))
    assert (run_dir / FINAL_MODEL).exists()
    assert (run_dir / FINAL_VECNORM).exists()
    assert (run_dir / "config.yaml").exists()
    assert "${" not in (run_dir / "config.yaml").read_text()
    assert load_wandb_id(run_dir) is not None
    assert any((run_dir / CHECKPOINT_DIR).glob("model_*_steps.zip"))
