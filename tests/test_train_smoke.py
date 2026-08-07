from pathlib import Path

import pytest
from conftest import smoke_cfg

from owm_envs.envs.iss.config import ISSConfig

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
    # The env the run actually trained on, spelled out: environments=
    # from_dataset would otherwise leave only a repo name behind.
    assert ISSConfig.from_yaml(run_dir / "env_config.yaml").dt == 0.05


def test_upload_without_a_repo_id_fails_at_launch(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    cfg = smoke_cfg(tmp_path, "ppo", extra=["hub.upload=true", "hub.repo_id=null"])

    with pytest.raises(SystemExit, match="hub.repo_id is empty"):
        run_training(cfg)

    # Refused before the run dir was claimed, so the fixed launch can reuse it.
    assert not Path(cfg.run_dir).exists()
