from pathlib import Path

import pytest
from conftest import smoke_cfg
from stable_baselines3 import PPO

from owm.baselines.rl.run_state import FINAL_MODEL, FINAL_VECNORM
from owm.baselines.rl.train import run_training

STATE_DIM = 25
RESNET18_EMBED_DIM = 512


@pytest.mark.render
def test_train_smoke_with_resnet_observations(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    run_dir = run_training(
        smoke_cfg(
            tmp_path,
            "ppo",
            extra=[
                "environments=iss_coop_goal_ports_render224",
                "rl.obs=vector_resnet",
            ],
        )
    )
    assert (run_dir / FINAL_MODEL).exists()
    assert (run_dir / FINAL_VECNORM).exists()
    # What the policy was actually trained over: the state vector with the
    # frozen ResNet's embedding concatenated onto it, as one flat Box.
    model = PPO.load(run_dir / FINAL_MODEL, device="cpu")
    assert model.observation_space.shape == (STATE_DIM + RESNET18_EMBED_DIM,)


def test_video_capture_is_refused_with_resnet_observations(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    # Assigned rather than overridden: conftest's SMOKE list already pins
    # video.enabled, and hydra rejects the same key twice.
    cfg = smoke_cfg(tmp_path, "ppo", extra=["rl.obs=vector_resnet"])
    cfg.video.enabled = True

    with pytest.raises(SystemExit, match="video.enabled=true"):
        run_training(cfg)

    # Refused before the run dir was claimed, so the fixed launch can reuse it.
    assert not Path(cfg.run_dir).exists()
