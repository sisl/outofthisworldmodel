from pathlib import Path

import pytest
from conftest import smoke_cfg

from owm_envs.envs.iss.config import ISSConfig

from owm.baselines.rl import train
from owm.baselines.rl.run_state import (
    CHECKPOINT_DIR,
    FINAL_MODEL,
    FINAL_REPLAY_BUFFER,
    FINAL_VECNORM,
    load_wandb_id,
)
from owm.baselines.rl.train import run_training
from owm.envs import factory


def _capture_env_conf(monkeypatch) -> list[dict]:
    """Record what the vec env is actually built from, still building it."""
    seen: list[dict] = []
    real = train.make_vec_env

    def spy(env_conf, *args, **kwargs):
        seen.append(env_conf)
        return real(env_conf, *args, **kwargs)

    monkeypatch.setattr(train, "make_vec_env", spy)
    return seen


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
    # Only off-policy runs carry a buffer, and only locally — an extend that
    # resumes from the finals needs it, the hub upload never sees it.
    assert (run_dir / FINAL_REPLAY_BUFFER).exists() == (algo == "sac")
    # The env the run actually trained on, spelled out: environments=
    # from_dataset would otherwise leave only a repo name behind.
    assert ISSConfig.from_yaml(run_dir / "env_config.yaml").dt == 0.05


def test_workers_get_the_resolved_env_not_the_dataset_ref(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    shipped = tmp_path / "shipped_env_config.yaml"
    ISSConfig(max_steps=200).to_yaml(shipped)
    downloads = []

    def fake_download(*, repo_id, filename, repo_type, revision):
        downloads.append(repo_id)
        return str(shipped)

    monkeypatch.setattr(factory, "hf_hub_download", fake_download)
    seen = _capture_env_conf(monkeypatch)

    run_dir = run_training(smoke_cfg(tmp_path, "ppo", extra=["environments=from_dataset"]))

    # One resolution, by the launcher. Handing workers the repo name instead
    # would have each of them download it again, and an unpinned ref that
    # moves mid-run would leave them training on different dynamics.
    assert len(downloads) == 1
    assert "from_dataset_repo" not in seen[0]
    assert seen[0]["max_steps"] == 200
    assert ISSConfig.from_yaml(run_dir / "env_config.yaml").max_steps == 200


def test_resume_trains_the_recorded_env_and_leaves_it_alone(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    run_dir = run_training(smoke_cfg(tmp_path, "ppo"))
    record = run_dir / "env_config.yaml"
    # Stand in for a dataset ref that moved between the two legs: the record
    # and what a fresh resolution would produce now disagree, and the record
    # is the one the first leg actually trained under.
    recorded = ISSConfig.from_yaml(record)
    noisier = recorded.sensor_noise.model_copy(update={"sigma_pos_m": 0.123})
    recorded.model_copy(update={"sensor_noise": noisier}).to_yaml(record)
    before = record.read_text()

    seen = _capture_env_conf(monkeypatch)
    run_training(smoke_cfg(tmp_path, "ppo", extra=["resume=true", "extend_timesteps=600"]))

    assert seen[0]["sensor_noise"]["sigma_pos_m"] == 0.123
    assert record.read_text() == before


def test_upload_without_a_repo_id_fails_at_launch(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    cfg = smoke_cfg(tmp_path, "ppo", extra=["hub.upload=true", "hub.repo_id=null"])

    with pytest.raises(SystemExit, match="hub.repo_id is empty"):
        run_training(cfg)

    # Refused before the run dir was claimed, so the fixed launch can reuse it.
    assert not Path(cfg.run_dir).exists()
