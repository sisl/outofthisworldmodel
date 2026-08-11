import shutil
from pathlib import Path

import pytest
from conftest import smoke_cfg
from huggingface_hub.utils import EntryNotFoundError
from omegaconf import DictConfig

from owm.baselines.rl import evaluate
from owm.baselines.rl.evaluate import resolve_checkpoint, run_eval, stats_for_checkpoint
from owm.baselines.rl.run_state import FINAL_MODEL, FINAL_VECNORM
from owm.baselines.rl.train import run_training

HF_SPEC = f"hf:org/repo/rl/ppo_x/{FINAL_MODEL}"


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory) -> Path:
    """One smoke run whose finals stand in for a published checkpoint."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("WANDB_MODE", "offline")
        return run_training(smoke_cfg(tmp_path_factory.mktemp("trained"), "ppo"))


def eval_cfg(tmp_path: Path, checkpoint: str) -> DictConfig:
    cfg = smoke_cfg(tmp_path, "ppo")
    cfg.eval.checkpoint = checkpoint
    cfg.eval.episodes = 1
    # Eval correctness doesn't depend on horizon; cap it so a smoke policy
    # that fails to trigger early termination can't stall the test.
    cfg.environments.max_steps = 500
    return cfg


class _EnvSpy:
    """Delegates to a real env, recording close() and optionally blowing up."""

    def __init__(self, env, fail: bool):
        self._env = env
        self._fail = fail
        self.closed = False

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self, **kwargs):
        if self._fail:
            raise RuntimeError("episode blew up")
        return self._env.reset(**kwargs)

    def close(self):
        self.closed = True
        self._env.close()


@pytest.mark.parametrize("fail", [False, True])
def test_eval_closes_its_env(trained_run, tmp_path: Path, monkeypatch, fail: bool):
    spied = []
    real_make = evaluate.make_env

    def spy(cfg, seed, render=False):
        spied.append(_EnvSpy(real_make(cfg, seed=seed, render=render), fail))
        return spied[-1]

    monkeypatch.setattr(evaluate, "make_env", spy)
    cfg = eval_cfg(tmp_path, str(trained_run / FINAL_MODEL))

    # A rendering eval holds a GL context, so the env has to be released on
    # the way out whether the episodes finished or raised.
    if fail:
        with pytest.raises(RuntimeError, match="episode blew up"):
            run_eval(cfg)
    else:
        run_eval(cfg)
    assert spied and spied[0].closed


def test_resolve_checkpoint_local(tmp_path: Path):
    f = tmp_path / "m.zip"
    f.touch()
    assert resolve_checkpoint(str(f)) == f


def test_resolve_checkpoint_rejects_malformed_hf_spec():
    # Without a path inside the repo there is nothing to download; falling
    # through to the local-path branch would report a missing file instead.
    with pytest.raises(SystemExit, match="malformed hf: spec"):
        resolve_checkpoint("hf:org/repo")


def test_hf_checkpoint_fetches_its_stats_sibling(trained_run, tmp_path, monkeypatch):
    requested = []

    def fake_download(*, repo_id, filename, repo_type):
        requested.append(filename)
        assert repo_id == "org/repo" and repo_type == "model"
        return str(trained_run / Path(filename).name)

    monkeypatch.setattr(evaluate, "hf_hub_download", fake_download)
    results = run_eval(eval_cfg(tmp_path, HF_SPEC))

    # Both halves of the checkpoint come from the repo: the model alone would
    # leave the policy running on unnormalized observations.
    assert requested == [f"rl/ppo_x/{FINAL_MODEL}", f"rl/ppo_x/{FINAL_VECNORM}"]
    assert results["episodes"] == 1


def test_hf_checkpoint_without_stats_refuses_to_evaluate(trained_run, tmp_path, monkeypatch):
    lonely = tmp_path / "lonely"
    lonely.mkdir()
    shutil.copy(trained_run / FINAL_MODEL, lonely / FINAL_MODEL)

    def fake_download(*, repo_id, filename, repo_type):
        if filename.endswith(FINAL_VECNORM):
            raise EntryNotFoundError(f"{filename} not found in {repo_id}")
        return str(lonely / FINAL_MODEL)

    monkeypatch.setattr(evaluate, "hf_hub_download", fake_download)

    with pytest.raises(SystemExit, match="no VecNormalize stats"):
        run_eval(eval_cfg(tmp_path, HF_SPEC))


def test_eval_smoke_model_on_modified_env(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    run_dir = run_training(smoke_cfg(tmp_path, "ppo"))
    ckpt = run_dir / FINAL_MODEL
    assert stats_for_checkpoint(ckpt) is not None

    cfg = eval_cfg(tmp_path, str(ckpt))
    cfg.eval.episodes = 2
    # "New environment": same task, sensor noise off — a different ISSConfig.
    cfg.environments.sensor_noise.enabled = False
    results = run_eval(cfg)
    assert results["episodes"] == 2
    assert 0.0 <= results["success_rate"] <= 1.0
    assert results["mean_length"] > 0
