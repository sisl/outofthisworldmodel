from pathlib import Path

import pytest
from conftest import CONF_DIR, SMOKE, smoke_cfg
from hydra import compose, initialize_config_dir
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.save_util import load_from_pkl
from stable_baselines3.common.vec_env import VecNormalize

from owm.baselines.rl import train
from owm.baselines.rl.run_state import (
    CHECKPOINT_DIR,
    FINAL_MODEL,
    FINAL_REPLAY_BUFFER,
    FINAL_STEPS,
    FINAL_VECNORM,
    checkpoint_steps,
    latest_checkpoint,
    load_wandb_id,
    replay_buffer_for,
    vecnormalize_for,
)
from owm.baselines.rl.train import run_training


def _refuse_to_train(self, *, total_timesteps, **kwargs):
    raise AssertionError(f"resume asked for {total_timesteps} more steps")


def _spy_on_loads(monkeypatch, algo) -> list[Path]:
    """Record what the resumed model is rebuilt from, still loading for real."""
    loaded: list[Path] = []
    original = algo.load

    def spy(path, *args, **kwargs):
        loaded.append(Path(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(algo, "load", spy)
    return loaded


def _drop_finals(run_dir: Path) -> None:
    """Leave the run looking like a crash before its final save."""
    for name in (FINAL_MODEL, FINAL_VECNORM, FINAL_REPLAY_BUFFER, FINAL_STEPS):
        (run_dir / name).unlink(missing_ok=True)


def _keep_only_newest_checkpoint(run_dir: Path) -> Path:
    newest = latest_checkpoint(run_dir)
    for path in (run_dir / CHECKPOINT_DIR).iterdir():
        if f"_{checkpoint_steps(newest)}_steps." not in path.name:
            path.unlink()
    return newest


def bare_resume_cfg(run_dir: Path):
    """What `just resume RUN_DIR` composes: smoke settings, no budget given."""
    overrides = [o for o in SMOKE if not o.startswith("rl.total_timesteps")]
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=["rl=ppo", f"run_dir={run_dir}", "resume=true", *overrides],
        )
    assert cfg.rl.total_timesteps == 5_000_000 and cfg.extend_timesteps is None
    return cfg


def test_resume_continues_same_run(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    cfg = smoke_cfg(tmp_path, "ppo", extra=["rl.checkpoint.save_freq=64"])
    run_dir = run_training(cfg)
    first_id = load_wandb_id(run_dir)
    assert first_id is not None
    seen_before = load_from_pkl(vecnormalize_for(latest_checkpoint(run_dir))).obs_rms.count

    cfg2 = smoke_cfg(tmp_path, "ppo", extra=["resume=true", "extend_timesteps=600"])
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


def test_resume_ignores_the_composed_default_budget(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    run_dir = run_training(smoke_cfg(tmp_path, "ppo"))
    trained = PPO.load(run_dir / FINAL_MODEL, device="cpu").num_timesteps

    # What `just resume RUN_DIR` composes: no budget on the command line, so
    # rl.total_timesteps falls back to conf/rl/ppo.yaml's 5M. Adopting that as
    # the budget would silently sign the run up for another 5M steps — which
    # the stub turns into an immediate failure rather than an hours-long one.
    with pytest.MonkeyPatch.context() as no_training:
        no_training.setattr(PPO, "learn", _refuse_to_train)
        run_training(bare_resume_cfg(run_dir))
    assert PPO.load(run_dir / FINAL_MODEL, device="cpu").num_timesteps == trained

    # extend_timesteps is the one way to raise it, and it is absolute, not an
    # increment: the run stops at the first rollout boundary past 600.
    run_training(smoke_cfg(tmp_path, "ppo", extra=["resume=true", "extend_timesteps=600"]))
    extended = PPO.load(run_dir / FINAL_MODEL, device="cpu").num_timesteps
    assert 600 <= extended < 600 + 64 * 2  # n_steps * n_envs


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
    # save_freq=320 the last checkpoint is at 320. Nothing is left to train, so
    # the finals must be left exactly as the finished leg wrote them — rewriting
    # them would at best reproduce them and at worst (had the resume come off
    # that trailing checkpoint) roll them back 64 steps.
    run_dir = run_training(smoke_cfg(tmp_path, "ppo", extra=["rl.checkpoint.save_freq=320"]))
    before = PPO.load(run_dir / FINAL_MODEL, device="cpu").num_timesteps
    assert PPO.load(latest_checkpoint(run_dir), device="cpu").num_timesteps < before
    vecnorm_written = (run_dir / FINAL_VECNORM).stat().st_mtime_ns

    run_training(smoke_cfg(tmp_path, "ppo", extra=["resume=true"]))

    assert PPO.load(run_dir / FINAL_MODEL, device="cpu").num_timesteps == before
    assert (run_dir / FINAL_VECNORM).stat().st_mtime_ns == vecnorm_written


def test_extending_a_finished_run_starts_from_its_finals(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    # save_freq=320 against a 384-step rollout boundary leaves the finals ahead
    # of the last checkpoint. Rebuilding from that checkpoint would silently
    # drop the steps the finished run had already trained and published.
    run_dir = run_training(smoke_cfg(tmp_path, "ppo", extra=["rl.checkpoint.save_freq=320"]))
    final_steps = PPO.load(run_dir / FINAL_MODEL, device="cpu").num_timesteps
    assert PPO.load(latest_checkpoint(run_dir), device="cpu").num_timesteps < final_steps

    loaded = _spy_on_loads(monkeypatch, PPO)
    run_training(smoke_cfg(tmp_path, "ppo", extra=["resume=true", "extend_timesteps=600"]))
    assert loaded == [run_dir / FINAL_MODEL]

    extended = PPO.load(run_dir / FINAL_MODEL, device="cpu").num_timesteps
    assert 600 <= extended < 600 + 64 * 2  # n_steps * n_envs


def test_resume_falls_back_to_the_last_complete_checkpoint(
    tmp_path: Path, monkeypatch, capsys
):
    monkeypatch.setenv("WANDB_MODE", "offline")
    run_dir = run_training(smoke_cfg(tmp_path, "ppo"))
    _drop_finals(run_dir)
    # A kill between the newest checkpoint's zip and its siblings leaves that
    # checkpoint unusable, but the one before it is intact: resuming from it
    # costs one interval of re-training, where failing costs the whole run.
    ckpts = sorted((run_dir / CHECKPOINT_DIR).glob("model_*_steps.zip"), key=checkpoint_steps)
    newest, fallback = ckpts[-1], ckpts[-2]
    vecnormalize_for(newest).unlink()

    loaded = _spy_on_loads(monkeypatch, PPO)
    run_training(smoke_cfg(tmp_path, "ppo", extra=["resume=true"]))

    assert loaded == [fallback]
    warning = capsys.readouterr().out
    assert newest.name in warning and "vecnormalize sibling" in warning
    assert fallback.name in warning  # names the source it actually used


def test_a_crashed_extension_withdraws_the_final_marker(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    run_dir = run_training(smoke_cfg(tmp_path, "ppo"))
    assert (run_dir / FINAL_STEPS).exists()
    original_save = VecNormalize.save

    def crash_rewriting_the_finals(self, path):
        # The checkpoint siblings keep saving; only the finals blow up, which
        # is the window where final_model.zip is new and vecnormalize.pkl old.
        if str(path).endswith(FINAL_VECNORM):
            raise RuntimeError("crashed rewriting the finals")
        return original_save(self, path)

    with pytest.MonkeyPatch.context() as crashing:
        crashing.setattr(VecNormalize, "save", crash_rewriting_the_finals)
        with pytest.raises(RuntimeError, match="crashed rewriting the finals"):
            run_training(smoke_cfg(tmp_path, "ppo", extra=["resume=true", "extend_timesteps=600"]))

    # The marker vouches for a whole generation of finals. Half-replaced ones
    # must not inherit the old count, or the next resume trusts a model from
    # this leg paired with normalization statistics from the previous one.
    assert not (run_dir / FINAL_STEPS).exists()

    loaded = _spy_on_loads(monkeypatch, PPO)
    run_training(smoke_cfg(tmp_path, "ppo", extra=["resume=true"]))
    assert loaded and loaded[0].parent.name == CHECKPOINT_DIR


def test_usable_finals_rescue_a_run_whose_checkpoints_are_all_broken(
    tmp_path: Path, monkeypatch, capsys
):
    monkeypatch.setenv("WANDB_MODE", "offline")
    run_dir = run_training(smoke_cfg(tmp_path, "ppo"))
    newest = _keep_only_newest_checkpoint(run_dir)
    vecnormalize_for(newest).unlink()

    # Finals with their marker are a complete, self-consistent generation, so
    # unreadable checkpoints are no reason to refuse: they only mean the steps
    # past the finals have to be trained again.
    loaded = _spy_on_loads(monkeypatch, PPO)
    run_training(smoke_cfg(tmp_path, "ppo", extra=["resume=true", "extend_timesteps=600"]))

    assert loaded == [run_dir / FINAL_MODEL]
    warning = capsys.readouterr().out
    assert newest.name in warning and FINAL_MODEL in warning


def test_resume_refuses_when_every_checkpoint_is_incomplete(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    run_dir = run_training(smoke_cfg(tmp_path, "ppo"))
    _drop_finals(run_dir)
    vecnormalize_for(_keep_only_newest_checkpoint(run_dir)).unlink()

    # Only a run dir with no checkpoints at all may restart from scratch under
    # the same wandb id; one that has trained state it cannot read must stop.
    with pytest.raises(SystemExit, match="vecnormalize sibling"):
        run_training(smoke_cfg(tmp_path, "ppo", extra=["resume=true"]))


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


def test_sac_resume_reloads_replay_buffer(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    run_dir = run_training(smoke_cfg(tmp_path, "sac", extra=["rl.checkpoint.save_freq=64"]))
    ckpt = latest_checkpoint(run_dir)
    assert ckpt is not None and replay_buffer_for(ckpt) is not None
    filled_before = load_from_pkl(replay_buffer_for(ckpt)).pos

    run_training(smoke_cfg(tmp_path, "sac", extra=["resume=true", "extend_timesteps=400"]))
    model = SAC.load(run_dir / FINAL_MODEL, device="cpu")
    assert model.num_timesteps >= 400
    # The finals are what a further extend would resume from, so off-policy
    # runs have to leave a buffer beside them too.
    assert (run_dir / FINAL_REPLAY_BUFFER).exists()
    # A buffer that had been dropped instead of reloaded would hold only the
    # transitions collected after the resume, i.e. fewer than it already had.
    filled_after = load_from_pkl(replay_buffer_for(latest_checkpoint(run_dir))).pos
    assert filled_after > filled_before

    # Resuming SAC from a checkpoint whose buffer went missing would silently
    # restart from an empty one, so it has to fail instead.
    _drop_finals(run_dir)
    replay_buffer_for(_keep_only_newest_checkpoint(run_dir)).unlink()
    with pytest.raises(SystemExit, match="replay_buffer sibling"):
        run_training(smoke_cfg(tmp_path, "sac", extra=["resume=true", "extend_timesteps=800"]))


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
