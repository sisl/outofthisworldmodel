from pathlib import Path

from owm.baselines.rl.run_state import (
    CHECKPOINT_DIR,
    latest_checkpoint,
    load_wandb_id,
    replay_buffer_for,
    save_wandb_id,
    vecnormalize_for,
)


def test_wandb_id_roundtrip(tmp_path: Path):
    assert load_wandb_id(tmp_path) is None
    save_wandb_id(tmp_path, "abc123")
    assert load_wandb_id(tmp_path) == "abc123"


def test_latest_checkpoint_picks_highest_step(tmp_path: Path):
    ckpts = tmp_path / CHECKPOINT_DIR
    ckpts.mkdir()
    for n in (1000, 9000, 20000):
        (ckpts / f"model_{n}_steps.zip").touch()
    (ckpts / "model_replay_buffer_20000_steps.pkl").touch()
    (ckpts / "model_vecnormalize_9000_steps.pkl").touch()
    latest = latest_checkpoint(tmp_path)
    assert latest is not None and latest.name == "model_20000_steps.zip"
    assert replay_buffer_for(latest).name == "model_replay_buffer_20000_steps.pkl"
    assert vecnormalize_for(latest) is None  # only the 9000-step stats exist


def test_latest_checkpoint_empty(tmp_path: Path):
    assert latest_checkpoint(tmp_path) is None


def test_sibling_lookups_none_for_non_checkpoint_names(tmp_path: Path):
    assert replay_buffer_for(tmp_path / "final_model.zip") is None
    assert vecnormalize_for(tmp_path / "not_a_checkpoint.zip") is None
