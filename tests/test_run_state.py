from pathlib import Path

from owm.baselines.rl.run_state import (
    CHECKPOINT_DIR,
    latest_checkpoint,
    latest_complete_checkpoint,
    load_final_steps,
    load_wandb_id,
    replay_buffer_for,
    save_final_steps,
    save_wandb_id,
    vecnormalize_for,
)


def test_wandb_id_roundtrip(tmp_path: Path):
    assert load_wandb_id(tmp_path) is None
    save_wandb_id(tmp_path, "abc123")
    assert load_wandb_id(tmp_path) == "abc123"


def test_final_steps_roundtrip(tmp_path: Path):
    assert load_final_steps(tmp_path) is None
    save_final_steps(tmp_path, 5_000_192)
    assert load_final_steps(tmp_path) == 5_000_192


def test_latest_complete_checkpoint_skips_a_half_written_newest(tmp_path: Path):
    ckpts = tmp_path / CHECKPOINT_DIR
    ckpts.mkdir()
    for n in (128, 256):
        (ckpts / f"model_{n}_steps.zip").touch()
    # A crash between the zip and its siblings leaves 256 unusable; 128 is
    # older but complete, and a resume from it loses only one interval.
    (ckpts / "model_vecnormalize_128_steps.pkl").touch()
    (ckpts / "model_replay_buffer_128_steps.pkl").touch()

    assert latest_checkpoint(tmp_path).name == "model_256_steps.zip"
    assert latest_complete_checkpoint(tmp_path, need_replay_buffer=False).name == (
        "model_128_steps.zip"
    )
    assert latest_complete_checkpoint(tmp_path, need_replay_buffer=True).name == (
        "model_128_steps.zip"
    )


def test_latest_complete_checkpoint_needs_the_buffer_only_for_off_policy(tmp_path: Path):
    ckpts = tmp_path / CHECKPOINT_DIR
    ckpts.mkdir()
    (ckpts / "model_128_steps.zip").touch()
    (ckpts / "model_vecnormalize_128_steps.pkl").touch()

    assert latest_complete_checkpoint(tmp_path, need_replay_buffer=False) is not None
    assert latest_complete_checkpoint(tmp_path, need_replay_buffer=True) is None


def test_latest_complete_checkpoint_empty(tmp_path: Path):
    assert latest_complete_checkpoint(tmp_path, need_replay_buffer=False) is None


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
