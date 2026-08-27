import json
from pathlib import Path

# Ahead of the owm_envs imports below: importing it pins JAX to CPU, which XLA
# reads when its backend first comes up, and owm_envs brings that backend up.
from owm.envs.factory import env_config, make_env  # isort: skip

import numpy as np
import pytest
from conftest import smoke_cfg
from owm_envs.datasets.trajectory import load_trajectory, save_trajectory, start_fingerprint

from owm.baselines.rl.eval_matrix import at_rate, base_env_conf, for_port
from owm.baselines.rl.evaluate import load_normalizer
from owm.baselines.rl.film import (
    DEFAULT_CHECKPOINT,
    EVAL_DROPS,
    PAPER_EVALS,
    classify_outcome,
    eval_outcome,
    fly_episode,
    resolve_evals,
    run_film,
)
from owm.baselines.rl.run_state import FINAL_MODEL, load_run_config
from owm.baselines.rl.train import ALGOS, run_training

PORT = "harmony_fwd_pma2"


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory) -> Path:
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("WANDB_MODE", "offline")
        return run_training(smoke_cfg(tmp_path_factory.mktemp("trained"), "ppo"))


def short_cfg(run_dir: Path, rate_hz: float):
    base = at_rate(base_env_conf(run_dir, load_run_config(run_dir)), rate_hz)
    # 2 s of flight is enough to record a handful of steps; the horizon is not under test.
    base = {**base, "max_steps": int(2 * rate_hz)}
    return env_config(for_port(base, PORT))


def test_classify_outcome_orders_the_flags():
    assert classify_outcome(docked=True, escaped=False, ever_collided=True) == "docked"
    assert classify_outcome(docked=False, escaped=True, ever_collided=True) == "escaped"
    assert classify_outcome(docked=False, escaped=False, ever_collided=True) == "collision"
    assert classify_outcome(docked=False, escaped=False, ever_collided=False) == "truncated"


def test_fly_records_every_integration_step(trained_run, tmp_path):
    ckpt = trained_run / FINAL_MODEL
    model = ALGOS["ppo"].load(ckpt, device="cpu")
    vecnorm = load_normalizer(ckpt, allow_unnormalized=True)
    cfg = short_cfg(trained_run, rate_hz=20)
    traj = fly_episode(model, vecnorm, cfg, seed=7, action_repeat=20, rate_hz=20,
                       port=PORT, lighting="unknown", produced_by=str(ckpt))
    assert traj.steps == 40                     # 2 s at 20 Hz, whatever the decision cadence
    assert traj.meta["dt"] == pytest.approx(0.05)
    assert traj.meta["rate_hz"] == 20 and traj.meta["action_repeat"] == 20
    assert traj.meta["method"] == "rl" and traj.meta["port"] == PORT and traj.meta["seed"] == 7
    # The action is held across each repeat.
    np.testing.assert_array_equal(traj.action_norm[0], traj.action_norm[19])
    save_trajectory(traj, tmp_path)
    assert load_trajectory(tmp_path).steps == 40


def test_start_fingerprint_matches_a_direct_reset(trained_run):
    ckpt = trained_run / FINAL_MODEL
    model = ALGOS["ppo"].load(ckpt, device="cpu")
    vecnorm = load_normalizer(ckpt, allow_unnormalized=True)
    cfg = short_cfg(trained_run, rate_hz=20)
    traj = fly_episode(model, vecnorm, cfg, seed=11, action_repeat=20, rate_hz=20,
                       port=PORT, lighting="unknown", produced_by="test")
    env = make_env(cfg, seed=11)
    _, info = env.reset(seed=11)
    env.close()
    assert traj.meta["start_fingerprint"] == start_fingerprint(info["state"])
    # And the 1 Hz variant starts from the same state, which is what lets the
    # film be paired with the 1 Hz evaluation.
    slow = make_env(short_cfg(trained_run, rate_hz=1), seed=11)
    _, slow_info = slow.reset(seed=11)
    slow.close()
    assert start_fingerprint(slow_info["state"]) == traj.meta["start_fingerprint"]


def test_eval_outcome_reads_the_matching_row(tmp_path):
    (tmp_path / "episodes.csv").write_text(
        "port,split,trial,seed,steps,outcome\n"
        "harmony_fwd_pma2,train,0,100000,360,truncated\n"
        "harmony_fwd_pma2,train,1,100001,200,docked\n"
    )
    assert eval_outcome(tmp_path, PORT, 100001) == "docked"
    assert eval_outcome(tmp_path, PORT, 999) is None
    assert eval_outcome(None, PORT, 100001) is None
    assert eval_outcome(tmp_path / "absent", PORT, 100001) is None


def test_run_film_writes_a_trajectory_per_rl_row(trained_run, tmp_path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "rollouts:\n"
        "  - {name: a, port: harmony_fwd_pma2, seed: 3, lighting: sunlit, distribution: train,\n"
        "     methods: {rl: {rate_hz: 20, action_repeat: 20}}}\n"
        "  - {name: b, port: rassvet_nadir, seed: 4, lighting: eclipse, distribution: train,\n"
        "     methods: {wm: {rate_hz: 20, action_repeat: 5}}}\n"
    )
    out = tmp_path / "rollouts"
    results = run_film(manifest, out, str(trained_run / FINAL_MODEL), views="fpv",
                       stride=1, force=False, only=[], evals_dir=None, render=False,
                       max_steps=40)
    assert [r["name"] for r in results] == ["a"]
    assert (out / "a" / "trajectory.npz").exists()
    assert (out / "a" / "meta.json").exists()
    assert not (out / "b").exists()
    # A second run skips the finished row unless forced.
    again = run_film(manifest, out, str(trained_run / FINAL_MODEL), views="fpv",
                     stride=1, force=False, only=[], evals_dir=None, render=False,
                     max_steps=40)
    assert again[0]["skipped"] is True
    assert results[0]["skipped"] is False
    assert again[0]["start_fingerprint"] == results[0]["start_fingerprint"]


def test_run_film_only_refuses_a_row_it_cannot_film(trained_run, tmp_path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "rollouts:\n"
        "  - {name: a, port: harmony_fwd_pma2, seed: 3, lighting: sunlit, distribution: train,\n"
        "     methods: {rl: {rate_hz: 20, action_repeat: 20}}}\n"
        "  - {name: b, port: rassvet_nadir, seed: 4, lighting: eclipse, distribution: train,\n"
        "     methods: {wm: {rate_hz: 20, action_repeat: 5}}}\n"
    )
    with pytest.raises(SystemExit, match="b"):
        run_film(manifest, tmp_path / "rollouts", str(trained_run / FINAL_MODEL),
                 render=False, only=["b"], max_steps=40)


def test_default_checkpoint_is_the_terminal_shaping_run():
    assert DEFAULT_CHECKPOINT == (
        "runs/best/owm-iss-numerical-v1-coop-terminal-ppo-vector/final_model.zip")
    assert "-terminal-" in DEFAULT_CHECKPOINT


def test_ppo_eval_drops_name_the_paper_drop_first():
    assert EVAL_DROPS["ppo"][0] == PAPER_EVALS / "ppo_coop"
    assert Path("runs/evals/paper/ppo_coop") in EVAL_DROPS["ppo"]
    assert PAPER_EVALS / "owm_coop" in EVAL_DROPS["owm"]


def test_resolve_evals_takes_the_first_drop_that_exists(tmp_path):
    present = tmp_path / "present"
    present.mkdir()
    resolved = resolve_evals({
        "a": (tmp_path / "absent", present),
        "b": (tmp_path / "gone", tmp_path / "also_gone"),
    })
    assert resolved == {"a": present}


def test_eval_outcome_classifies_the_flag_columns(tmp_path):
    (tmp_path / "episodes.csv").write_text(
        "port,split,trial,seed,steps,outcome,env_docked,ever_collided,escaped\n"
        f"{PORT},train,0,100000,360,truncated,False,True,False\n"
        f"{PORT},train,1,100001,360,truncated,False,False,False\n"
        f"{PORT},train,2,100002,200,docked,True,True,False\n"
    )
    # The drops never say "collision": a truncated row that grazed the zone is
    # a collision in film's vocabulary.
    assert eval_outcome(tmp_path, PORT, 100000) == "collision"
    assert eval_outcome(tmp_path, PORT, 100001) == "truncated"
    assert eval_outcome(tmp_path, PORT, 100002) == "docked"


def _one_row_manifest(tmp_path: Path, lighting: str = "sunlit") -> Path:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "rollouts:\n"
        f"  - {{name: a, port: {PORT}, seed: 3, lighting: {lighting}, distribution: train,\n"
        "     methods: {rl: {rate_hz: 20, action_repeat: 20}}}\n"
    )
    return manifest


def test_skip_refuses_a_row_flown_by_another_checkpoint(trained_run, tmp_path):
    manifest = _one_row_manifest(tmp_path)
    out = tmp_path / "rollouts"
    run_film(manifest, out, str(trained_run / FINAL_MODEL), views="fpv", render=False,
             max_steps=40)
    meta_path = out / "a" / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["produced_by"] = "runs/best/somewhere-else/final_model.zip"
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(SystemExit, match="produced_by"):
        run_film(manifest, out, str(trained_run / FINAL_MODEL), views="fpv", render=False,
                 max_steps=40)
    # --force refilms it rather than refusing.
    forced = run_film(manifest, out, str(trained_run / FINAL_MODEL), views="fpv", force=True,
                      render=False, max_steps=40)
    assert forced[0]["skipped"] is False


def test_skip_refuses_a_row_flown_at_another_cadence(trained_run, tmp_path):
    manifest = _one_row_manifest(tmp_path)
    out = tmp_path / "rollouts"
    run_film(manifest, out, str(trained_run / FINAL_MODEL), views="fpv", render=False,
             max_steps=40)
    manifest.write_text(manifest.read_text().replace("action_repeat: 20", "action_repeat: 10"))
    with pytest.raises(SystemExit, match="action_repeat"):
        run_film(manifest, out, str(trained_run / FINAL_MODEL), views="fpv", render=False,
                 max_steps=40)


def test_skip_rewrites_the_stored_lighting(trained_run, tmp_path, capsys):
    manifest = _one_row_manifest(tmp_path, lighting="sunlit")
    out = tmp_path / "rollouts"
    run_film(manifest, out, str(trained_run / FINAL_MODEL), views="fpv", render=False,
             max_steps=40)
    manifest.write_text(manifest.read_text().replace("lighting: sunlit", "lighting: eclipse"))
    again = run_film(manifest, out, str(trained_run / FINAL_MODEL), views="fpv", render=False,
                     max_steps=40)
    assert again[0]["skipped"] is True and again[0]["relit"] is True
    assert again[0]["lighting"] == "eclipse"
    assert json.loads((out / "a" / "meta.json").read_text())["lighting"] == "eclipse"
    assert "lighting updated" in capsys.readouterr().out


def test_summary_prints_the_collision_flag(trained_run, tmp_path, capsys):
    manifest = _one_row_manifest(tmp_path)
    results = run_film(manifest, tmp_path / "rollouts", str(trained_run / FINAL_MODEL),
                       views="fpv", render=False, max_steps=40)
    expected = "Y" if results[0]["ever_collided"] else "N"
    assert f"collided={expected}" in capsys.readouterr().out
