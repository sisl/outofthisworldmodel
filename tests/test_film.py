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
from owm.baselines.rl.film import classify_outcome, fly_episode
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
