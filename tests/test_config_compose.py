import inspect

import pytest
from conftest import CONF_DIR
from hydra import compose, initialize_config_dir

from owm.baselines.rl.train import ALGOS
from owm.envs.factory import iss_config

# The published 500k dataset's train-split port set, in the order
# owm-envs configs/generation_500k.yaml's splits.train.policy.dock.ports
# lists them. Order maps the recorded positional dock_port_index, so this
# is a hard assertion protecting that index mapping, not just membership.
TRAIN_PORT_NAMES = (
    "harmony_fwd_pma2",
    "harmony_nadir_cbm",
    "zvezda_aft",
    "pirs_nadir",
    "rassvet_nadir",
)
HELDOUT_PORT_NAMES = ("harmony_zenith_cbm", "poisk_zenith")


@pytest.mark.parametrize("rl", ["ppo", "sac", "ppo_tuned", "sac_tuned"])
def test_root_config_composes(rl):
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"rl={rl}"])
    assert cfg.environments.dt == 0.05
    assert cfg.rl.algo == rl.removesuffix("_tuned")
    assert cfg.rl.checkpoint.save_freq > 0
    assert cfg.video.enabled is False


@pytest.mark.parametrize("rl", ["ppo_tuned", "sac_tuned"])
def test_tuned_config_carries_a_winners_hyperparams(rl):
    # A tuned config with an empty or missing hyperparams block would silently
    # fall back to whatever conf/rl's base defaults are, defeating the freeze.
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"rl={rl}"])
    hyperparams = dict(cfg.rl.hyperparams)
    assert hyperparams
    # Every key must be a real SB3 constructor arg, same bar the sweep specs
    # are held to (tests/test_sweep_specs.py), since this file is not one of
    # the sweeps/*.yaml globs that guard covers.
    accepted = set(inspect.signature(ALGOS[cfg.rl.algo].__init__).parameters)
    assert set(hyperparams) <= accepted, set(hyperparams) - accepted


def test_ports_environment_composes_with_the_train_port_set():
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=["environments=iss_coop_goal_ports"])
    resolved = iss_config(cfg.environments)
    assert tuple(port.name for port in resolved.dock.ports) == TRAIN_PORT_NAMES


def test_heldout_ports_environment_composes_with_the_val_only_port_set():
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(
            config_name="config", overrides=["environments=iss_coop_goal_ports_heldout"]
        )
    resolved = iss_config(cfg.environments)
    assert tuple(port.name for port in resolved.dock.ports) == HELDOUT_PORT_NAMES
