import pytest
from conftest import CONF_DIR
from hydra import compose, initialize_config_dir

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


@pytest.mark.parametrize("rl", ["ppo", "sac"])
def test_root_config_composes(rl):
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"rl={rl}"])
    assert cfg.environments.dt == 0.05
    assert cfg.rl.algo == rl
    assert cfg.rl.checkpoint.save_freq > 0
    assert cfg.video.enabled is False


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
