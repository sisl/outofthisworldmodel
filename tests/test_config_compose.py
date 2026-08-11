import inspect

import pytest
from conftest import CONF_DIR
from hydra import compose, initialize_config_dir

from owm.baselines.rl.train import ALGOS
from owm.envs.factory import env_config

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
    resolved = env_config(cfg.environments)
    assert tuple(port.name for port in resolved.dock.ports) == TRAIN_PORT_NAMES


def test_heldout_ports_environment_composes_with_the_val_only_port_set():
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(
            config_name="config", overrides=["environments=iss_coop_goal_ports_heldout"]
        )
    resolved = env_config(cfg.environments)
    assert tuple(port.name for port in resolved.dock.ports) == HELDOUT_PORT_NAMES


def test_numerical_ports_environment_composes_with_the_train_port_set():
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=["environments=iss_numerical_ports"])
    resolved = env_config(cfg.environments)
    assert tuple(port.name for port in resolved.dock.ports) == TRAIN_PORT_NAMES
    # The start shell this env actually disperses over lives under `orbit`;
    # physics.start_radius_range_m is never read here, so asserting through
    # start_shell() is the only way to catch the shell being written to the
    # field that does nothing.
    assert resolved.start_shell() == (100.0, 500.0)
    assert resolved.observation.mode == "relative"
    assert resolved.observation.goal_error is True
    # The force model is the point of this env; a config that silently lost it
    # would train two-body dynamics under a numerical env's name.
    assert resolved.perturbations.zonal_max_degree == 4
    assert resolved.perturbations.third_body_sun
    assert resolved.perturbations.third_body_moon
    assert resolved.perturbations.drag
    # Reward semantics are held identical to the iss port config: this
    # migration changes the dynamics and the observation, never the reward.
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        iss_cfg = compose(config_name="config", overrides=["environments=iss_coop_goal_ports"])
    iss_resolved = env_config(iss_cfg.environments)
    assert resolved.reward_weights == iss_resolved.reward_weights
    assert resolved.reward_goal_position == iss_resolved.reward_goal_position
    assert resolved.dock.max_distance_m == iss_resolved.dock.max_distance_m
    assert resolved.dock.max_velocity_m_s == iss_resolved.dock.max_velocity_m_s
