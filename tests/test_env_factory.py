from dataclasses import replace

import numpy as np
import pytest
from conftest import CONF_DIR, env_conf
from hydra import compose, initialize_config_dir
from owm_envs.envs.common.goal import GOAL_ERROR_DIM
from owm_envs.envs.iss.config import ISSConfig
from owm_envs.envs.iss_numerical.config import OBS_MODE_DIM, NumericalConfig
from stable_baselines3.common.env_checker import check_env

from owm.baselines.rl.metrics import GOAL_ERROR_KEYS

from owm.envs import factory
from owm.envs.factory import (
    env_config,
    env_config_from_dataset,
    make_env,
    make_vec_env,
)


def test_env_config_matches_preset_values():
    cfg = env_config(env_conf())
    assert cfg.dt == 0.05
    assert cfg.max_steps == 7200
    assert cfg.observation.goal_error is True
    assert cfg.sensor_noise.enabled is True
    assert cfg.sensor_noise.sigma_pos_m == 0.05
    assert cfg.reward_weights.collision == -1_000_000.0


def test_make_env_shapes_and_check():
    env = make_env(env_config(env_conf()), seed=0)
    assert env.observation_space.shape == (25,)  # 13 state + 12 goal-error
    assert env.action_space.shape == (6,)
    assert np.allclose(env.action_space.low, -1.0)
    assert np.allclose(env.action_space.high, 1.0)
    check_env(env.unwrapped, warn=True, skip_render_check=True)
    obs, _ = env.reset(seed=0)
    obs2, reward, term, trunc, info = env.step(np.zeros(6, dtype=np.float32))
    assert obs2.shape == (25,)
    assert {"success", "collision", "escaped"} <= info.keys()


def test_make_vec_env_dummy():
    venv = make_vec_env(env_conf(), n_envs=2, seed=0, vec="dummy")
    obs = venv.reset()
    assert obs.shape == (2, 25)
    venv.close()


def test_env_config_from_dataset_reads_shipped_yaml(tmp_path, monkeypatch):
    cfg_path = tmp_path / "env_config.yaml"
    ISSConfig(dt=0.1).to_yaml(cfg_path)

    seen = []

    def fake_hf_hub_download(*, repo_id, filename, repo_type, revision):
        seen.append((repo_id, revision))
        assert filename == "env_config.yaml"
        assert repo_type == "dataset"
        return str(cfg_path)

    monkeypatch.setattr(factory, "hf_hub_download", fake_hf_hub_download)

    cfg = env_config({"from_dataset_repo": "org/name"})
    assert cfg.dt == 0.1
    # No revision key: the default branch, whatever the dataset's HEAD is.
    assert seen == [("org/name", None)]

    env_config({"from_dataset_repo": "org/name", "from_dataset_revision": "v2"})
    assert seen[-1] == ("org/name", "v2")


@pytest.mark.network
def test_env_config_from_dataset_matches_main_repo():
    cfg = env_config_from_dataset("sislaboratory/owm-iss-coop-goal-dt50ms")
    assert cfg.max_steps == 7200


def test_from_dataset_environments_group_composes():
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=["environments=from_dataset"])
    assert cfg.environments.from_dataset_repo == "sislaboratory/owm-iss-coop-goal-dt50ms"


def _env_conf_for(group: str):
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        return compose(config_name="config", overrides=[f"environments={group}"]).environments


def test_ports_config_draws_a_varying_dock_port_across_resets():
    cfg = env_config(_env_conf_for("iss_coop_goal_ports"))
    env = make_env(cfg, seed=0)

    ports_seen = set()
    goal_poses = set()
    for seed in range(12):
        _, info = env.reset(seed=seed)
        ports_seen.add(info["dock_port"])
        goal_poses.add(tuple(np.asarray(info["goal_pose"]).tolist()))

    assert len(ports_seen) > 1
    assert len(goal_poses) > 1


def test_no_ports_config_never_reports_a_dock_port():
    cfg = env_config(env_conf())
    env = make_env(cfg, seed=0)

    _, info = env.reset(seed=0)
    assert "dock_port" not in info
    assert np.allclose(info["goal_pose"][:3], cfg.dock.position)


NUMERICAL_GROUP = "iss_numerical_ports"
# 2 epoch + 13 relative_view + 12 goal-error. The relative_view block is
# element for element what the iss env carries in-state, so this is the 25-dim
# iss observation plus the [jd, sec] epoch prefix.
NUMERICAL_OBS_DIM = 27


def test_env_name_routes_to_the_registry_config_class():
    cfg = env_config(_env_conf_for(NUMERICAL_GROUP))
    assert type(cfg) is NumericalConfig
    assert factory.env_name_of(cfg) == "iss-numerical"
    # The reserved key is consumed, not passed through to a config that
    # forbids extra fields.
    assert not hasattr(cfg, factory.ENV_NAME_KEY)


def test_env_conf_dict_round_trips_through_env_config():
    """What training hands a worker must rebuild the same config, env included.

    The worker gets a plain dict, not the config object, so `env_name` has to
    survive the trip or the far side would validate a NumericalConfig's fields
    as an ISSConfig's.
    """
    for group in ("iss_coop_goal_ports", NUMERICAL_GROUP):
        cfg = env_config(_env_conf_for(group))
        assert env_config(factory.env_conf_dict(cfg)) == cfg


def test_unknown_env_name_names_the_registered_suite():
    with pytest.raises(ValueError, match="unknown env_name 'iss-imaginary'"):
        env_config({factory.ENV_NAME_KEY: "iss-imaginary"})


def test_numerical_env_observation_matches_the_configured_mode():
    cfg = env_config(_env_conf_for(NUMERICAL_GROUP))
    assert cfg.observation.mode == "relative"
    # The width the upstream mode table documents, before the goal block.
    assert OBS_MODE_DIM[cfg.observation.mode] + GOAL_ERROR_DIM == NUMERICAL_OBS_DIM

    env = make_env(cfg, seed=0)
    assert env.observation_space.shape == (NUMERICAL_OBS_DIM,)
    assert env.action_space.shape == (6,)
    check_env(env.unwrapped, warn=True, skip_render_check=True)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (NUMERICAL_OBS_DIM,)
    obs2, _, _, _, info = env.step(np.zeros(6, dtype=np.float32))
    assert obs2.shape == (NUMERICAL_OBS_DIM,)
    assert {"success", "collision", "escaped"} <= info.keys()


def test_numerical_env_emits_the_telemetry_the_docking_callback_reads():
    """DockingMetricsCallback self-disables when these keys are missing."""
    cfg = env_config(_env_conf_for(NUMERICAL_GROUP))
    env = make_env(cfg, seed=0)
    _, info = env.reset(seed=0)
    assert set(GOAL_ERROR_KEYS) == set(info["goal_error_true"])
    assert np.asarray(info["goal_pose"]).shape == (7,)
    assert info["dock_port"] in {p.name for p in cfg.dock.ports}
    assert info["dock_port_index"] == [p.name for p in cfg.dock.ports].index(
        info["dock_port"]
    )


def test_numerical_env_reset_is_reproducible_for_a_seed():
    cfg = env_config(_env_conf_for(NUMERICAL_GROUP))
    first = make_env(cfg, seed=0)
    second = make_env(cfg, seed=123)  # a different construction seed

    obs_a, info_a = first.reset(seed=7)
    obs_b, info_b = second.reset(seed=7)
    np.testing.assert_array_equal(obs_a, obs_b)
    assert info_a["dock_port"] == info_b["dock_port"]
    np.testing.assert_array_equal(info_a["goal_pose"], info_b["goal_pose"])

    # ... and a different seed is a different episode, so the equality above
    # is reproducibility rather than a constant start.
    obs_c, _ = first.reset(seed=8)
    assert not np.array_equal(obs_a, obs_c)


def test_numerical_ports_config_draws_a_varying_dock_port_across_resets():
    cfg = env_config(_env_conf_for(NUMERICAL_GROUP))
    env = make_env(cfg, seed=0)

    ports_seen = set()
    goal_poses = set()
    for seed in range(12):
        _, info = env.reset(seed=seed)
        ports_seen.add(info["dock_port"])
        goal_poses.add(tuple(np.asarray(info["goal_pose"]).tolist()))

    assert len(ports_seen) > 1
    assert len(goal_poses) == len(ports_seen)


def test_numerical_train_ports_match_the_iss_train_split():
    """One port mechanism, not two: the same names must mean the same poses.

    dock.ports is a field of the shared DockConfig and the per-episode draw is
    the shared PortGoalMixin, so a port set written for the iss env and one
    written here name the same episodes. This pins that, since it is the whole
    basis for reading a numerical run's port-conditioned numbers against an
    iss run's.
    """
    iss_cfg = env_config(_env_conf_for("iss_coop_goal_ports"))
    num_cfg = env_config(_env_conf_for(NUMERICAL_GROUP))
    assert [p.name for p in num_cfg.dock.ports] == [p.name for p in iss_cfg.dock.ports]
    for iss_port, num_port in zip(iss_cfg.dock.ports, num_cfg.dock.ports):
        assert iss_port == num_port


def test_make_vec_env_dummy_on_the_numerical_env():
    venv = make_vec_env(
        factory.env_conf_dict(env_config(_env_conf_for(NUMERICAL_GROUP))),
        n_envs=2,
        seed=0,
        vec="dummy",
    )
    obs = venv.reset()
    assert obs.shape == (2, NUMERICAL_OBS_DIM)
    venv.close()


def test_env_name_of_refuses_an_ambiguous_config_class(monkeypatch):
    """Two envs sharing a config class must raise, not pick by iteration order.

    env_name_of recovers the env from the config alone, which only works while
    the registry keeps one config class per env. That is an upstream invariant
    this repo cannot enforce, so the failure mode worth pinning is what happens
    when it breaks: a named error rather than a silent route to whichever env
    the registry listed first.
    """
    from owm_envs.envs import ENV_REGISTRY

    spec = ENV_REGISTRY["iss"]
    doubled = {**ENV_REGISTRY, "iss-twin": replace(spec, name="iss-twin")}
    monkeypatch.setattr(factory, "ENV_REGISTRY", doubled)

    with pytest.raises(ValueError, match="more than one env"):
        factory.env_name_of(env_config(env_conf()))


def test_env_name_of_rejects_a_config_class_the_registry_does_not_know():
    class NotAnEnvConfig(ISSConfig):
        pass

    # A subclass, deliberately: it is not the registered class, and matching it
    # by isinstance would route it to `iss` while carrying fields ISSConfig
    # cannot round-trip.
    with pytest.raises(ValueError, match="not the config class of any env"):
        factory.env_name_of(NotAnEnvConfig())
