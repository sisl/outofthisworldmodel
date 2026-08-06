import numpy as np
import pytest
from conftest import CONF_DIR, env_conf
from hydra import compose, initialize_config_dir
from owm_envs.envs.iss.config import ISSConfig
from stable_baselines3.common.env_checker import check_env

from owm.envs import factory
from owm.envs.factory import (
    iss_config,
    iss_config_from_dataset,
    make_iss_env,
    make_vec_env,
)


def test_iss_config_matches_preset_values():
    cfg = iss_config(env_conf())
    assert cfg.dt == 0.05
    assert cfg.max_steps == 7200
    assert cfg.observation.goal_error is True
    assert cfg.sensor_noise.enabled is True
    assert cfg.sensor_noise.sigma_pos_m == 0.05
    assert cfg.reward_weights.collision == -1_000_000.0


def test_make_iss_env_shapes_and_check():
    env = make_iss_env(iss_config(env_conf()), seed=0)
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


def test_iss_config_from_dataset_reads_shipped_yaml(tmp_path, monkeypatch):
    cfg_path = tmp_path / "env_config.yaml"
    ISSConfig(dt=0.1).to_yaml(cfg_path)

    seen_repo_ids = []

    def fake_hf_hub_download(*, repo_id, filename, repo_type):
        seen_repo_ids.append(repo_id)
        assert filename == "env_config.yaml"
        assert repo_type == "dataset"
        return str(cfg_path)

    monkeypatch.setattr(factory, "hf_hub_download", fake_hf_hub_download)

    cfg = iss_config({"from_dataset_repo": "org/name"})
    assert cfg.dt == 0.1
    assert seen_repo_ids == ["org/name"]


@pytest.mark.network
def test_iss_config_from_dataset_matches_main_repo():
    cfg = iss_config_from_dataset("sislaboratory/owm-iss-coop-goal-dt50ms")
    assert cfg.max_steps == 7200


def test_from_dataset_environments_group_composes():
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(
            config_name="config", overrides=["~rl", "environments=from_dataset"]
        )
    assert cfg.environments.from_dataset_repo == "sislaboratory/owm-iss-coop-goal-dt50ms"
