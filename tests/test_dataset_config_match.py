"""Cross-check conf/environments/iss_coop_goal.yaml against the published
dataset's as-run env_config.yaml (uploads mirror the generation run).

Checked against sislaboratory/owm-iss-coop-goal-dt50ms (main), not the
-trial variant: the trial dataset predates owm-envs' "shorten the horizon
to 360 s" change and records max_steps 12000, so it is not config-identical
to main and would fail this check on that field alone."""

import os

import pytest
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from owm_envs.envs.iss.config import ISSConfig

from owm.envs.factory import env_config
from conftest import CONF_DIR
from hydra import compose, initialize_config_dir

DATASET_REPO = os.environ.get(
    "OWM_HF_DATASET_REPO", "sislaboratory/owm-iss-coop-goal-dt50ms"
)


@pytest.mark.network
def test_env_config_matches_dataset():
    path = hf_hub_download(
        repo_id=DATASET_REPO, filename="env_config.yaml", repo_type="dataset"
    )
    shipped = OmegaConf.to_container(OmegaConf.load(path))
    if "angular_velocity" in shipped.get("reward_weights", {}):
        pytest.xfail(
            "published dataset's env_config.yaml carries the pre-reshape "
            "reward schema; regenerate the dataset under the current "
            "owm-envs reward to reinstate this check"
        )
    published = ISSConfig.from_yaml(path)
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        env_conf = compose(
            config_name="config", overrides=["environments=iss_coop_goal"]
        ).environments
    ours = env_config(env_conf)
    assert ours == published
