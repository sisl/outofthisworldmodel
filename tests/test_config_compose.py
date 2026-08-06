import pytest
from conftest import CONF_DIR
from hydra import compose, initialize_config_dir


@pytest.mark.parametrize("rl", ["ppo", "sac"])
def test_root_config_composes(rl):
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"rl={rl}"])
    assert cfg.environments.dt == 0.05
    assert cfg.rl.algo == rl
    assert cfg.rl.checkpoint.save_freq > 0
    assert cfg.video.enabled is False
