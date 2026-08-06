from conftest import CONF_DIR
from hydra import compose, initialize_config_dir


def test_root_config_composes():
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=["~rl"])
    assert cfg.environments.dt == 0.05
    assert cfg.logging.mode in ("online", "offline", "disabled")
