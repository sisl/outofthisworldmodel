from pathlib import Path

from hydra import compose, initialize_config_dir

CONF_DIR = str(Path(__file__).resolve().parent.parent / "conf")


def env_conf():
    # Temporary: rl group has no real config yet, so compose can't resolve
    # it. Drop the ~rl override once conf/rl/ppo.yaml exists (Task 4).
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        return compose(config_name="config", overrides=["~rl"]).environments
