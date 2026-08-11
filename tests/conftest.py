from pathlib import Path

from hydra import compose, initialize_config_dir

CONF_DIR = str(Path(__file__).resolve().parent.parent / "conf")

SMOKE = [
    "rl.n_envs=2",
    "rl.vec=dummy",
    "rl.device=cpu",
    "rl.total_timesteps=300",
    "rl.checkpoint.save_freq=128",
    "video.enabled=false",
    "hub.upload=false",
]


def env_conf():
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        return compose(config_name="config").environments


def smoke_cfg(tmp_path: Path, algo: str, extra: list[str] = ()):
    overrides = [f"rl={algo}", f"run_dir={tmp_path / 'run'}", *SMOKE, *extra]
    if algo == "ppo":
        overrides += ["rl.hyperparams.n_steps=64", "rl.hyperparams.batch_size=64"]
    else:
        overrides += ["rl.hyperparams.learning_starts=50", "rl.hyperparams.buffer_size=1000"]
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        return compose(config_name="config", overrides=overrides)
