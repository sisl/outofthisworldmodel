from pathlib import Path

import wandb
from conftest import smoke_cfg

from owm.baselines.rl import train
from owm.baselines.rl.run_state import FINAL_MODEL
from owm.baselines.rl.sweep_callbacks import OBJECTIVE, EvalReportCallback
from owm.baselines.rl.train import run_training


def test_a_trial_trains_inside_the_run_its_agent_opened(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    cfg = smoke_cfg(tmp_path, "ppo", extra=["external_wandb=true"])

    # Stands in for the wandb agent, which opens the trial's run before the
    # trial's code gets to say anything.
    run = wandb.init(project="owm-sweep-test", dir=str(tmp_path), mode="offline")

    logged: list[dict] = []
    real_log = wandb.log

    def spy(data, *args, **kwargs):
        logged.append(data)
        return real_log(data, *args, **kwargs)

    monkeypatch.setattr(wandb, "log", spy)

    def second_run(*args, **kwargs):
        raise AssertionError("run_training opened a second wandb run")

    monkeypatch.setattr(train.wandb, "init", second_run)

    callback = EvalReportCallback(
        run_dir=Path(cfg.run_dir),
        every_steps=100,
        episodes=1,
        final_episodes=1,
        seed=99,
        max_episode_steps=3,  # the env's own limit is 7200 steps
        vec="dummy",
        env_name=str(cfg.environments.env_name),
    )
    run_dir = run_training(cfg, extra_callbacks=[callback])

    # Still the agent's run, still open: run_training must not finish what it
    # did not start, or the trial's later logging would go nowhere.
    assert wandb.run is run
    wandb.finish()

    assert (run_dir / FINAL_MODEL).exists()
    # An external run's id belongs to the agent; recording it would invite a
    # resume to reattach to a run nothing holds open.
    assert not (run_dir / "wandb_run_id.txt").exists()

    objective = [payload for payload in logged if OBJECTIVE in payload]
    assert len(objective) >= 2, "expected periodic reports and a final one"
    assert sum("sweep/final_mean_return" in payload for payload in objective) == 1
