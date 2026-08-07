import inspect
from pathlib import Path

import pytest
import yaml

from owm.baselines.rl.sweep_callbacks import OBJECTIVE
from owm.baselines.rl.sweep_trial import RESERVED_KEYS, RESOURCES
from owm.baselines.rl.train import ALGOS

SWEEPS = Path(__file__).resolve().parent.parent / "sweeps"
# One spec per (algo, obs mode); the pixel pair lands with rl.obs.
VECTOR_SPECS = {"ppo": "ppo_vector", "sac": "sac_vector"}


def spec(algo: str) -> dict:
    return yaml.safe_load((SWEEPS / f"{VECTOR_SPECS[algo]}.yaml").read_text())


@pytest.mark.parametrize("algo", ["ppo", "sac"])
def test_sweep_spec_asks_for_a_bayes_search_on_the_eval_objective(algo):
    body = spec(algo)
    assert body["method"] == "bayes"
    assert body["metric"] == {"name": OBJECTIVE, "goal": "maximize"}
    assert body["early_terminate"]["type"] == "hyperband"
    assert body["early_terminate"]["min_iter"] >= 1
    assert body["parameters"]["algo"]["value"] == algo
    # The agent has to reach the trial entry point, not train.py: a command
    # pointing at plain training would run a sweep of identical default runs.
    assert "owm.baselines.rl.sweep_trial" in body["command"]


@pytest.mark.parametrize("algo", ["ppo", "sac"])
def test_every_swept_parameter_is_searchable(algo):
    for name, body in spec(algo)["parameters"].items():
        assert {"value", "values", "distribution"} & set(body), name
        if "distribution" in body:
            assert body["min"] < body["max"], name
        if "values" in body:
            assert len(body["values"]) > 1, name


@pytest.mark.parametrize("algo", ["ppo", "sac"])
def test_every_swept_parameter_is_a_real_sb3_argument(algo):
    # A typo here costs a whole trial: the name lands in rl.hyperparams and
    # SB3 rejects it, hours after the agent picked the value.
    accepted = set(inspect.signature(ALGOS[algo].__init__).parameters)
    swept = set(spec(algo)["parameters"]) - RESERVED_KEYS
    assert swept <= accepted, swept - accepted


@pytest.mark.parametrize("algo", ["ppo", "sac"])
def test_every_spec_pins_its_own_horizon(algo):
    # Without it the trial would run conf/rl's multi-million-step budget, and
    # a sweep of 5M-step trials looks like a sweep, just a far slower one.
    assert spec(algo)["parameters"]["trial_timesteps"]["value"] == 500_000


def test_every_ppo_batch_size_divides_the_rollout_it_is_drawn_from():
    # PPO's last minibatch of an epoch is a short one when batch_size does not
    # divide n_steps * n_envs. SB3 only warns, so a bad pair costs a whole
    # trial's worth of slightly-wrong updates instead of failing.
    params = spec("ppo")["parameters"]
    n_envs = RESOURCES["ppo"]["n_envs"]
    for n_steps in params["n_steps"]["values"]:
        for batch_size in params["batch_size"]["values"]:
            assert (n_steps * n_envs) % batch_size == 0, (n_steps, batch_size)
