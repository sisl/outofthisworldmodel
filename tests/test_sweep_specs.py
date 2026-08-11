import inspect
from pathlib import Path

import pytest
import yaml

from owm.baselines.rl.sweep_callbacks import OBJECTIVE
from owm.baselines.rl.sweep_trial import PIXEL_N_ENVS, RESERVED_KEYS, RESOURCES
from owm.baselines.rl.train import ALGOS

SWEEPS = Path(__file__).resolve().parent.parent / "sweeps"
# Discovered, not listed: one spec per (algo, obs mode), and the pixel pair
# that lands with rl.obs has to meet the same bar without editing this file.
SPEC_NAMES = sorted(path.stem for path in SWEEPS.glob("*.yaml"))


def spec(name: str) -> dict:
    return yaml.safe_load((SWEEPS / f"{name}.yaml").read_text())


def algo_of(name: str) -> str:
    return spec(name)["parameters"]["algo"]["value"]


def test_a_spec_exists_for_every_algo():
    assert {algo_of(name) for name in SPEC_NAMES} == set(ALGOS)


@pytest.mark.parametrize("name", SPEC_NAMES)
def test_sweep_spec_asks_for_a_bayes_search_on_the_eval_objective(name):
    body = spec(name)
    assert body["method"] == "bayes"
    assert body["metric"] == {"name": OBJECTIVE, "goal": "maximize"}
    assert body["early_terminate"]["type"] == "hyperband"
    assert body["early_terminate"]["min_iter"] >= 1
    assert body["parameters"]["algo"]["value"] in ALGOS
    # The recipe pins the device on this prefix, so a spec named otherwise
    # would run wherever the shell happened to point it.
    assert name.startswith(f"{algo_of(name)}_")
    # The agent has to reach the trial entry point, not train.py: a command
    # pointing at plain training would run a sweep of identical default runs.
    assert "owm.baselines.rl.sweep_trial" in body["command"]


@pytest.mark.parametrize("name", SPEC_NAMES)
def test_every_swept_parameter_is_searchable(name):
    for key, body in spec(name)["parameters"].items():
        assert {"value", "values", "distribution"} & set(body), key
        if "distribution" in body:
            assert body["min"] < body["max"], key
        if "values" in body:
            assert len(body["values"]) > 1, key


@pytest.mark.parametrize("name", SPEC_NAMES)
def test_every_swept_parameter_is_a_real_sb3_argument(name):
    # A typo here costs a whole trial: the name lands in rl.hyperparams and
    # SB3 rejects it, hours after the agent picked the value.
    accepted = set(inspect.signature(ALGOS[algo_of(name)].__init__).parameters)
    swept = set(spec(name)["parameters"]) - RESERVED_KEYS
    assert swept <= accepted, swept - accepted


@pytest.mark.parametrize("name", SPEC_NAMES)
def test_every_spec_pins_its_own_horizon(name):
    # Without it the trial would run conf/rl's multi-million-step budget, and
    # a sweep of 5M-step trials looks like a sweep, just a far slower one.
    assert spec(name)["parameters"]["trial_timesteps"]["value"] > 0


PPO_SPEC_NAMES = [name for name in SPEC_NAMES if algo_of(name) == "ppo"]


def effective_n_envs(name: str) -> int:
    """The width sweep_trial will actually run this spec at.

    Derived the way build_cfg derives it, from the imported constants rather
    than a copy of the numbers: a pixel spec runs PIXEL_N_ENVS regardless of
    its algo's vector width, because every one of its envs renders.
    """
    obs_param = spec(name)["parameters"].get("obs", {"value": "vector"})
    # A spec that swept obs would run at two widths and this guard would have
    # to check both. None does; pin the assumption rather than quietly
    # checking one branch of it.
    assert "value" in obs_param, f"{name} sweeps obs; its width is not static"
    if obs_param["value"] == "vector_resnet":
        return PIXEL_N_ENVS
    return RESOURCES[algo_of(name)]["n_envs"]


@pytest.mark.parametrize("name", PPO_SPEC_NAMES)
def test_every_ppo_batch_size_divides_the_rollout_it_is_drawn_from(name):
    # PPO's last minibatch of an epoch is a short one when batch_size does not
    # divide n_steps * n_envs. SB3 only warns, so a bad pair costs a whole
    # trial's worth of slightly-wrong updates instead of failing. Checked per
    # spec at that spec's own width: the pixel lane runs 4 envs, not 8, so the
    # vector lane passing says nothing about it.
    params = spec(name)["parameters"]
    n_envs = effective_n_envs(name)
    for n_steps in params["n_steps"]["values"]:
        for batch_size in params["batch_size"]["values"]:
            assert (n_steps * n_envs) % batch_size == 0, (
                name, n_steps, batch_size, n_envs
            )
