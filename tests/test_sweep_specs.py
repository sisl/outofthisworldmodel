import inspect
from pathlib import Path

import pytest
import yaml

from owm.baselines.rl.sweep_callbacks import OBJECTIVES
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


# Two kinds of spec live here, and they answer different questions. A search
# ranks configurations against each other and wants Bayes and banding. A
# replication asks how often ONE configuration works, which is a measurement
# rather than a search: it enumerates its draws and must not prune them.
# Keyed on method rather than on a naming convention so that a spec cannot
# opt out of a bar by what it is called.
SEARCH_METHOD = "bayes"
REPLICATION_METHOD = "grid"
SEARCH_NAMES = [n for n in SPEC_NAMES if spec(n)["method"] == SEARCH_METHOD]
REPLICATION_NAMES = [n for n in SPEC_NAMES if spec(n)["method"] == REPLICATION_METHOD]


def test_a_spec_exists_for_every_algo():
    assert {algo_of(name) for name in SPEC_NAMES} == set(ALGOS)


@pytest.mark.parametrize("name", SPEC_NAMES)
def test_every_spec_is_one_of_the_two_kinds(name):
    # A typo'd method would otherwise drop a spec out of both of the tests
    # below and leave it checked by neither.
    assert spec(name)["method"] in {SEARCH_METHOD, REPLICATION_METHOD}


@pytest.mark.parametrize("name", SPEC_NAMES)
def test_sweep_spec_optimises_an_objective_the_trial_reports(name):
    body = spec(name)
    # One of the objectives the trial actually reports, paired with the only
    # direction that objective can be read in: a spec asking to maximize a
    # distance-to-goal, or to optimise a key nothing logs, would run a whole
    # sweep and rank it on nothing.
    metric = body["metric"]
    assert metric["name"] in OBJECTIVES, metric["name"]
    assert metric["goal"] == OBJECTIVES[metric["name"]], metric
    assert body["parameters"]["algo"]["value"] in ALGOS
    # The recipe pins the device on this prefix, so a spec named otherwise
    # would run wherever the shell happened to point it.
    assert name.startswith(f"{algo_of(name)}_")
    # The agent has to reach the trial entry point, not train.py: a command
    # pointing at plain training would run a sweep of identical default runs.
    assert "owm.baselines.rl.sweep_trial" in body["command"]


@pytest.mark.parametrize("name", SEARCH_NAMES)
def test_a_search_spec_bands_its_trials(name):
    body = spec(name)
    assert body["early_terminate"]["type"] == "hyperband"
    assert body["early_terminate"]["min_iter"] >= 1


@pytest.mark.parametrize("name", REPLICATION_NAMES)
def test_a_replication_spec_varies_only_its_repeat_axis_and_keeps_every_draw(name):
    body = spec(name)
    # Banding a replication would discard the draws that failed, which are
    # precisely the observations it exists to count.
    assert "early_terminate" not in body, name
    # Everything else pinned: a second free parameter would make the spread it
    # measures a mixture of that parameter and the repeat, and the failure rate
    # would belong to no single configuration. Free means anything the agent
    # draws from -- an enumerated `values` or a continuous `distribution` --
    # against `value`, which is the only form that pins.
    varied = sorted(
        key
        for key, param in body["parameters"].items()
        if {"values", "distribution"} & set(param)
    )
    assert varied == ["seed"], varied


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
