# Ahead of the owm_envs imports below: importing it pins JAX to CPU, which XLA
# reads when its backend first comes up, and owm_envs brings that backend up.
from owm.envs.factory import (  # isort: skip
    env_conf_dict,
    env_config,
    env_name_of,
    env_spec,
    make_env,
)

import numpy as np
from conftest import smoke_cfg
from omegaconf import OmegaConf

from owm.baselines.rl.eval_matrix import at_rate, for_port
from owm.baselines.rl.scout import (
    default_seeds,
    illumination_profile,
    lighting_tag,
    scout_seeds,
)


def _smoke_env_conf(tmp_path, rate_hz, max_steps, port):
    """The smoke run's environment, re-timed and narrowed the way scout does.

    Round-tripped through the task config first, so `dock.ports` carries the
    resolved entries `base_env_conf` hands `for_port` in a real run.
    """
    raw = OmegaConf.to_container(smoke_cfg(tmp_path, "ppo").environments, resolve=True)
    base = env_conf_dict(env_config(raw))
    return env_config(for_port({**at_rate(base, rate_hz), "max_steps": max_steps}, port))


def test_lighting_tag_thresholds():
    assert lighting_tag(1.0) == "sunlit"
    assert lighting_tag(0.96) == "sunlit"
    assert lighting_tag(0.0) == "eclipse"
    assert lighting_tag(0.04) == "eclipse"
    assert lighting_tag(0.5) == "transition"


def test_default_seeds_follow_eval_matrix_blocks():
    assert default_seeds("harmony_fwd_pma2")[:3] == [100000, 100001, 100002]
    assert default_seeds("poisk_zenith")[0] == 100000 + 4 * 10_000
    assert len(default_seeds("pirs_nadir")) == 50


def test_default_seeds_cover_the_manifest_scenarios():
    assert {100003, 100020} <= set(default_seeds("harmony_fwd_pma2"))
    assert {160025, 160040} <= set(default_seeds("rassvet_nadir"))
    assert {140015, 140017} <= set(default_seeds("poisk_zenith"))


def test_illumination_profile_is_a_fraction_per_state(tmp_path):
    cfg = _smoke_env_conf(tmp_path, 1, 5, "harmony_fwd_pma2")
    env = make_env(cfg, seed=0)
    _, info = env.reset(seed=0)
    states = [np.asarray(info["state"])]
    for _ in range(5):
        _, _, term, trunc, info = env.step(np.zeros(6, dtype=np.float32))
        states.append(np.asarray(info["state"]))
        if term or trunc:
            break
    env.close()
    profile = illumination_profile(np.stack(states), env_spec(env_name_of(cfg)).layout)
    assert profile.shape == (len(states),)
    assert np.all((profile >= 0.0) & (profile <= 1.0))


def test_scout_reports_one_row_per_seed(tmp_path):
    cfg = _smoke_env_conf(tmp_path, 1, 3, "harmony_fwd_pma2")
    rows = scout_seeds(cfg, [100000, 100001], "harmony_fwd_pma2", evals_dirs={})
    assert [row["seed"] for row in rows] == [100000, 100001]
    assert set(rows[0]) >= {"seed", "lighting", "illumination", "start_range_m"}
    assert rows[0]["start_range_m"] > 0.0
    assert rows[0]["lighting"] in {"sunlit", "eclipse", "transition"}


def test_scout_reads_an_outcome_out_of_an_eval_drop(tmp_path):
    drop = tmp_path / "drop"
    drop.mkdir()
    (drop / "episodes.csv").write_text(
        "port,seed,outcome\nharmony_fwd_pma2,100000,docked\nharmony_fwd_pma2,100001,escaped\n"
    )
    cfg = _smoke_env_conf(tmp_path, 1, 3, "harmony_fwd_pma2")
    rows = scout_seeds(cfg, [100000, 100001], "harmony_fwd_pma2", evals_dirs={"ppo": drop})
    assert [row["ppo_outcome"] for row in rows] == ["docked", "escaped"]
