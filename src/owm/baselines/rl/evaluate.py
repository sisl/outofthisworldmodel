"""Run a saved checkpoint on an arbitrary env config and report metrics.

    python -m owm.baselines.rl.evaluate eval.checkpoint=runs/ppo_x/final_model.zip rl=ppo
    python -m owm.baselines.rl.evaluate \\
        eval.checkpoint=hf:sislaboratory/owm-models/rl/ppo_x/final_model.zip \\
        rl=ppo environments.sensor_noise.enabled=false
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path, PurePosixPath

import hydra
import imageio
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from omegaconf import DictConfig

from owm.baselines.rl.run_state import vecnormalize_name_for
from owm.baselines.rl.train import ALGOS
from owm.envs.factory import iss_config, make_iss_env

load_dotenv()

_HF_RE = re.compile(r"^hf:(?P<org>[^/]+)/(?P<repo>[^/]+)/(?P<path>.+)$")


def resolve_checkpoint(spec: str) -> Path:
    if spec.startswith("hf:"):
        match = _HF_RE.match(spec)
        if match is None:
            raise SystemExit(
                f"malformed hf: spec {spec!r}, expected hf:<org>/<repo>/<path>"
            )
        repo_id = f"{match['org']}/{match['repo']}"
        ckpt = Path(hf_hub_download(
            repo_id=repo_id, filename=match["path"], repo_type="model"
        ))
        # The normalization stats are a separate file in the repo, so they have
        # to be asked for separately; both land in the same snapshot dir, where
        # the local sibling lookup then finds them.
        stats_name = vecnormalize_name_for(ckpt.name)
        if stats_name is not None:
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=str(PurePosixPath(match["path"]).with_name(stats_name)),
                    repo_type="model",
                )
            except EntryNotFoundError:
                pass  # reported by run_eval, which knows whether they're needed
        return ckpt
    path = Path(spec)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {spec}")
    return path


def stats_for_checkpoint(ckpt: Path) -> Path | None:
    name = vecnormalize_name_for(ckpt.name)
    if name is None:
        return None
    sibling = ckpt.parent / name
    return sibling if sibling.exists() else None


def run_eval(cfg: DictConfig) -> dict:
    ckpt = resolve_checkpoint(str(cfg.eval.checkpoint))
    model = ALGOS[cfg.rl.algo].load(ckpt, device="cpu")

    stats = stats_for_checkpoint(ckpt)
    if stats is None and not bool(cfg.eval.allow_unnormalized):
        raise SystemExit(
            f"no VecNormalize stats found next to {ckpt}; a policy trained on "
            "normalized observations cannot be evaluated on raw ones — pass "
            "eval.allow_unnormalized=true to override"
        )
    vecnorm = None
    if stats is not None:
        # The pickled VecNormalize normalizes standalone (__setstate__ drops
        # only its venv), so eval reuses training's own transform rather than
        # a hand-rolled copy that could drift from it.
        with open(stats, "rb") as f:
            vecnorm = pickle.load(f)
        vecnorm.training = False

    record = cfg.eval.video_path is not None
    env = make_iss_env(iss_config(cfg.environments), seed=int(cfg.seed), render=record)

    returns, lengths, successes, collisions = [], [], 0, 0
    frames: list[np.ndarray] = []
    # A rendering env holds a GL context; leaking it on an episode that raises
    # would keep the window and its worker alive for the rest of the process.
    try:
        for episode in range(int(cfg.eval.episodes)):
            obs, _ = env.reset(seed=int(cfg.seed) + episode)
            done, ep_return, ep_len = False, 0.0, 0
            while not done:
                norm = vecnorm.normalize_obs(obs) if vecnorm is not None else obs
                action, _ = model.predict(norm, deterministic=bool(cfg.eval.deterministic))
                obs, reward, term, trunc, info = env.step(action)
                ep_return += float(reward)
                ep_len += 1
                if record and episode == 0:
                    frames.append(env.render())
                done = term or trunc
            returns.append(ep_return)
            lengths.append(ep_len)
            successes += int(bool(info.get("success")))
            collisions += int(bool(info.get("collision")))

        if frames:
            imageio.mimsave(cfg.eval.video_path, frames,
                            fps=env.metadata.get("render_fps", 20))
    finally:
        env.close()

    n = int(cfg.eval.episodes)
    results = {
        "episodes": n,
        "mean_return": float(np.mean(returns)),
        "success_rate": successes / n,
        "collision_rate": collisions / n,
        "mean_length": float(np.mean(lengths)),
    }
    for key, value in results.items():
        print(f"{key:>16}: {value}")
    return results


@hydra.main(config_path="../../../../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    run_eval(cfg)


if __name__ == "__main__":
    main()
