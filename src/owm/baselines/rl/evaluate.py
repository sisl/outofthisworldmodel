"""Run a saved checkpoint on an arbitrary env config and report metrics.

    python -m owm.baselines.rl.evaluate eval.checkpoint=runs/ppo_x/final_model.zip rl=ppo
    python -m owm.baselines.rl.evaluate \\
        eval.checkpoint=hf:sislaboratory/owm-models/rl/ppo_x/final_model.zip \\
        rl=ppo environments.sensor_noise.enabled=false
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import hydra
import imageio
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig

from owm.baselines.rl.run_state import FINAL_MODEL, FINAL_VECNORM, vecnormalize_for
from owm.baselines.rl.train import ALGOS
from owm.envs.factory import iss_config, make_iss_env

load_dotenv()

_HF_RE = re.compile(r"^hf:(?P<org>[^/]+)/(?P<repo>[^/]+)/(?P<path>.+)$")


def resolve_checkpoint(spec: str) -> Path:
    match = _HF_RE.match(spec)
    if match:
        return Path(hf_hub_download(
            repo_id=f"{match['org']}/{match['repo']}",
            filename=match["path"],
            repo_type="model",
        ))
    path = Path(spec)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {spec}")
    return path


def stats_for_checkpoint(ckpt: Path) -> Path | None:
    if ckpt.name == FINAL_MODEL:
        sibling = ckpt.parent / FINAL_VECNORM
        return sibling if sibling.exists() else None
    return vecnormalize_for(ckpt)


def run_eval(cfg: DictConfig) -> dict:
    ckpt = resolve_checkpoint(str(cfg.eval.checkpoint))
    model = ALGOS[cfg.rl.algo].load(ckpt, device="cpu")

    stats = stats_for_checkpoint(ckpt)
    obs_rms = None
    if stats is not None:
        # A pickled VecNormalize; only its obs running stats matter here.
        with open(stats, "rb") as f:
            obs_rms = pickle.load(f).obs_rms

    record = cfg.eval.video_path is not None
    env = make_iss_env(iss_config(cfg.environments), seed=int(cfg.seed), render=record)

    returns, lengths, successes, collisions = [], [], 0, 0
    frames: list[np.ndarray] = []
    for episode in range(int(cfg.eval.episodes)):
        obs, _ = env.reset(seed=int(cfg.seed) + episode)
        done, ep_return, ep_len = False, 0.0, 0
        while not done:
            norm = obs
            if obs_rms is not None:
                norm = np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8),
                               -10.0, 10.0).astype(np.float32)
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
