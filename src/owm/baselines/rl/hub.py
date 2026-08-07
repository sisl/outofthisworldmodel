"""Upload a finished RL run's model to the Hugging Face Hub.

train.py publishes automatically at the end of a run; if that publish step
failed or was skipped, republish it manually:

    python -m owm.baselines.rl.hub runs/ppo_a
    python -m owm.baselines.rl.hub runs/ppo_a my-org/my-model
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

from owm.baselines.rl.run_state import FINAL_MODEL, FINAL_VECNORM

_UPLOAD_FILES = (FINAL_MODEL, FINAL_VECNORM, "config.yaml")


def upload_run(run_dir: Path, repo_id: str) -> str:
    # allow_patterns silently uploads whatever subset happens to be there, so a
    # crashed or half-written run would publish a model with no normalization
    # statistics — loadable, and wrong. Check before the repo is even created.
    missing = [name for name in _UPLOAD_FILES if not (run_dir / name).exists()]
    if missing:
        raise SystemExit(
            f"{run_dir} is missing {', '.join(missing)}; refusing to publish a "
            "partial run"
        )

    api = HfApi()  # token from HF_TOKEN or the local login
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(run_dir),
        path_in_repo=f"rl/{run_dir.name}",
        allow_patterns=list(_UPLOAD_FILES),
        commit_message=f"Upload RL run {run_dir.name}",
    )
    return f"https://huggingface.co/{repo_id}/tree/main/rl/{run_dir.name}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish a finished RL run's finals to the HF Hub.")
    parser.add_argument("run_dir", type=Path, help="run directory, e.g. runs/ppo_a")
    parser.add_argument(
        "repo_id",
        nargs="?",
        default=os.environ.get("OWM_HF_MODEL_REPO"),
        help="HF model repo id; defaults to $OWM_HF_MODEL_REPO",
    )
    args = parser.parse_args()
    if not args.repo_id:
        parser.error("repo_id not given and OWM_HF_MODEL_REPO is not set")
    print(upload_run(args.run_dir, args.repo_id))


if __name__ == "__main__":
    main()
