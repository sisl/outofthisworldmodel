"""Upload a finished RL run's model to the Hugging Face Hub.

train.py publishes automatically at the end of a run; if that publish step
failed or was skipped, republish it manually:

    python -m owm.baselines.rl.hub runs/ppo_a
    python -m owm.baselines.rl.hub runs/ppo_a my-org/my-model
"""

from __future__ import annotations

import os
from pathlib import Path

import click
from dotenv import load_dotenv
from huggingface_hub import HfApi

from owm.baselines.rl.run_state import FINAL_MODEL, FINAL_VECNORM

# The manual-retry CLI must see the same .env the training entry points load;
# without this, OWM_HF_MODEL_REPO is only visible under `just` (dotenv-load).
load_dotenv()

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


@click.command()
@click.argument("run_dir", type=click.Path(path_type=Path))
@click.argument("repo_id", required=False, default=None)
def main(run_dir: Path, repo_id: str | None) -> None:
    """Publish a finished RL run's finals to the HF Hub.

    RUN_DIR is the run directory, e.g. runs/ppo_a. REPO_ID is the HF model
    repo id; defaults to $OWM_HF_MODEL_REPO.
    """
    if not repo_id:
        repo_id = os.environ.get("OWM_HF_MODEL_REPO")
    if not repo_id:
        raise click.UsageError("repo_id not given and OWM_HF_MODEL_REPO is not set")
    print(upload_run(run_dir, repo_id))


if __name__ == "__main__":
    main()
