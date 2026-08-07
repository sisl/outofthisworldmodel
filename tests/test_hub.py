from pathlib import Path

import pytest

from owm.baselines.rl import hub
from owm.baselines.rl.hub import upload_run
from owm.baselines.rl.run_state import FINAL_MODEL, FINAL_VECNORM


class _FakeApi:
    """Stands in for HfApi; records what a complete run dir would publish."""

    calls: list[dict] = []

    def create_repo(self, repo_id, **kwargs):
        self.calls.append({"create_repo": repo_id})

    def upload_folder(self, **kwargs):
        self.calls.append(kwargs)


def _complete_run(tmp_path: Path) -> Path:
    for name in (FINAL_MODEL, FINAL_VECNORM, "config.yaml"):
        (tmp_path / name).touch()
    return tmp_path


def test_upload_refuses_a_run_missing_its_stats(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(hub, "HfApi", _FakeApi)
    _complete_run(tmp_path)
    (tmp_path / FINAL_VECNORM).unlink()

    # Publishing the model alone would leave a checkpoint that evaluates
    # against raw observations; allow_patterns would have shipped it happily.
    with pytest.raises(SystemExit, match=FINAL_VECNORM):
        upload_run(tmp_path, "org/repo")


def test_upload_refuses_before_creating_the_repo(tmp_path: Path, monkeypatch):
    api = _FakeApi()
    api.calls = []
    monkeypatch.setattr(hub, "HfApi", lambda: api)

    with pytest.raises(SystemExit, match=f"{FINAL_MODEL}, {FINAL_VECNORM}, config.yaml"):
        upload_run(tmp_path, "org/repo")
    assert api.calls == []


def test_upload_publishes_a_complete_run(tmp_path: Path, monkeypatch):
    api = _FakeApi()
    api.calls = []
    monkeypatch.setattr(hub, "HfApi", lambda: api)

    url = upload_run(_complete_run(tmp_path), "org/repo")

    assert url.endswith(f"/rl/{tmp_path.name}")
    assert api.calls[0] == {"create_repo": "org/repo"}
    assert api.calls[1]["allow_patterns"] == [FINAL_MODEL, FINAL_VECNORM, "config.yaml"]
