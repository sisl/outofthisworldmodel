from pathlib import Path

import pytest

from owm.baselines.rl.manifest import MethodSettings, load_manifest

GOOD = """
rollouts:
  - name: harmony_fwd_sunlit_a
    port: harmony_fwd_pma2
    seed: 100003
    lighting: sunlit
    distribution: train
    methods:
      rl: {rate_hz: 20, action_repeat: 20}
      wm: {rate_hz: 20, action_repeat: 5}
  - name: poisk_zenith_eclipse_a
    port: poisk_zenith
    seed: 140007
    lighting: eclipse
    distribution: heldout
    methods:
      wm: {rate_hz: 20, action_repeat: 5}
clips:
  - path: anomaly/nominal.mp4
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "manifest.yaml"
    path.write_text(text)
    return path


def test_rows_are_parsed_in_order(tmp_path):
    rows = load_manifest(write(tmp_path, GOOD))
    assert [row.name for row in rows] == ["harmony_fwd_sunlit_a", "poisk_zenith_eclipse_a"]
    assert rows[0].rl == MethodSettings(rate_hz=20.0, action_repeat=20)
    assert rows[0].wm == MethodSettings(rate_hz=20.0, action_repeat=5)
    assert rows[1].rl is None
    assert rows[1].seed == 140007 and rows[1].distribution == "heldout"


def test_unknown_port_is_refused(tmp_path):
    text = GOOD.replace("port: poisk_zenith", "port: nowhere")
    with pytest.raises(ValueError, match="nowhere"):
        load_manifest(write(tmp_path, text))


def test_duplicate_name_is_refused(tmp_path):
    text = GOOD.replace("name: poisk_zenith_eclipse_a", "name: harmony_fwd_sunlit_a")
    with pytest.raises(ValueError, match="harmony_fwd_sunlit_a"):
        load_manifest(write(tmp_path, text))


def test_bad_lighting_is_refused(tmp_path):
    text = GOOD.replace("lighting: eclipse", "lighting: dusk")
    with pytest.raises(ValueError, match="dusk"):
        load_manifest(write(tmp_path, text))


def test_unknown_row_keys_are_ignored(tmp_path):
    text = GOOD.replace(
        "seed: 140007",
        "seed: 140007\n    trial: 3\n    wm_outcome: docked",
    )
    rows = load_manifest(write(tmp_path, text))
    assert rows[1].seed == 140007


def test_missing_method_key_keeps_the_underlying_cause(tmp_path):
    text = GOOD.replace("rl: {rate_hz: 20, action_repeat: 20}", "rl: {rate_hz: 20}")
    with pytest.raises(ValueError, match="action_repeat") as excinfo:
        load_manifest(write(tmp_path, text))
    assert isinstance(excinfo.value.__cause__, KeyError)
