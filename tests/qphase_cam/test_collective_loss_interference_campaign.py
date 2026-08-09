from __future__ import annotations

import csv
import importlib.util
import sys
from itertools import combinations
from pathlib import Path

import numpy as np


def _script_module(name: str):
    path = Path(__file__).parents[2] / "scripts" / name
    module_name = Path(name).stem
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _campaign_module():
    return _script_module("generate_collective_loss_kerr_interference_campaign.py")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_interference_campaign_covers_all_real_sign_sectors_and_control_pairs():
    campaign = _campaign_module()

    assert set(campaign.SECTORS) == {"acn_bcp", "acp_bcn", "acn_bcn"}
    assert {frozenset(scheme.controls) for scheme in campaign.SCHEMES} == {
        frozenset(pair) for pair in combinations(campaign.CONTROL_PARAMETERS, 2)
    }


def test_interference_campaign_uses_disjoint_nested_sobol_blocks():
    campaign = _campaign_module()
    scheme = campaign.SCHEMES[0]
    rows = campaign._rows("acn_bcp", scheme)

    blocks = [
        {row["point_id"] for row in rows[start:stop]}
        for start, stop in campaign.STAGE_SLICES.values()
    ]
    assert [len(block) for block in blocks] == [8, 8, 16]
    assert not (blocks[0] & blocks[1] or blocks[0] & blocks[2] or blocks[1] & blocks[2])


def test_interference_campaign_preserves_sector_signs_for_outer_and_controls():
    campaign = _campaign_module()
    scheme = next(
        item for item in campaign.SCHEMES if item.controls == ("g_ab", "g_ac")
    )
    rows = campaign._rows("acn_bcn", scheme)
    g_bc = np.asarray([row["g_bc"] for row in rows])
    g_ac_control = campaign._control_range("g_ac", campaign.SECTORS["acn_bcn"])

    assert np.all(g_bc < 0.0)
    assert g_ac_control["min"] < g_ac_control["max"] < 0.0
    assert g_ac_control["sampling"] == "log_abs"


def test_interference_campaign_summary_collects_s1_diagnostics(tmp_path: Path):
    analysis = _script_module(
        "analyze_collective_loss_kerr_interference_campaign.py"
    )
    job = tmp_path / (
        "collective_loss_kerr_3mode_interference_"
        "acn_bcp_detuning_mixing_s1"
    )
    job.mkdir()
    stem = job.name
    _write_csv(
        job / f"{stem}_cases.csv",
        [{"case": 0, "case_status": "complete", "near_miss_saved": 2}],
    )
    _write_csv(job / f"{stem}_candidates.csv", [{"case": 0, "candidate": 0}])
    _write_csv(
        job / f"{stem}_response_summary.csv",
        [{"case": 0, "candidate_index": 0, "minimum_rayleigh_visibility": 0.1}],
    )
    _write_csv(
        job / f"{stem}_stochastic_validity.csv",
        [{"case": 0, "candidate_index": 0, "epsilon_crossover": 1e-4}],
    )

    coverage, candidates, responses, stochastic = analysis.summarize([tmp_path])

    assert len(coverage) == len(candidates) == len(responses) == len(stochastic) == 1
    assert responses[0]["sector"] == "acn_bcp"
    assert responses[0]["scheme"] == "detuning_mixing"
    assert stochastic[0]["epsilon_crossover"] == "0.0001"
