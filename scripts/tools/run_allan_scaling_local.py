"""Run the scan-level AllanScalingAnalyzer on a saved typed SDE bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "tools"))

from prepare_collective_loss_kerr_3mode_refined_dark import (  # noqa: E402
    _product_payload,
)
from qphase.backend.numpy_backend import NumpyBackend  # noqa: E402
from qphase.data import load_bundle  # noqa: E402
from qphase_sde.analyser.allan_scaling import AllanScalingAnalyzer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sde-session", type=Path, required=True)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--critical-value", type=float, required=True)
    parser.add_argument(
        "--config-json",
        type=Path,
        help=(
            "optional analyzer config overrides as JSON "
            "(workflow values are the default)"
        ),
    )
    args = parser.parse_args()

    sim_dir = args.sde_session / f"{args.stem}_sim"
    bundle = load_bundle(sim_dir)

    config: dict[str, object] = {
        "scan_param": "omega_c",
        "critical_value": args.critical_value,
        "side": "positive",
        "mode": 0,
        "window_selection": "latest",
        "white_slope_min": -1.15,
        "white_slope_max": -0.85,
        "min_local_r2": 0.98,
        "min_tau_decades": 0.4,
        "min_averaging_samples": 16,
        "max_relative_sem": 0.25,
        "min_independent_windows": 32,
        "min_scaling_points": 8,
        "target_scaling_decades": 2.0,
        "max_scaling_reduced_chi2": 10.0,
        "bootstrap_samples": 5000,
        "bootstrap_seed": 20260820,
        "min_frequency_rss_improvement": 0.05,
        "normal_form": {"n": 3, "k": 1, "m": 0, "observable_order": 1},
    }
    if args.config_json is not None:
        config.update(json.loads(args.config_json.read_text(encoding="utf-8")))
    output_dir = args.sde_session / f"{args.stem}_allan"
    analyzer_config = AllanScalingAnalyzer.config_schema(
        **{**config, "output_dir": str(output_dir)}
    )
    analyzer = AllanScalingAnalyzer(analyzer_config)
    products = analyzer.execute(
        bundle.products,
        backend=NumpyBackend(),
        label="allan_scaling",
        metadata=bundle.metadata,
    )
    summary = _product_payload(products["allan_scaling"])["summary"]
    print(json.dumps(summary, indent=2, default=str))
    print(f"written to {output_dir}")


if __name__ == "__main__":
    main()
