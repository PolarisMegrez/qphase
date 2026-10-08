"""Run the fig-1c ridge-conditioned carrier analysis on the refined-dark scan.

Mirrors the ``band_limited_carrier`` + ``spectral_ridge`` configuration of the
``collective_loss_kerr_3mode_ridge_psd_26`` workflow (used for the fig-1c
positive-branch carrier) and applies it to the 21-point refined-dark SDE scan,
writing ``carrier_results.csv`` / ``carrier_candidates.csv`` /
``carrier_platforms.csv`` into the given output directory.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from qphase.backend.numpy_backend import NumpyBackend  # noqa: E402
from qphase.data import load_bundle  # noqa: E402
from qphase_sde.analyser.band_limited_carrier import (  # noqa: E402
    BandLimitedCarrierAnalyzer,
    BandLimitedCarrierConfig,
    RidgeConditionedCenterConfig,
)
from qphase_sde.analyser.spectral_ridge import SpectralRidgeConfig  # noqa: E402
from qphase_sde.result import SDEDataBundle  # noqa: E402

DEFAULT_BUNDLE = (
    ROOT
    / "runs/2026/08/2026-08-20T11-04-09_18d6a9"
    / "collective_loss_kerr_3mode_refined_dark_medium_21_sim"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--freq-min", type=float, default=-0.35)
    parser.add_argument("--freq-max", type=float, default=0.1)
    args = parser.parse_args()

    bundle = load_bundle(args.bundle)
    if not isinstance(bundle, SDEDataBundle):
        raise TypeError(f"{args.bundle} is not an SDEDataBundle")

    config = BandLimitedCarrierConfig(
        scan_param="omega_c",
        readout="trace",
        freq_min=args.freq_min,
        freq_max=args.freq_max,
        maximum_lag=256.0,
        tracking_enabled=False,
        center=RidgeConditionedCenterConfig(
            maximum_neighbor_fraction=0.45,
            spectral_ridge=SpectralRidgeConfig(
                scan_param="omega_c",
                readouts=["trace"],
                freq_min=args.freq_min,
                freq_max=args.freq_max,
                minimum_prominence_fraction=0.03,
                tracking_path_count=2,
                tracking_max_cost_delta=2.0,
            ),
        ),
        output_dir=str(args.output_dir),
    )
    products = BandLimitedCarrierAnalyzer(config).execute(
        bundle.products,
        backend=NumpyBackend(),
        label="band_limited_carrier",
        metadata=bundle.metadata,
    )
    rows = list(
        products["band_limited_carrier"].attributes["payload_meta"]["carrier_rows"]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda row: (float(row["omega_c"]), int(row["ridge_order"])))
    fieldnames = list(rows[0])
    with (args.output_dir / "carrier_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    primary = [row for row in rows if int(row["ridge_order"]) == 0]
    print("idx  omega_c    status  ridge_center   frequency    diag_unc   correction")
    for index, row in enumerate(primary):
        print(
            f"{index:3d}  {float(row['omega_c']):.6f}  {row['status']:<6}  "
            f"{float(row['ridge_center']):+.6f}   {float(row['frequency']):+.6f}   "
            f"{float(row['diagnostic_uncertainty']):.2e}   "
            f"{float(row['ridge_carrier_correction']):+.2e}"
        )


if __name__ == "__main__":
    main()
