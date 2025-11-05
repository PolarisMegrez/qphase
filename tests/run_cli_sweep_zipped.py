# Test: CLI run with params sweep (zipped)
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / 'packages' / 'QPhaseSDE_cli'))
sys.path.append(str(ROOT / 'packages' / 'QPhaseSDE'))

from QPhaseSDE_cli.commands.run import sde

def main():
    cfg_path = ROOT / 'tests' / 'tmp_zipped.yaml'
    # Minimal sweep config (zipped)
    cfg_text = """
profile:
  backend: numpy
  solver: euler
  save:
    root: runs
    save_every: 10
    save_timeseries: true
    save_psd_complex: false
    save_psd_modular: false
run:
  time:
    dt: 0.001
    steps: 120
  trajectories:
    n_traj: 4
    master_seed: 321
jobs:
  - name: sweep_zip
    module: models.vdp_two_mode
    function: build_sde
    combinator: zipped
    params:
      g: { values: [0.4, 0.5, 0.6] }
      D: { values: [0.01, 0.02, 0.03] }
      omega_a: 0.005
      omega_b: 0.0
      gamma_a: 2.0
      gamma_b: 1.0
      Gamma: 0.01
    ic:
      - ["7.0+0.0j", "0.0-7.0j"]
    noise:
      kind: independent
"""
    cfg_path.write_text(cfg_text, encoding='utf-8')

    runs_root = ROOT / 'runs'
    before = set(p.name for p in runs_root.glob('*')) if runs_root.exists() else set()
    sde(config=str(cfg_path))
    after = set(p.name for p in runs_root.glob('*'))
    new = sorted(after - before)
    assert len(new) == 3, f"Expected 3 runs for zipped sweep, got {len(new)}"  # 3 pairs
    print('OK: CLI zipped sweep produced 3 runs:', ', '.join(new))

if __name__ == '__main__':
    main()
