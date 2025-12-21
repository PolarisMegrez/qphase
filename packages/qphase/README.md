# qphase

Typer-based command-line interface (CLI) to configure, run, and analyze phase-space SDE simulations using YAML/JSON configuration files.

- Easy-to-use CLI for running and analyzing quantum optics SDE simulations
- Supports configuration-driven workflows and batch processing
- Generates time-series, phase portraits, and PSD plots
- Integrates with the qphase-sde core engine

## Installation

```sh
pip install qphase
```

## Usage

Example:

```sh
qps run sde --config configs/vdp_run.yaml
qps analyze phase --from-run runs/<run_id>
```

See the main project documentation for configuration file details and advanced usage.

## License
MIT
