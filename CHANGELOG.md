# CHANGELOG

<!-- version list -->

## v1.1.0 (2026-07-12)

### Bug Fixes

- **tools**: Align pre-commit ruff with project version and fix E501
  ([`69c615d`](https://github.com/PolarisMegrez/qphase/commit/69c615d729441df8e97685fd654648c12937ee64))

### Continuous Integration

- **release**: Fix python-semantic-release branches config
  ([`80d6a82`](https://github.com/PolarisMegrez/qphase/commit/80d6a8286ba69ac3569348f61a3e2189936269ce))

### Documentation

- Add dedicated `docs/api/qphase_sde/` section (en + zh)
  ([`455efec`](https://github.com/PolarisMegrez/qphase/commit/455efecac40572b52cbf3946fa6b095e10208b04))

### Features

- Add new models and configurations for van der Pol oscillator
  ([`6acba3d`](https://github.com/PolarisMegrez/qphase/commit/6acba3d68002f08b18e6b77cbcd468632812e680))

- Enhance vdp_sde and vdp_viz configurations, add parameter evolution plotting, and improve peak
  analysis in PSD
  ([`c461541`](https://github.com/PolarisMegrez/qphase/commit/c4615412df8f67254eb243887f4aa5aa59eb4cc0))

- Implement peak finding analyzers and postprocessing utilities
  ([`9cee710`](https://github.com/PolarisMegrez/qphase/commit/9cee71013012ba14542d9d7c7be4bea2a922682b))

- Implement service layer for configuration, registry, and scheduling
  ([`3fbcc93`](https://github.com/PolarisMegrez/qphase/commit/3fbcc93857a2ea0ec8bc2eb2a204c17d877d4519))

- Refactor dependencies and add optional Cupy support
  ([`3bdd69f`](https://github.com/PolarisMegrez/qphase/commit/3bdd69f381ceab640e727801424bda02376239ec))

- Update vdp_sde and vdp_viz configurations, enhance save control, and add vdp_psd configuration for
  power spectrum visualization
  ([`135491b`](https://github.com/PolarisMegrez/qphase/commit/135491b0d7a15023241a4dc576117cdefd4d9a28))

- Update vdp_sde configuration, enhance dtype handling, and add utility functions for complex noise
  ([`0d941a7`](https://github.com/PolarisMegrez/qphase/commit/0d941a725a9efa8d465ce3d81e8d05bda51ea191))

- **sde, docs, tests**: Add qphase_sde docs, save_stride tests, and Welch/multitaper PSD methods
  ([`455efec`](https://github.com/PolarisMegrez/qphase/commit/455efecac40572b52cbf3946fa6b095e10208b04))

### Testing

- **lorentz**: Use np.isclose for fitted center assertion
  ([`2f59e84`](https://github.com/PolarisMegrez/qphase/commit/2f59e84b54787500a6befdebef43dec1b084dccf))


## v1.0.0 (2026-01-01)

- Initial Release
