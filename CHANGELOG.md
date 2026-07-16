# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-07-16

### Added

- `seed` parameter on `Simulator`/`Simulator.build` for reproducible random
  simulations (the global NumPy random state no longer has any effect).
- `horizon` parameter on `Simulator.step`: restricts the shortest-path refresh
  to the routing rows readable before the next refresh (identical results,
  much cheaper); a guard raises if the promised refresh does not happen.
- `refresh_interval` and `refresh_workers` build parameters: automatic
  shortest-path refreshes, optionally parallelised over worker processes.
- `examples/simulate.py`: header-ful CSV support, `--seed`, `--arrivals`, and
  `--workers` options.

### Changed

- **Behaviour**: edge admission is decided against the lane state at the start
  of each step; lane space vacated during a step becomes available only in the
  following step (the previous same-step reuse was unintended). Random-mode
  results for a given seed also differ from earlier versions.
- Performance: ~10x faster and O(edges) routing memory (sparse routing
  structures, destination-restricted Dijkstra, active-vehicle tracking,
  vectorized admission and trip logging).

## [0.1.1] - 2025-04-20

### Fixed

- GitHub repo URLs fixed to use correct lowercase name (`fast-traffic-simulator`).

### Added

- PyPI publish workflow (`.github/workflows/publish.yml`).

[0.2.0]: https://github.com/fbk-most/fast-traffic-simulator/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/fbk-most/fast-traffic-simulator/compare/v0.1.0...v0.1.1
