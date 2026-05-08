# nmrkit

General-purpose NMR processing toolkit: the first MVP targets automated kinetic classification of peaks in time-resolved 2D NMR spectra; the package name leaves room for future modules (chemical shift prediction, 1D processing, assignment).

## Quick start

```bash
uv sync
pre-commit install
pre-commit install --hook-type commit-msg
make check
```

Use `pre-commit install --hook-type commit-msg` so [Commitizen](https://github.com/commitizen-tools/commitizen) can validate commit messages (see [Conventional commits](#conventional-commits) below).

## Quick demo

The MVP pipeline loads a time-resolved 2D stack, detects and tracks peaks across frames, fits non-negative matrix factorization (NMF) to separate kinetic components and cluster trajectories, optionally refines clusters with diffusion data from DOSY, and writes a multi-panel summary figure plus JSON and Markdown reports. After `uv sync`, run the bundled synthetic Bruker-style fixture end-to-end:

```bash
make validate-synthetic
```

This fills `validation_output_synthetic/` with `summary.png`, `clusters.json`, and `report.md` (that directory is gitignored). The synthetic series lives under `tests/fixtures/synthetic_timeseries/`; regenerate it with `uv run python tests/fixtures/build_synthetic_timeseries.py` if needed.

## Using your own data

Point the validation script at any directory that [`load_timeseries`](src/nmrkit/io/nmr.py) accepts. Each top-level child (after sorting) should be either a NMRPipe 2D `.ft2` file, a Bruker **pdata** directory that contains a `2rr` file, or a Bruker experiment directory whose `pdata/<n>/` subfolder contains `2rr`. Timestamps are read from spectrum metadata when possible; otherwise the loader falls back to synthetic `0, 1, 2, …` seconds (with a warning).

```bash
uv run python examples/validate_dataset.py \
  --data-dir /path/to/stack_or_series \
  --output-dir ./validation_output \
  [--dosy-path /path/to/dosy.npz] \
  [--n-components 3]
```

For Bruker data, `--data-dir` is often an experiment folder listing multiple `pdata` entries or repeated acquisitions; for NMRPipe, it may be a folder of sequentially numbered `.ft2` files. Optional DOSY input must be a NumPy `.npz` with the keys expected by `load_dosy` (see `src/nmrkit/dosy/fit.py`).

## Conventional commits

**All commit messages must follow [Conventional Commits](https://www.conventionalcommits.org/).** Non-compliant messages are rejected by the `commit-msg` hook ([Commitizen](https://github.com/commitizen-tools/commitizen)), configured in [`.pre-commit-config.yaml`](.pre-commit-config.yaml).

After cloning, run both:

```bash
pre-commit install
pre-commit install --hook-type commit-msg
```

The second command is required: without it, Conventional Commits are not enforced. To author messages interactively in the expected format, use `cz commit` (from the dev dependencies installed by `uv sync`).

## Development

- **Install:** `make install` (or `uv sync`). Optional Apple/accelerator-oriented profile: `make install-mps` (currently equivalent to `install`; reserved for optional accelerated stacks).
- **Quality gate:** `make check` runs Ruff (lint + format check), Mypy on `src/`, and pytest.
- **Format / lint only:** `make format`, `make lint`.
- **Tests:** `make test`.
- **Synthetic pipeline demo:** `make validate-synthetic` runs `examples/validate_dataset.py` on `tests/fixtures/synthetic_timeseries` and checks that `validation_output_synthetic/summary.png` exists.

## Project status

**Alpha / work in progress.** APIs and behavior may change without notice until a stable release.

## License

Licensed under the Apache License, Version 2.0. See [`LICENSE`](LICENSE) for the full text.
