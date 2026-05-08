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

## Project status

**Alpha / work in progress.** APIs and behavior may change without notice until a stable release.

## License

Licensed under the Apache License, Version 2.0. See [`LICENSE`](LICENSE) for the full text.
