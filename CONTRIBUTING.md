# Contributing

Report bugs and propose changes in the [issue tracker](https://github.com/Red-Eyed/DatasetRT/issues).
Include the DatasetRT version or commit, Python version, operating system and
architecture, a minimal reproducer, and expected and observed behavior. For
reader or writer issues, include worker counts, prefetch settings, compression,
and whether caches are local or on a network filesystem. Use synthetic records
when reproducing issues with private training data.

## Development

Install Rust stable, uv, and just, then run:

```bash
git clone https://github.com/Red-Eyed/DatasetRT.git
cd DatasetRT
just sync
just check
```

`just check` builds the native extension before Python tests and runs formatting,
linting, type checks, documentation checks, Rust tests, and Clippy. Use `just fmt`
to format changes and `just develop` to rebuild the extension during development.
See the [Justfile](Justfile) and [repository conventions](AGENTS.md).

## Scope and validation

Rust owns storage integrity, sampling, authoritative state, and bounded worker
pipelines. Python owns source discovery, serialization, configuration, and
framework integration. Keep dataset-scale operations columnar across that
boundary and preserve deterministic ordering and stable sample identities.

Describe compatibility and migration for public API or storage changes. Add
focused regression coverage for changed behavior; include corruption cases for
storage changes and determinism cases for sampling changes. Mirror native API
changes in `dataset_rt/_dataset_rt.pyi` and update the relevant guides.

For performance claims, provide a reproducible command, hardware, package
versions, payload sizes, sample/cache counts, storage, worker settings, and
measurement boundaries. The [benchmark guide](https://red-eyed.github.io/DatasetRT/benchmarks/)
documents the existing evidence and its limits.

Package artifacts are built locally with `just build-all`; only the maintainer
publishes them. CI checks changes and deploys documentation.
