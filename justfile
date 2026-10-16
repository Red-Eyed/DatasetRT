set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

python := "3.11"
py := "uv run --python " + python + " --extra dev"

default:
    just --list

sync:
    uv sync --python {{python}} --extra dev

develop:
    {{py}} maturin develop

fmt:
    {{py}} ruff format dataset_rt tests scripts
    cargo fmt

fmt-check:
    {{py}} ruff format --check dataset_rt tests scripts
    cargo fmt --check

lint:
    {{py}} ruff check dataset_rt tests scripts
    {{py}} scripts/check_precise_types.py

typecheck:
    {{py}} pyrefly check

test-rust:
    cargo test

clippy:
    cargo clippy --all-targets -- -D warnings

test-python: develop
    {{py}} pytest

test-correctness: develop
    {{py}} pytest -n 0 tests/integrations/test_loader_correctness.py --durations=5

sync-api-docs:
    {{py}} scripts/sync_api_docs.py

api-docs-check:
    {{py}} scripts/sync_api_docs.py --check

test: test-rust test-python

docs-build: sync-api-docs
    uv run --python {{python}} --locked --only-group docs mkdocs build --strict

docs-serve: sync-api-docs
    uv run --python {{python}} --locked --only-group docs mkdocs serve

docs-check: api-docs-check
    uv run --python {{python}} --locked --only-group docs mkdocs build --strict

check: fmt-check lint typecheck api-docs-check docs-check test-rust clippy test-python

smoke: develop
    {{py}} scripts/bench_smoke.py

bench-reader *args: develop
    {{py}} scripts/bench_reader.py {{args}}

build-all:
    {{py}} scripts/build_local.py

build-sdist:
    DATASETRT_SKIP_MACOS=true DATASETRT_SKIP_LINUX=true {{py}} scripts/build_local.py

build-macos:
    DATASETRT_SKIP_LINUX=true {{py}} scripts/build_local.py

build-linux:
    DATASETRT_SKIP_MACOS=true {{py}} scripts/build_local.py
