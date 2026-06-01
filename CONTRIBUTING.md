# Contributing to Clean

Thanks for your interest! Clean is small, local, and easy to hack on.

## Get set up

```bash
git clone https://github.com/cleanmcp/clean-mcp.git
cd clean-mcp
make install   # creates .venv and installs Clean + dev deps
make test      # runs the test suite
```

Requires Python 3.10-3.13. If you are not using `make`, create or activate your
own virtualenv and run:

```bash
python -m pip install -e ".[dev]"
```

The `dev` extra installs local tooling such as `pytest` and `ruff`.

## Running locally

Clean is a stdio MCP server. For a quick boot check, run either:

```bash
python -m clean.local.mcp_server
clean
```

The process waits for an MCP client over stdin/stdout. It is not an HTTP server
and will not print a URL.

## Project layout

```
src/clean/
  core/        Config and error types
  parsing/     Tree-sitter parsers (Python, JavaScript, TypeScript)
  indexing/    File scanning, code indexing, incremental reindex
  embedding/   Local embedding model (all-MiniLM-L6-v2)
  search/      Semantic search + context expansion
  storage/     LanceDB vector store
  formatting/  Output formatters (TOON, JSON, Rich)
  db/          SQLite metadata store
  repo/        Git clone/pull manager
  mcp/         MCP shared helpers
  local/       Stdio MCP server entry point
  services/    ServiceContainer (dependency wiring)
  stats/       Token savings tracker
  util/        Logging, security, hashing

tests/
  unit/        Fast, isolated tests
  integration/ Tests with real indexing/search
  e2e/         End-to-end MCP server tests
```

## Common tasks

```bash
make install   Create .venv and install Clean + dev deps
make test      Run all tests
make lint      Lint + format check (ruff)
make format    Auto-format with ruff
make clean     Remove .venv and caches
```

Before opening a PR, run `make lint` and `make test`. The first search or index
may download the local embedding model (`all-MiniLM-L6-v2`) into your Hugging
Face cache if it is not already present.

## Adding a language parser

Tree-sitter does the heavy lifting. To add a new language:

1. Add the tree-sitter grammar to `pyproject.toml` `dependencies`.
2. Create `src/clean/parsing/yourlang.py` following `python.py` or `javascript.py`.
3. Register it in `src/clean/parsing/registry.py`.
4. Add tests under `tests/unit/parsing/`.

## Code style

- Type hints on public functions.
- Small focused functions; three similar lines beats a premature helper.
- No dead code, no commented-out blocks, no speculative abstractions.

## Submitting a PR

1. Open an issue first if it's a non-trivial change.
2. Fork, branch from `main` (`fix/...`, `feat/...`, `docs/...`).
3. Keep commits focused - one logical change each.
4. Make sure `make test` and `make lint` pass.
5. Open the PR with a short description of what and why.

## Reporting bugs

Open an issue with: what you did, what you expected, what happened, your OS + Python version.

## License

By contributing, you agree your contributions will be MIT-licensed.
