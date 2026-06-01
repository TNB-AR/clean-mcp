# Clean

> Local semantic code search for AI coding agents — runs on your laptop, indexes stay on disk.

Clean is an [MCP](https://modelcontextprotocol.io) server that gives Claude Code, Cursor, and other AI tools **meaning-aware** code search. It parses your repositories with tree-sitter, builds a call graph, embeds every function with a local sentence-transformer model, and stores everything in [LanceDB](https://lancedb.com) — no cloud, no API keys, no telemetry.

```text
"find the function that validates email on signup"
   ↓
search_code(query="email validation on signup")
   ↓
returns the right function with full source + callers/callees
```

## Features

- **Semantic search** — describes *behaviour*, not keywords; finds code by what it does.
- **Local-only** — embeddings, metadata, and source files live in `~/.clean/`. Nothing leaves your machine.
- **MCP-native** — drops into Claude Code / Cursor / any MCP client over stdio.
- **Index anything** — point it at a local folder *or* a public GitHub repo.
- **Tree-sitter parsing** — Python, JavaScript, TypeScript.
- **Call graph aware** — search results include direct callers and callees.

## Installation

Requires Python 3.10–3.13.

```bash
git clone https://github.com/cleanmcp/clean-mcp.git
python -m pip install -e ".[dev]"
```

This project uses a `src/` layout, so the module command only works after the
package is installed into the Python environment you are using. If you create or
activate a virtualenv, run `python -m pip install -e ".[dev]"` again inside
that environment before starting the server. The `dev` extra installs the
project's local tooling, including `pytest` and `ruff`.

First search or index downloads `all-MiniLM-L6-v2` (~90 MB) into
`~/.cache/huggingface/` if it is not already cached. After that, the model is
reused from disk and does not download again.

The server starts with heavy dependencies lazy-loaded. Launching `clean` should
reach `Starting Clean MCP server` quickly; the embedding model and LanceDB stack
load only when you call `search_code` or `index_repo` for the first time in that
process.

## Running the server

Clean is a **stdio** MCP server. It speaks the Model Context Protocol over stdin/stdout — there is no HTTP port, no web framework, and nothing to run under `uvicorn`/`gunicorn`. In normal use your MCP client (Claude Code, Cursor, …) **launches the process for you** based on the config below; you don't start it by hand.

To run it manually — for debugging, or to confirm it boots — use either of these equivalent commands:

```bash
# Module form
python -m clean.local.mcp_server

# Console script (installed by `python -m pip install -e ".[dev]"`)
clean
```

The process then waits silently for an MCP client to connect over stdin/stdout. That silence is expected — it is not a web server and will not print a URL. Do not type into that terminal; blank lines or other text are treated as MCP input and will produce JSON-RPC parse errors. Press `Ctrl+C` to stop it. To talk to it interactively, use the [MCP Inspector](https://github.com/modelcontextprotocol/inspector):

```bash
npx @modelcontextprotocol/inspector python -m clean.local.mcp_server
```

### Pin a default repo

Pass `--repo owner/repo` so `search_code` targets that repo without callers specifying it each time:

```bash
python -m clean.local.mcp_server --repo facebook/react
```

### Where it stores data

Override the on-disk locations with environment variables (see [Where your data lives](#where-your-data-lives)):

```bash
CLEAN_PERSIST_PATH=/data/clean/index python -m clean.local.mcp_server
```

## Wire it into your MCP client

### Claude Code / Cursor — `.mcp.json` (project-scoped)

```json
{
  "mcpServers": {
    "clean": {
      "command": "python",
      "args": ["-m", "clean.local.mcp_server"]
    }
  }
}
```

Or globally via the CLI: `claude mcp add clean -- python -m clean.local.mcp_server`

## Tools

| Tool | Key inputs | What it does |
|------|------------|--------------|
| `index_repo` | `path` or `repo`, optional `branch`, `force`, `background`, `timeout` | Index a local folder or clone+index a GitHub repo. Starts in the background by default. |
| `search_code` | `query`, optional `repo`, `branch`, `cwd`, `top_k`, `depth` | Semantic search across indexed code, returning source, callers/callees, and neighbouring functions. |
| `list_repos` | none | Show every indexed repository with branch, status, entity count, and detected metadata. |
| `get_file_tree` | optional `repo`, `branch`, `depth`, `include_hidden` | Print the directory tree of an indexed repo. |
| `get_source` | `file`, optional `repo`, `branch`, `start_line`, `end_line`, `function` | Read a file or exact indexed function from an indexed repo. |
| `expand_result` | `rank` | Get full source for a truncated result from the last `search_code` call in the same session. |
| `delete_repo` | `repo`, optional `branch`, `remove_files` | Remove an index, metadata record, and optionally cloned source files. |
| `get_token_savings` | optional `reset` | Show or reset TOON-format token savings for the current server session. |

### Indexing modes

Ask your MCP client to index code in one of two ways:

- **Index this local directory**
  - Calls `index_repo` with `path`.
  - Indexes a folder already on disk.
  - Does not clone anything.
  - Uses the folder basename as the repo name unless it can detect a GitHub remote, in which case it uses `owner/repo`.
  - Starts in the background by default; use `list_repos` to check for `ready`.

- **Index this GitHub repo `owner/repo`**
  - Calls `index_repo` with `repo`.
  - Expects a GitHub repo in `owner/repo` format.
  - Uses `RepoManager` to clone or locate the repo under the configured repos directory, usually `~/.clean/repos`.
  - Then indexes that checked-out local copy.
  - Starts clone and indexing in the background by default; use `list_repos` to check for `ready`.

Useful prompt variants:

```text
Index this local directory
Index this GitHub repo clarsbyte/obs-assistant
Force re-index this GitHub repo clarsbyte/obs-assistant
Index the main branch of this repo in the foreground
```

## Example usage in Claude Code

> "Index this directory" → calls `index_repo` with the current path
>
> "Find the function that handles login redirects" → `search_code`
>
> "Show me how the indexer entry point works" → `search_code` then `get_source`

## Where your data lives

```
~/.clean/
├── index/         LanceDB vector store
├── metadata.db    SQLite — which repos are indexed, status
└── repos/         git clones (only for GitHub-mode indexing)
```

Back up that folder to keep your indexes. Delete it to start fresh.

Override the location with env vars:

| Variable | Default |
|----------|---------|
| `CLEAN_REPOS_DIR` | `~/.clean/repos` |
| `CLEAN_DB_PATH` | `~/.clean/metadata.db` |
| `CLEAN_PERSIST_PATH` | `~/.clean/index` |
| `CLEAN_SHOW_PROGRESS_BAR` | `false` |

## Development

```bash
make install   # creates .venv and installs deps
make test      # runs the test suite
make lint      # ruff check + format check
make format    # apply ruff fixes
```

## Contributing

PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).
