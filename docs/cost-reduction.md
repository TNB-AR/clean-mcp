# How Clean Reduces Cost — The Token-Economy Deep Dive

This document explains, in full detail, **how Clean minimises the cost of using an
AI coding agent**. "Cost" here is almost entirely about one scarce resource: the
**tokens** that flow into and out of a large language model. Every token an agent
reads or writes costs money and, just as importantly, consumes a slice of the
model's finite context window. Clean is engineered, end to end, to deliver the
*maximum useful understanding of a codebase per token spent*.

There is no single "cost-reducing algorithm." Instead Clean stacks **seven
cooperating mechanisms**, each attacking a different source of token (or compute)
waste. This document covers all of them, with references to the exact code that
implements each one.

---

## Table of contents

1. [The cost model: why tokens are the thing that matters](#1-the-cost-model)
2. [Mechanism 1 — Retrieval instead of exploration](#2-mechanism-1--retrieval-instead-of-exploration)
3. [Mechanism 2 — Tiered result formatting](#3-mechanism-2--tiered-result-formatting)
4. [Mechanism 3 — On-demand expansion and the session cache](#4-mechanism-3--on-demand-expansion-and-the-session-cache)
5. [Mechanism 4 — TOON encoding and the measurement harness](#5-mechanism-4--toon-encoding-and-the-measurement-harness)
6. [Mechanism 5 — Hybrid retrieval precision (fewer round-trips)](#6-mechanism-5--hybrid-retrieval-precision)
7. [Mechanism 6 — Batched call-graph context expansion](#7-mechanism-6--batched-call-graph-context-expansion)
8. [Mechanism 7 — Incremental indexing and staleness detection](#8-mechanism-7--incremental-indexing-and-staleness-detection)
9. [The accounting layer: how savings are measured](#9-the-accounting-layer)
10. [A worked end-to-end example](#10-a-worked-end-to-end-example)
11. [Tuning knobs](#11-tuning-knobs)
12. [Summary table](#12-summary-table)

---

## 1. The cost model

Before the mechanisms, fix the unit of cost firmly in mind.

When an AI agent (Claude Code, Cursor, …) works on a codebase, the dominant cost
is **the agent's context window**. Three things fill it up:

1. **Input tokens** — everything the agent reads: file contents, search results,
   tool outputs.
2. **Output tokens** — everything the agent writes, including the tool calls it
   emits to go fetch more.
3. **Round-trips** — each "search → read → search again" loop re-sends a growing
   conversation back to the model. Long conversations are quadratically
   expensive because the whole transcript is re-processed on every turn.

Clean's internal accounting uses a deliberately simple, provider-agnostic
approximation that you will see throughout the code:

```
tokens ≈ characters / 4
```

This is the standard rule-of-thumb for English-and-code text under byte-pair
encodings. It is implemented literally in the stats tracker
(`src/clean/stats/tracker.py:64`):

```python
json_tokens = json_chars // 4
toon_tokens = toon_chars // 4
```

Every mechanism below is ultimately judged by how it moves that
`characters / 4` number — either by shrinking what is returned, or by
eliminating whole round-trips that would otherwise have happened.

---

## 2. Mechanism 1 — Retrieval instead of exploration

**This is the single largest cost lever, and it is behavioural, not numeric.**

### The expensive default

Without a semantic index, an agent answering "where is login handled?" does
something like:

```
grep -r "login" .         →   200 matching lines across 40 files
read file A                →   1,800 tokens
read file B                →   2,400 tokens
read file C ...            →   (repeat 10–20 times)
```

Each `read` pours an entire file — most of it irrelevant — into the context
window. The agent burns tens of thousands of tokens *just locating* the code,
before it has done any actual reasoning. Worse, text matching (`grep`) misses
code that is semantically relevant but lexically different ("auth", "signin",
"session start"), so the agent often has to widen the search and read *more*.

### What Clean does

Clean replaces that whole loop with a **single semantic `search_code` call** that
returns *exactly the functions that matter*, ranked, with file paths, line
numbers, and call-graph relationships already attached. One tool call, one
bounded response.

The mechanism is enforced two ways:

- **Technically** — the index is built by parsing every function/class/method
  with tree-sitter, embedding each one with a local sentence-transformer
  (`all-MiniLM-L6-v2`, 384-dim), and storing the vectors in LanceDB. A query is
  embedded once and matched by cosine similarity. Meaning, not keywords.
- **Behaviourally** — the `search_code` tool *description* is itself a
  cost-control device. It is an unusually long, directive prompt
  (`src/clean/local/mcp_server.py:146`) that repeatedly instructs the agent:

  > "THIS TOOL REPLACES grep, find, glob, and manual file reading…"
  > "BEFORE reading any source files — search first, read only if needed"
  > "DO NOT skip this tool and grep/read files instead. That approach is slower,
  > misses semantic connections, and wastes context window tokens."

  This is intentional. The cheapest token is the one never read, so the product
  spends words steering the agent away from the read-everything anti-pattern.

The net effect: the agent's path to the right code goes from *"read 15 files to
find 1 function"* to *"receive 1 function (plus its neighbours) directly."* Every
file that is **not** read is pure savings.

### Why no grep is needed: the index stores the location, not just the vector

The reason a single semantic query can replace grep entirely is that **LanceDB
does not store bare vectors — it stores the full code record next to each
vector.** Every entity row carries its own location and relationship metadata, so
the answer to "where is this?" travels back *with* the similarity match. There is
never a second step where the agent has to go find the code on disk.

The on-disk schema is defined in `_entity_schema`
(`src/clean/storage/lancedb.py:28`). Each row holds:

| Column | Purpose |
|--------|---------|
| `vector` | The 384-dim embedding that the similarity search matches against. |
| `file_path` | **The exact file** the entity lives in — returned directly, no grep. |
| `line_start`, `line_end` | **The exact line range** — so the agent gets `path:start-end` immediately. |
| `name` | The function/class/method name. |
| `code` | The full source text of the entity (used later by `expand_result`/tier summaries). |
| `calls`, `called_by` | JSON-encoded call-graph edges — the neighbours, precomputed. |
| `language`, `kind`, `sub_kind`, `class_name`, `decorators`, `exported` | Structural metadata for filtering and display. |
| `id`, `project_id` | Identity and per-repo scoping. |
| `chunk_index`, `parent_id`, `total_chunks` | Chunking bookkeeping for very large entities. |

So a single approximate-nearest-neighbour query against the `vector` column
returns rows that **already contain** `file_path`, `line_start`, `line_end`,
`name`, and the call edges. That is precisely why `search_code`'s response can
print `auth/signup.py:42-58` with callers and callees attached, with **zero**
filesystem scanning: the search index *is* the map of the codebase.

LanceDB ranks by L2 distance, which the store converts to a 0–1 similarity score
(`lancedb.py:329`):

```python
similarity = 1.0 / (1.0 + distance)
```

The same stored columns also power the keyword side of hybrid search — name and
path lookups run as indexed `WHERE` filters over `name` / `file_path`
(`get_by_name_substring`, `get_by_file_substring`), again with no grep over source
files. Everything the agent needs to *locate* code is answered from the index
itself; the actual source files are only ever touched on an explicit
`expand_result`/`get_source` read.

---

## 3. Mechanism 2 — Tiered result formatting

Finding the right functions is only half the battle. If the server then dumped
the full source of all of them, it would re-introduce the very bloat semantic
search just eliminated. So Clean **never returns full source code in a search
response.** Instead it returns a *tiered summary*, implemented in
`src/clean/formatting/tiered.py`.

The principle: **spend tokens in proportion to how likely a result is to be the
one the agent wants.** The top hit gets the richest summary; lower-ranked hits
get progressively terser ones.

### The three tiers

| Tier | Ranks | What is included |
|------|-------|------------------|
| **Tier 1** | #1 (best match) | Location + line count, the **signature line**, up to **4 lines of docstring**, and full **call-graph context** — `CALLS →`, `CALLED BY ←`, and `SAME FILE` neighbours — plus an `expand_result` hint. |
| **Tier 2** | #2–#5 | Location + line count, signature line, up to 4 docstring lines, and an `expand_result` hint. **No call-graph context.** |
| **Tier 3** | #6+ | The most compact form: location + line count + **signature only** + `expand_result` hint. |

This is exactly the dispatch in `format_tiered_results`
(`src/clean/formatting/tiered.py:278`):

```python
for i, result in enumerate(results):
    rank = i + 1
    if rank == 1:
        sections.append(_format_tier1(result, context, max_tier1_lines, max_sim))
    elif rank <= 5:
        sections.append(_format_tier2(rank, result, max_tier2_lines, max_sim))
    else:
        sections.append(_format_tier3(rank, result, max_sim))
```

A result that would have cost, say, 300 lines of source if dumped in full now
costs ~3–8 lines as a Tier-1 summary, and ~2 lines as a Tier-3 entry. With a
default `top_k` of 5 (and up to 50 allowed), the difference between "summarise
all" and "dump all" is enormous — typically an order of magnitude.

### Why only the docstring and signature?

Because the **signature + docstring is the highest-information-density slice of a
function.** It tells the agent the name, the parameters, the return shape, and
the author's own description of intent — usually enough to decide "yes this is
it" or "no, skip." The expensive body is fetched only if that decision is "yes."

The docstring extractor (`_docstring_block`, `tiered.py:62`) is language-aware: it
recognises Python triple-quoted strings (`"""`/`'''`), `#` line comments, and
JS/TS block comments (`/** … */`, `// …`), so the snippet is meaningful across
all supported languages.

### Normalised relevance scores

Each tier header shows a **normalised** score, not a raw cosine similarity
(`_normalize_score`, `tiered.py:20`). The top result's raw similarity becomes the
100% reference, and the rest are expressed relative to it, with qualitative
labels:

```python
normalized = similarity / max_similarity
if   normalized >= 0.80: label = "Strong"
elif normalized >= 0.60: label = "Good"
elif normalized >= 0.40: label = "Moderate"
else:                    label = "Weak"
```

This is itself a cost optimisation: a clear `Strong · 100%` vs `Weak · 32%`
signal helps the agent **stop early** and avoid expanding or re-searching results
that obviously do not match — which saves the tokens those follow-ups would cost.

A short footer is appended after the results (`mcp_server.py:1005`) reminding the
agent that rank #1 has the most detail, that `expand_result(rank=N)` exists, and
to "only call `search_code` again for a genuinely different concept" — again,
nudging away from wasteful repeat searches.

---

## 4. Mechanism 3 — On-demand expansion and the session cache

Tiered formatting deliberately withholds full source. The agent gets it back
**only when it explicitly asks**, via the `expand_result` tool. This is the
"lazy loading" half of the design.

### How expansion works

After every search, the server stores the full result set in an in-memory
session cache keyed by rank (`mcp_server.py:982`):

```python
search_cache.clear()
search_cache["repo"]       = project.repo_full_name
search_cache["branch"]     = getattr(project, "branch", None)
search_cache["local_path"] = project.local_path
search_cache["results"]    = results
```

When the agent calls `expand_result(rank=3)`, the handler
(`_handle_expand_result`, `mcp_server.py:1248`) looks up result #3 in the cache
and reads its **exact line range** from disk — no re-embedding, no re-searching,
no re-ranking. The expensive search pipeline runs once; expansions are cheap
disk reads of precisely the bytes requested.

### Why this saves money

The agent pays the full-source token cost **only for the handful of results it
genuinely needs to read in detail** — often just rank #1. The other four (or
forty-nine) results never cost more than their few-line summary. Compare this to
a naïve server that returns all bodies up front: there, the agent pays for 50
full function bodies to use 1.

`get_source` (`mcp_server.py:1132`) provides the same on-demand discipline for
arbitrary files: it returns at most **500 lines** per call, and supports
`start_line`/`end_line` windows and a `function=` lookup that reads a named
entity's exact span (up to 2000 lines). You never accidentally inhale a
10,000-line file.

---

## 5. Mechanism 4 — TOON encoding and the measurement harness

### TOON: Token-Optimized Object Notation

`src/clean/formatting/toon.py` implements a compact **tabular** encoding of
results. Its reason for existing is structural redundancy in JSON.

A JSON array of results repeats every field name on every element:

```json
[
  { "function_name": "validate_email", "file_path": "auth/signup.py", "line_start": 42, "similarity": 0.91 },
  { "function_name": "check_session",  "file_path": "auth/session.py", "line_start": 88, "similarity": 0.86 }
]
```

The keys `function_name`, `file_path`, `line_start`, `similarity` — plus braces,
quotes, and commas — are paid for *once per row*. TOON hoists the field names
into a **single header row** and emits the data as aligned columns:

```
results
  name             | file_path        | line   | similarity
  validate_email   | auth/signup.py   | 42     | 91%
  check_session    | auth/session.py  | 88     | 86%
```

The repeated structural punctuation is gone; each field name is paid for once
rather than N times. The module docstring states the design target: **30–40%
token savings** versus the equivalent JSON. Column widths, truncation, and which
fields appear are all driven by `ToonFormatterConfig`
(`src/clean/core/config.py:77`).

### The measurement harness

Here is the subtle and important part of how Clean *proves* its savings. On every
search, the server formats the results **twice** (`mcp_server.py:992`):

```python
text = format_tiered_results(results, context)   # what the agent actually receives
...
json_text = container.json_formatter.format_results(results)   # a full-fat JSON baseline
container.stats_tracker.record_search(json_text, text)
```

- `text` is the **compact tiered summary** that is actually returned to the agent.
- `json_text` is a **full JSON dump including every function's complete `code`
  field** (`src/clean/formatting/json.py:13`) — i.e. a faithful model of what a
  naïve MCP server *would* have sent.

`record_search` (`tracker.py:59`) then measures the delta between the two:

```python
json_chars = len(json_output)
toon_chars = len(toon_output)
saved      = json_chars - toon_chars
s.total_tokens_saved_est += saved // 4
```

So the headline "tokens saved" figure is a concrete, per-search measurement of
*the compact representation versus the full-source JSON baseline*, accumulated
and persisted to `~/.clean/stats.json`. It is not a marketing estimate; it is the
literal character-count difference between what was sent and what a dumb server
would have sent, divided by four.

> **Note on naming.** In the live search path the baseline (`json_text`) is
> compared against the **tiered** output. The standalone `ToonFormatter` is the
> compact tabular encoder used where a flat result table is wanted; the
> `record_search` parameter is named `toon_output` for historical reasons but
> receives whichever compact rendering was sent. Both share the same goal:
> measure compact-vs-JSON and bank the difference.

The agent (or you) can read the running total at any time with the
`get_token_savings` tool, which prints the formatted report from
`StatsTracker.get_summary` (`tracker.py:91`): total searches, JSON vs compact
character counts, estimated tokens saved, average savings percent, and a
per-session breakdown.

---

## 6. Mechanism 5 — Hybrid retrieval precision

Every **unnecessary follow-up search is a full round-trip** — the entire growing
conversation re-sent to the model. So making the *first* search land the right
results is itself a major cost saver. Clean does this with **hybrid retrieval**
(`src/clean/search/searcher.py` + `src/clean/search/hybrid.py`).

### The problem with pure semantic search

Embeddings are great at *behaviour* ("function that validates email before
signup") but can rank an **exact identifier** lower than expected. If the agent
already knows a name — `ClientLayout`, `get_user_by_id`, `src/components/auth` —
pure vector search may bury it, forcing a second query.

### The hybrid solution

`CodeSearcher.search` first runs `extract_identifiers` on the query
(`hybrid.py:66`), which detects identifier-shaped tokens:

- **PascalCase** (`ClientLayout`)
- **camelCase** (`getUserById`)
- **snake_case** / **UPPER_CASE** (`get_user_by_id`, `MAX_RETRIES`)
- **dotted paths** (`auth.session`)
- **file-path fragments** (`src/components/auth`)

If identifiers are present, the searcher widens the candidate pool —
`semantic_fetch = top_k * 2` (`searcher.py:85`) — and additionally runs direct
name-substring and path-substring lookups against the store. All three sources
are then fused by `merge_results` (`hybrid.py:103`) with additive weighting:

```python
semantic_weight = 0.6   # score = similarity * 0.6
name_weight     = 0.3   # score += 0.3 for a name match
path_weight     = 0.1   # score += 0.1 for a path match
```

An entity that is *both* a strong semantic match *and* a name match has its
scores **summed**, so it floats decisively to the top. When no identifiers are
present, the code falls straight back to pure semantic search — natural-language
queries behave exactly as before, with zero overhead.

The payoff is precision-at-rank-1. The more often the agent finds what it needs
in the first response, the fewer second and third searches it issues — and each
avoided search is a whole round-trip's worth of tokens never spent.

---

## 7. Mechanism 6 — Batched call-graph context expansion

When the agent needs to understand *how* the top result fits into the codebase,
Clean attaches its **call-graph neighbourhood** — its callees, its callers, and
its same-file siblings (the `CALLS / CALLED BY / SAME FILE` block in Tier 1).
This pre-empts the agent's next questions ("what does this call?", "who calls
this?") and so prevents the follow-up searches it would otherwise make.

Gathering that neighbourhood naïvely is itself expensive: walking a call graph to
`depth` levels with one lookup per node is `O(branching^depth)` database queries.
`ContextExpander.expand` (`src/clean/search/context.py:20`) instead issues **one
batched query per depth level** — `O(depth)` total — via `get_by_names`:

```python
# Uses batch queries per depth level instead of individual lookups,
# reducing total queries from O(branching^depth) to O(depth).
```

It also tracks a `visited` set to avoid revisiting nodes in cyclic graphs, and
caps the result at `MAX_CONTEXT_ENTITIES = 50` (`context.py:11`) per direction so
a pathologically connected function can never blow up either the query count or
the response size.

This mechanism saves **compute and latency cost** (far fewer DB round-trips) and,
by answering the agent's likely next question in the same response, saves the
**token cost** of the follow-up search that would otherwise be needed.

---

## 8. Mechanism 7 — Incremental indexing and staleness detection

The remaining cost source is **embedding work**: turning source code into vectors.
Embedding an entire repository on every change would be wasteful (CPU time
locally; potentially API dollars if a hosted embedder is ever configured). Clean
avoids re-embedding code that has not changed.

### Change detection

`IncrementalIndexer.detect_changes` (`src/clean/indexing/incremental.py:33`)
compares the current file set against the previously indexed state and classifies
every file as **added / modified / deleted / unchanged**, using the cheapest
reliable method available:

1. **Git diff first** — if the project is a git repo, `git diff --name-only
   <stored_head>..HEAD` (`incremental.py:58`) yields the changed set almost
   instantly, with no file reads at all.
2. **Content-hash fallback** — otherwise it SHA-256-hashes each file
   (`_hash_based_diff`, `incremental.py:107`) and compares against stored hashes.

Only **added** and **modified** files are re-parsed and re-embedded. **Unchanged**
files keep their existing vectors. **Deleted** files have their vectors dropped.
On a large repo where a commit touched three files, this turns a full re-embed
into a three-file re-embed.

### Staleness-triggered, non-blocking re-index

On every `search_code`, before searching, the server runs `check_staleness`
(`src/clean/indexing/staleness.py:73`). It decides whether the index is out of
date using:

- **Git** — current `HEAD` vs the stored head, plus `git status --porcelain` to
  catch uncommitted/staged/untracked edits (`staleness.py:22`); falling back to
- **mtime** — comparing each tracked file's modification time against when it was
  last indexed (`staleness.py:60`).

Crucially, if the index is stale the re-index is fired **fire-and-forget** and the
search proceeds **immediately against the existing index**
(`mcp_server.py:914`):

```python
if is_stale:
    # Fire-and-forget: do not await — search stale data immediately.
    asyncio.ensure_future(loop.run_in_executor(None, lambda: container.indexer.index(...)))
```

The agent is never billed (in latency or blocked turns) for the re-index; it gets
its answer now, and the fresh vectors are ready for next time. The staleness
check itself is bounded by a 10-second timeout, and falls through to "use what we
have" if it can't decide quickly.

---

## 9. The accounting layer

All of the above would be unfalsifiable without measurement. The accounting lives
in `src/clean/stats/tracker.py` and is wired into every search.

- **Per-search recording** — `record_search` runs on every `search_code`, banking
  the JSON-baseline-vs-compact character delta and the `delta / 4` token estimate.
- **Persistence** — totals are written to `~/.clean/stats.json` and survive across
  runs. Session counters (`session_searches`, `session_tokens_saved_est`) reset to
  zero on load (`tracker.py:43`) so you can see both lifetime and this-session
  impact.
- **Derived metric** — `avg_savings_percent = total_chars_saved /
  total_json_chars * 100` (`tracker.py:75`) gives the headline "we are X% smaller
  than a naïve server" number.
- **Reporting** — `get_token_savings` prints the full report; pass `reset: true`
  to zero the counters.

The fields tracked (`TokenStats`, `tracker.py:14`) are deliberately exhaustive:
total searches; total JSON chars; total compact chars; chars saved; estimated
JSON / compact / saved tokens; average savings percent; first/last search
timestamps; and the per-session trio.

---

## 10. A worked end-to-end example

Suppose an agent is asked: *"How does email validation work on signup?"* in a
50,000-line repo.

**Without Clean (the expensive path):**

| Step | Approx. tokens |
|------|----------------|
| `grep -r email` → scan output | ~1,500 |
| Read `signup.py` (whole file) | ~2,000 |
| Read `validators.py` (whole file) | ~2,500 |
| Read `user_model.py` to trace a call | ~2,200 |
| Re-grep "validate" after a miss | ~1,200 |
| Read 2 more files | ~4,000 |
| **Total to merely locate the code** | **~13,400** |

**With Clean:**

| Step | Approx. tokens |
|------|----------------|
| `search_code("email validation on signup")` → tiered summary of 5 results, with rank-1 call graph | ~600 |
| `expand_result(rank=1)` → full source of the one function that matters | ~450 |
| **Total** | **~1,050** |

The order-of-magnitude difference comes from stacking the mechanisms: semantic
retrieval skips the grep-and-read flailing (Mechanism 1), tiered formatting keeps
the five-result response tiny (Mechanism 2), expansion fetches exactly one body
on demand (Mechanism 3), the call graph pre-answers "what does it call" so no
third search is needed (Mechanism 6), and hybrid scoring put the right function
at rank 1 so there was no second search (Mechanism 5). The exact figures vary,
but the shape — **~10× fewer tokens to reach the same understanding** — is the
design goal, and `get_token_savings` lets you watch the real numbers accrue.

---

## 11. Tuning knobs

The cost/recall trade-off is adjustable. The most relevant levers:

| Knob | Where | Effect on cost |
|------|-------|----------------|
| `top_k` (search arg) | default **5**, clamped to **50** (`mcp_server.py:941`, `searcher.py:73`) | More results = larger response. Use 3 for targeted lookup, 10–15 for exploration. |
| `depth` (search arg) | default **1**, clamped to **3** in the handler / **5** in config (`mcp_server.py:936`) | Deeper call-graph context = more `CALLS/CALLED BY` entries in Tier 1. |
| `MAX_CONTEXT_ENTITIES` | **50** (`context.py:11`) | Hard ceiling on call-graph entries per direction; bounds worst-case Tier-1 size. |
| Tier boundaries | ranks `1`, `2–5`, `6+` (`tiered.py:280`) | Where detail drops off. Lower-ranked = terser. |
| TOON column set / widths | `ToonFormatterConfig` (`config.py:77`) | Which fields appear and how aggressively values are truncated. |
| `get_source` line cap | **500** lines/call (2000 for `function=`) (`mcp_server.py:1171`) | Prevents whole-file inhalation. |
| Embedding model | `all-MiniLM-L6-v2`, 384-dim; `CLEAN_EMBEDDING_MODEL` env | Smaller model = faster, cheaper indexing. |

---

## 12. Summary table

| # | Mechanism | Cost attacked | Core idea | Key code |
|---|-----------|---------------|-----------|----------|
| 1 | Retrieval instead of exploration | Input tokens + round-trips | One semantic search replaces grep + many file reads; tool prompt steers the agent to use it | `mcp_server.py:146` |
| 2 | Tiered result formatting | Input tokens | Never dump full source; spend tokens in proportion to rank (signature + docstring, call graph for #1 only) | `formatting/tiered.py` |
| 3 | On-demand expansion + session cache | Input tokens | Full bodies fetched only when asked, by rank, from a per-search cache | `mcp_server.py:1248` |
| 4 | TOON encoding + measurement | Input tokens | Tabular encoding kills JSON's repeated keys (~30–40%); every search measures compact-vs-JSON and banks the delta | `formatting/toon.py`, `stats/tracker.py` |
| 5 | Hybrid retrieval precision | Round-trips | Identifier detection + weighted fusion lands the right hit at rank 1, avoiding repeat searches | `search/hybrid.py` |
| 6 | Batched context expansion | Compute + round-trips | `O(depth)` batched queries with a 50-entity cap; pre-answers the agent's next question | `search/context.py` |
| 7 | Incremental indexing + staleness | Embedding compute | Git-diff / hash change detection re-embeds only changed files; stale re-index is non-blocking | `indexing/incremental.py`, `indexing/staleness.py` |

**In one sentence:** Clean reduces cost by *retrieving instead of reading*,
*summarising instead of dumping*, *expanding only on demand*, *landing the right
result first try*, and *never re-embedding code that hasn't changed* — and it
measures the token delta on every single search so the savings are a fact, not a
claim.
