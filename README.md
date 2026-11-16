# Acme Support GenAI

A support chatbot for a fictional cloud product, "Acme Cloud". It answers questions about **refunds, plans, pricing,
SLA and orders**, and nothing else. It is a FastAPI service around a LangGraph agent, with retrieval over a small
knowledge base, multi-turn sessions, tracing, and a design that runs as several replicas behind a load balancer.

## What it does

- **Answers only from evidence.** Every answer needs a tool result from the current turn: a knowledge-base search
  (`search_knowledge_base`) or an order lookup (`get_order_status`). A deterministic guard replaces any answer that
  has no evidence with a fixed refusal, whatever the model says.
- **Refuses off-topic questions cheaply.** A scope gate embeds the question and compares it with the knowledge base.
  Questions that are clearly unrelated are refused before the model is called.
- **Remembers conversations.** Send a `session_id` back and the next message sees the earlier turns. History sent to
  the model is trimmed to the last 10 messages.
- **Handles concurrency.** Two messages for the same session cannot run at once: the second gets `409 session_busy`
  immediately. Different sessions run in parallel.
- **Cleans up on a schedule.** The purge job deletes conversations idle for 30 days and retires old knowledge-base
  indexes. It only runs when you schedule it (see [Scheduled jobs](#scheduled-jobs)).
- **Fails cleanly.** Provider, database and agent failures map to specific status codes with a request ID.

## Quick start

You need Python 3.14.7 or newer, [uv](https://docs.astral.sh/uv/), Docker, an OpenAI API key, and a
[Langfuse](https://langfuse.com) project (cloud, or self-hosted on `http://localhost:3000`).

```bash
# 1. install
uv sync

# 2. configure
cp env.example .env
#    then fill in OPENAI_API_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and LANGFUSE_HOST

# 3. start Postgres with pgvector (port 5433)
docker compose up -d

# 4. run the API
uv run uvicorn app.main:app --reload
```

`DATABASE_URL` in `env.example` already points at the compose database
(`postgresql://acme:acme@localhost:5433/acme_support`). Leave it empty to run with no database: sessions then live in
memory, and the knowledge-base index is a local file in `.index/`.

Try it:

```bash
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"message": "Can I get a refund for order A-1001?"}'
```

```json
{
  "answer": "You can get a full refund for order A-1001 ...",
  "tools_used": ["get_order_status", "search_knowledge_base"],
  "session_id": "5f0c1c2e-7a53-4a55-9a52-1f0f6c7b2d11",
  "trace_id": "..."
}
```

Send the returned `session_id` with your next message to continue the conversation. The demo orders are `A-1001`
(within the refund window) and `A-1002` (past it). Questions outside Acme Cloud get
`"I can only help with Acme Cloud policy and order questions."`

## API

### `POST /chat`

| Field | Type | Notes |
|---|---|---|
| `message` | string, 1 to 2000 characters | required |
| `session_id` | UUID version 4 | optional. Omit it to start a new conversation; the response returns the id to reuse. Ids you invent are rejected with 422, so they cannot be guessed. |

Response: `answer`, `tools_used` (tools called **this turn**), `session_id`, `trace_id` (Langfuse).

### `GET /health`

Returns `{"status": "ok", "langfuse": true}`.

### Errors

Every error has the same shape, and the `request_id` also comes back in the `x-request-id` response header. Send your
own `x-request-id` (letters, digits and hyphens, up to 64 characters) to correlate with your logs.

```json
{"error": {"code": "session_busy", "message": "...", "request_id": "..."}}
```

| Status | `code` | When |
|---|---|---|
| 409 | `session_busy` | another message for this session is still being answered |
| 422 | (FastAPI validation) | empty or oversized message, `session_id` not a UUID4 |
| 500 | `agent_loop` | the agent hit its step limit without finishing |
| 500 | `internal_error` | anything unexpected; details are in the logs, never in the response |
| 502 | `upstream_rejected` | OpenAI refused the request (bad key, bad model, bad parameters) |
| 503 | `upstream_unavailable` | OpenAI is unreachable, rate limiting, or failing |
| 503 | `storage_unavailable` | Postgres is unreachable |
| 503 | `knowledge_base_unavailable` | the knowledge-base index was removed while a replica was using it |
| 504 | `upstream_timeout` | OpenAI took longer than `LLM_TIMEOUT_SECONDS` |

A 409 is safe to retry once the first message has been answered. A UI should disable "send" while a turn runs.

## How a request is handled

1. **Lock.** The session lock is taken (a Postgres advisory lock, or an in-process lock without a database). If it is
   already held, the request ends with 409.
2. **Load.** The session's retention clock is reset and its conversation is loaded from the checkpointer.
3. **Gate.** The last two user messages are embedded and compared with the knowledge base. Too dissimilar and no
   model call is made.
4. **Agent.** The first model call of every turn is forced to use a tool. Tools run, and the model writes the answer
   from their results.
5. **Guard.** If no tool returned usable data this turn, the answer is replaced with the refusal.
6. **Save and release.** State is saved and the lock is released.

## Configuration

Settings are read from the environment and from `.env` (see `app/config.py`). The names are the field names in
upper case.

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | required | model and embeddings |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` | required | tracing |
| `LANGFUSE_HOST` | `http://localhost:3000` | Langfuse server |
| `DATABASE_URL` | unset | Postgres with pgvector. Unset means in-memory sessions and a local file index |
| `CHAT_MODEL` | `gpt-4o-mini` | chat model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | embeddings |
| `TEMPERATURE` | `0.0` | chat model temperature |
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | `300`, `30` | how documents are split. Changing them (or the model, or a document) builds a new index |
| `KB_DIR`, `INDEX_DIR` | `knowledge_base/`, `.index/` | where the documents are, and where the local file index is written |
| `LLM_TIMEOUT_SECONDS`, `LLM_MAX_RETRIES` | `30`, `2` | per model call |
| `SCOPE_SCORE_THRESHOLD` | `0.2` | minimum similarity for the scope gate |
| `RETRIEVAL_SCORE_THRESHOLD`, `RETRIEVAL_K` | `0.2`, `2` | what the search tool returns |
| `HISTORY_MAX_MESSAGES` | `10` | messages sent to the model |
| `SESSION_RETENTION_DAYS` | `30` | idle time before a conversation is purged |
| `KB_INDEXES_TO_KEEP` | `2` | indexes the purge job keeps |
| `DB_POOL_MIN_SIZE`, `DB_POOL_MAX_SIZE` | `1`, `10` | Postgres pool for state and search |
| `DB_LOCK_POOL_MAX_SIZE` | `20` | one connection per running turn: keep it above your concurrent chats |
| `DB_TIMEOUT_SECONDS` | `5` | how long a request waits for a database connection |
| `LOG_LEVEL` | `INFO` | logs are JSON, one object per line, with the request ID |

Changing `CHAT_MODEL`, the thresholds or the prompt can change behavior: run the evals afterwards.

## Running several replicas

The app keeps no state in the process when `DATABASE_URL` is set, so you can run any number of replicas.

- All conversations, locks, retention rows and the vector index live in Postgres.
- Replicas can start at the same moment on an empty database. Schema setup is serialized with an advisory lock, and
  the knowledge-base index is built once, by whichever replica gets there first.
- Run `python -m app.jobs.ingest` in your deploy step if you would rather build the index before any replica starts.
- Use a **direct** connection to Postgres, or a pooler in **session** mode. The session lock is an advisory lock, and
  transaction-mode poolers such as PgBouncer break it.
- Roll out a changed knowledge base without downtime: the new index is built next to the old one, and old replicas
  keep using theirs until they are replaced.

## Scheduled jobs

```bash
uv run python -m app.jobs.ingest   # build the knowledge-base index if it is missing (a no-op otherwise)
uv run python -m app.jobs.purge    # delete idle conversations and retired indexes
```

Run `purge` on a schedule, for example daily from cron:

```
15 3 * * *  cd /path/to/genai && uv run python -m app.jobs.purge
```

It is safe to run from several places at once.

## Evals

```bash
uv run python -m evals.eval
```

Runs a fixed set of questions (in-scope, off-topic and mixed) through the agent and reports keyword recall and tool
correctness to Langfuse under the dataset `acme-support-eval`. It calls the real model and needs your Langfuse keys.
It does not need Postgres: it uses an in-memory checkpointer and the local file index.

## Tests

```bash
uv run pytest                        # everything
uv run pytest -m "not integration"   # fast tests only, no database needed
```

Most tests use a fake model and need no network. Tests marked `integration` need Postgres, and the pgvector ones need
the `vector` extension; they skip when it is not there. They run against
`postgresql://acme:acme@localhost:5433/acme_support` by default. Point them elsewhere with `TEST_DATABASE_URL`.

Some tests start six replicas at once on a brand-new database, so the database user needs permission to create and
drop databases. The compose user can.

## Project layout

```
app/
  main.py           app, lifespan, middleware, routers
  bootstrap.py      builds the agent and the single run() entry point
  graph.py          the LangGraph graph: gate -> agent <-> tools -> guard
  config.py         typed settings
  errors.py         error types with their status codes and messages
  logging.py  middleware.py   JSON logs, request IDs, access log
  dependencies.py   FastAPI dependencies
  api/              routes, request and response schemas, error handler
  services/         the chat use case: lock, touch, run, map failures
  ai/
    prompts.py  llm.py  guardrails.py
    tools/          search_knowledge_base, get_order_status
    rag/            loading, chunking, the local file index, the store protocol
  db/               everything that talks to Postgres
    advisory.py       advisory locks and the schema lock
    checkpointer.py   conversation storage (Postgres, or in memory)
    sessions.py       retention rows and purging
    session_locks.py  the per-session lock
    pgvector.py       the shared vector index
  jobs/             ingest.py, purge.py
evals/              the eval dataset and runner
tests/              unit and integration tests
knowledge_base/     the documents the bot answers from (pricing, refunds, SLA)
```

Dependencies point inward: `api` calls `services`, which use `ai` and `db`. `ai` depends on small interfaces such as
`SimilarityStore`, never on a database.

## Development

```bash
uv run ruff check --fix
uv run ruff format
```

Add or change a knowledge-base document and the index is rebuilt automatically: its fingerprint covers the files, the
embedding model and the chunk settings.

## Troubleshooting

- **App fails to start with "this Postgres has no pgvector extension".** The database is not using the pgvector
  image. Check `docker-compose.yml` says `pgvector/pgvector:pg17`, then run `docker compose pull && docker compose up -d`.
- **Collation version mismatch after switching images.** If you changed an existing Postgres data volume from the
  plain `postgres` image to the pgvector one, Postgres warns, and refuses to create new databases. For each of
  `template1`, `postgres` and `acme_support`, run `REINDEX DATABASE <db>; ALTER DATABASE <db> REFRESH COLLATION VERSION;`.
  Or recreate the volume with `docker compose down -v && docker compose up -d` (this deletes the data).
- **Port 5433 is in use.** Change the host port in `docker-compose.yml` and in `DATABASE_URL`.
- **Everything is refused after a deploy.** Check the logs for `knowledge_base_unavailable`: a replica is running an
  older knowledge base than the purge job keeps. Restart it.

## Current limits

- The order lookup is a mock (two fixed orders).
- The service is synchronous: no streaming, and each request holds a worker thread for the length of the model call.
- The database schema is created by code on startup. There are no migrations yet.
