# LLM KV-Cache Service

A deterministic PyTorch serving simulator focused on the difficult parts of stateful LLM inference: exact prefix reuse, request isolation, cache eviction, interrupted generation, and crash-safe recovery.

The model is intentionally small and frozen, so cache behavior can be tested without downloading model weights or relying on nondeterministic outputs. Cached and uncached execution produce the same generated tokens.

## What it demonstrates

- Namespace-safe request and prefix identity
- Exact longest-prefix reuse with correct sequence positions
- Immutable cache snapshots across logical batches
- Entry-count and resident-token LRU budgets
- Interrupted generation and exact session resume
- Stop-sequence handling across stored and newly generated tokens
- Checksummed checkpoints with tensor reconstruction from tokens
- Resumable journals and a lock-protected shared cache store
- Strict JSON validation, deterministic reports, and atomic file replacement

## Architecture

`kv_service/model.py` contains a frozen two-layer causal attention model. `cache.py` owns the namespace-aware LRU and immutable attention states. `service.py` coordinates batching, shared prefix evaluation, generation, and resume behavior. `checkpoint.py` validates portable persisted state before rebuilding tensors. `cli.py` exposes normal, checkpointed, journaled, and shared-store execution.

## Quick start

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e .

kv-cache-service \
  --input examples/requests.json \
  --output generation.json
```

Run without cache reuse:

```bash
kv-cache-service --input examples/requests.json --output generation.json --no-cache
```

Persist cache state between invocations:

```bash
kv-cache-service --input examples/requests.json --output generation.json \
  --checkpoint-out cache.checkpoint.json
```

For crash recovery, pass `--journal run.journal.json`. For a cache shared by concurrent worker processes, pass `--store cache.checkpoint.json`; access is protected by an OS file lock.

## Input format

```json
{
  "cache_capacity": 16,
  "cache_token_budget": 256,
  "prefix_stride": 4,
  "batch_size": 1,
  "requests": [
    {
      "request_id": "first",
      "namespace": "demo",
      "prompt": "shared prompt",
      "max_new_tokens": 6
    },
    {
      "request_id": "extended",
      "namespace": "demo",
      "prompt": "shared prompt extended",
      "max_new_tokens": 8,
      "stop": ["cc"]
    }
  ]
}
```

The generated report includes token IDs, decoded text, finish reason, reuse status, cache counters, and the final LRU snapshot.

## Tests

```bash
python -m unittest discover -s tests -v
```

The public regression suite checks cached/uncached equivalence, exact prefix reuse, namespace isolation, padded positions, interrupted resume, and checkpoint recovery.

## Development approach

This project was developed through an AI-assisted engineering workflow using Claude Code, with human direction over the architecture, debugging strategy, edge-case design, and verification. The work emphasized iterative diagnosis and executable regression tests rather than one-shot code generation.

