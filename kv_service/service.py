"""Generation orchestration and the public JSON contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .cache import CacheEntry, KVCache
from .checkpoint import document_digest, load_checkpoint, load_journal, save_checkpoint, save_journal
from .model import FrozenCausalLM, ModelState
from .protocol import (CacheConfig, RequestFailure, RequestSpec, item_identity, parse_document,
                       parse_request)
from .tokenizer import Tokenizer


@dataclass
class _Prepared:
    spec: RequestSpec
    prompt_tokens: tuple[int, ...]
    generated: tuple[int, ...]
    state: ModelState
    logits: torch.Tensor
    response: dict[str, Any]
    observed: CacheEntry | None
    prefixes: list[tuple[tuple[int, ...], ModelState, torch.Tensor]]


@dataclass
class RunState:
    """Progress of one document: every logical batch commits before the cursor advances."""

    cache_enabled: bool
    batch_size: int
    requests: list[Any]
    responses: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    next_index: int = 0

    @property
    def done(self) -> bool:
        return self.next_index >= len(self.requests)


class _SharedEvaluator:
    """Evaluates each (namespace, token sequence) state at most once within one logical batch."""

    def __init__(self, model: FrozenCausalLM) -> None:
        self.model = model
        self.nodes: dict[tuple[str, tuple[int, ...]], tuple[ModelState, torch.Tensor]] = {}

    def step(self, namespace: str, sequence: tuple[int, ...], state: ModelState | None
             ) -> tuple[ModelState, torch.Tensor]:
        key = (namespace, sequence)
        found = self.nodes.get(key)
        if found is None:
            found = self.model.advance([sequence[-1]], state)
            self.nodes[key] = found
        return found


class GenerationService:
    def __init__(self, cache_capacity: int = 16, cache_token_budget: int = 1024,
                 prefix_stride: int = 4) -> None:
        self.tokenizer = Tokenizer()
        self.model = FrozenCausalLM(self.tokenizer.vocab_size)
        config, _, _ = parse_document(
            {"requests": [], "cache_capacity": cache_capacity, "cache_token_budget": cache_token_budget,
             "prefix_stride": prefix_stride},
            CacheConfig(),
        )
        self.cache = KVCache(config)

    def save_checkpoint(self, path: str | Path) -> None:
        save_checkpoint(path, self.model.model_name, self.tokenizer.fingerprint, self.cache)

    def load_checkpoint(self, path: str | Path) -> None:
        restored = load_checkpoint(path, self.model, self.tokenizer.fingerprint)
        self.cache = restored

    @staticmethod
    def _slice_state(state: ModelState, length: int) -> ModelState:
        """Keys and values of a state hold every prefix of its sequence; keep the first ``length``."""
        return ModelState(tuple(keys[:, :length, :] for keys in state.keys),
                          tuple(values[:, :length, :] for values in state.values))

    def _advance_prompt(
        self, namespace: str, prompt_tokens: tuple[int, ...], prefix: CacheEntry | None, reused: int,
        evaluator: _SharedEvaluator,
    ) -> tuple[ModelState, torch.Tensor, list[tuple[tuple[int, ...], ModelState, torch.Tensor]]]:
        checkpoints: list[tuple[tuple[int, ...], ModelState, torch.Tensor]] = []
        state: ModelState | None = None
        logits: torch.Tensor | None = None
        offset = 0
        if prefix is not None:
            offset = reused
            if reused == len(prefix.tokens):
                state, logits = prefix.state, prefix.logits
            else:
                state = self._slice_state(prefix.state, reused)
        for position in range(offset, len(prompt_tokens)):
            count = position + 1
            state, logits = evaluator.step(namespace, prompt_tokens[:count], state)
            if count % self.cache.config.prefix_stride == 0 or count == len(prompt_tokens):
                checkpoints.append((prompt_tokens[:count], state, logits))
        if state is None or logits is None:
            raise ValueError("tokenizer produced an empty prompt")
        return state, logits, checkpoints

    def _stops_at(self, generated: list[int] | tuple[int, ...], stop: tuple[str, ...]) -> int | None:
        """Smallest p >= 1 whose decoded continuation generated[:p] ends with a stop string."""
        if not stop:
            return None
        text = ""
        for count, token in enumerate(generated, start=1):
            text += self.tokenizer.decode([token])
            if any(text.endswith(candidate) for candidate in stop):
                return count
        return None

    def _finish(self, namespace: str, sequence: tuple[int, ...], state: ModelState, logits: torch.Tensor,
                generated: list[int], total: int, stop: tuple[str, ...], evaluator: _SharedEvaluator):
        text = self.tokenizer.decode(generated)
        while len(generated) < total:
            token = self.model.select_next(logits)
            generated.append(token)
            sequence = sequence + (token,)
            state, logits = evaluator.step(namespace, sequence, state)
            text += self.tokenizer.decode([token])
            if any(text.endswith(candidate) for candidate in stop):
                return tuple(generated), state, logits, "stop"
        return tuple(generated), state, logits, "length"

    def _prepare(self, spec: RequestSpec, view: tuple[CacheEntry, ...], cache_enabled: bool,
                 evaluator: _SharedEvaluator) -> _Prepared:
        prompt_tokens = tuple(self.tokenizer.encode(spec.prompt))
        observed: CacheEntry | None = None
        stopped_at: int | None = None
        prefixes: list[tuple[tuple[int, ...], ModelState, torch.Tensor]] = []
        if spec.resume:
            if not cache_enabled:
                raise RequestFailure("resume_not_found")
            observed = self.cache.peek_session(spec.namespace, spec.request_id, view)
            if observed is None:
                raise RequestFailure("resume_not_found", observed)
            if observed.prompt_tokens != prompt_tokens or spec.max_new_tokens < len(observed.generated_tokens):
                raise RequestFailure("resume_conflict", observed)
            stopped_at = self._stops_at(observed.generated_tokens, spec.stop)
            if stopped_at is not None and stopped_at < len(observed.generated_tokens):
                raise RequestFailure("resume_conflict", observed)
            state, logits = observed.state, observed.logits
            generated = list(observed.generated_tokens)
            status, reused = "resume_hit", len(observed.tokens)
        else:
            observed, reused = (self.cache.peek_reusable(spec.namespace, prompt_tokens, view)
                                if cache_enabled else (None, 0))
            state, logits, prefixes = self._advance_prompt(spec.namespace, prompt_tokens, observed, reused,
                                                           evaluator)
            generated = []
            status = "prefix_hit" if observed is not None else ("miss" if cache_enabled else "disabled")
        sequence = prompt_tokens + tuple(generated)
        if spec.resume and stopped_at is not None:
            generated_tokens, finish_reason = tuple(generated), "stop"
        else:
            generated_tokens, state, logits, finish_reason = self._finish(
                spec.namespace, sequence, state, logits, generated, spec.max_new_tokens, spec.stop, evaluator)
        response = {
            "request_id": spec.request_id,
            "namespace": spec.namespace,
            "prompt_token_ids": list(prompt_tokens),
            "generated_token_ids": list(generated_tokens),
            "generated_text": self.tokenizer.decode(list(generated_tokens)),
            "finish_reason": finish_reason,
            "cache": {"status": status, "reused_tokens": reused},
        }
        return _Prepared(spec, prompt_tokens, generated_tokens, state, logits, response, observed, prefixes)

    @staticmethod
    def _error(index: int, item: Any, code: str) -> dict[str, Any]:
        request_id, namespace = item_identity(item)
        return {"index": index, "request_id": request_id, "namespace": namespace, "code": code}

    def _is_pristine(self) -> bool:
        return (not self.cache.freeze() and self.cache.hits == self.cache.misses == self.cache.evictions
                == self.cache.invalidations == self.cache.oversize_skips == self.cache.restores == 0)

    def begin_run(self, document: Any, cache_enabled: bool = True) -> RunState:
        config, batch_size, requests = parse_document(document, self.cache.config)
        if config != self.cache.config:
            if not self._is_pristine():
                raise ValueError("cache configuration cannot change while cache state exists")
            self.cache = KVCache(config)
        return RunState(cache_enabled, batch_size, requests)

    def resume_run(self, document: Any, journal_path: str | Path) -> RunState:
        config, batch_size, requests = parse_document(document, CacheConfig())
        cache, next_index, responses, errors = load_journal(
            journal_path, self.model, self.tokenizer, requests, batch_size, document_digest(document))
        if cache.config != config:
            raise ValueError("journal cache configuration does not match the input")
        self.cache = cache
        return RunState(True, batch_size, requests, responses, errors, next_index)

    def save_journal(self, path: str | Path, document: Any, run: RunState) -> None:
        save_journal(path, self.model.model_name, self.tokenizer.fingerprint, self.cache,
                     document_digest(document), run.next_index, run.responses, run.errors)

    def run_next_batch(self, run: RunState) -> None:
        start = run.next_index
        stop = min(start + run.batch_size, len(run.requests))
        view = self.cache.freeze() if run.cache_enabled else ()
        evaluator = _SharedEvaluator(self.model)
        operations: list[tuple[_Prepared | None, CacheEntry | None]] = []
        for index in range(start, stop):
            item = run.requests[index]
            try:
                spec = parse_request(item)
            except RequestFailure as exc:
                run.errors.append(self._error(index, item, exc.code))
                continue
            try:
                prepared = self._prepare(spec, view, run.cache_enabled, evaluator)
                run.responses.append(prepared.response)
                operations.append((prepared, prepared.observed))
            except RequestFailure as exc:
                run.errors.append(self._error(index, item, exc.code))
                operations.append((None, exc.observed))
        if run.cache_enabled:
            for prepared, observed in operations:
                self.cache.record_lookup(observed)
                if prepared is None:
                    continue
                spec = prepared.spec
                if not spec.resume:
                    self.cache.invalidate_session(spec.namespace, spec.request_id)
                    for tokens, state, logits in prepared.prefixes:
                        self.cache.put_prefix(spec.namespace, tokens, state, logits)
                self.cache.put_session(spec.namespace, spec.request_id, prepared.prompt_tokens,
                                       prepared.generated, prepared.state, prepared.logits)
        run.next_index = stop

    def report(self, run: RunState) -> dict[str, Any]:
        if run.cache_enabled:
            summary = {"capacity": self.cache.config.capacity,
                       "token_budget": self.cache.config.token_budget,
                       "prefix_stride": self.cache.config.prefix_stride,
                       "resident_tokens": self.cache.resident_tokens,
                       "hits": self.cache.hits, "misses": self.cache.misses,
                       "evictions": self.cache.evictions, "invalidations": self.cache.invalidations,
                       "oversize_skips": self.cache.oversize_skips, "restores": self.cache.restores,
                       "entries": self.cache.snapshot()}
        else:
            summary = {"capacity": 0, "token_budget": 0, "prefix_stride": 0, "resident_tokens": 0,
                       "hits": 0, "misses": 0, "evictions": 0, "invalidations": 0,
                       "oversize_skips": 0, "restores": 0, "entries": []}
        return {"schema_version": 2, "model": self.model.model_name, "cache_enabled": run.cache_enabled,
                "responses": run.responses, "errors": run.errors, "cache_summary": summary}

    def run_document(self, document: Any, cache_enabled: bool = True) -> dict[str, Any]:
        run = self.begin_run(document, cache_enabled)
        while not run.done:
            self.run_next_batch(run)
        return self.report(run)
