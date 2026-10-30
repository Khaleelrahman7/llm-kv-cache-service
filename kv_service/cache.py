"""Namespace-aware, dual-budget LRU for immutable attention states."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
from typing import Iterable

import torch

from .model import ModelState
from .protocol import CacheConfig


@dataclass(frozen=True)
class CacheEntry:
    entry_id: str
    kind: str
    namespace: str
    request_id: str | None
    tokens: tuple[int, ...]
    prompt_tokens: tuple[int, ...]
    generated_tokens: tuple[int, ...]
    state: ModelState
    logits: torch.Tensor
    last_access: int


class KVCache:
    def __init__(self, config: CacheConfig) -> None:
        self.config = config
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.invalidations = 0
        self.oversize_skips = 0
        self.restores = 0
        self._clock = 0
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()

    @staticmethod
    def _digest(parts: object) -> str:
        encoded = json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:20]

    @classmethod
    def prefix_id(cls, namespace: str, tokens: tuple[int, ...]) -> str:
        return "p:" + cls._digest([namespace, list(tokens)])

    @classmethod
    def session_id(cls, namespace: str, request_id: str) -> str:
        return "s:" + cls._digest([namespace, request_id])

    @staticmethod
    def distinct_tokens(entries: Iterable[CacheEntry]) -> int:
        """Resident memory: entries of one namespace share the keys and values of common prefixes."""
        nodes: set[tuple[str, tuple[int, ...]]] = set()
        for entry in entries:
            for length in range(len(entry.tokens), 0, -1):
                node = (entry.namespace, entry.tokens[:length])
                if node in nodes:
                    break
                nodes.add(node)
        return len(nodes)

    @property
    def resident_tokens(self) -> int:
        return self.distinct_tokens(self._entries.values())

    def freeze(self) -> tuple[CacheEntry, ...]:
        return tuple(self._entries.values())

    @staticmethod
    def reusable_length(entry: CacheEntry, tokens: tuple[int, ...]) -> int:
        """Longest prefix of ``tokens`` whose state (and, at the prompt end, logits) the entry can supply."""
        common = 0
        for mine, theirs in zip(entry.tokens, tokens):
            if mine != theirs:
                break
            common += 1
        if common == len(tokens) and len(entry.tokens) > len(tokens):
            common -= 1
        return common

    @classmethod
    def peek_reusable(cls, namespace: str, tokens: tuple[int, ...],
                      view: Iterable[CacheEntry]) -> tuple[CacheEntry | None, int]:
        best: CacheEntry | None = None
        best_key = (0, 0)
        for entry in view:
            if entry.namespace != namespace:
                continue
            key = (cls.reusable_length(entry, tokens), entry.last_access)
            if key[0] > 0 and key > best_key:
                best, best_key = entry, key
        return best, best_key[0]

    @classmethod
    def peek_session(cls, namespace: str, request_id: str, view: Iterable[CacheEntry]) -> CacheEntry | None:
        entry_id = cls.session_id(namespace, request_id)
        return next((entry for entry in view if entry.entry_id == entry_id and entry.kind == "session"), None)

    def _stamp(self, entry: CacheEntry) -> CacheEntry:
        self._clock += 1
        return CacheEntry(entry.entry_id, entry.kind, entry.namespace, entry.request_id, entry.tokens,
                          entry.prompt_tokens, entry.generated_tokens, entry.state, entry.logits, self._clock)

    def _touch_if_unchanged(self, observed: CacheEntry) -> None:
        current = self._entries.get(observed.entry_id)
        if current is None or current.tokens != observed.tokens or current.last_access != observed.last_access:
            return
        self._entries.pop(observed.entry_id)
        self._entries[observed.entry_id] = self._stamp(current)

    def record_lookup(self, observed: CacheEntry | None) -> None:
        if observed is None:
            self.misses += 1
            return
        self.hits += 1
        self._touch_if_unchanged(observed)

    def _store(self, entry: CacheEntry) -> CacheEntry | None:
        self._entries.pop(entry.entry_id, None)
        if len(entry.tokens) > self.config.token_budget:
            self.oversize_skips += 1
            return None
        stamped = self._stamp(entry)
        self._entries[entry.entry_id] = stamped
        while True:
            if len(self._entries) > self.config.capacity:
                self._entries.popitem(last=False)
            elif self.resident_tokens > self.config.token_budget:
                del self._entries[self._least_recent_freeing_entry()]
            else:
                break
            self.evictions += 1
        return self._entries.get(entry.entry_id)

    def _least_recent_freeing_entry(self) -> str:
        """Least recent entry whose removal frees memory: no other entry of its namespace extends it."""
        resident = list(self._entries.values())
        for candidate in resident:
            covered = any(other is not candidate and other.namespace == candidate.namespace
                          and other.tokens[:len(candidate.tokens)] == candidate.tokens for other in resident)
            if not covered:
                return candidate.entry_id
        raise AssertionError("an over-budget cache always holds a freeing entry")

    def invalidate_session(self, namespace: str, request_id: str) -> bool:
        removed = self._entries.pop(self.session_id(namespace, request_id), None)
        if removed is None:
            return False
        self.invalidations += 1
        return True

    def put_prefix(self, namespace: str, tokens: tuple[int, ...], state: ModelState,
                   logits: torch.Tensor) -> CacheEntry | None:
        return self._store(CacheEntry(self.prefix_id(namespace, tokens), "prefix", namespace, None,
                                      tokens, tokens, (), state, logits.clone(), 0))

    def put_session(self, namespace: str, request_id: str, prompt_tokens: tuple[int, ...],
                    generated_tokens: tuple[int, ...], state: ModelState,
                    logits: torch.Tensor) -> CacheEntry | None:
        return self._store(CacheEntry(self.session_id(namespace, request_id), "session", namespace, request_id,
                                      prompt_tokens + generated_tokens, prompt_tokens, generated_tokens,
                                      state, logits.clone(), 0))

    def install_restored(self, entries: list[CacheEntry], *, clock: int, hits: int, misses: int,
                         evictions: int, invalidations: int, oversize_skips: int, restores: int) -> None:
        if len({entry.entry_id for entry in entries}) != len(entries):
            raise ValueError("checkpoint has duplicate cache entries")
        if any(entry.last_access <= 0 for entry in entries):
            raise ValueError("checkpoint has invalid access counters")
        if len({entry.last_access for entry in entries}) != len(entries):
            raise ValueError("checkpoint has duplicate access counters")
        if entries != sorted(entries, key=lambda entry: entry.last_access):
            raise ValueError("checkpoint entries are not in LRU order")
        if len(entries) > self.config.capacity or self.distinct_tokens(entries) > self.config.token_budget:
            raise ValueError("checkpoint exceeds cache budgets")
        if not isinstance(clock, int) or isinstance(clock, bool) or clock < max((e.last_access for e in entries), default=0):
            raise ValueError("checkpoint clock is invalid")
        counters = (hits, misses, evictions, invalidations, oversize_skips, restores)
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counters):
            raise ValueError("checkpoint contains an invalid counter")
        self._entries = OrderedDict((entry.entry_id, entry) for entry in entries)
        self._clock = clock
        self.hits, self.misses, self.evictions = hits, misses, evictions
        self.invalidations, self.oversize_skips = invalidations, oversize_skips
        self.restores = restores + 1

    def checkpoint_manifest(self) -> dict[str, object]:
        return {
            "config": {"capacity": self.config.capacity, "token_budget": self.config.token_budget,
                       "prefix_stride": self.config.prefix_stride},
            "clock": self._clock,
            "counters": {"hits": self.hits, "misses": self.misses, "evictions": self.evictions,
                         "invalidations": self.invalidations, "oversize_skips": self.oversize_skips,
                         "restores": self.restores},
            "entries": [{"entry_id": entry.entry_id, "kind": entry.kind, "namespace": entry.namespace,
                         "request_id": entry.request_id, "tokens": list(entry.tokens),
                         "prompt_tokens": list(entry.prompt_tokens),
                         "generated_tokens": list(entry.generated_tokens), "last_access": entry.last_access}
                        for entry in self._entries.values()],
        }

    def snapshot(self) -> list[dict[str, object]]:
        return [{"entry_id": entry.entry_id, "kind": entry.kind, "namespace": entry.namespace,
                 "request_id": entry.request_id, "token_count": len(entry.tokens),
                 "prompt_token_count": len(entry.prompt_tokens), "last_access": entry.last_access}
                for entry in self._entries.values()]
