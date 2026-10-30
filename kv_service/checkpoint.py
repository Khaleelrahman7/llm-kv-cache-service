"""Portable, checksummed cache checkpoints and run journals that never deserialize tensors."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any, Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX development hosts
    fcntl = None
    import msvcrt

from .cache import CacheEntry, KVCache
from .model import FrozenCausalLM
from .protocol import (CacheConfig, ERROR_CODES, RequestFailure, item_identity, parse_request)
from .tokenizer import Tokenizer


_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_NAMESPACE = re.compile(r"^[A-Za-z0-9_.-]{1,32}$")
_RESPONSE_FIELDS = {"request_id", "namespace", "prompt_token_ids", "generated_token_ids",
                    "generated_text", "finish_reason", "cache"}


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def document_digest(document: object) -> str:
    return hashlib.sha256(_canonical(document)).hexdigest()


def _atomic_write(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    envelope = {"checksum": hashlib.sha256(_canonical(payload)).hexdigest(), "payload": payload}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(envelope, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(destination)


def save_checkpoint(path: str | Path, model_name: str, tokenizer_fingerprint: str, cache: KVCache) -> None:
    _atomic_write(path, {"schema_version": 1, "model": model_name, "tokenizer": tokenizer_fingerprint,
                         "cache": cache.checkpoint_manifest()})


def save_journal(path: str | Path, model_name: str, tokenizer_fingerprint: str, cache: KVCache,
                 input_sha256: str, next_index: int, responses: list[dict[str, Any]],
                 errors: list[dict[str, Any]]) -> None:
    _atomic_write(path, {"schema_version": 2, "model": model_name, "tokenizer": tokenizer_fingerprint,
                         "cache": cache.checkpoint_manifest(),
                         "run": {"input_sha256": input_sha256, "next_index": next_index,
                                 "responses": responses, "errors": errors}})


def _exact_dict(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"checkpoint {label} has invalid fields")
    return value


def _plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _token_tuple(value: Any, vocab_size: int, label: str) -> tuple[int, ...]:
    if (not isinstance(value, list) or not value
            or any(not _plain_int(token) or not 0 <= token < vocab_size for token in value)):
        raise ValueError(f"checkpoint {label} is invalid")
    return tuple(value)


def _read_payload(path: str | Path, model: FrozenCausalLM, tokenizer_fingerprint: str,
                  schema_version: int, fields: set[str]) -> dict[str, Any]:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            envelope = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("checkpoint is unreadable") from exc
    envelope = _exact_dict(envelope, {"checksum", "payload"}, "envelope")
    if not isinstance(envelope["checksum"], str) or len(envelope["checksum"]) != 64:
        raise ValueError("checkpoint checksum is invalid")
    if hashlib.sha256(_canonical(envelope["payload"])).hexdigest() != envelope["checksum"]:
        raise ValueError("checkpoint checksum mismatch")
    payload = _exact_dict(envelope["payload"], fields, "payload")
    if (not _plain_int(payload["schema_version"]) or payload["schema_version"] != schema_version
            or payload["model"] != model.model_name):
        raise ValueError("checkpoint model identity mismatch")
    if payload["tokenizer"] != tokenizer_fingerprint:
        raise ValueError("checkpoint tokenizer identity mismatch")
    return payload


def _rebuild_cache(payload_cache: Any, model: FrozenCausalLM) -> KVCache:
    manifest = _exact_dict(payload_cache, {"config", "clock", "counters", "entries"}, "cache")
    config_data = _exact_dict(manifest["config"], {"capacity", "token_budget", "prefix_stride"}, "config")
    if any(not _plain_int(value) for value in config_data.values()):
        raise ValueError("checkpoint cache configuration is invalid")
    config = CacheConfig(**config_data)
    if not (1 <= config.capacity <= 64 and 16 <= config.token_budget <= 4096 and 1 <= config.prefix_stride <= 16):
        raise ValueError("checkpoint cache configuration is invalid")
    counters = _exact_dict(
        manifest["counters"],
        {"hits", "misses", "evictions", "invalidations", "oversize_skips", "restores"},
        "counters",
    )
    if not isinstance(manifest["entries"], list):
        raise ValueError("checkpoint entries must be an array")
    if len(manifest["entries"]) > config.capacity:
        raise ValueError("checkpoint exceeds cache budgets")
    entries: list[CacheEntry] = []
    pending: list[tuple] = []
    rebuilt: dict[tuple[str, tuple[int, ...]], tuple[Any, Any]] = {}
    fields = {"entry_id", "kind", "namespace", "request_id", "tokens", "prompt_tokens",
              "generated_tokens", "last_access"}
    for raw in manifest["entries"]:
        raw = _exact_dict(raw, fields, "entry")
        namespace, request_id, kind = raw["namespace"], raw["request_id"], raw["kind"]
        if not isinstance(namespace, str) or _NAMESPACE.fullmatch(namespace) is None:
            raise ValueError("checkpoint namespace is invalid")
        tokens = _token_tuple(raw["tokens"], model.vocab_size, "tokens")
        prompt_tokens = _token_tuple(raw["prompt_tokens"], model.vocab_size, "prompt tokens")
        generated_raw = raw["generated_tokens"]
        if (not isinstance(generated_raw, list)
                or any(not _plain_int(token) or not 0 <= token < model.vocab_size for token in generated_raw)):
            raise ValueError("checkpoint generated tokens are invalid")
        generated_tokens = tuple(generated_raw)
        if tokens[0] != 1 or prompt_tokens[0] != 1:
            raise ValueError("checkpoint sequence is missing BOS")
        if any(token < 3 for token in prompt_tokens[1:]) or any(token < 4 for token in generated_tokens):
            raise ValueError("checkpoint sequence contains an impossible token")
        if kind == "prefix":
            expected_id = KVCache.prefix_id(namespace, tokens)
            if request_id is not None or generated_tokens or prompt_tokens != tokens:
                raise ValueError("checkpoint prefix entry is inconsistent")
        elif kind == "session":
            if not isinstance(request_id, str) or _IDENTIFIER.fullmatch(request_id) is None:
                raise ValueError("checkpoint request identity is invalid")
            expected_id = KVCache.session_id(namespace, request_id)
            if tokens != prompt_tokens + generated_tokens:
                raise ValueError("checkpoint session entry is inconsistent")
        else:
            raise ValueError("checkpoint entry kind is invalid")
        if raw["entry_id"] != expected_id:
            raise ValueError("checkpoint entry identity mismatch")
        if not _plain_int(raw["last_access"]) or raw["last_access"] <= 0:
            raise ValueError("checkpoint access counter is invalid")
        if len(tokens) > config.token_budget:
            raise ValueError("checkpoint exceeds cache budgets")
        pending.append((expected_id, kind, namespace, request_id, tokens, prompt_tokens, generated_tokens,
                        raw["last_access"]))
    footprint = [SimpleNamespace(namespace=item[2], tokens=item[4]) for item in pending]
    if KVCache.distinct_tokens(footprint) > config.token_budget:
        raise ValueError("checkpoint exceeds cache budgets")
    for expected_id, kind, namespace, request_id, tokens, prompt_tokens, generated_tokens, last_access in pending:
        state = logits = None
        for count in range(1, len(tokens) + 1):
            key = (namespace, tokens[:count])
            if key not in rebuilt:
                rebuilt[key] = model.advance([tokens[count - 1]], state)
            state, logits = rebuilt[key]
        entries.append(CacheEntry(expected_id, kind, namespace, request_id, tokens, prompt_tokens,
                                  generated_tokens, state, logits, last_access))
    cache = KVCache(config)
    cache.install_restored(entries, clock=manifest["clock"], **counters)
    return cache


def load_checkpoint(path: str | Path, model: FrozenCausalLM, tokenizer_fingerprint: str) -> KVCache:
    payload = _read_payload(path, model, tokenizer_fingerprint, 1,
                            {"schema_version", "model", "tokenizer", "cache"})
    return _rebuild_cache(payload["cache"], model)


def _first_stop(tokenizer: Tokenizer, generated: list[int], stop: tuple[str, ...]) -> int | None:
    text = ""
    for count, token in enumerate(generated, start=1):
        text += tokenizer.decode([token])
        if any(text.endswith(candidate) for candidate in stop):
            return count
    return None


def _validate_run(run: Any, tokenizer: Tokenizer, requests: list[Any], batch_size: int,
                  input_sha256: str) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]]]:
    run = _exact_dict(run, {"input_sha256", "next_index", "responses", "errors"}, "run")
    if run["input_sha256"] != input_sha256:
        raise ValueError("journal belongs to a different input")
    next_index = run["next_index"]
    if (not _plain_int(next_index) or not 1 <= next_index <= len(requests)
            or (next_index != len(requests) and next_index % batch_size != 0)):
        raise ValueError("journal cursor is invalid")
    responses, errors = run["responses"], run["errors"]
    if not isinstance(responses, list) or not isinstance(errors, list) or len(responses) + len(errors) != next_index:
        raise ValueError("journal outcomes do not cover the cursor")
    failed: set[int] = set()
    previous = -1
    for error in errors:
        error = _exact_dict(error, {"index", "request_id", "namespace", "code"}, "error")
        index = error["index"]
        if not _plain_int(index) or not previous < index < next_index:
            raise ValueError("journal error index is invalid")
        previous = index
        request_id, namespace = item_identity(requests[index])
        if error["request_id"] != request_id or error["namespace"] != namespace:
            raise ValueError("journal error identity mismatch")
        try:
            spec = parse_request(requests[index])
        except RequestFailure as exc:
            if error["code"] != exc.code:
                raise ValueError("journal error code mismatch") from None
        else:
            if not spec.resume or error["code"] not in ("resume_not_found", "resume_conflict"):
                raise ValueError("journal error code mismatch")
        if error["code"] not in ERROR_CODES:
            raise ValueError("journal error code mismatch")
        failed.add(index)
    active = [index for index in range(next_index) if index not in failed]
    for index, response in zip(active, responses, strict=True):
        try:
            spec = parse_request(requests[index])
        except RequestFailure:
            raise ValueError("journal response does not match a valid request") from None
        response = _exact_dict(response, _RESPONSE_FIELDS, "response")
        generated = response["generated_token_ids"]
        cache_info = _exact_dict(response["cache"], {"status", "reused_tokens"}, "response cache")
        prompt_tokens = tokenizer.encode(spec.prompt)
        if (not isinstance(generated, list)
                or any(not _plain_int(token) or not 4 <= token < tokenizer.vocab_size for token in generated)):
            raise ValueError("journal response does not match its request")
        stop_point = _first_stop(tokenizer, generated, spec.stop)
        expected_length = spec.max_new_tokens if stop_point is None else stop_point
        expected_reason = "length" if stop_point is None else "stop"
        prompt_ids = response["prompt_token_ids"]
        if (not isinstance(prompt_ids, list) or any(not _plain_int(token) for token in prompt_ids)
                or prompt_ids != prompt_tokens):
            raise ValueError("journal response does not match its request")
        reused, status = cache_info["reused_tokens"], cache_info["status"]
        if not _plain_int(reused):
            raise ValueError("journal response does not match its request")
        if spec.resume:
            reuse_ok = status == "resume_hit" and len(prompt_tokens) < reused <= len(prompt_tokens) + len(generated)
        else:
            reuse_ok = ((status == "miss" and reused == 0)
                        or (status == "prefix_hit" and 1 <= reused <= len(prompt_tokens)))
        if (not reuse_ok or response["request_id"] != spec.request_id or response["namespace"] != spec.namespace
                or response["finish_reason"] != expected_reason or len(generated) != expected_length
                or response["generated_text"] != tokenizer.decode(generated)):
            raise ValueError("journal response does not match its request")
    return next_index, responses, errors


def load_journal(path: str | Path, model: FrozenCausalLM, tokenizer: Tokenizer, requests: list[Any],
                 batch_size: int, input_sha256: str) -> tuple[KVCache, int, list[dict[str, Any]], list[dict[str, Any]]]:
    payload = _read_payload(path, model, tokenizer.fingerprint, 2,
                            {"schema_version", "model", "tokenizer", "cache", "run"})
    next_index, responses, errors = _validate_run(payload["run"], tokenizer, requests, batch_size, input_sha256)
    return _rebuild_cache(payload["cache"], model), next_index, responses, errors


@contextmanager
def store_lock(store_path: str | Path) -> Iterator[None]:
    """Exclusive, crash-safe ownership of a shared store.

    The lock lives on a sibling file that is never replaced, so the atomic rename of the store
    cannot split waiters across inodes, and the operating system releases it when the owner dies.
    """
    lock_path = Path(store_path).with_name(Path(store_path).name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "a+b")
    try:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        else:  # pragma: no cover - non-POSIX development hosts
            while True:
                try:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    continue
        yield
    finally:
        handle.close()
