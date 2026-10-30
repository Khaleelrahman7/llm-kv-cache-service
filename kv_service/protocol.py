"""Validation and normalized request types for the serving protocol."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_NAMESPACE = re.compile(r"^[A-Za-z0-9_.-]{1,32}$")
_REQUEST_FIELDS = {"request_id", "namespace", "prompt", "max_new_tokens", "resume", "stop"}
_DOCUMENT_FIELDS = {
    "requests",
    "cache_capacity",
    "cache_token_budget",
    "prefix_stride",
    "batch_size",
}


ERROR_CODES = ("invalid_request", "invalid_request_id", "invalid_namespace", "invalid_prompt",
               "invalid_max_new_tokens", "invalid_resume", "invalid_stop", "unknown_fields", "resume_not_found",
               "resume_conflict")


def item_identity(item: Any) -> tuple[str | None, str | None]:
    """Identities echoed in an error entry: supplied strings, the default namespace, else null."""
    request_id = item.get("request_id") if isinstance(item, dict) else None
    namespace = item.get("namespace", "default") if isinstance(item, dict) else None
    return (request_id if isinstance(request_id, str) else None,
            namespace if isinstance(namespace, str) else None)


class RequestFailure(Exception):
    def __init__(self, code: str, observed: object | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.observed = observed


@dataclass(frozen=True)
class CacheConfig:
    capacity: int = 16
    token_budget: int = 1024
    prefix_stride: int = 4


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    namespace: str
    prompt: str
    max_new_tokens: int
    resume: bool
    stop: tuple[str, ...] = ()


def _bounded_int(value: Any, lower: int, upper: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer from {lower} through {upper}")
    return value


def parse_document(document: Any, defaults: CacheConfig) -> tuple[CacheConfig, int, list[Any]]:
    if not isinstance(document, dict) or set(document) - _DOCUMENT_FIELDS:
        raise ValueError("input contains unknown fields or is not an object")
    requests = document.get("requests")
    if not isinstance(requests, list):
        raise ValueError("requests must be an array")
    config = CacheConfig(
        capacity=_bounded_int(document.get("cache_capacity", defaults.capacity), 1, 64, "cache_capacity"),
        token_budget=_bounded_int(
            document.get("cache_token_budget", defaults.token_budget), 16, 4096, "cache_token_budget"
        ),
        prefix_stride=_bounded_int(document.get("prefix_stride", defaults.prefix_stride), 1, 16, "prefix_stride"),
    )
    batch_size = _bounded_int(document.get("batch_size", 4), 1, 8, "batch_size")
    return config, batch_size, requests


def parse_request(item: Any) -> RequestSpec:
    if not isinstance(item, dict):
        raise RequestFailure("invalid_request")
    if set(item) - _REQUEST_FIELDS:
        raise RequestFailure("unknown_fields")
    request_id = item.get("request_id")
    namespace = item.get("namespace", "default")
    prompt = item.get("prompt")
    max_new_tokens = item.get("max_new_tokens")
    resume = item.get("resume", False)
    if not isinstance(request_id, str) or _IDENTIFIER.fullmatch(request_id) is None:
        raise RequestFailure("invalid_request_id")
    if not isinstance(namespace, str) or _NAMESPACE.fullmatch(namespace) is None:
        raise RequestFailure("invalid_namespace")
    if not isinstance(prompt, str) or not prompt or len(prompt) > 256:
        raise RequestFailure("invalid_prompt")
    if (not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool)
            or not 1 <= max_new_tokens <= 32):
        raise RequestFailure("invalid_max_new_tokens")
    if not isinstance(resume, bool):
        raise RequestFailure("invalid_resume")
    stop = item.get("stop", [])
    if (not isinstance(stop, list) or len(stop) > 4
            or any(not isinstance(text, str) or not 1 <= len(text) <= 16 for text in stop)):
        raise RequestFailure("invalid_stop")
    return RequestSpec(request_id, namespace, prompt, max_new_tokens, resume, tuple(stop))
