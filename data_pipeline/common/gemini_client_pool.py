from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

RATE_LIMIT_TOKENS = (
    "429",
    "RESOURCE_EXHAUSTED",
    "RATE_LIMIT",
    "TOO_MANY_REQUESTS",
    "TOO MANY REQUESTS",
)


def _split_keys(value: str) -> List[str]:
    tokens = re.split(r"[,\s]+", value.strip())
    return [token.strip() for token in tokens if token.strip()]


def _dedupe_preserve(keys: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for key in keys:
        if key and key not in seen:
            seen.add(key)
            result.append(key)
    return result


def gather_api_keys(
    explicit_keys: Optional[Sequence[str]] = None,
    key_files: Optional[Sequence[Path]] = None,
) -> List[str]:
    """Return a deduplicated list of API keys from CLI, files, or env vars."""
    collected: List[str] = []

    for value in explicit_keys or []:
        collected.extend(_split_keys(str(value)))

    if key_files is None:
        file_iterable: Iterable[Path] = ()
    elif isinstance(key_files, (str, os.PathLike)):  # allow single CLI Path defaults
        file_iterable = (Path(key_files),)
    else:
        file_iterable = key_files

    for file_path in file_iterable:
        path = Path(file_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Gemini API key file not found: {path}")
        collected.extend(_split_keys(path.read_text()))

    if not collected:
        single_env = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if single_env:
            collected.extend(_split_keys(single_env))

    if not collected:
        multi_env = os.environ.get("GEMINI_API_KEYS") or os.environ.get("GOOGLE_API_KEYS")
        if multi_env:
            collected.extend(_split_keys(multi_env))

    return _dedupe_preserve(collected)


def is_rate_limit_error(exc: Exception) -> bool:
    code = getattr(exc, "code", None)
    if isinstance(code, int) and code == 429:
        return True
    status = str(getattr(exc, "status", "")).upper()
    message = str(getattr(exc, "message", "")).upper()
    details = getattr(exc, "details", None)
    detail_text = ""
    if isinstance(details, (list, tuple)):
        detail_text = " ".join(str(item) for item in details).upper()
    combined = " ".join([status, message, detail_text, str(exc).upper()])
    return any(token in combined for token in RATE_LIMIT_TOKENS if token)


class GeminiClientPool:
    """Round-robin Gemini client that rotates across multiple API keys."""

    def __init__(self, api_keys: Sequence[str], client_factory: Callable[[str], object]):
        if not api_keys:
            raise ValueError("GeminiClientPool requires at least one API key.")
        self._clients = [client_factory(key) for key in api_keys]
        self._lock = threading.Lock()
        self._cursor = 0
        self.models = _ModelsProxy(self)

    def __getattr__(self, name: str):
        return getattr(self._clients[0], name)

    def _acquire_order(self) -> List[int]:
        with self._lock:
            start = self._cursor
            self._cursor = (self._cursor + 1) % len(self._clients)
        total = len(self._clients)
        return [(start + offset) % total for offset in range(total)]


class _ModelsProxy:
    def __init__(self, pool: GeminiClientPool):
        self._pool = pool

    def generate_content(self, *args, **kwargs):
        last_exc: Optional[Exception] = None
        for idx in self._pool._acquire_order():
            client = self._pool._clients[idx]
            try:
                return client.models.generate_content(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                if not is_rate_limit_error(exc):
                    raise
                last_exc = exc
                continue
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("GeminiClientPool has no available clients.")
