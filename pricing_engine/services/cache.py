"""Simple in-memory cache for pricing results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Hashable, Optional


@dataclass
class CacheEntry:
    value: object
    metadata: Dict[str, str] = field(default_factory=dict)


class SimpleCache:
    def __init__(self) -> None:
        self._store: Dict[Hashable, CacheEntry] = {}

    def get(self, key: Hashable) -> Optional[CacheEntry]:
        return self._store.get(key)

    def set(self, key: Hashable, entry: CacheEntry) -> None:
        self._store[key] = entry
