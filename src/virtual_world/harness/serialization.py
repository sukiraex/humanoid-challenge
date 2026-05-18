"""JSON serialization helpers for harness types."""

from __future__ import annotations

import json
from enum import Enum
from typing import Any


class HarnessJSONEncoder(json.JSONEncoder):
    """Encode enums and nested structures produced by the harness."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, Enum):
            return obj.name
        return super().default(obj)


def to_json(data: Any, *, indent: int | None = 2) -> str:
    """Serialize a JSON-compatible structure to a string."""
    return json.dumps(data, cls=HarnessJSONEncoder, indent=indent)


def from_json(text: str) -> Any:
    """Parse JSON text."""
    return json.loads(text)
