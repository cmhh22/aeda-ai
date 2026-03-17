from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_JSON_LOG_LOCK = threading.Lock()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_json_log(path: str | Path, event: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with _JSON_LOG_LOCK:
        if destination.exists():
            try:
                payload = json.loads(destination.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = []
        else:
            payload = []

        if not isinstance(payload, list):
            payload = []

        payload.append(event)
        destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
