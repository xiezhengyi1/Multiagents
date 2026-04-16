from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4


def ensure_directory(path: Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_json_file(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_file_atomic(path: Path, payload: Any) -> Path:
    destination = Path(path)
    ensure_directory(destination.parent)
    temp_path = destination.with_suffix(f"{destination.suffix}.tmp-{uuid4().hex}")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp_path, destination)
    return destination


__all__ = ["ensure_directory", "read_json_file", "write_json_file_atomic"]
