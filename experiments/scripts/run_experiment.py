from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import cast

from experiments.contracts import ExperimentCase
from experiments.runner import ExperimentBackend, run_cases
from training.artifacts import read_jsonl, write_jsonl


def load_backend(import_path: str) -> ExperimentBackend:
    module_name, separator, attribute_name = import_path.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("backend must use module:attribute syntax")
    factory = getattr(importlib.import_module(module_name), attribute_name)
    return cast(ExperimentBackend, factory())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = [ExperimentCase.model_validate(row) for row in read_jsonl(args.cases)]
    if not cases:
        raise ValueError("case file must contain at least one experiment case")
    write_jsonl(args.output, run_cases(cases, backend=load_backend(args.backend)))


if __name__ == "__main__":
    main()


__all__ = ["load_backend"]
