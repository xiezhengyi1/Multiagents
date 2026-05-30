from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from experiments.scripts.common import EXPERIMENT_ROOT, LEDGER_ROOT, TASK_ROOT, load_json

TASK_CATALOG_PATH = TASK_ROOT / "task_catalog.json"
LEDGER_PATH = LEDGER_ROOT / "run_ledger.csv"
FAILURE_LEDGER_PATH = LEDGER_ROOT / "failure_cases.csv"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        payload = json.loads(text)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _task_map() -> Dict[str, Dict[str, Any]]:
    payload = load_json(TASK_CATALOG_PATH)
    return {
        str(item.get("task_id") or "").strip(): item
        for item in payload.get("tasks", [])
        if str(item.get("task_id") or "").strip()
    }


def _extract_task_id(item: Dict[str, Any]) -> str:
    direct = str(item.get("task_id") or "").strip()
    if direct:
        return direct
    task_metadata = item.get("task_metadata")
    if isinstance(task_metadata, dict):
        nested = str(task_metadata.get("task_id") or "").strip()
        if nested:
            return nested
    return str(item.get("scenario_id") or "").strip()


def _extract_status_code(item: Dict[str, Any]) -> str:
    for key in ("qos_feedback", "mobility_feedback"):
        feedback = item.get(key)
        if not isinstance(feedback, dict):
            continue
        receipts = feedback.get("dispatch_receipts")
        if not isinstance(receipts, list):
            continue
        for receipt in receipts:
            if isinstance(receipt, dict) and receipt.get("status_code") is not None:
                return str(receipt.get("status_code"))
    return ""


def _extract_failure_type(item: Dict[str, Any]) -> str:
    diagnosis = item.get("diagnosis")
    if isinstance(diagnosis, dict):
        category = str(diagnosis.get("root_cause_category") or "").strip()
        if category:
            return category
    if item.get("status") == "error":
        return str(item.get("error_type") or "runtime_error").strip()
    return ""


def _append_csv_rows(path: Path, fieldnames: Iterable[str], rows: List[Dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate run_workflow_experiment JSONL into experiment ledgers.")
    parser.add_argument("--result-jsonl", type=Path, required=True, help="Path to workflow_experiment_runs.jsonl")
    parser.add_argument("--method", required=True, help="Method id, e.g. Ours")
    parser.add_argument("--experiment", required=True, help="Experiment id, e.g. E1")
    parser.add_argument("--scenario", required=True, help="Scenario id, e.g. S2")
    args = parser.parse_args()

    task_lookup = _task_map()
    results = _read_jsonl(args.result_jsonl)
    if not results:
        raise RuntimeError(f"No run rows found in {args.result_jsonl}")

    ledger_rows: List[Dict[str, Any]] = []
    failure_rows: List[Dict[str, Any]] = []
    for index, item in enumerate(results, start=1):
        task_id = _extract_task_id(item)
        task = task_lookup.get(task_id, {})
        completed = bool(item.get("completed"))
        status = str(item.get("status") or ("success" if completed else "failed")).strip()
        failure_type = _extract_failure_type(item)
        row = {
            "run_id": f"{args.method}-{args.experiment}-{args.scenario}-{index:03d}",
            "experiment_id": args.experiment,
            "method_id": args.method,
            "scenario_id": args.scenario,
            "task_id": task_id,
            "run_index": 1,
            "status": status,
            "completed": completed,
            "isr": "",
            "ga": "",
            "sv": "",
            "esr": 1 if completed else 0,
            "crr": "",
            "ssi": "",
            "round_count": item.get("round_count", ""),
            "retry_count": item.get("retry_count", ""),
            "avg_latency_ms": "",
            "execution_status": (
                (item.get("qos_feedback") or {}).get("execution_status")
                if isinstance(item.get("qos_feedback"), dict)
                else ""
            ),
            "status_code": _extract_status_code(item),
            "failure_type": failure_type,
            "snapshot_before": item.get("snapshot_id", ""),
            "snapshot_after": (
                (item.get("qos_feedback") or {}).get("committed_snapshot_id")
                if isinstance(item.get("qos_feedback"), dict)
                else ""
            ),
            "result_path": str(args.result_jsonl),
            "summary_path": "",
            "notes": str(task.get("user_input") or item.get("user_input") or "").strip(),
        }
        ledger_rows.append(row)

        if not completed:
            diagnosis = item.get("diagnosis") if isinstance(item.get("diagnosis"), dict) else {}
            failure_rows.append(
                {
                    "run_id": row["run_id"],
                    "experiment_id": args.experiment,
                    "method_id": args.method,
                    "scenario_id": args.scenario,
                    "task_id": task_id,
                    "run_index": 1,
                    "failure_type": failure_type,
                    "error_type": str(item.get("error_type") or "").strip(),
                    "status_code": row["status_code"],
                    "root_cause_category": str(diagnosis.get("root_cause_category") or "").strip(),
                    "user_input": str(item.get("user_input") or "").strip(),
                    "diagnosis_summary": str(diagnosis.get("reason_summary") or item.get("error") or "").strip(),
                    "repair_action": "; ".join(diagnosis.get("recommended_actions") or []) if isinstance(diagnosis, dict) else "",
                    "result_path": str(args.result_jsonl),
                }
            )

    _append_csv_rows(
        LEDGER_PATH,
        [
            "run_id",
            "experiment_id",
            "method_id",
            "scenario_id",
            "task_id",
            "run_index",
            "status",
            "completed",
            "isr",
            "ga",
            "sv",
            "esr",
            "crr",
            "ssi",
            "round_count",
            "retry_count",
            "avg_latency_ms",
            "execution_status",
            "status_code",
            "failure_type",
            "snapshot_before",
            "snapshot_after",
            "result_path",
            "summary_path",
            "notes",
        ],
        ledger_rows,
    )
    if failure_rows:
        _append_csv_rows(
            FAILURE_LEDGER_PATH,
            [
                "run_id",
                "experiment_id",
                "method_id",
                "scenario_id",
                "task_id",
                "run_index",
                "failure_type",
                "error_type",
                "status_code",
                "root_cause_category",
                "user_input",
                "diagnosis_summary",
                "repair_action",
                "result_path",
            ],
            failure_rows,
        )
    print(
        json.dumps(
            {
                "ledger_rows": len(ledger_rows),
                "failure_rows": len(failure_rows),
                "result_jsonl": str(args.result_jsonl),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
