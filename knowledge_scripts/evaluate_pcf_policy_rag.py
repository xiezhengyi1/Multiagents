from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from agents.tools import knowledge_tool
from database.langchain_pg import (
    PCF_AM_POLICY_CLAUSES_COLLECTION,
    PCF_AM_POLICY_SCHEMA_COLLECTION,
    PCF_POLICY_GLOSSARY_COLLECTION,
    PCF_SM_POLICY_CLAUSES_COLLECTION,
    PCF_SM_POLICY_SCHEMA_COLLECTION,
    PCF_URSP_CLAUSES_COLLECTION,
    PCF_URSP_SCHEMA_COLLECTION,
)
from knowledge_scripts.build_pcf_policy_kb import (
    CLAUSE_EXACT_INDEX_JSON,
    CLAUSE_JSONL,
    GLOSSARY_EXACT_INDEX_JSON,
    GLOSSARY_JSONL,
    RETRIEVAL_EVAL_JSON,
    SCHEMA_EXACT_INDEX_JSON,
    SCHEMA_JSONL,
    SPEC_OBJECT_MAP_JSON,
    TERM_ALIAS_MAP_JSON,
)
from knowledge_scripts.build_pcf_policy_kb_docling import (
    DOCLING_CLAUSE_EXACT_INDEX_JSON,
    DOCLING_CLAUSE_JSONL,
    DOCLING_GLOSSARY_EXACT_INDEX_JSON,
    DOCLING_GLOSSARY_JSONL,
    DOCLING_RETRIEVAL_EVAL_JSON,
    DOCLING_SCHEMA_EXACT_INDEX_JSON,
    DOCLING_SCHEMA_JSONL,
    DOCLING_SPEC_OBJECT_MAP_JSON,
    DOCLING_TERM_ALIAS_MAP_JSON,
    PCF_AM_POLICY_CLAUSES_DOCLING_COLLECTION,
    PCF_AM_POLICY_SCHEMA_DOCLING_COLLECTION,
    PCF_POLICY_GLOSSARY_DOCLING_COLLECTION,
    PCF_SM_POLICY_CLAUSES_DOCLING_COLLECTION,
    PCF_SM_POLICY_SCHEMA_DOCLING_COLLECTION,
    PCF_URSP_CLAUSES_DOCLING_COLLECTION,
    PCF_URSP_SCHEMA_DOCLING_COLLECTION,
)


DEFAULT_CASES_PATH = Path(current_dir) / "data" / "pcf_policy_rag_agent_expanded_cases.json"
DEFAULT_IEA_GOLD_CASES_PATH = Path(current_dir) / "data" / "pcf_policy_rag_iea_gold_cases.json"
DEFAULT_EXTENDED_CASES_PATH = Path(current_dir) / "data" / "pcf_policy_rag_extended_cases.json"
DEFAULT_STRICT_CASES_PATH = Path(current_dir) / "data" / "pcf_policy_rag_strict_cases.json"
DEFAULT_AGENT_EXPANDED_CASES_PATH = Path(current_dir) / "data" / "pcf_policy_rag_agent_expanded_cases.json"
DEFAULT_IEA_TRACE_PATHS = (
    Path(parent_dir) / "sft_data" / "main_control" / "raw_traces" / "main_control.jsonl",
    Path(parent_dir) / "sft_data" / "intent_encoding" / "raw_traces" / "intent_encoding.jsonl",
    Path(parent_dir) / "sft_data" / "optimization_strategy" / "raw_traces" / "optimization_strategy.jsonl",
)

PROFILE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "base": {
        "clause_jsonl": CLAUSE_JSONL,
        "schema_jsonl": SCHEMA_JSONL,
        "glossary_jsonl": GLOSSARY_JSONL,
        "clause_exact_index": CLAUSE_EXACT_INDEX_JSON,
        "schema_exact_index": SCHEMA_EXACT_INDEX_JSON,
        "glossary_exact_index": GLOSSARY_EXACT_INDEX_JSON,
        "spec_object_map": SPEC_OBJECT_MAP_JSON,
        "term_alias_map": TERM_ALIAS_MAP_JSON,
        "retrieval_eval_json": RETRIEVAL_EVAL_JSON,
        "collections": {
            "am_clause": PCF_AM_POLICY_CLAUSES_COLLECTION,
            "am_schema": PCF_AM_POLICY_SCHEMA_COLLECTION,
            "sm_clause": PCF_SM_POLICY_CLAUSES_COLLECTION,
            "sm_schema": PCF_SM_POLICY_SCHEMA_COLLECTION,
            "ursp_clause": PCF_URSP_CLAUSES_COLLECTION,
            "ursp_schema": PCF_URSP_SCHEMA_COLLECTION,
            "glossary": PCF_POLICY_GLOSSARY_COLLECTION,
        },
    },
    "docling": {
        "clause_jsonl": DOCLING_CLAUSE_JSONL,
        "schema_jsonl": DOCLING_SCHEMA_JSONL,
        "glossary_jsonl": DOCLING_GLOSSARY_JSONL,
        "clause_exact_index": DOCLING_CLAUSE_EXACT_INDEX_JSON,
        "schema_exact_index": DOCLING_SCHEMA_EXACT_INDEX_JSON,
        "glossary_exact_index": DOCLING_GLOSSARY_EXACT_INDEX_JSON,
        "spec_object_map": DOCLING_SPEC_OBJECT_MAP_JSON,
        "term_alias_map": DOCLING_TERM_ALIAS_MAP_JSON,
        "retrieval_eval_json": DOCLING_RETRIEVAL_EVAL_JSON,
        "collections": {
            "am_clause": PCF_AM_POLICY_CLAUSES_DOCLING_COLLECTION,
            "am_schema": PCF_AM_POLICY_SCHEMA_DOCLING_COLLECTION,
            "sm_clause": PCF_SM_POLICY_CLAUSES_DOCLING_COLLECTION,
            "sm_schema": PCF_SM_POLICY_SCHEMA_DOCLING_COLLECTION,
            "ursp_clause": PCF_URSP_CLAUSES_DOCLING_COLLECTION,
            "ursp_schema": PCF_URSP_SCHEMA_DOCLING_COLLECTION,
            "glossary": PCF_POLICY_GLOSSARY_DOCLING_COLLECTION,
        },
    },
}


def load_eval_cases(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("Evaluation cases JSON must be a list.")
    return payload


def debug_tool_invocation(
    query: str,
    *,
    category: Optional[str] = None,
    limit: int = 5,
    verbose: bool = True,
    stage_timeout_seconds: int = 30,
) -> Dict[str, Any]:
    """单条调试入口：拆解 knowledge_tool 的检索流水线并返回阶段计时。"""
    return knowledge_tool.debug_search_semantic_knowledge_pipeline(
        query=query,
        category=category,
        limit=limit,
        verbose=verbose,
        stage_timeout_seconds=stage_timeout_seconds,
    )


def _flatten_trace_runs(run: Dict[str, Any]) -> List[Dict[str, Any]]:
    flattened: List[Dict[str, Any]] = []
    stack = [run]
    while stack:
        current = stack.pop()
        flattened.append(current)
        stack.extend(reversed(current.get("child_runs") or []))
    return flattened


def _record_lookup_by_result_key() -> Dict[str, List[Dict[str, Any]]]:
    lookup: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in knowledge_tool._record_catalog().values():
        metadata = record.get("metadata") or {}
        result_key = knowledge_tool._candidate_key(metadata, record.get("id") or "")
        if result_key:
            lookup[result_key].append(record)
    return lookup


def _target_kind_from_record(record: Dict[str, Any]) -> str:
    doc_type = str((record.get("metadata") or {}).get("doc_type") or "").strip().lower()
    if doc_type == "openapi":
        return "schema"
    if doc_type == "glossary":
        return "glossary"
    if doc_type in {"stage2", "stage3", "table"}:
        return "clause"
    return "openapi"


def _trace_case_domain_matches(category: Optional[str], expected_domain: str, inferred_domain: str) -> bool:
    normalized_category = str(category or "").strip().lower()
    normalized_expected = str(expected_domain or "").strip().lower()
    normalized_inferred = str(inferred_domain or "").strip().lower()

    if normalized_expected == "shared":
        return normalized_inferred in {"all", "mobility", "shared", ""} or normalized_category in {"", "none", "all", "shared"}
    if normalized_category in {"", "none", "all", "shared"}:
        return normalized_inferred in {"", "all", "mobility", normalized_expected}
    if normalized_category == "mobility":
        return normalized_expected in {"am_policy", "ursp"}
    return normalized_category == normalized_expected


def extract_iea_trace_eval_cases(
    *,
    trace_paths: Iterable[Path] = DEFAULT_IEA_TRACE_PATHS,
    min_frequency: int = 1,
    max_cases: Optional[int] = None,
) -> List[Dict[str, Any]]:
    result_lookup = _record_lookup_by_result_key()
    aggregated: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    for trace_path in trace_paths:
        with trace_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                trace = json.loads(line)
                runs = _flatten_trace_runs(trace)
                for index, run in enumerate(runs):
                    if run.get("name") != "search_semantic_knowledge":
                        continue
                    args = run.get("inputs", {}).get("args") or {}
                    query = str(args.get("query") or "").strip()
                    if not query:
                        continue
                    category = args.get("category")
                    chosen_key = ""
                    for later in runs[index + 1 :]:
                        later_name = later.get("name")
                        if later_name == "search_semantic_knowledge":
                            break
                        if later_name != "get_knowledge_by_key":
                            continue
                        later_args = later.get("inputs", {}).get("args") or {}
                        candidate_key = str(later_args.get("key") or "").strip()
                        if not candidate_key:
                            continue
                        result_text = str(((later.get("outputs") or {}).get("result") or {}).get("content") or "")
                        if "Knowledge item not found" in result_text:
                            continue
                        chosen_key = candidate_key
                        break
                    if not chosen_key:
                        continue
                    matched_records = result_lookup.get(chosen_key) or []
                    if not matched_records:
                        continue
                    record = matched_records[0]
                    metadata = record.get("metadata") or {}
                    expected_domain = str(metadata.get("policy_domain") or "").strip()
                    inferred_domain = knowledge_tool._infer_policy_domain(query, str(category or ""))
                    if not _trace_case_domain_matches(category, expected_domain, inferred_domain):
                        continue
                    policy_domain = expected_domain
                    if expected_domain == "shared" and inferred_domain in {"am_policy", "sm_policy", "ursp"}:
                        policy_domain = inferred_domain

                    aggregate_key = (query, str(category or ""), chosen_key)
                    bucket = aggregated.setdefault(
                        aggregate_key,
                        {
                            "query": query,
                            "category": category,
                            "policy_domain": policy_domain,
                            "target_kind": _target_kind_from_record(record),
                            "expected_result_keys": [chosen_key],
                            "frequency": 0,
                            "trace_sources": set(),
                        },
                    )
                    bucket["frequency"] += 1
                    bucket["trace_sources"].add(trace_path.name)

    ordered = sorted(
        aggregated.values(),
        key=lambda item: (-int(item["frequency"]), item["policy_domain"], item["target_kind"], item["query"]),
    )
    cases: List[Dict[str, Any]] = []
    for offset, item in enumerate(ordered, start=1):
        if int(item["frequency"]) < int(min_frequency):
            continue
        case = {
            "id": f"iea-trace-{offset:03d}",
            "query": item["query"],
            "category": item["category"],
            "policy_domain": item["policy_domain"],
            "target_kind": item["target_kind"],
            "expected_result_keys": item["expected_result_keys"],
            "frequency": item["frequency"],
            "trace_sources": sorted(item["trace_sources"]),
        }
        cases.append(case)
        if max_cases is not None and len(cases) >= max_cases:
            break
    return cases


def validate_trace_eval_cases(cases: Iterable[Dict[str, Any]]) -> None:
    cases = list(cases)
    seen_ids: set[str] = set()
    if not cases:
        raise ValueError("Trace-derived evaluation cases are empty.")
    for case in cases:
        case_id = str(case.get("id") or "").strip()
        if not case_id:
            raise ValueError("Each trace-derived evaluation case requires a non-empty id.")
        if case_id in seen_ids:
            raise ValueError(f"Duplicate trace-derived evaluation case id: {case_id}")
        seen_ids.add(case_id)
        if not str(case.get("query") or "").strip():
            raise ValueError(f"Trace-derived evaluation case {case_id} requires a non-empty query.")
        expected_keys = [str(item).strip() for item in case.get("expected_result_keys") or [] if str(item).strip()]
        if not expected_keys:
            raise ValueError(f"Trace-derived evaluation case {case_id} requires expected_result_keys.")


def write_eval_cases(path: Path, cases: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(cases), ensure_ascii=False, indent=2), encoding="utf-8")


def validate_eval_cases(cases: Iterable[Dict[str, Any]]) -> None:
    domain_coverage: Dict[str, set[str]] = defaultdict(set)
    seen_ids: set[str] = set()
    for case in cases:
        case_id = str(case.get("id") or "").strip()
        if not case_id:
            raise ValueError("Each evaluation case requires a non-empty id.")
        if case_id in seen_ids:
            raise ValueError(f"Duplicate evaluation case id: {case_id}")
        seen_ids.add(case_id)

        query = str(case.get("query") or "").strip()
        if not query:
            raise ValueError(f"Evaluation case {case_id} requires a non-empty query.")

        expected_keys = [str(item).strip() for item in case.get("expected_result_keys") or [] if str(item).strip()]
        if not expected_keys:
            raise ValueError(f"Evaluation case {case_id} requires expected_result_keys.")

        domain = str(case.get("policy_domain") or "").strip()
        target_kind = str(case.get("target_kind") or "").strip()
        if domain not in {"am_policy", "sm_policy", "ursp"}:
            raise ValueError(f"Evaluation case {case_id} has unsupported policy_domain: {domain}")
        if target_kind not in {"schema", "clause", "glossary", "openapi"}:
            raise ValueError(f"Evaluation case {case_id} has unsupported target_kind: {target_kind}")
        domain_coverage[domain].add(target_kind)

    for domain in ("am_policy", "sm_policy", "ursp"):
        covered_kinds = domain_coverage.get(domain, set())
        if "schema" not in covered_kinds:
            raise ValueError(f"Evaluation cases do not cover schema retrieval for {domain}.")
        if not (covered_kinds - {"schema"}):
            raise ValueError(f"Evaluation cases do not cover non-schema knowledge for {domain}.")


@contextmanager
def knowledge_profile(profile_name: str) -> Iterator[None]:
    if profile_name not in PROFILE_CONFIGS:
        raise ValueError(f"Unsupported profile: {profile_name}")
    config = PROFILE_CONFIGS[profile_name]
    collections = config["collections"]
    required_paths = (
        config["clause_jsonl"],
        config["schema_jsonl"],
        config["glossary_jsonl"],
        config["clause_exact_index"],
        config["schema_exact_index"],
        config["glossary_exact_index"],
        config["spec_object_map"],
        config["term_alias_map"],
    )

    overrides = {
        "CLAUSE_JSONL": config["clause_jsonl"],
        "SCHEMA_JSONL": config["schema_jsonl"],
        "GLOSSARY_JSONL": config["glossary_jsonl"],
        "CLAUSE_EXACT_INDEX_JSON": config["clause_exact_index"],
        "SCHEMA_EXACT_INDEX_JSON": config["schema_exact_index"],
        "GLOSSARY_EXACT_INDEX_JSON": config["glossary_exact_index"],
        "SPEC_OBJECT_MAP_JSON": config["spec_object_map"],
        "TERM_ALIAS_MAP_JSON": config["term_alias_map"],
        "REQUIRED_KB_PATHS": required_paths,
        "PCF_AM_POLICY_CLAUSES_COLLECTION": collections["am_clause"],
        "PCF_AM_POLICY_SCHEMA_COLLECTION": collections["am_schema"],
        "PCF_SM_POLICY_CLAUSES_COLLECTION": collections["sm_clause"],
        "PCF_SM_POLICY_SCHEMA_COLLECTION": collections["sm_schema"],
        "PCF_URSP_CLAUSES_COLLECTION": collections["ursp_clause"],
        "PCF_URSP_SCHEMA_COLLECTION": collections["ursp_schema"],
    }
    original_values = {name: getattr(knowledge_tool, name) for name in overrides}
    for name, value in overrides.items():
        setattr(knowledge_tool, name, value)
    for cache_func in (
        knowledge_tool._ensure_processed_corpus,
        knowledge_tool._record_catalog,
        knowledge_tool._records_by_citation,
        knowledge_tool._clause_exact_index,
        knowledge_tool._schema_exact_index,
        knowledge_tool._glossary_exact_index,
        knowledge_tool._spec_object_map,
        knowledge_tool._term_alias_map,
    ):
        cache_func.cache_clear()
    try:
        yield
    finally:
        for name, value in original_values.items():
            setattr(knowledge_tool, name, value)
        for cache_func in (
            knowledge_tool._ensure_processed_corpus,
            knowledge_tool._record_catalog,
            knowledge_tool._records_by_citation,
            knowledge_tool._clause_exact_index,
            knowledge_tool._schema_exact_index,
            knowledge_tool._glossary_exact_index,
            knowledge_tool._spec_object_map,
            knowledge_tool._term_alias_map,
        ):
            cache_func.cache_clear()


def _safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return round(mean(values), 4) if values else 0.0


def evaluate_case(case: Dict[str, Any], *, limit: int) -> Dict[str, Any]:
    query = str(case["query"]).strip()
    category = str(case.get("category") or "").strip() or None
    expected_keys = [str(item).strip() for item in case["expected_result_keys"]]
    domain = knowledge_tool._infer_policy_domain(query, category)
    candidates, errors = knowledge_tool._assemble_response(query, domain, limit=limit)
    retrieved_keys = [
        knowledge_tool._candidate_key(candidate["metadata"], candidate.get("id") or "")
        for candidate in candidates
    ]
    retrieved_top_k = retrieved_keys[:limit]
    expected_set = set(expected_keys)
    relevant_hits = [key for key in retrieved_top_k if key in expected_set]

    precision_at_k = len(relevant_hits) / len(retrieved_top_k) if retrieved_top_k else 0.0
    recall_at_k = len(expected_set.intersection(retrieved_top_k)) / len(expected_set) if expected_set else 0.0
    accuracy_at_1 = 1.0 if retrieved_top_k and retrieved_top_k[0] in expected_set else 0.0
    first_relevant_rank = 0
    for index, key in enumerate(retrieved_top_k, start=1):
        if key in expected_set:
            first_relevant_rank = index
            break
    reciprocal_rank = 1.0 / first_relevant_rank if first_relevant_rank else 0.0

    return {
        "id": case["id"],
        "query": query,
        "category": category,
        "policy_domain": case["policy_domain"],
        "target_kind": case["target_kind"],
        "expected_result_keys": expected_keys,
        "retrieved_result_keys": retrieved_top_k,
        "relevant_hits": relevant_hits,
        "precision_at_k": round(precision_at_k, 4),
        "recall_at_k": round(recall_at_k, 4),
        "accuracy_at_1": round(accuracy_at_1, 4),
        "reciprocal_rank": round(reciprocal_rank, 4),
        "success": bool(relevant_hits),
        "warnings": errors,
    }


def summarize_results(results: List[Dict[str, Any]], *, profile: str, limit: int, case_source: Path) -> Dict[str, Any]:
    by_domain: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_target_kind: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_domain[str(result["policy_domain"])].append(result)
        by_target_kind[str(result["target_kind"])].append(result)

    def _group_summary(group_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "cases": len(group_results),
            "precision_at_k": _safe_mean(item["precision_at_k"] for item in group_results),
            "recall_at_k": _safe_mean(item["recall_at_k"] for item in group_results),
            "accuracy_at_1": _safe_mean(item["accuracy_at_1"] for item in group_results),
            "mrr": _safe_mean(item["reciprocal_rank"] for item in group_results),
            "success_rate": _safe_mean(1.0 if item["success"] else 0.0 for item in group_results),
        }

    return {
        "profile": profile,
        "top_k": limit,
        "case_source": str(case_source),
        "overall": _group_summary(results),
        "by_domain": {name: _group_summary(group) for name, group in sorted(by_domain.items())},
        "by_target_kind": {name: _group_summary(group) for name, group in sorted(by_target_kind.items())},
        "results": results,
    }


def render_summary(report: Dict[str, Any]) -> str:
    lines = [
        f"Profile: {report['profile']}",
        f"Cases: {len(report['results'])}",
        f"Top-K: {report['top_k']}",
        f"Case Source Type: {report.get('case_source_type', 'curated')}",
        "",
        "**Overall**",
        (
            f"precision@{report['top_k']}={report['overall']['precision_at_k']:.4f} "
            f"recall@{report['top_k']}={report['overall']['recall_at_k']:.4f} "
            f"accuracy@1={report['overall']['accuracy_at_1']:.4f} "
            f"mrr={report['overall']['mrr']:.4f} "
            f"success_rate={report['overall']['success_rate']:.4f}"
        ),
        "",
        "**By Domain**",
    ]
    for domain, summary in report["by_domain"].items():
        lines.append(
            (
                f"{domain}: cases={summary['cases']} "
                f"precision@{report['top_k']}={summary['precision_at_k']:.4f} "
                f"recall@{report['top_k']}={summary['recall_at_k']:.4f} "
                f"accuracy@1={summary['accuracy_at_1']:.4f}"
            )
        )
    lines.append("")
    lines.append("**By Target Kind**")
    for target_kind, summary in report["by_target_kind"].items():
        lines.append(
            (
                f"{target_kind}: cases={summary['cases']} "
                f"precision@{report['top_k']}={summary['precision_at_k']:.4f} "
                f"recall@{report['top_k']}={summary['recall_at_k']:.4f} "
                f"accuracy@1={summary['accuracy_at_1']:.4f}"
            )
        )
    lines.append("")
    lines.append("**Misses**")
    misses = [item for item in report["results"] if not item["success"]]
    if not misses:
        lines.append("none")
    else:
        for miss in misses:
            lines.append(
                f"{miss['id']}: expected={miss['expected_result_keys']} retrieved={miss['retrieved_result_keys']}"
            )
    return "\n".join(lines)


def run_evaluation(
    *,
    profile: str,
    cases_path: Path,
    limit: int,
    case_source_type: str = "curated",
    trace_paths: Iterable[Path] = DEFAULT_IEA_TRACE_PATHS,
    trace_min_frequency: int = 1,
    trace_max_cases: Optional[int] = None,
) -> Dict[str, Any]:
    with knowledge_profile(profile):
        if case_source_type == "curated":
            cases = load_eval_cases(cases_path)
            validate_eval_cases(cases)
            case_source = cases_path
        elif case_source_type == "extended":
            cases = load_eval_cases(DEFAULT_EXTENDED_CASES_PATH)
            validate_trace_eval_cases(cases)
            case_source = DEFAULT_EXTENDED_CASES_PATH
        elif case_source_type == "iea_trace":
            cases = load_eval_cases(DEFAULT_IEA_GOLD_CASES_PATH)
            validate_trace_eval_cases(cases)
            case_source = DEFAULT_IEA_GOLD_CASES_PATH
        elif case_source_type == "iea_trace_weak":
            cases = extract_iea_trace_eval_cases(
                trace_paths=trace_paths,
                min_frequency=trace_min_frequency,
                max_cases=trace_max_cases,
            )
            validate_trace_eval_cases(cases)
            case_source = Path("iea_trace_weak://semantic_queries")
        else:
            raise ValueError(f"Unsupported case source type: {case_source_type}")
        total_cases = len(cases)
        print(
            f"[rag-eval] profile={profile} case_source={case_source_type} "
            f"cases={total_cases} top_k={limit}",
            flush=True,
        )
        results: List[Dict[str, Any]] = []
        for index, case in enumerate(cases, start=1):
            print(
                f"[rag-eval] ({index}/{total_cases}) start case={case['id']} "
                f"domain={case['policy_domain']} target={case['target_kind']} "
                f"query={case['query']}",
                flush=True,
            )
            result = evaluate_case(case, limit=limit)
            print(
                f"[rag-eval] ({index}/{total_cases}) done case={case['id']} "
                f"success={result['success']} top1={result['accuracy_at_1']:.0f} "
                f"mrr={result['reciprocal_rank']:.4f}",
                flush=True,
            )
            results.append(result)
    report = summarize_results(results, profile=profile, limit=limit, case_source=case_source)
    report["case_source_type"] = case_source_type
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the PCF policy RAG knowledge base retrieval quality.")
    parser.add_argument("--profile", choices=sorted(PROFILE_CONFIGS.keys()), default="docling")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--case-source", choices=("curated", "extended", "iea_trace", "iea_trace_weak"), default="curated")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--trace-min-frequency", type=int, default=1)
    parser.add_argument("--trace-max-cases", type=int)
    parser.add_argument("--case-output", type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    if args.case_source == "iea_trace_weak" and args.case_output:
        cases = extract_iea_trace_eval_cases(
            min_frequency=max(1, int(args.trace_min_frequency)),
            max_cases=args.trace_max_cases,
        )
        validate_trace_eval_cases(cases)
        write_eval_cases(args.case_output, cases)

    report = run_evaluation(
        profile=args.profile,
        cases_path=args.cases,
        limit=max(1, int(args.top_k)),
        case_source_type=args.case_source,
        trace_min_frequency=max(1, int(args.trace_min_frequency)),
        trace_max_cases=args.trace_max_cases,
    )
    print(render_summary(report))
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
