from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from docling.document_converter import DocumentConverter
from docling_core.types.io import DocumentStream
from docling_core.types.doc import TableItem, TextItem
from pypdf import PdfReader, PdfWriter

SCRIPT_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BUILD_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = KNOWLEDGE_BUILD_ROOT.parent
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

from database.langchain_pg import build_pgvector_document, rebuild_pgvector_collection
from knowledge_build.scripts.build_pcf_policy_kb import (
    SPEC_OBJECT_MAP_JSON as LEGACY_SPEC_OBJECT_MAP_JSON,
    TERM_ALIAS_MAP_JSON as LEGACY_TERM_ALIAS_MAP_JSON,
    _infer_strategy_domains,
    MANIFEST_PATH,
    build_auxiliary_maps,
    build_eval_queries,
    build_exact_index,
    build_glossary_records,
    collection_scoped_ids,
    default_sources,
    extract_openapi_chunks,
    infer_object_tags,
    is_minimal_source,
    is_probable_heading,
    load_jsonl,
    make_metadata,
    normalize_query_terms,
    normalize_records_for_embedding,
    estimate_tokens,
    sanitize_text,
    split_large_section_units,
    TOKEN_RE,
    write_jsonl,
)
from shared.logging import setup_logger

logger = setup_logger(__name__)

DOCLING_DATA_ROOT = KNOWLEDGE_BUILD_ROOT / "data" / "pcf_policy_r18_docling"
DOCLING_PROCESSED_ROOT = DOCLING_DATA_ROOT / "processed"
DOCLING_CLAUSE_JSONL = DOCLING_PROCESSED_ROOT / "clauses.jsonl"
DOCLING_SCHEMA_JSONL = DOCLING_PROCESSED_ROOT / "schema.jsonl"
DOCLING_GLOSSARY_JSONL = DOCLING_PROCESSED_ROOT / "glossary.jsonl"
DOCLING_BUILD_STATS_JSON = DOCLING_PROCESSED_ROOT / "build_stats.json"
DOCLING_SPEC_OBJECT_MAP_JSON = DOCLING_PROCESSED_ROOT / "spec_object_map.json"
DOCLING_TERM_ALIAS_MAP_JSON = DOCLING_PROCESSED_ROOT / "term_alias_map.json"
DOCLING_CLAUSE_EXACT_INDEX_JSON = DOCLING_PROCESSED_ROOT / "clause_exact_index.json"
DOCLING_SCHEMA_EXACT_INDEX_JSON = DOCLING_PROCESSED_ROOT / "schema_exact_index.json"
DOCLING_GLOSSARY_EXACT_INDEX_JSON = DOCLING_PROCESSED_ROOT / "glossary_exact_index.json"
DOCLING_RETRIEVAL_EVAL_JSON = DOCLING_PROCESSED_ROOT / "retrieval_eval_queries.json"

PCF_AM_POLICY_CLAUSES_DOCLING_COLLECTION = "pcf_am_policy_clauses_r18_docling"
PCF_AM_POLICY_SCHEMA_DOCLING_COLLECTION = "pcf_am_policy_schema_r18_docling"
PCF_SM_POLICY_CLAUSES_DOCLING_COLLECTION = "pcf_sm_policy_clauses_r18_docling"
PCF_SM_POLICY_SCHEMA_DOCLING_COLLECTION = "pcf_sm_policy_schema_r18_docling"
PCF_URSP_CLAUSES_DOCLING_COLLECTION = "pcf_ursp_clauses_r18_docling"
PCF_URSP_SCHEMA_DOCLING_COLLECTION = "pcf_ursp_schema_r18_docling"
PCF_POLICY_GLOSSARY_DOCLING_COLLECTION = "pcf_policy_glossary_r18_docling"


def ensure_docling_directories() -> None:
    DOCLING_PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)


def _write_auxiliary_maps_to_docling_outputs(
    clause_records: List[Dict[str, Any]],
    schema_records: List[Dict[str, Any]],
    glossary_records: List[Dict[str, Any]],
) -> None:
    original_targets = (
        LEGACY_SPEC_OBJECT_MAP_JSON,
        LEGACY_TERM_ALIAS_MAP_JSON,
    )
    try:
        globals_map = build_auxiliary_maps.__globals__
        globals_map["SPEC_OBJECT_MAP_JSON"] = DOCLING_SPEC_OBJECT_MAP_JSON
        globals_map["TERM_ALIAS_MAP_JSON"] = DOCLING_TERM_ALIAS_MAP_JSON
        build_auxiliary_maps(clause_records, schema_records, glossary_records)
    finally:
        globals_map["SPEC_OBJECT_MAP_JSON"], globals_map["TERM_ALIAS_MAP_JSON"] = original_targets


def _load_minimal_sources_by_doc_type(
    doc_types: set[str],
    manifest: Optional[Iterable[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    if manifest is None:
        if MANIFEST_PATH.exists():
            manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        else:
            manifest = []
            for source in default_sources():
                source_dict = {
                    "source_id": source.source_id,
                    "spec_id": source.spec_id,
                    "title": source.title,
                    "release": source.release,
                    "version": source.version,
                    "source_url": source.source_url,
                    "doc_type": source.doc_type,
                    "policy_domain": source.policy_domain,
                    "local_name": source.local_name,
                    "local_path": str(KNOWLEDGE_BUILD_ROOT / "data" / "pcf_policy_r18" / "raw" / source.local_name),
                }
                manifest.append(source_dict)

    sources: List[Dict[str, Any]] = []
    for source in manifest:
        if source.get("doc_type") not in doc_types:
            continue
        local_path = Path(str(source.get("local_path") or "")).expanduser()
        if not local_path.exists():
            raise FileNotFoundError(f"PCF PDF source not found: {local_path}")
        normalized = dict(source)
        normalized["local_path"] = str(local_path)
        sources.append(normalized)
    return [source for source in sources if is_minimal_source(source)]


def _load_pdf_sources(manifest: Optional[Iterable[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    pdf_sources = _load_minimal_sources_by_doc_type({"stage2", "stage3"}, manifest)
    if not pdf_sources:
        raise RuntimeError("No PCF PDF sources were found for docling processing.")
    return pdf_sources


def _load_openapi_sources(manifest: Optional[Iterable[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    return _load_minimal_sources_by_doc_type({"openapi"}, manifest)


def _get_docling_converter() -> DocumentConverter:
    return DocumentConverter()


def _contains_non_ascii(text: str) -> bool:
    return any(ord(char) > 127 for char in str(text or ""))


def _pick_ascii_drive_letter() -> str:
    for letter in ("X", "Y", "Z", "W", "V", "U"):
        if not Path(f"{letter}:\\").exists():
            return letter
    raise RuntimeError("No free drive letter is available for docling ASCII path remapping.")


def _map_under_drive(path: Path, root: Path, drive: str) -> str:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    if resolved_path.is_relative_to(resolved_root):
        suffix = str(resolved_path.relative_to(resolved_root))
        return f"{drive}:\\{suffix}" if suffix else f"{drive}:\\"
    return str(resolved_path)


def _ensure_ascii_runtime_or_reexec(argv: List[str]) -> None:
    if os.name != "nt":
        return
    if os.getenv("DOCLING_ASCII_REEXEC") == "1":
        return

    repo_root = PROJECT_ROOT.resolve()
    executable = Path(sys.executable).resolve()
    script_path = Path(__file__).resolve()
    problematic_paths = [str(repo_root), str(executable), str(script_path)]
    if not any(_contains_non_ascii(item) for item in problematic_paths):
        return

    drive = _pick_ascii_drive_letter()
    subst_target = str(repo_root)
    mapped_python = _map_under_drive(executable, repo_root, drive)
    mapped_script = _map_under_drive(script_path, repo_root, drive)
    remapped_args = [
        _map_under_drive(Path(arg), repo_root, drive) if Path(arg).is_absolute() and Path(arg).exists() else arg
        for arg in argv[1:]
    ]
    env = dict(os.environ)
    env["DOCLING_ASCII_REEXEC"] = "1"

    try:
        subprocess.run(["subst", f"{drive}:", subst_target], check=True)
        completed = subprocess.run([mapped_python, mapped_script, *remapped_args], env=env)
    finally:
        subprocess.run(["subst", f"{drive}:", "/D"], check=False)
    raise SystemExit(completed.returncode)


def _item_page_no(item: Any, *, page_offset: int = 0) -> int:
    prov = getattr(item, "prov", None) or []
    page_numbers = [int(getattr(entry, "page_no", 0) or 0) for entry in prov if getattr(entry, "page_no", None)]
    base_page = min(page_numbers) if page_numbers else 1
    return page_offset + base_page


def _iter_single_page_pdf_streams(path: str) -> Iterable[tuple[int, DocumentStream]]:
    source_path = Path(path)
    with source_path.open("rb") as source_file:
        reader = PdfReader(source_file)
        for page_no, page in enumerate(reader.pages, start=1):
            writer = PdfWriter()
            writer.add_page(page)
            page_stream = BytesIO()
            writer.write(page_stream)
            page_stream.seek(0)
            yield page_no, DocumentStream(
                name=f"{source_path.stem}-page-{page_no}.pdf",
                stream=page_stream,
            )


def _table_title(table_item: TableItem, document: Any) -> str:
    title = sanitize_text(table_item.caption_text(document))
    if title:
        return title
    table_markdown = sanitize_text(table_item.export_to_markdown(document))
    first_line = next((line.strip() for line in table_markdown.splitlines() if line.strip()), "")
    return first_line[:120]


def _append_body_segment(
    *,
    segments: List[Dict[str, Any]],
    body_buffer: List[str],
    page_no: int,
) -> None:
    text = sanitize_text("\n".join(body_buffer))
    if text:
        segments.append({"kind": "body", "title": "", "text": text, "page_no": page_no})
    body_buffer.clear()


def extract_pdf_sections_with_docling(source: Dict[str, Any]) -> List[Dict[str, Any]]:
    converter = _get_docling_converter()
    section_heading = ""
    section_number = ""
    section_start_page = 1
    body_buffer: List[str] = []
    segments: List[Dict[str, Any]] = []
    records: List[Dict[str, Any]] = []
    current_page = 1
    record_counter = 0

    def flush_section(last_page: int) -> None:
        nonlocal segments, body_buffer, record_counter
        _append_body_segment(segments=segments, body_buffer=body_buffer, page_no=last_page)
        if not section_number:
            segments = []
            return
        related_specs = [source["spec_id"]]
        if source["policy_domain"] == "shared":
            related_specs.extend(["29.512", "29.525", "24.526", "23.503"])
        for segment_index, segment in enumerate(segments, start=1):
            doc_type = source["doc_type"] if segment["kind"] == "body" else "table"
            canonical_title = f"{source['spec_id']} {section_number} {section_heading}".strip()
            if segment["title"]:
                canonical_title = f"{canonical_title} {segment['title']}".strip()
            object_tags = infer_object_tags(segment["text"])
            metadata = make_metadata(
                source=source,
                doc_type=doc_type,
                clause_path=section_number,
                clause_title=section_heading,
                page_start=section_start_page,
                page_end=max(section_start_page, int(segment["page_no"])),
                canonical_title=canonical_title,
                object_tags=object_tags,
                table_id=segment["title"],
                citation_anchor=f"{source['spec_id']}:{section_number}",
                related_specs=related_specs,
                normalized_terms=normalize_query_terms(f"{canonical_title} {segment['text'][:240]}"),
            )
            kind = "tbl" if segment["kind"] == "table" else "clause"
            chunk_units = split_large_section_units(
                segment["text"],
                enable_overlap=(segment["kind"] == "body"),
            )
            for chunk_index, chunk_unit in enumerate(chunk_units, start=1):
                record_counter += 1
                chunk_metadata = {
                    **metadata,
                    "chunk_strategy": chunk_unit["chunk_strategy"],
                    "chunk_overlap_from": chunk_unit["chunk_overlap_from"],
                }
                records.append(
                    {
                        "id": f"{source['spec_id']}-{section_number}-{kind}-{segment_index}-{chunk_index}-{record_counter}",
                        "page_content": chunk_unit["text"],
                        "metadata": chunk_metadata,
                    }
                )
        segments = []

    for page_no, page_stream in _iter_single_page_pdf_streams(source["local_path"]):
        current_page = page_no
        conversion = converter.convert(page_stream)
        document = conversion.document
        for item, _level in document.iterate_items():
            item_page_no = _item_page_no(item, page_offset=page_no - 1)

            if isinstance(item, TextItem):
                text = sanitize_text(getattr(item, "text", ""))
                if not text:
                    continue
                if is_probable_heading(text):
                    flush_section(item_page_no)
                    heading_match = text.split(maxsplit=1)
                    section_number = heading_match[0].strip()
                    section_heading = heading_match[1].strip() if len(heading_match) > 1 else ""
                    section_start_page = item_page_no
                    continue
                if section_number:
                    body_buffer.append(text)
                continue

            if isinstance(item, TableItem):
                if not section_number:
                    continue
                _append_body_segment(segments=segments, body_buffer=body_buffer, page_no=item_page_no)
                table_markdown = sanitize_text(item.export_to_markdown(document))
                if not table_markdown:
                    continue
                segments.append(
                    {
                        "kind": "table",
                        "title": _table_title(item, document),
                        "text": table_markdown,
                        "page_no": item_page_no,
                    }
                )

    flush_section(current_page)
    return records


# ---------------------------------------------------------------------------
# 中文标注：数据清洗 — docling 解析 PDF 后会混入大量无关内容
# ---------------------------------------------------------------------------

# 模板化/非技术内容的关键词（命中即判定为噪声）
_BOILERPLATE_KEYWORDS = (
    "intellectual property",
    "copyright",
    "legal notice",
    "modal verb",
    "shall indicates",
    "third digit is incremented",
    "second digit is incremented",
    "editorial only",
    "numbering convention",
    "etsi deliverable",
    "not normative",
    "non-normative",
    "registration policies",
    "terms of reference",
    "iprs essential",
    "essential patents",
    "present document has been",
    "revision history",
    "change history",
)

# 3GPP/ETSI 文档中的非技术前言章节（章节号匹配）
_BOILERPLATE_CLAUSE_PREFIXES = (
    "foreword",
    "history",
)

# 最小有效内容长度（低于此阈值认为信息量不足）
_MIN_CONTENT_LENGTH = 40
_SINGLE_DIGIT_TOKEN_RE = re.compile(r"^\d$")
_GENERIC_NOISE_TITLE_RE = re.compile(r"^(?:\d+\s+){5,}\d+$")
_EMBEDDED_TABLE_MARKER_RE = re.compile(
    r"(?P<section>\d+(?:\.\d+)+)\s+Type\s+(?P<type>[A-Za-z][A-Za-z0-9_]*)\s+"
    r"Table\s+(?P<section_repeat>\d+(?:\.\d+)+)-\d+:\s+Definition of type\s+(?P=type)",
    re.IGNORECASE | re.MULTILINE,
)


def _is_boilerplate(record: Dict[str, Any]) -> bool:
    """判定记录是否为模板化/非技术内容。"""
    content = str(record.get("page_content") or "").lower()[:400]
    # 关键词匹配
    if any(kw in content for kw in _BOILERPLATE_KEYWORDS):
        return True
    # 章节号匹配
    metadata = record.get("metadata") or {}
    clause_title = str(metadata.get("clause_title") or "").strip().lower()
    if clause_title in _BOILERPLATE_CLAUSE_PREFIXES:
        return True
    return False


def _is_low_information(record: Dict[str, Any]) -> bool:
    """判定记录是否信息量过低。"""
    content = str(record.get("page_content") or "").strip()
    if len(content) < _MIN_CONTENT_LENGTH:
        return True
    # 纯标点/空白/单词
    if not any(c.isalpha() for c in content):
        return True
    # "None." 类占位内容
    if content.lower() in {"none.", "none", "n/a", "：", ".", "-"}:
        return True
    return False


def _is_ocr_noise(record: Dict[str, Any]) -> bool:
    """判定记录是否更像 OCR/版面噪声而不是标准条文。"""
    content = str(record.get("page_content") or "").strip()
    metadata = record.get("metadata") or {}
    clause_title = str(metadata.get("clause_title") or "").strip()
    if _GENERIC_NOISE_TITLE_RE.match(clause_title):
        return True

    tokens = TOKEN_RE.findall(content)
    if len(tokens) < 8:
        return False

    single_digit_tokens = sum(1 for token in tokens if _SINGLE_DIGIT_TOKEN_RE.match(token))
    alpha_tokens = sum(1 for token in tokens if any(char.isalpha() for char in token))
    long_alpha_tokens = sum(1 for token in tokens if len(token) >= 4 and any(char.isalpha() for char in token))
    unique_tokens = len(set(token.lower() for token in tokens))

    if single_digit_tokens >= 6 and single_digit_tokens / len(tokens) >= 0.35 and long_alpha_tokens <= 2:
        return True
    if alpha_tokens <= max(2, len(tokens) // 6) and unique_tokens <= max(4, len(tokens) // 5):
        return True
    return False


def _deduplicate_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """基于内容前缀去重，保留第一次出现的记录。"""
    seen: Dict[str, bool] = {}
    deduped: List[Dict[str, Any]] = []
    for record in records:
        content = str(record.get("page_content") or "").strip()
        # 用前200字符作为去重键
        dedup_key = content[:200].lower()
        if dedup_key in seen:
            continue
        seen[dedup_key] = True
        deduped.append(record)
    return deduped


def _promote_embedded_table_markers(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    promoted: List[Dict[str, Any]] = list(records)
    seen_ids = {str(record.get("id") or "").strip() for record in records}
    for record in records:
        metadata = record.get("metadata") or {}
        spec_id = str(metadata.get("spec_id") or "").strip()
        page_content = str(record.get("page_content") or "")
        page_start = int(metadata.get("page_start") or 1)
        page_end = int(metadata.get("page_end") or page_start)
        for match in _EMBEDDED_TABLE_MARKER_RE.finditer(page_content):
            section = str(match.group("section") or "").strip()
            repeated = str(match.group("section_repeat") or "").strip()
            type_name = str(match.group("type") or "").strip()
            if not section or section != repeated or not type_name:
                continue
            synthetic_id = f"{record['id']}-embedded-table-{section}-{type_name}"
            if synthetic_id in seen_ids:
                continue
            canonical_title = f"{spec_id} {section} Type {type_name} Table {section}-1: Definition of type {type_name}".strip()
            synthetic_metadata = {
                **metadata,
                "doc_type": "table",
                "clause_path": section,
                "clause_title": f"Type {type_name}",
                "table_id": f"Table {section}-1: Definition of type {type_name}",
                "canonical_title": canonical_title,
                "citation_anchor": f"{spec_id}:{section}",
                "page_start": page_start,
                "page_end": page_end,
                "object_tags": infer_object_tags(f"{type_name} {page_content}"),
                "normalized_terms": normalize_query_terms(canonical_title),
            }
            promoted.append(
                {
                    "id": synthetic_id,
                    "page_content": f"Table {section}-1: Definition of type {type_name}",
                    "metadata": synthetic_metadata,
                }
            )
            seen_ids.add(synthetic_id)
    return promoted


def _clean_docling_records_with_summary(records: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    中文标注：完整的数据清洗流水线。
    1. 移除模板化/非技术内容 (版权、前言、版本说明)
    2. 移除低信息量记录 (<40字符、纯标点)
    3. 去除重复内容
    """
    summary = {
        "input_records": len(records),
        "boilerplate": 0,
        "low_information": 0,
        "ocr_noise": 0,
        "duplicate": 0,
        "embedded_table_promotions": 0,
        "output_records": 0,
    }
    filtered: List[Dict[str, Any]] = []
    for record in records:
        if _is_boilerplate(record):
            summary["boilerplate"] += 1
            continue
        if _is_low_information(record):
            summary["low_information"] += 1
            continue
        if _is_ocr_noise(record):
            summary["ocr_noise"] += 1
            continue
        filtered.append(record)
    promoted = _promote_embedded_table_markers(filtered)
    summary["embedded_table_promotions"] = max(0, len(promoted) - len(filtered))
    deduped = _deduplicate_records(promoted)
    summary["duplicate"] = max(0, len(promoted) - len(deduped))
    summary["output_records"] = len(deduped)
    return deduped, summary


def clean_docling_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered, _summary = _clean_docling_records_with_summary(records)
    return filtered


def _record_summary(records: Iterable[Dict[str, Any]], *, cleaning_summary: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    records = list(records)
    by_doc_type: Dict[str, int] = {}
    by_policy_domain: Dict[str, int] = {}
    token_counts: List[int] = []
    hard_split_count = 0
    for record in records:
        metadata = record.get("metadata") or {}
        doc_type = str(metadata.get("doc_type") or "unknown").strip() or "unknown"
        domain = str(metadata.get("policy_domain") or "unknown").strip() or "unknown"
        by_doc_type[doc_type] = by_doc_type.get(doc_type, 0) + 1
        by_policy_domain[domain] = by_policy_domain.get(domain, 0) + 1
        token_counts.append(estimate_tokens(str(record.get("page_content") or "")))
        if str(metadata.get("chunk_strategy") or "") == "hard_split":
            hard_split_count += 1
    sorted_tokens = sorted(token_counts)
    avg_chunk_tokens = round(sum(sorted_tokens) / len(sorted_tokens), 2) if sorted_tokens else 0.0
    if sorted_tokens:
        p95_index = max(0, min(len(sorted_tokens) - 1, int(len(sorted_tokens) * 0.95) - 1))
        p95_chunk_tokens = sorted_tokens[p95_index]
    else:
        p95_chunk_tokens = 0
    return {
        "total_records": len(records),
        "by_doc_type": dict(sorted(by_doc_type.items())),
        "by_policy_domain": dict(sorted(by_policy_domain.items())),
        "avg_chunk_tokens": avg_chunk_tokens,
        "p95_chunk_tokens": p95_chunk_tokens,
        "hard_split_count": hard_split_count,
        "cleaning_summary": dict(cleaning_summary or {}),
    }


def _namespace_docling_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    namespaced: List[Dict[str, Any]] = []
    for record in records:
        normalized = dict(record)
        normalized["id"] = f"docling-{str(record.get('id') or '').strip()}"
        namespaced.append(normalized)
    return namespaced


def build_corpus_from_docling(manifest: Optional[Iterable[Dict[str, Any]]] = None) -> Dict[str, int]:
    ensure_docling_directories()
    pdf_sources = _load_pdf_sources(manifest)
    openapi_sources = _load_openapi_sources(manifest)
    clause_records: List[Dict[str, Any]] = []
    for source in pdf_sources:
        clause_records.extend(extract_pdf_sections_with_docling(source))

    # 中文标注：数据清洗 — 移除 docling 解析产生的噪声数据
    clause_records, cleaning_summary = _clean_docling_records_with_summary(clause_records)
    raw_count = int(cleaning_summary.get("input_records") or 0)
    logger.info("Data cleaning: %d -> %d records (removed %d noise entries).", raw_count, len(clause_records), raw_count - len(clause_records))

    schema_records: List[Dict[str, Any]] = []
    for source in openapi_sources:
        schema_records.extend(extract_openapi_chunks(source))
    clause_records = _namespace_docling_records(clause_records)
    schema_records = _namespace_docling_records(schema_records)
    glossary_records = build_glossary_records(clause_records, schema_records)
    glossary_records = _namespace_docling_records(glossary_records)

    write_jsonl(DOCLING_CLAUSE_JSONL, clause_records)
    write_jsonl(DOCLING_SCHEMA_JSONL, schema_records)
    write_jsonl(DOCLING_GLOSSARY_JSONL, glossary_records)
    _write_auxiliary_maps_to_docling_outputs(clause_records, schema_records, glossary_records)

    clause_exact = build_exact_index(clause_records, index_name="clause_exact")
    schema_exact = build_exact_index(schema_records, index_name="schema_exact")
    glossary_exact = build_exact_index(glossary_records, index_name="glossary_exact")
    DOCLING_CLAUSE_EXACT_INDEX_JSON.write_text(json.dumps(clause_exact, ensure_ascii=False, indent=2), encoding="utf-8")
    DOCLING_SCHEMA_EXACT_INDEX_JSON.write_text(json.dumps(schema_exact, ensure_ascii=False, indent=2), encoding="utf-8")
    DOCLING_GLOSSARY_EXACT_INDEX_JSON.write_text(json.dumps(glossary_exact, ensure_ascii=False, indent=2), encoding="utf-8")

    eval_queries = []
    collection_rewrites = {
        "pcf_am_policy_clauses_r18": PCF_AM_POLICY_CLAUSES_DOCLING_COLLECTION,
        "pcf_am_policy_schema_r18": PCF_AM_POLICY_SCHEMA_DOCLING_COLLECTION,
        "pcf_sm_policy_clauses_r18": PCF_SM_POLICY_CLAUSES_DOCLING_COLLECTION,
        "pcf_sm_policy_schema_r18": PCF_SM_POLICY_SCHEMA_DOCLING_COLLECTION,
        "pcf_ursp_clauses_r18": PCF_URSP_CLAUSES_DOCLING_COLLECTION,
        "pcf_ursp_schema_r18": PCF_URSP_SCHEMA_DOCLING_COLLECTION,
        "pcf_policy_glossary_r18": PCF_POLICY_GLOSSARY_DOCLING_COLLECTION,
    }
    for item in build_eval_queries():
        target = str(item["target_collection"])
        for base_collection, docling_collection in collection_rewrites.items():
            target = target.replace(base_collection, docling_collection)
        eval_queries.append({**item, "target_collection": target})
    DOCLING_RETRIEVAL_EVAL_JSON.write_text(json.dumps(eval_queries, ensure_ascii=False, indent=2), encoding="utf-8")
    DOCLING_BUILD_STATS_JSON.write_text(
        json.dumps(
            _record_summary(
                clause_records + schema_records + glossary_records,
                cleaning_summary=cleaning_summary,
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {"clauses": len(clause_records), "schema": len(schema_records), "glossary": len(glossary_records)}


def _records_to_documents(records: Iterable[Dict[str, Any]]) -> List[Any]:
    return [build_pgvector_document(page_content=record["page_content"], metadata=record["metadata"]) for record in records]


def _with_strategy_domains(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized_records: List[Dict[str, Any]] = []
    for record in records:
        normalized = dict(record)
        metadata = dict(normalized.get("metadata") or {})
        if not metadata.get("strategy_domains"):
            searchable_text = "\n".join(
                part
                for part in (
                    str(metadata.get("canonical_title") or "").strip(),
                    str(metadata.get("clause_title") or "").strip(),
                    str(normalized.get("page_content") or "")[:1200],
                )
                if part
            )
            metadata["strategy_domains"] = _infer_strategy_domains(
                source_policy_domain=str(metadata.get("policy_domain") or ""),
                object_tags=list(metadata.get("object_tags") or []),
                searchable_text=searchable_text,
            )
        normalized["metadata"] = metadata
        normalized_records.append(normalized)
    return normalized_records


def _filter_minimal_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    allowed_spec_ids = {"23.503", "24.526", "29.507", "29.512", "29.525", "29.571", "glossary"}
    filtered: List[Dict[str, Any]] = []
    for record in records:
        metadata = record.get("metadata") or {}
        source_id = str(metadata.get("source_id") or "").strip()
        spec_id = str(metadata.get("spec_id") or "").strip()
        if source_id and is_minimal_source({"source_id": source_id}):
            filtered.append(record)
            continue
        if spec_id in allowed_spec_ids:
            filtered.append(record)
    return filtered


def _dedupe_records_by_id(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chosen: Dict[str, Dict[str, Any]] = {}
    for record in records:
        record_id = str(record.get("id") or "").strip()
        if not record_id:
            continue
        current = chosen.get(record_id)
        if current is None or len(str(record.get("page_content") or "")) > len(str(current.get("page_content") or "")):
            chosen[record_id] = record
    return list(chosen.values())


def _prepare_records_for_ingest(path: Path) -> List[Dict[str, Any]]:
    return _dedupe_records_by_id(
        _filter_minimal_records(
            _with_strategy_domains(
                normalize_records_for_embedding(load_jsonl(path))
            )
        )
    )


def ingest_docling_corpus() -> Dict[str, int]:
    clause_records = _prepare_records_for_ingest(DOCLING_CLAUSE_JSONL)
    schema_records = _prepare_records_for_ingest(DOCLING_SCHEMA_JSONL)
    glossary_records = _prepare_records_for_ingest(DOCLING_GLOSSARY_JSONL)

    if not clause_records or not schema_records or not glossary_records:
        raise RuntimeError("Docling processed corpus is incomplete. Run build first.")

    am_clause_records = [record for record in clause_records if "am_policy" in (record["metadata"].get("strategy_domains") or [])]
    sm_clause_records = [record for record in clause_records if "sm_policy" in (record["metadata"].get("strategy_domains") or [])]
    ursp_clause_records = [record for record in clause_records if "ursp" in (record["metadata"].get("strategy_domains") or [])]
    am_schema_records = [record for record in schema_records if "am_policy" in (record["metadata"].get("strategy_domains") or [])]
    sm_schema_records = [record for record in schema_records if "sm_policy" in (record["metadata"].get("strategy_domains") or [])]
    ursp_schema_records = [record for record in schema_records if "ursp" in (record["metadata"].get("strategy_domains") or [])]

    targets = [
        (PCF_AM_POLICY_CLAUSES_DOCLING_COLLECTION, am_clause_records),
        (PCF_AM_POLICY_SCHEMA_DOCLING_COLLECTION, am_schema_records),
        (PCF_SM_POLICY_CLAUSES_DOCLING_COLLECTION, sm_clause_records),
        (PCF_SM_POLICY_SCHEMA_DOCLING_COLLECTION, sm_schema_records),
        (PCF_URSP_CLAUSES_DOCLING_COLLECTION, ursp_clause_records),
        (PCF_URSP_SCHEMA_DOCLING_COLLECTION, ursp_schema_records),
        (PCF_POLICY_GLOSSARY_DOCLING_COLLECTION, glossary_records),
    ]
    stats: Dict[str, int] = {}
    for collection_name, records in targets:
        store = rebuild_pgvector_collection(collection_name=collection_name)
        if records:
            store.add_documents(_records_to_documents(records), ids=collection_scoped_ids(collection_name, records))
        stats[collection_name] = len(records)
        logger.info("Ingested %s documents into %s.", len(records), collection_name)
    return stats


def run_docling_pipeline() -> Dict[str, Any]:
    processed = build_corpus_from_docling()
    ingested = ingest_docling_corpus()
    return {"processed": processed, "ingested": ingested}


def repair_processed_docling_corpus() -> Dict[str, int]:
    clause_records = load_jsonl(DOCLING_CLAUSE_JSONL)
    schema_records = load_jsonl(DOCLING_SCHEMA_JSONL)
    glossary_records = load_jsonl(DOCLING_GLOSSARY_JSONL)
    repaired_clause_records = _namespace_docling_records(
        clean_docling_records(
            [
                {**record, "id": str(record.get("id") or "").removeprefix("docling-")}
                if str(record.get("id") or "").startswith("docling-")
                else record
                for record in clause_records
            ]
        )
    )
    write_jsonl(DOCLING_CLAUSE_JSONL, repaired_clause_records)
    clause_exact = build_exact_index(repaired_clause_records, index_name="clause_exact")
    DOCLING_CLAUSE_EXACT_INDEX_JSON.write_text(json.dumps(clause_exact, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_auxiliary_maps_to_docling_outputs(repaired_clause_records, schema_records, glossary_records)
    return {"clauses": len(repaired_clause_records)}


def main() -> None:
    _ensure_ascii_runtime_or_reexec(sys.argv)
    parser = argparse.ArgumentParser(description="Build the docling-based PCF SM/AM/UE policy knowledge base.")
    parser.add_argument("command", choices=["build", "ingest", "all", "repair"])
    args = parser.parse_args()

    ensure_docling_directories()
    if args.command == "build":
        result = build_corpus_from_docling()
    elif args.command == "ingest":
        result = ingest_docling_corpus()
    elif args.command == "repair":
        result = repair_processed_docling_corpus()
    else:
        result = run_docling_pipeline()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
