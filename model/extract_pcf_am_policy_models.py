from __future__ import annotations

import json
import keyword
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "model" / "openapi" / "models"
GO_OUTPUT_DIR = ROOT / "model" / "go" / "pcf_am_policy"
PY_OUTPUT_DIR = ROOT / "model" / "pydantic"
PY_OUTPUT_FILE = PY_OUTPUT_DIR / "pcf_am_policy_models.py"
MANIFEST_FILE = ROOT / "model" / "pcf_am_policy_manifest.json"
ROOT_TYPE = "PcfAmPolicyControlPolicyAssociation"

BUILTIN_TYPE_MAP = {
    "string": "str",
    "bool": "bool",
    "int": "int",
    "int32": "int",
    "int64": "int",
    "float32": "float",
    "float64": "float",
    "time.Time": "datetime",
    "interface{}": "Any",
    "interface": "Any",
    "byte": "int",
    "rune": "str",
}


@dataclass
class FieldDef:
    go_name: str
    go_type: str
    json_name: str
    omitempty: bool
    comment: str = ""


@dataclass
class TypeDef:
    name: str
    file: Path
    comment: str
    kind: str
    alias_expr: str = ""
    fields: list[FieldDef] = field(default_factory=list)
    enum_values: list[tuple[str, str]] = field(default_factory=list)


def _normalize_comment(lines: Iterable[str]) -> str:
    return " ".join(part.strip() for part in lines if part.strip())


def _collect_leading_comment(lines: list[str], index: int) -> str:
    comment_lines: list[str] = []
    cursor = index - 1
    while cursor >= 0:
        stripped = lines[cursor].strip()
        if stripped.startswith("//"):
            comment_lines.append(stripped[2:].strip())
            cursor -= 1
            continue
        if stripped == "":
            if comment_lines:
                break
            cursor -= 1
            continue
        break
    comment_lines.reverse()
    return _normalize_comment(comment_lines)


def _parse_struct_body(body: str) -> list[FieldDef]:
    fields: list[FieldDef] = []
    pending_comments: list[str] = []

    for raw_line in body.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            pending_comments.clear()
            continue
        if stripped.startswith("//"):
            pending_comments.append(stripped[2:].strip())
            continue

        inline_comment = ""
        if "//" in raw_line:
            code_part, inline_part = raw_line.split("//", 1)
            inline_comment = inline_part.strip()
        else:
            code_part = raw_line

        tag = ""
        if "`" in code_part:
            code_part, tag_part = code_part.split("`", 1)
            tag = tag_part.split("`", 1)[0]

        code_part = code_part.strip()
        if not code_part:
            pending_comments.clear()
            continue

        json_match = re.search(r'json:"([^"]+)"', tag)
        if not json_match:
            pending_comments.clear()
            continue
        json_tag = json_match.group(1)
        json_name = json_tag.split(",", 1)[0]
        if json_name == "-":
            pending_comments.clear()
            continue

        match = re.match(r"^(?P<name>\w+)\s+(?P<type>[^\s].*)$", code_part)
        if not match:
            raise ValueError(f"Unable to parse tagged struct field line: {raw_line}")

        omitempty = "omitempty" in json_tag.split(",")[1:] if "," in json_tag else False
        comment = _normalize_comment([*pending_comments, inline_comment])
        pending_comments.clear()

        fields.append(
            FieldDef(
                go_name=match.group("name"),
                go_type=match.group("type").strip(),
                json_name=json_name,
                omitempty=omitempty,
                comment=comment,
            )
        )

    return fields


def _parse_type_defs(source_dir: Path) -> dict[str, TypeDef]:
    type_index: dict[str, TypeDef] = {}

    for file_path in sorted(source_dir.glob("model_*.go")):
        text = file_path.read_text(encoding="utf-8")
        lines = text.splitlines()
        for match in re.finditer(r"^type\s+(?P<name>\w+)\s+(?P<rest>struct\s*\{|[^\n]+)", text, re.M):
            name = match.group("name")
            rest = match.group("rest").strip()
            line_index = text[: match.start()].count("\n")
            comment = _collect_leading_comment(lines, line_index)

            if rest.startswith("struct"):
                brace_start = match.end("rest") - 1
                depth = 0
                cursor = brace_start
                while cursor < len(text):
                    char = text[cursor]
                    if char == "{":
                        depth += 1
                    elif char == "}":
                        depth -= 1
                        if depth == 0:
                            body = text[brace_start + 1 : cursor]
                            type_index[name] = TypeDef(
                                name=name,
                                file=file_path,
                                comment=comment,
                                kind="struct",
                                fields=_parse_struct_body(body),
                            )
                            break
                    cursor += 1
                else:
                    raise ValueError(f"Unclosed struct body in {file_path}")
                continue

            enum_values: list[tuple[str, str]] = []
            if rest == "string":
                for const_block in re.finditer(r"const\s*\((?P<body>.*?)\n\)", text, re.S):
                    for const_line in const_block.group("body").splitlines():
                        const_match = re.match(
                            rf"^\s*(?P<name>\w+)\s+{re.escape(name)}\s+=\s+\"(?P<value>[^\"]*)\"",
                            const_line,
                        )
                        if const_match:
                            enum_values.append((const_match.group("name"), const_match.group("value")))

            type_index[name] = TypeDef(
                name=name,
                file=file_path,
                comment=comment,
                kind="alias",
                alias_expr=rest,
                enum_values=enum_values,
            )

    if not type_index:
        raise RuntimeError(f"No model_*.go files found in {source_dir}")

    return type_index


def _extract_refs_from_expr(expr: str, type_names: set[str]) -> list[str]:
    refs: list[str] = []
    for token in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expr):
        if token in type_names and token not in refs:
            refs.append(token)
    return refs


def _type_refs(type_def: TypeDef, type_names: set[str]) -> list[str]:
    refs: list[str] = []

    if type_def.kind == "alias":
        refs.extend(_extract_refs_from_expr(type_def.alias_expr, type_names))
    else:
        for field_def in type_def.fields:
            refs.extend(_extract_refs_from_expr(field_def.go_type, type_names))

    return [ref for ref in refs if ref != type_def.name]


def _resolve_closure(root_type: str, type_index: dict[str, TypeDef]) -> list[TypeDef]:
    if root_type not in type_index:
        raise KeyError(f"Root type {root_type} not found in source index.")

    resolved: list[TypeDef] = []
    seen: set[str] = set()
    queue = [root_type]
    type_names = set(type_index)

    while queue:
        current = queue.pop(0)
        if current in seen:
            continue
        seen.add(current)
        type_def = type_index[current]
        resolved.append(type_def)
        for ref in _type_refs(type_def, type_names):
            if ref not in seen and ref not in queue:
                queue.append(ref)

    return resolved


def _safe_identifier(name: str) -> str:
    candidate = re.sub(r"\W+", "_", name)
    if not candidate:
        candidate = "field_"
    if candidate[0].isdigit():
        candidate = f"field_{candidate}"
    if keyword.iskeyword(candidate):
        candidate = f"{candidate}_"
    return candidate


def _enum_member_name(type_name: str, const_name: str, value: str) -> str:
    prefix = f"{type_name}_"
    raw = const_name[len(prefix) :] if const_name.startswith(prefix) else value
    member = _safe_identifier(raw.upper())
    if not member:
        raise ValueError(f"Unable to derive enum member from {const_name}={value}")
    return member


def _go_expr_to_python(expr: str) -> tuple[str, bool]:
    expr = expr.strip()
    if expr.startswith("*"):
        inner, _ = _go_expr_to_python(expr[1:])
        return inner, True
    if expr.startswith("[]"):
        inner, _ = _go_expr_to_python(expr[2:])
        return f"List[{inner}]", False
    if expr.startswith("map["):
        close_idx = expr.index("]")
        value_expr = expr[close_idx + 1 :]
        value_type, _ = _go_expr_to_python(value_expr)
        return f"Dict[str, {value_type}]", False
    return BUILTIN_TYPE_MAP.get(expr, expr), False


def _python_field_name(field_def: FieldDef) -> str:
    return _safe_identifier(field_def.json_name)


def _render_field(field_def: FieldDef) -> str:
    field_name = _python_field_name(field_def)
    annotation, pointer_optional = _go_expr_to_python(field_def.go_type)
    optional = pointer_optional or field_def.omitempty
    rendered_annotation = f"Optional[{annotation}]" if optional else annotation
    default = "None" if optional else "..."

    field_args = [default]
    if field_name != field_def.json_name:
        field_args.append(f'alias="{field_def.json_name}"')
    if field_def.comment:
        description = field_def.comment.replace("\\", "\\\\").replace('"', '\\"')
        field_args.append(f'description="{description}"')

    return f"    {field_name}: {rendered_annotation} = Field({', '.join(field_args)})"


def _render_alias(type_def: TypeDef) -> str:
    if type_def.enum_values:
        lines = [f"class {type_def.name}(str, Enum):"]
        if type_def.comment:
            lines.append(f'    """{type_def.comment}"""')
        for const_name, value in type_def.enum_values:
            lines.append(f'    {_enum_member_name(type_def.name, const_name, value)} = "{value}"')
        return "\n".join(lines)

    rhs = BUILTIN_TYPE_MAP.get(type_def.alias_expr, type_def.alias_expr)
    if type_def.comment:
        return f"# {type_def.comment}\n{type_def.name} = {rhs}"
    return f"{type_def.name} = {rhs}"


def _render_struct(type_def: TypeDef) -> str:
    lines = [f"class {type_def.name}(OpenAPIBaseModel):"]
    if type_def.comment:
        lines.append(f'    """{type_def.comment}"""')
    if not type_def.fields:
        lines.append("    pass")
        return "\n".join(lines)
    for field_def in type_def.fields:
        lines.append(_render_field(field_def))
    return "\n".join(lines)


def _render_python_module(type_defs: list[TypeDef]) -> str:
    sections = [
        "from __future__ import annotations",
        "",
        "from datetime import datetime",
        "from enum import Enum",
        "from typing import Any, Dict, List, Optional",
        "",
        "from pydantic import BaseModel, ConfigDict, Field",
        "",
        "",
        "class OpenAPIBaseModel(BaseModel):",
        "    model_config = ConfigDict(populate_by_name=True, extra='forbid')",
    ]

    rebuildable_models: list[str] = []

    for type_def in type_defs:
        sections.append("")
        sections.append("")
        if type_def.kind == "struct":
            sections.append(_render_struct(type_def))
            rebuildable_models.append(type_def.name)
        else:
            sections.append(_render_alias(type_def))

    if rebuildable_models:
        sections.append("")
        sections.append("")
        sections.append("# Resolve forward references across generated models.")
        for model_name in rebuildable_models:
            sections.append(f"{model_name}.model_rebuild()")

    sections.append("")
    sections.append("")
    exports = ", ".join(f'"{type_def.name}"' for type_def in type_defs)
    sections.append(f"__all__ = [{exports}]")
    sections.append("")

    return "\n".join(sections)


def _ensure_package_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    init_file = path / "__init__.py"
    if not init_file.exists():
        init_file.write_text("", encoding="utf-8")


def main() -> None:
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Source directory not found: {SOURCE_DIR}")

    type_index = _parse_type_defs(SOURCE_DIR)
    resolved = _resolve_closure(ROOT_TYPE, type_index)

    GO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for old_file in GO_OUTPUT_DIR.glob("*.go"):
        old_file.unlink()

    for type_def in resolved:
        shutil.copy2(type_def.file, GO_OUTPUT_DIR / type_def.file.name)

    _ensure_package_dir(ROOT / "model")
    _ensure_package_dir(PY_OUTPUT_DIR)

    python_module = _render_python_module(resolved)
    PY_OUTPUT_FILE.write_text(python_module, encoding="utf-8")

    (PY_OUTPUT_DIR / "__init__.py").write_text(
        "from .pcf_am_policy_models import *\n",
        encoding="utf-8",
    )

    manifest = {
        "root_type": ROOT_TYPE,
        "source_dir": str(SOURCE_DIR),
        "go_output_dir": str(GO_OUTPUT_DIR),
        "python_output_file": str(PY_OUTPUT_FILE),
        "types": [
            {
                "name": type_def.name,
                "kind": type_def.kind,
                "source_file": str(type_def.file),
                "copied_file": str(GO_OUTPUT_DIR / type_def.file.name),
            }
            for type_def in resolved
        ],
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Resolved {len(resolved)} model types.")
    print(f"Copied Go files to: {GO_OUTPUT_DIR}")
    print(f"Generated Pydantic models at: {PY_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
