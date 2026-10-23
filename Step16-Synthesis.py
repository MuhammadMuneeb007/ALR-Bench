#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================================
# Step16-Synthesis.py
# ============================================================
#
# Parallel-safe Step16 synthesis/extraction.
#
# Main changes:
#   - Processes ALL chunks by default.
#   - Model is passed as an argument.
#   - Supports SLURM array indexing.
#   - Each array task processes N global articles.
#   - Safe parallel output: one JSON per article, one JSONL per task.
#   - Optional merge mode to combine article JSON files later.
#
# Examples:
#
#   Single task, first 3 global articles:
#     python Step16-Synthesis.py \
#       --model-folder-name Qwen__Qwen2.5-3B-Instruct \
#       --task-index 1 \
#       --articles-per-task 3
#
#   Direct one article:
#     python Step16-Synthesis.py \
#       --model-folder-name Qwen__Qwen2.5-3B-Instruct \
#       --project-index 1 \
#       --article-index 1
#
#   Merge outputs after all array jobs finish:
#     python Step16-Synthesis.py \
#       --model-folder-name Qwen__Qwen2.5-3B-Instruct \
#       --merge-only
#
# ============================================================

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable


# ============================================================
# DEFAULT SETTINGS
# ============================================================

DEFAULT_PROJECTS_DIR = "Projects"
DEFAULT_DATABASE_DIR = "DataBase_Strategy3_Top1000"
DEFAULT_RANKINGS_DIR = "Rankings"
DEFAULT_INPUT_FILE = "full_rank_strategy_10.csv"

DEFAULT_TOP_N = 200

DEFAULT_MODEL_FOLDER_NAME = "Qwen__Qwen2.5-3B-Instruct"

DEFAULT_CHUNK_CHARS = 5000
DEFAULT_CHUNK_OVERLAP = 500

# Important:
#   0 means process ALL chunks.
#   This replaces the old bad default of 2.
DEFAULT_MAX_CHUNKS_PER_ARTICLE = 0

DEFAULT_MAX_NEW_TOKENS = 1200
DEFAULT_TEMPERATURE = 0.0

MAX_REVIEW_CONTEXT_CHARS = 6000
MAX_FULLTEXT_CHARS = 24000

RANDOM_SEED = 42

SEARCH_DIRS = [
    Path.cwd(),
    Path.cwd().parent,
    Path("/data/ascher02/uqmmune1/ALR/QA/LOCAL_MODEL_CACHE/models"),
    Path("/data/ascher02/uqmmune1/ALR/QA/LOCAL_MODEL_CACHE"),
    Path("/data/ascher02/uqmmune1/ALR/FinalCode/LOCAL_MODEL_CACHE/models"),
    Path("/data/ascher02/uqmmune1/LOCAL_MODEL_CACHE/models"),
    Path("/data/ascher02/uqmmune1/_MODEL_CACHE/models"),
]


# ============================================================
# BASIC HELPERS
# ============================================================

def set_reproducible(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def clean_text(x: Any) -> str:
    if x is None:
        return ""

    text = str(x)

    if text.lower().strip() in {"nan", "none", "null", "na", "n/a"}:
        return ""

    text = text.replace("\r\n", " ")
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_fulltext(text: Any) -> str:
    if text is None:
        return ""

    text = str(text)

    if text.lower().strip() in {"nan", "none", "null", "na", "n/a"}:
        return ""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def truncate_text(text: str, max_chars: int) -> str:
    text = str(text)

    if max_chars <= 0:
        return text

    if len(text) <= max_chars:
        return text

    cut = text[:max_chars]
    cut = cut.rsplit(" ", 1)[0]

    return cut.strip()


def safe_int(x: Any, default: int = 0) -> int:
    try:
        s = clean_text(x)
        if not s:
            return default
        return int(float(s))
    except Exception:
        return default


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        s = clean_text(x)
        if not s:
            return default
        return float(s)
    except Exception:
        return default


def safe_model_name(model_folder_name: str) -> str:
    x = clean_text(model_folder_name)
    x = x.replace("/", "__")
    x = re.sub(r"[^A-Za-z0-9_.-]+", "_", x)
    return x.strip("_") or "model"


def read_json(path: Path, default: Any = None) -> Any:
    if default is None:
        default = {}

    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return default


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []

    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def write_json(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    with tmp.open("w", encoding="utf-8", errors="ignore") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    tmp.replace(path)


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8", errors="ignore") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", errors="ignore") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="", errors="ignore") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
            quoting=csv.QUOTE_MINIMAL,
            escapechar="\\",
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def flatten_values(value: Any) -> List[str]:
    out: List[str] = []

    if value is None:
        return out

    if isinstance(value, str):
        t = clean_text(value)
        if t:
            out.append(t)
        return out

    if isinstance(value, (int, float, bool)):
        t = clean_text(value)
        if t:
            out.append(t)
        return out

    if isinstance(value, list):
        for item in value:
            out.extend(flatten_values(item))
        return out

    if isinstance(value, dict):
        for item in value.values():
            out.extend(flatten_values(item))
        return out

    t = clean_text(value)
    if t:
        out.append(t)

    return out


def unique_keep_order(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []

    for item in items:
        item = clean_text(item)
        if not item:
            continue

        key = item.lower()
        if key in seen:
            continue

        seen.add(key)
        out.append(item)

    return out


def get_config_values(config: Dict[str, Any], keys: List[str]) -> List[str]:
    vals: List[str] = []

    for key in keys:
        vals.extend(flatten_values(config.get(key, [])))

    return unique_keep_order(vals)


def normalize_title(x: Any) -> str:
    title = clean_text(x).lower()
    title = re.sub(r"[^a-z0-9]+", " ", title)
    title = re.sub(r"\s+", " ", title)
    return title.strip()


def normalize_doi(x: Any) -> str:
    doi = clean_text(x).lower()

    for prefix in [
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "doi.org/",
        "doi:",
        "doi ",
    ]:
        if doi.startswith(prefix):
            doi = doi[len(prefix):]

    m = re.search(r"(10\.\d{4,9}/[^\s\"<>]+)", doi, flags=re.I)
    if m:
        doi = m.group(1)

    doi = re.sub(r"\s+", "", doi)
    doi = doi.strip().strip(".,;:()[]{}<>")

    if doi in {"", "none", "null", "nan", "na", "n/a"}:
        return ""

    return doi


def get_first_existing(row: Dict[str, Any], keys: List[str]) -> str:
    for key in keys:
        if key in row and clean_text(row.get(key, "")):
            return clean_text(row.get(key, ""))

    lower_map = {str(k).lower(): k for k in row.keys()}

    for key in keys:
        real_key = lower_map.get(key.lower())
        if real_key is not None and clean_text(row.get(real_key, "")):
            return clean_text(row.get(real_key, ""))

    return ""


def article_key(row: Dict[str, Any], fallback_i: int = 0) -> str:
    doi = normalize_doi(
        get_first_existing(
            row,
            ["doi", "DOI", "article_doi", "work_doi", "clean_doi", "normalized_doi"],
        )
    )

    if doi:
        return "doi:" + doi

    title = normalize_title(
        get_first_existing(
            row,
            ["title", "Title", "article_title", "display_name"],
        )
    )

    if title:
        return "title:" + title

    return "row:" + str(fallback_i)


def dedupe_limit_rows(rows: List[Dict[str, str]], limit: int) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen: Set[str] = set()

    for i, row in enumerate(rows, start=1):
        key = article_key(row, i)

        if key in seen:
            continue

        seen.add(key)

        row["_input_rank"] = safe_int(
            get_first_existing(
                row,
                ["rank", "final_rank", "full_rank", "hybrid_rank", "input_rank"],
            ),
            i,
        )

        row["_article_key"] = key
        out.append(row)

        if len(out) >= limit:
            break

    return out


# ============================================================
# PROJECT DISCOVERY
# ============================================================

def discover_projects(projects_dir: Path) -> List[Path]:
    if not projects_dir.exists():
        return []

    projects: List[Path] = []

    for p in sorted(projects_dir.iterdir()):
        if not p.is_dir():
            continue

        if p.name.lower() in {"rankings", "search", "evaluation", "__pycache__", "extra"}:
            continue

        if (p / "config.json").exists():
            projects.append(p)

    return projects


# ============================================================
# FULL TEXT HELPERS
# ============================================================

def slugify(x: Any, max_len: int = 80) -> str:
    text = normalize_title(x)
    text = text.replace(" ", "_")
    return text[:max_len].strip("_")


def find_project_fulltext_root(database_dir: Path, project_name: str) -> Optional[Path]:
    if not database_dir.exists():
        return None

    exact = database_dir / project_name
    if exact.exists():
        return exact

    project_norm = normalize_title(project_name)

    for p in sorted(database_dir.iterdir()):
        if not p.is_dir():
            continue

        p_norm = normalize_title(p.name)

        if p_norm.startswith(project_norm[:40]):
            return p

        if project_norm[:50] in p_norm:
            return p

    return None


def collect_text_files(root: Optional[Path]) -> List[Path]:
    if root is None or not root.exists():
        return []

    exts = {".txt", ".md", ".markdown", ".json", ".xml", ".tei"}

    files: List[Path] = []

    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)

    return files


def remove_references_section(text: str) -> str:
    text = normalize_fulltext(text)

    if not text:
        return ""

    lines = text.splitlines()
    total_chars = len(text)
    char_pos = 0

    heading_re = re.compile(
        r"^\s*(?:#{1,6}\s*)?(?:\d+\.?\s*)?"
        r"(references|bibliography|works cited|literature cited|cited literature)"
        r"\s*$",
        flags=re.I,
    )

    for idx, line in enumerate(lines):
        current_pos = char_pos
        char_pos += len(line) + 1

        if not heading_re.match(line.strip()):
            continue

        if current_pos < max(5000, int(total_chars * 0.45)):
            continue

        before = "\n".join(lines[:idx]).strip()
        after = "\n".join(lines[idx:]).strip()

        if len(after) >= 500:
            return before

    return text


def read_text_file(path: Path, max_chars: int) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        text = normalize_fulltext(text)
        text = remove_references_section(text)
        return truncate_text(text, max_chars)
    except Exception:
        return ""


def find_fulltext_for_article(
    row: Dict[str, Any],
    project_fulltext_root: Optional[Path],
    fulltext_files: List[Path],
    max_chars: int,
) -> Tuple[str, str]:

    for col in [
        "fulltext_path",
        "full_text_path",
        "matched_fulltext_path",
        "fulltext_file",
        "full_text_file",
        "tei_path",
        "markdown_path",
        "text_path",
    ]:
        value = clean_text(row.get(col, ""))

        if not value:
            continue

        p = Path(value)

        if p.exists():
            text = read_text_file(p, max_chars)
            if text:
                return text, str(p)

        if project_fulltext_root is not None:
            p2 = project_fulltext_root / value
            if p2.exists():
                text = read_text_file(p2, max_chars)
                if text:
                    return text, str(p2)

    if not fulltext_files:
        return "", ""

    doi = normalize_doi(
        get_first_existing(
            row,
            ["doi", "DOI", "article_doi", "work_doi", "clean_doi", "normalized_doi"],
        )
    )

    title = get_first_existing(
        row,
        ["title", "Title", "article_title", "display_name"],
    )

    title_norm = normalize_title(title)
    title_slug = slugify(title, 70)

    doi_safe = ""
    if doi:
        doi_safe = doi.replace("/", "_").replace(".", "_").replace("-", "_").lower()

    candidates: List[Tuple[int, Path]] = []

    for p in fulltext_files:
        ptext = str(p).lower()
        stem_norm = normalize_title(p.stem)

        score = 0

        if doi and doi.lower() in ptext:
            score += 100

        if doi_safe and doi_safe in ptext.replace("/", "_").replace(".", "_").replace("-", "_"):
            score += 80

        if title_slug and title_slug in ptext:
            score += 50

        if title_norm and len(title_norm) > 20:
            title_tokens = set(title_norm.split())
            stem_tokens = set(stem_norm.split())

            if title_tokens:
                overlap = len(title_tokens & stem_tokens) / max(1, len(title_tokens))
                if overlap >= 0.55:
                    score += int(overlap * 50)

        if score > 0:
            candidates.append((score, p))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_path = candidates[0][1]
        text = read_text_file(best_path, max_chars)

        if text:
            return text, str(best_path)

    return "", ""


def split_text_chunks(text: str, chunk_chars: int, overlap_chars: int) -> List[str]:
    text = clean_text(text)

    if not text:
        return []

    if chunk_chars <= 0 or len(text) <= chunk_chars:
        return [text]

    chunks: List[str] = []
    start = 0
    n = len(text)

    overlap_chars = max(0, min(overlap_chars, chunk_chars // 2))

    while start < n:
        end = min(start + chunk_chars, n)

        if end < n:
            sentence_break = max(
                text.rfind(". ", start, end),
                text.rfind("; ", start, end),
                text.rfind("\n", start, end),
            )

            if sentence_break > start + int(chunk_chars * 0.55):
                end = sentence_break + 1
            else:
                space_break = text.rfind(" ", start, end)
                if space_break > start + int(chunk_chars * 0.55):
                    end = space_break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break

        next_start = max(0, end - overlap_chars)

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


# ============================================================
# REVIEW CONTEXT
# ============================================================

def build_review_profile(config: Dict[str, Any]) -> Dict[str, Any]:
    profile = {
        "title": get_config_values(config, ["title"]),
        "review_type": get_config_values(config, ["review_type"]),
        "objective": get_config_values(config, ["objective"]),
        "research_questions": get_config_values(config, ["research_questions", "review_questions", "rqs"]),

        "concepts": get_config_values(
            config,
            [
                "concepts_keywords",
                "keywords",
                "main_concepts",
                "review_concepts",
                "generated_search_keywords",
                "prisma_s_search_keywords",
                "prisma_s_free_text_terms",
                "prisma_s_controlled_vocabulary_terms",
            ],
        ),

        "inclusion": get_config_values(
            config,
            [
                "prisma_eligibility_inclusion",
                "inclusion_criteria",
                "eligibility_inclusion",
                "generated_abstract_include_keywords",
                "generated_full_text_include_keywords",
                "prisma_title_abstract_include_keywords",
                "prisma_title_abstract_must_have_keywords",
                "prisma_full_text_include_keywords",
                "prisma_full_text_must_have_keywords",
            ],
        ),

        "exclusion": get_config_values(
            config,
            [
                "prisma_eligibility_exclusion",
                "exclusion_criteria",
                "eligibility_exclusion",
                "generated_abstract_exclude_keywords",
                "generated_full_text_exclude_keywords",
                "prisma_title_abstract_exclude_keywords",
                "prisma_full_text_exclude_keywords",
            ],
        ),

        "quality": get_config_values(
            config,
            [
                "prisma_quality_assessment_questions",
                "generated_quality_assessment_questions",
                "quality_assessment_questions",
                "prisma_quality_assessment_keywords",
                "generated_quality_assessment_keywords",
                "quality_assessment_keywords",
            ],
        ),

        "date_start": clean_text(config.get("prisma_date_coverage_start", "")) or clean_text(config.get("generated_date_start", "")),
        "date_end": clean_text(config.get("prisma_date_coverage_end", "")) or clean_text(config.get("generated_date_end", "")),
        "language": clean_text(config.get("prisma_s_language_limits", "")) or clean_text(config.get("generated_language", "")),
    }

    return profile


def build_review_context(profile: Dict[str, Any]) -> str:
    context = f"""
REVIEW TITLE:
{"; ".join(profile.get("title", []))}

REVIEW TYPE:
{"; ".join(profile.get("review_type", []))}

OBJECTIVE:
{"; ".join(profile.get("objective", []))}

RESEARCH QUESTIONS:
{"; ".join(profile.get("research_questions", []))}

CORE CONCEPTS / KEYWORDS:
{"; ".join(profile.get("concepts", [])[:80])}

INCLUSION CRITERIA:
{"; ".join(profile.get("inclusion", [])[:60])}

EXCLUSION CRITERIA:
{"; ".join(profile.get("exclusion", [])[:40])}

QUALITY / EXTRACTION QUESTIONS:
{"; ".join(profile.get("quality", [])[:40])}

DATE RANGE:
{profile.get("date_start", "")} to {profile.get("date_end", "")}

LANGUAGE LIMIT:
{profile.get("language", "")}
""".strip()

    return truncate_text(context, MAX_REVIEW_CONTEXT_CHARS)


def build_article_text(
    row: Dict[str, Any],
    project_fulltext_root: Optional[Path],
    fulltext_files: List[Path],
    max_fulltext_chars: int,
) -> Dict[str, str]:

    title = get_first_existing(row, ["title", "Title", "article_title", "display_name"])
    abstract = get_first_existing(row, ["abstract", "Abstract", "summary", "description"])
    journal = get_first_existing(row, ["journal", "Journal", "container_title", "source", "venue"])
    year = get_first_existing(row, ["year", "publication_year", "published_year", "date", "publication_date"])
    doi = normalize_doi(get_first_existing(row, ["doi", "DOI", "article_doi", "work_doi", "clean_doi", "normalized_doi"]))
    source_db = get_first_existing(row, ["source_db", "database", "source_database", "source"])

    fulltext, fulltext_path = find_fulltext_for_article(
        row=row,
        project_fulltext_root=project_fulltext_root,
        fulltext_files=fulltext_files,
        max_chars=max_fulltext_chars,
    )

    article_text = f"""
TITLE:
{title}

ABSTRACT:
{abstract}

JOURNAL / VENUE:
{journal}

YEAR / DATE:
{year}

DOI:
{doi}

SOURCE:
{source_db}

FULL TEXT IF AVAILABLE:
{fulltext}
""".strip()

    return {
        "title": title,
        "abstract": abstract,
        "journal": journal,
        "year": year,
        "doi": doi,
        "source_db": source_db,
        "fulltext": fulltext,
        "fulltext_path": fulltext_path,
        "article_text": article_text,
    }


# ============================================================
# PROMPTING
# ============================================================

def build_chunk_prompt(
    review_context: str,
    article_text: str,
    chunk_i: int,
    chunk_n: int,
) -> str:

    return f"""
You are helping with a systematic literature review.

Use ONLY the REVIEW CONTEXT and ARTICLE TEXT below.
Do not invent information.
If something is missing, write "not_reported" or "uncertain".
Return ONLY valid JSON.
Do not include markdown.
Do not include explanation outside JSON.

Important:
- This is chunk {chunk_i} of {chunk_n}.
- Use only this chunk.
- Do not assume evidence from other chunks.
- Extract evidence that is actually present in this chunk.

Return this JSON schema exactly:

{{
  "chunk_index": {chunk_i},
  "chunk_count": {chunk_n},
  "screening": {{
    "decision": "include|exclude|uncertain",
    "confidence": 0.0,
    "reason": "",
    "inclusion_evidence": [],
    "exclusion_evidence": []
  }},
  "extraction": {{
    "article_type": "",
    "aim_or_purpose": "",
    "study_domain_or_context": "",
    "datasets_or_data_sources": [],
    "sample_size_or_scale": "",
    "methods_models_tools": [],
    "outcomes_or_targets": [],
    "validation_or_evaluation": "",
    "evaluation_metrics": [],
    "main_numerical_results": [],
    "important_tables_figures_taxonomies": [],
    "main_findings": [],
    "limitations_or_future_work": [],
    "relevance_to_review": "",
    "quality_evidence": []
  }},
  "synthesis_card": {{
    "one_sentence_summary": "",
    "key_points": [],
    "why_it_matters": ""
  }}
}}

REVIEW CONTEXT:
{review_context}

ARTICLE TEXT:
{article_text}
""".strip()


def build_article_synthesis_prompt(
    review_context: str,
    article_meta: Dict[str, str],
    chunk_jsons: List[Dict[str, Any]],
) -> str:

    compact_chunks = []

    for obj in chunk_jsons:
        if not isinstance(obj, dict):
            continue

        compact_chunks.append({
            "chunk_index": obj.get("chunk_index", ""),
            "chunk_count": obj.get("chunk_count", ""),
            "screening": obj.get("screening", {}),
            "extraction": obj.get("extraction", {}),
            "synthesis_card": obj.get("synthesis_card", {}),
        })

    chunk_text = json.dumps(compact_chunks, ensure_ascii=False, indent=2)

    return f"""
You are helping with a systematic literature review.

You are now doing ARTICLE-LEVEL synthesis.

You will receive:
1. REVIEW CONTEXT
2. ARTICLE METADATA
3. CHUNK-LEVEL JSON EXTRACTIONS from the same article

Use ONLY the provided chunk-level evidence.
Do not invent information.
If evidence is weak or contradictory, mark it uncertain.
Return ONLY valid JSON.
Do not include markdown.
Do not include explanation outside JSON.

Your task:
- Combine all chunks into one final article-level extraction.
- Resolve conflicting chunk decisions.
- Do not duplicate repeated evidence.
- Make final inclusion/exclusion/uncertain decision.
- Ground the final answer in the chunk evidence.

Return this JSON schema exactly:

{{
  "final_screening": {{
    "decision": "include|exclude|uncertain",
    "confidence": 0.0,
    "reason": "",
    "strongest_inclusion_evidence": [],
    "strongest_exclusion_evidence": []
  }},
  "final_extraction": {{
    "article_type": "",
    "aim_or_purpose": "",
    "study_domain_or_context": "",
    "datasets_or_data_sources": [],
    "sample_size_or_scale": "",
    "methods_models_tools": [],
    "outcomes_or_targets": [],
    "validation_or_evaluation": "",
    "evaluation_metrics": [],
    "main_numerical_results": [],
    "important_tables_figures_taxonomies": [],
    "main_findings": [],
    "limitations_or_future_work": [],
    "relevance_to_review": "",
    "quality_evidence": []
  }},
  "final_synthesis_card": {{
    "one_sentence_summary": "",
    "key_points": [],
    "why_it_matters": ""
  }}
}}

REVIEW CONTEXT:
{review_context}

ARTICLE METADATA:
Title: {article_meta.get("title", "")}
DOI: {article_meta.get("doi", "")}
Journal: {article_meta.get("journal", "")}
Year: {article_meta.get("year", "")}
Source: {article_meta.get("source_db", "")}

CHUNK-LEVEL JSON EXTRACTIONS:
{chunk_text}
""".strip()


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    text = text.strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        candidate = text[start:end + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None

    return None


def aggregate_chunk_results_python_fallback(chunk_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    parsed = [r.get("parsed_response") for r in chunk_records if isinstance(r.get("parsed_response"), dict)]

    if not parsed:
        return {
            "final_screening": {
                "decision": "uncertain",
                "confidence": 0.0,
                "reason": "No valid chunk JSON could be parsed.",
                "strongest_inclusion_evidence": [],
                "strongest_exclusion_evidence": [],
            },
            "final_extraction": {},
            "final_synthesis_card": {
                "one_sentence_summary": "",
                "key_points": [],
                "why_it_matters": "",
            },
        }

    decisions: List[str] = []
    confidences: List[float] = []

    inclusion_evidence: List[str] = []
    exclusion_evidence: List[str] = []
    key_points: List[str] = []
    methods: List[str] = []
    datasets: List[str] = []
    findings: List[str] = []
    limitations: List[str] = []
    metrics: List[str] = []
    numerical: List[str] = []
    quality: List[str] = []
    summaries: List[str] = []

    def extend_clean(target: List[str], values: Any) -> None:
        if isinstance(values, list):
            for v in values:
                t = clean_text(v)
                if t:
                    target.append(t)
        else:
            t = clean_text(values)
            if t:
                target.append(t)

    for obj in parsed:
        screening = obj.get("screening", {})
        extraction = obj.get("extraction", {})
        synth = obj.get("synthesis_card", {})

        decision = clean_text(screening.get("decision", "")).lower()
        if decision:
            decisions.append(decision)

        conf = safe_float(screening.get("confidence", 0), 0.0)
        if conf:
            confidences.append(conf)

        extend_clean(inclusion_evidence, screening.get("inclusion_evidence", []))
        extend_clean(exclusion_evidence, screening.get("exclusion_evidence", []))
        extend_clean(key_points, synth.get("key_points", []))
        extend_clean(methods, extraction.get("methods_models_tools", []))
        extend_clean(datasets, extraction.get("datasets_or_data_sources", []))
        extend_clean(findings, extraction.get("main_findings", []))
        extend_clean(limitations, extraction.get("limitations_or_future_work", []))
        extend_clean(metrics, extraction.get("evaluation_metrics", []))
        extend_clean(numerical, extraction.get("main_numerical_results", []))
        extend_clean(quality, extraction.get("quality_evidence", []))

        s = clean_text(synth.get("one_sentence_summary", ""))
        if s:
            summaries.append(s)

    if "include" in decisions:
        final_decision = "include"
    elif "exclude" in decisions and "uncertain" not in decisions and "include" not in decisions:
        final_decision = "exclude"
    else:
        final_decision = "uncertain"

    return {
        "final_screening": {
            "decision": final_decision,
            "confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
            "reason": "Python fallback aggregation from chunk-level outputs.",
            "strongest_inclusion_evidence": unique_keep_order(inclusion_evidence)[:20],
            "strongest_exclusion_evidence": unique_keep_order(exclusion_evidence)[:20],
        },
        "final_extraction": {
            "article_type": "",
            "aim_or_purpose": "",
            "study_domain_or_context": "",
            "datasets_or_data_sources": unique_keep_order(datasets)[:40],
            "sample_size_or_scale": "",
            "methods_models_tools": unique_keep_order(methods)[:40],
            "outcomes_or_targets": [],
            "validation_or_evaluation": "",
            "evaluation_metrics": unique_keep_order(metrics)[:30],
            "main_numerical_results": unique_keep_order(numerical)[:30],
            "important_tables_figures_taxonomies": [],
            "main_findings": unique_keep_order(findings)[:40],
            "limitations_or_future_work": unique_keep_order(limitations)[:30],
            "relevance_to_review": "",
            "quality_evidence": unique_keep_order(quality)[:30],
        },
        "final_synthesis_card": {
            "one_sentence_summary": summaries[0] if summaries else "",
            "key_points": unique_keep_order(key_points)[:30],
            "why_it_matters": "",
        },
    }


# ============================================================
# MODEL
# ============================================================

def find_local_model(model_folder_name: str) -> str:
    p = Path(model_folder_name)
    if p.exists():
        return str(p)

    for base in SEARCH_DIRS:
        candidate = base / model_folder_name
        if candidate.exists():
            return str(candidate)

    root = Path("/data/ascher02/uqmmune1/ALR")
    if root.exists():
        for found in root.rglob(model_folder_name):
            if found.exists():
                return str(found)

    raise FileNotFoundError(f"Could not find local model folder: {model_folder_name}")


def load_local_llm(model_path: str):
    print("=" * 140)
    print("Loading local LLM")
    print("=" * 140)
    print("Model path:", model_path)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("WARNING: CUDA is False. This will be slow on CPU.")

    print("=" * 140)
    print("Loading tokenizer...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    model.eval()

    print("Model loaded successfully.", flush=True)
    print("=" * 140)

    return tokenizer, model


def generate_response(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> Tuple[str, int, int, float, float]:

    messages = [{"role": "user", "content": prompt}]

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    max_input_len = getattr(tokenizer, "model_max_length", 32768)

    if not max_input_len or max_input_len > 100000:
        max_input_len = 32768

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_len,
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    input_tokens = int(inputs["input_ids"].shape[1])

    start = time.time()

    with torch.no_grad():
        if temperature and temperature > 0:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

    elapsed = time.time() - start

    generated_tokens = int(outputs.shape[1] - input_tokens)

    response = tokenizer.decode(
        outputs[0][input_tokens:],
        skip_special_tokens=True,
    )

    tps = generated_tokens / elapsed if elapsed > 0 else 0.0

    return response, input_tokens, generated_tokens, elapsed, tps


# ============================================================
# GLOBAL ARTICLE INDEX
# ============================================================

def build_global_article_index(args: argparse.Namespace) -> List[Dict[str, Any]]:
    projects_dir = Path(args.projects_dir)
    projects = discover_projects(projects_dir)

    global_items: List[Dict[str, Any]] = []
    global_i = 0

    for project_i, project_dir in enumerate(projects, start=1):
        ranking_path = project_dir / args.rankings_dir / args.input_file
        rows_raw = read_csv_rows(ranking_path)

        if not rows_raw:
            print(f"WARNING: ranking file missing or empty: {ranking_path}", flush=True)
            continue

        rows = dedupe_limit_rows(rows_raw, args.top_n)

        for article_i, row in enumerate(rows, start=1):
            global_i += 1
            global_items.append({
                "global_index": global_i,
                "project_index": project_i,
                "project_name": project_dir.name,
                "project_dir": str(project_dir),
                "article_index": article_i,
                "row": row,
            })

    return global_items


def select_items_for_this_run(
    items: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:

    # Direct mode: exact project/article.
    if args.project_index > 0 and args.article_index > 0:
        return [
            item for item in items
            if item["project_index"] == args.project_index
            and item["article_index"] == args.article_index
        ]

    # Direct project + article range.
    if args.project_index > 0 and args.article_start_index > 0:
        end_i = args.article_end_index if args.article_end_index > 0 else args.article_start_index

        return [
            item for item in items
            if item["project_index"] == args.project_index
            and args.article_start_index <= item["article_index"] <= end_i
        ]

    # Global direct range.
    if args.global_start_index > 0:
        end_i = args.global_end_index if args.global_end_index > 0 else args.global_start_index

        return [
            item for item in items
            if args.global_start_index <= item["global_index"] <= end_i
        ]

    # SLURM array mode.
    task_index = args.task_index

    if task_index <= 0:
        env_task = os.environ.get("SLURM_ARRAY_TASK_ID", "")
        task_index = safe_int(env_task, 1)

    articles_per_task = max(1, args.articles_per_task)

    start_0 = (task_index - 1) * articles_per_task
    end_0 = start_0 + articles_per_task

    return items[start_0:end_0]


# ============================================================
# OUTPUT PATHS
# ============================================================

def get_model_output_root(project_dir: Path, model_folder_name: str) -> Path:
    return project_dir / "Extractions" / f"Step16_{safe_model_name(model_folder_name)}"


def get_article_output_path(
    project_dir: Path,
    model_folder_name: str,
    global_index: int,
    article_index: int,
) -> Path:
    root = get_model_output_root(project_dir, model_folder_name)
    return root / "article_json" / f"global_{global_index:06d}_article_{article_index:04d}.json"


def get_task_output_path(
    projects_dir: Path,
    model_folder_name: str,
    task_index: int,
) -> Path:
    root = projects_dir / "Rankings" / f"Step16_{safe_model_name(model_folder_name)}"
    return root / "task_jsonl" / f"task_{task_index:06d}.jsonl"


# ============================================================
# PROCESS ONE ARTICLE
# ============================================================

def process_one_article(
    item: Dict[str, Any],
    args: argparse.Namespace,
    tokenizer,
    model,
) -> Dict[str, Any]:

    global_index = int(item["global_index"])
    project_index = int(item["project_index"])
    article_index = int(item["article_index"])

    project_dir = Path(item["project_dir"])
    project_name = item["project_name"]
    row = item["row"]

    output_path = get_article_output_path(
        project_dir=project_dir,
        model_folder_name=args.model_folder_name,
        global_index=global_index,
        article_index=article_index,
    )

    if output_path.exists() and not args.force:
        existing = read_json(output_path, {})
        if existing:
            existing["_status"] = "cached_article_json"
            return existing

    config_path = project_dir / "config.json"
    config = read_json(config_path, {})

    if not config:
        raise RuntimeError(f"Cannot read config: {config_path}")

    profile = build_review_profile(config)
    review_context = build_review_context(profile)

    project_fulltext_root = find_project_fulltext_root(Path(args.database_dir), project_name)
    fulltext_files = collect_text_files(project_fulltext_root)

    article_info = build_article_text(
        row=row,
        project_fulltext_root=project_fulltext_root,
        fulltext_files=fulltext_files,
        max_fulltext_chars=args.max_fulltext_chars,
    )

    article_text = article_info["article_text"]

    chunks = split_text_chunks(
        article_text,
        chunk_chars=args.chunk_chars,
        overlap_chars=args.chunk_overlap,
    )

    # Important:
    #   max_chunks_per_article = 0 means all chunks.
    if args.max_chunks_per_article > 0:
        chunks = chunks[:args.max_chunks_per_article]

    if not chunks:
        chunks = [article_text]

    print()
    print("=" * 140)
    print(f"GLOBAL ARTICLE {global_index}")
    print("=" * 140)
    print("Project index :", project_index)
    print("Project       :", project_name)
    print("Article index :", article_index)
    print("Title         :", article_info["title"][:180])
    print("DOI           :", article_info["doi"])
    print("Has full text :", 1 if article_info["fulltext"] else 0)
    print("Fulltext path :", article_info["fulltext_path"])
    print("Chunks        :", len(chunks))
    print("Output        :", output_path)
    print("=" * 140, flush=True)

    chunk_records: List[Dict[str, Any]] = []

    for chunk_i, chunk_text in enumerate(tqdm(chunks, desc=f"Article {global_index} chunks", unit="chunk"), start=1):
        prompt = build_chunk_prompt(
            review_context=review_context,
            article_text=chunk_text,
            chunk_i=chunk_i,
            chunk_n=len(chunks),
        )

        prompt_hash = hashlib.md5(prompt.encode("utf-8", errors="ignore")).hexdigest()

        rec: Dict[str, Any] = {
            "global_index": global_index,
            "project_index": project_index,
            "project": project_name,
            "article_index": article_index,
            "input_rank": safe_int(row.get("_input_rank", article_index), article_index),
            "article_key": clean_text(row.get("_article_key", "")) or article_key(row, article_index),
            "chunk_index": chunk_i,
            "chunk_count": len(chunks),
            "title": article_info["title"],
            "doi": article_info["doi"],
            "journal": article_info["journal"],
            "year": article_info["year"],
            "source_db": article_info["source_db"],
            "has_fulltext": 1 if article_info["fulltext"] else 0,
            "fulltext_path": article_info["fulltext_path"],
            "prompt_hash": prompt_hash,
            "prompt_chars": len(prompt),
            "model_folder_name": args.model_folder_name,
            "timestamp": time.time(),
        }

        try:
            raw_response, input_tokens, generated_tokens, elapsed, tps = generate_response(
                tokenizer=tokenizer,
                model=model,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )

            parsed = extract_json_object(raw_response)

            rec.update({
                "input_tokens": input_tokens,
                "generated_tokens": generated_tokens,
                "elapsed_seconds": round(elapsed, 4),
                "tokens_per_second": round(tps, 4),
                "raw_response": raw_response,
                "parsed_response": parsed,
                "json_parse_success": 1 if parsed else 0,
                "error": "",
            })

        except Exception as e:
            rec.update({
                "input_tokens": 0,
                "generated_tokens": 0,
                "elapsed_seconds": 0,
                "tokens_per_second": 0,
                "raw_response": "",
                "parsed_response": None,
                "json_parse_success": 0,
                "error": repr(e),
            })

            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        chunk_records.append(rec)

    parsed_chunks = [
        r.get("parsed_response")
        for r in chunk_records
        if isinstance(r.get("parsed_response"), dict)
    ]

    final_synthesis_raw = ""
    final_synthesis_parsed: Optional[Dict[str, Any]] = None
    final_synthesis_error = ""

    if parsed_chunks and args.article_level_synthesis:
        try:
            synthesis_prompt = build_article_synthesis_prompt(
                review_context=review_context,
                article_meta=article_info,
                chunk_jsons=parsed_chunks,
            )

            raw_response, input_tokens, generated_tokens, elapsed, tps = generate_response(
                tokenizer=tokenizer,
                model=model,
                prompt=synthesis_prompt,
                max_new_tokens=args.synthesis_max_new_tokens,
                temperature=args.temperature,
            )

            final_synthesis_raw = raw_response
            final_synthesis_parsed = extract_json_object(raw_response)

            if final_synthesis_parsed is None:
                final_synthesis_error = "article_level_json_parse_failed"

        except Exception as e:
            final_synthesis_error = repr(e)
            final_synthesis_parsed = None

            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    if final_synthesis_parsed is None:
        final_synthesis_parsed = aggregate_chunk_results_python_fallback(chunk_records)

    final_screening = final_synthesis_parsed.get("final_screening", {}) if isinstance(final_synthesis_parsed, dict) else {}
    final_extraction = final_synthesis_parsed.get("final_extraction", {}) if isinstance(final_synthesis_parsed, dict) else {}
    final_card = final_synthesis_parsed.get("final_synthesis_card", {}) if isinstance(final_synthesis_parsed, dict) else {}

    article_out: Dict[str, Any] = {
        "global_index": global_index,
        "project_index": project_index,
        "project": project_name,
        "article_index": article_index,
        "input_rank": safe_int(row.get("_input_rank", article_index), article_index),
        "article_key": clean_text(row.get("_article_key", "")) or article_key(row, article_index),

        "title": article_info["title"],
        "doi": article_info["doi"],
        "journal": article_info["journal"],
        "year": article_info["year"],
        "source_db": article_info["source_db"],

        "has_fulltext": 1 if article_info["fulltext"] else 0,
        "fulltext_path": article_info["fulltext_path"],

        "model_folder_name": args.model_folder_name,

        "chunk_count": len(chunks),
        "processed_all_chunks": 1 if args.max_chunks_per_article == 0 else 0,
        "max_chunks_per_article": args.max_chunks_per_article,

        "final_decision": clean_text(final_screening.get("decision", "uncertain")),
        "decision_confidence": safe_float(final_screening.get("confidence", 0.0), 0.0),
        "final_reason": clean_text(final_screening.get("reason", "")),

        "one_sentence_summary": clean_text(final_card.get("one_sentence_summary", "")),
        "final_key_points": final_card.get("key_points", []),

        "final_screening": final_screening,
        "final_extraction": final_extraction,
        "final_synthesis_card": final_card,

        "article_level_synthesis_enabled": 1 if args.article_level_synthesis else 0,
        "article_level_synthesis_raw_response": final_synthesis_raw,
        "article_level_synthesis_parse_success": 1 if final_synthesis_error == "" and final_synthesis_raw and isinstance(final_synthesis_parsed, dict) else 0,
        "article_level_synthesis_error": final_synthesis_error,

        "chunk_records": chunk_records,

        "output_path": str(output_path),
        "timestamp": time.time(),
        "_status": "done",
    }

    write_json(output_path, article_out)

    return article_out


# ============================================================
# MERGE MODE
# ============================================================

def merge_outputs(args: argparse.Namespace) -> None:
    projects_dir = Path(args.projects_dir)
    projects = discover_projects(projects_dir)

    model_safe = safe_model_name(args.model_folder_name)

    merged_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for project_dir in projects:
        root = get_model_output_root(project_dir, args.model_folder_name)
        article_dir = root / "article_json"

        files = sorted(article_dir.glob("global_*_article_*.json"))

        project_rows: List[Dict[str, Any]] = []

        for f in files:
            rec = read_json(f, {})
            if not rec:
                continue

            small = {
                "global_index": rec.get("global_index", ""),
                "project_index": rec.get("project_index", ""),
                "project": rec.get("project", ""),
                "article_index": rec.get("article_index", ""),
                "input_rank": rec.get("input_rank", ""),
                "article_key": rec.get("article_key", ""),
                "title": rec.get("title", ""),
                "doi": rec.get("doi", ""),
                "journal": rec.get("journal", ""),
                "year": rec.get("year", ""),
                "source_db": rec.get("source_db", ""),
                "has_fulltext": rec.get("has_fulltext", ""),
                "fulltext_path": rec.get("fulltext_path", ""),
                "model_folder_name": rec.get("model_folder_name", ""),
                "chunk_count": rec.get("chunk_count", ""),
                "processed_all_chunks": rec.get("processed_all_chunks", ""),
                "final_decision": rec.get("final_decision", ""),
                "decision_confidence": rec.get("decision_confidence", ""),
                "final_reason": rec.get("final_reason", ""),
                "one_sentence_summary": rec.get("one_sentence_summary", ""),
                "final_key_points": rec.get("final_key_points", []),
                "final_screening": rec.get("final_screening", {}),
                "final_extraction": rec.get("final_extraction", {}),
                "final_synthesis_card": rec.get("final_synthesis_card", {}),
                "output_path": rec.get("output_path", ""),
            }

            project_rows.append(small)
            merged_rows.append(small)

        project_jsonl = root / f"step16_synthesis_{model_safe}_merged.jsonl"
        write_jsonl(project_jsonl, project_rows)

        summary_rows.append({
            "project": project_dir.name,
            "model": args.model_folder_name,
            "articles_merged": len(project_rows),
            "output_jsonl": str(project_jsonl),
        })

    merged_root = projects_dir / "Rankings" / f"Step16_{model_safe}"
    merged_jsonl = merged_root / f"step16_synthesis_{model_safe}_ALL_PROJECTS_merged.jsonl"
    summary_csv = merged_root / f"step16_synthesis_{model_safe}_summary.csv"

    write_jsonl(merged_jsonl, merged_rows)

    write_csv(
        summary_csv,
        summary_rows,
        fields=[
            "project",
            "model",
            "articles_merged",
            "output_jsonl",
        ],
    )

    print("=" * 140)
    print("MERGE DONE")
    print("=" * 140)
    print("Merged JSONL :", merged_jsonl)
    print("Summary CSV  :", summary_csv)
    print("Total rows   :", len(merged_rows))
    print("=" * 140)


# ============================================================
# ARGUMENTS / MAIN
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--projects-dir", default=DEFAULT_PROJECTS_DIR)
    parser.add_argument("--database-dir", default=DEFAULT_DATABASE_DIR)
    parser.add_argument("--rankings-dir", default=DEFAULT_RANKINGS_DIR)
    parser.add_argument("--input-file", default=DEFAULT_INPUT_FILE)

    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)

    parser.add_argument("--model-folder-name", default=DEFAULT_MODEL_FOLDER_NAME)

    parser.add_argument("--chunk-chars", type=int, default=DEFAULT_CHUNK_CHARS)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)

    # 0 means all chunks.
    parser.add_argument("--max-chunks-per-article", type=int, default=DEFAULT_MAX_CHUNKS_PER_ARTICLE)

    parser.add_argument("--max-fulltext-chars", type=int, default=MAX_FULLTEXT_CHARS)

    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--synthesis-max-new-tokens", type=int, default=1600)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)

    # SLURM/global indexing.
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--articles-per-task", type=int, default=3)

    # Direct modes.
    parser.add_argument("--global-start-index", type=int, default=0)
    parser.add_argument("--global-end-index", type=int, default=0)

    parser.add_argument("--project-index", type=int, default=0)
    parser.add_argument("--article-index", type=int, default=0)

    parser.add_argument("--article-start-index", type=int, default=0)
    parser.add_argument("--article-end-index", type=int, default=0)

    parser.add_argument("--article-level-synthesis", action="store_true", default=True)
    parser.add_argument("--no-article-level-synthesis", dest="article_level_synthesis", action="store_false")

    parser.add_argument("--force", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    return parser.parse_args()


def print_run_plan(items: List[Dict[str, Any]], selected: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    total_articles = len(items)
    articles_per_task = max(1, args.articles_per_task)
    total_tasks = math.ceil(total_articles / articles_per_task) if total_articles else 0

    task_index = args.task_index
    if task_index <= 0:
        task_index = safe_int(os.environ.get("SLURM_ARRAY_TASK_ID", ""), 1)

    print("=" * 180)
    print("STEP16 SYNTHESIS / EXTRACTION")
    print("=" * 180)
    print("Projects directory       :", Path(args.projects_dir).resolve())
    print("Database directory       :", Path(args.database_dir).resolve())
    print("Input ranking            :", args.input_file)
    print("Top N per project        :", args.top_n)
    print("Model folder             :", args.model_folder_name)
    print("Chunk chars              :", args.chunk_chars)
    print("Chunk overlap            :", args.chunk_overlap)
    print("Max chunks/article       :", args.max_chunks_per_article, "(0 means ALL chunks)")
    print("Max fulltext chars       :", args.max_fulltext_chars)
    print("Max new tokens/chunk     :", args.max_new_tokens)
    print("Max new tokens/synthesis :", args.synthesis_max_new_tokens)
    print("Article-level synthesis  :", args.article_level_synthesis)
    print("Temperature              :", args.temperature)
    print("Force refresh            :", args.force)
    print("-" * 180)
    print("Total global articles    :", total_articles)
    print("Articles per task        :", articles_per_task)
    print("Required array tasks     :", total_tasks)
    print("Current task index       :", task_index)
    print("Selected articles        :", len(selected))
    print("=" * 180)

    for item in selected:
        row = item["row"]
        title = get_first_existing(row, ["title", "Title", "article_title", "display_name"])
        print(
            f'GLOBAL={item["global_index"]:06d} | '
            f'PROJECT={item["project_index"]:02d}:{item["project_name"]} | '
            f'ARTICLE={item["article_index"]:04d} | '
            f'{title[:140]}'
        )

    print("=" * 180, flush=True)


def main() -> None:
    set_reproducible()

    args = parse_args()

    if args.merge_only:
        merge_outputs(args)
        return

    items = build_global_article_index(args)
    selected = select_items_for_this_run(items, args)

    print_run_plan(items, selected, args)

    if not items:
        raise RuntimeError("No articles found. Check Projects directory and ranking files.")

    if not selected:
        print("No selected articles for this task. Exiting cleanly.")
        return

    if args.dry_run:
        print("Dry run only. No model loaded.")
        return

    model_path = find_local_model(args.model_folder_name)
    tokenizer, model = load_local_llm(model_path)

    task_index = args.task_index
    if task_index <= 0:
        task_index = safe_int(os.environ.get("SLURM_ARRAY_TASK_ID", ""), 1)

    task_output_path = get_task_output_path(
        projects_dir=Path(args.projects_dir),
        model_folder_name=args.model_folder_name,
        task_index=task_index,
    )

    task_records: List[Dict[str, Any]] = []

    for item in selected:
        try:
            rec = process_one_article(
                item=item,
                args=args,
                tokenizer=tokenizer,
                model=model,
            )
        except Exception as e:
            rec = {
                "global_index": item.get("global_index", ""),
                "project_index": item.get("project_index", ""),
                "project": item.get("project_name", ""),
                "article_index": item.get("article_index", ""),
                "model_folder_name": args.model_folder_name,
                "_status": "failed",
                "error": repr(e),
                "timestamp": time.time(),
            }

        task_records.append(rec)
        append_jsonl(task_output_path, rec)

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    print()
    print("=" * 180)
    print("TASK DONE")
    print("=" * 180)
    print("Task output:", task_output_path)
    print("Articles processed:", len(task_records))
    for rec in task_records:
        print(
            f'GLOBAL={rec.get("global_index")} | '
            f'PROJECT={rec.get("project")} | '
            f'ARTICLE={rec.get("article_index")} | '
            f'STATUS={rec.get("_status")} | '
            f'DECISION={rec.get("final_decision", "")}'
        )
    print("=" * 180)


if __name__ == "__main__":
    main()