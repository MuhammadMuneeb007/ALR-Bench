# Step9A-GenerateConfiguration.py
#
# Generates one clean, flat PRISMA/PRISMA-S configuration file per review project.
#
# Design rules:
#   1. The final config.json is single-level / flat.
#   2. No run_id, paths, timestamps, ranking settings, top-k settings, or pipeline metadata in config.json.
#   3. Author-reported PRISMA / PRISMA-S fields are copied verbatim or left empty.
#   4. Generated fields are clearly prefixed with generated_.
#   5. concepts_keywords is one flat list. No concept_1_terms / concept_2_terms / nested concepts.
#   6. Gold references are NEVER read in this step.
#
# Expected input per project:
#   Projects/<project>/article_metadata.json
#   Projects/<project>/article_text.txt
#   Projects/<project>/section_files/*.txt      optional
#   Projects/<project>/references.json          optional, non-gold only
#
# Output per project:
#   Projects/<project>/config.json
#
# Environment variables:
#   LLM_API_KEY or DEEPSEEK_API_KEY
#   LLM_BASE_URL or DEEPSEEK_BASE_URL
#   LLM_MODEL
#   LLM_TEMPERATURE
#   LLM_TOP_P
#   LLM_MAX_TOKENS
#   LLM_REASONING_EFFORT

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


# ============================================================
# SETTINGS
# ============================================================

PROJECTS_DIR = Path("Projects")

FORCE_REBUILD = os.getenv("FORCE_REBUILD", "0").strip().lower() in {
    "1", "true", "yes", "y"
}

DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek").strip()

DEFAULT_BASE_URL = (
    os.getenv("LLM_BASE_URL", "").strip()
    or os.getenv("DEEPSEEK_BASE_URL", "").strip()
    or "https://api.deepseek.com"
)

DEFAULT_MODEL = os.getenv("LLM_MODEL", "deepseek-chat").strip()

DEFAULT_API_KEY = (
    os.getenv("LLM_API_KEY", "").strip()
    or os.getenv("DEEPSEEK_API_KEY", "").strip()
)

# Low temperature because this is extraction + conservative generation.
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
DEFAULT_TOP_P = float(os.getenv("LLM_TOP_P", "0.95"))

_raw_max_tokens = os.getenv("LLM_MAX_TOKENS", "").strip()
DEFAULT_MAX_TOKENS: Optional[int] = int(_raw_max_tokens) if _raw_max_tokens else 8192

DEFAULT_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "").strip()

REQUEST_TIMEOUT = int(os.getenv("LLM_REQUEST_TIMEOUT", "90"))
MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
RETRY_SLEEP = float(os.getenv("LLM_RETRY_SLEEP", "3.0"))

MAX_ARTICLE_CHARS = int(os.getenv("CONFIG_MAX_ARTICLE_CHARS", "45000"))
MAX_SECTION_CHARS = int(os.getenv("CONFIG_MAX_SECTION_CHARS", "12000"))
REPAIR_MAX_ARTICLE_CHARS = int(os.getenv("CONFIG_REPAIR_MAX_ARTICLE_CHARS", "22000"))


# ============================================================
# .ENV LOADING
# ============================================================

def load_dotenv(path: Path = Path(".env")) -> None:
    """Lightweight .env loader."""
    try:
        from dotenv import load_dotenv as _load_dotenv  # type: ignore
        _load_dotenv(path)
        return
    except Exception:
        pass

    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


def resolve_llm_settings() -> Dict[str, Any]:
    api_key = (
        os.getenv("LLM_API_KEY", "").strip()
        or os.getenv("DEEPSEEK_API_KEY", "").strip()
        or DEFAULT_API_KEY
    )

    base_url = (
        os.getenv("LLM_BASE_URL", "").strip()
        or os.getenv("DEEPSEEK_BASE_URL", "").strip()
        or DEFAULT_BASE_URL
    )

    model = os.getenv("LLM_MODEL", DEFAULT_MODEL).strip()
    temperature = float(os.getenv("LLM_TEMPERATURE", str(DEFAULT_TEMPERATURE)))
    top_p = float(os.getenv("LLM_TOP_P", str(DEFAULT_TOP_P)))

    raw_max_tokens = os.getenv("LLM_MAX_TOKENS", "").strip()
    max_tokens = int(raw_max_tokens) if raw_max_tokens else DEFAULT_MAX_TOKENS

    reasoning_effort = os.getenv(
        "LLM_REASONING_EFFORT", DEFAULT_REASONING_EFFORT
    ).strip()

    return {
        "provider": DEFAULT_PROVIDER,
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "reasoning_effort": reasoning_effort,
    }


# ============================================================
# BASIC HELPERS
# ============================================================

def clean_text(x: Any) -> str:
    if x is None:
        return ""
    text = str(x)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_spaces(x: Any) -> str:
    return re.sub(r"\s+", " ", clean_text(x)).strip()


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def read_json(path: Path, default: Any = None) -> Any:
    if default is None:
        default = {}
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def clip_middle(text: str, max_chars: int) -> str:
    text = clean_text(text)
    if len(text) <= max_chars:
        return text

    head = int(max_chars * 0.70)
    tail = max_chars - head
    return f"{text[:head]}\n\n[... TRUNCATED ...]\n\n{text[-tail:]}"


def discover_projects(projects_dir: Path, one_project: str = "") -> List[Path]:
    if one_project:
        p = Path(one_project)
        if not p.is_absolute():
            p = projects_dir / one_project
        if not p.exists() or not p.is_dir():
            raise FileNotFoundError(f"Project folder not found: {p}")
        return [p]

    if not projects_dir.exists():
        raise FileNotFoundError(f"Projects directory not found: {projects_dir.resolve()}")

    return [p for p in sorted(projects_dir.iterdir()) if p.is_dir()]


def normalize_string_list(x: Any) -> List[str]:
    """Return a de-duplicated list of clean strings."""
    if x is None:
        return []

    if isinstance(x, str):
        x = [x]

    if not isinstance(x, list):
        return []

    out: List[str] = []
    seen = set()
    for item in x:
        val = normalize_spaces(item)
        if not val:
            continue
        key = val.lower()
        if key not in seen:
            out.append(val)
            seen.add(key)
    return out


def normalize_int(x: Any, default: int = 0) -> int:
    try:
        if x is None or x == "":
            return default
        return int(x)
    except Exception:
        return default


def normalize_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "yes", "y"}
    return bool(x)


def extract_json_object(text: str) -> Dict[str, Any]:
    """Robust JSON extraction from model output."""
    text = clean_text(text)

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    if fence:
        text = fence.group(1)

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found in model output.")

    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(text)):
        ch = text[i]

        if esc:
            esc = False
            continue

        if ch == "\\":
            esc = True
            continue

        if ch == '"':
            in_str = not in_str
            continue

        if in_str:
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                block = text[start:i + 1]
                block = re.sub(r",\s*([}\]])", r"\1", block)
                obj = json.loads(block)
                if isinstance(obj, dict):
                    return obj
                raise ValueError("Extracted JSON is not an object.")

    raise ValueError("Could not find a balanced JSON object.")


# ============================================================
# PROJECT INPUT LOADING
# ============================================================

def load_article_metadata(project_dir: Path) -> Dict[str, Any]:
    candidates = [
        "article_metadata.json",
        "metadata.json",
        "config_evidence.json",
    ]

    for rel in candidates:
        p = project_dir / rel
        if not p.exists():
            continue

        data = read_json(p, default={}) or {}

        if "article_identity" in data and isinstance(data["article_identity"], dict):
            return data["article_identity"]

        return data

    return {}


def load_sections(project_dir: Path) -> Dict[str, str]:
    sections: Dict[str, str] = {}

    section_dir = project_dir / "section_files"
    for sec in [
        "abstract",
        "introduction",
        "methods",
        "results",
        "discussion",
        "conclusion",
        "limitations",
    ]:
        p = section_dir / f"{sec}.txt"
        txt = read_text(p)
        if txt:
            sections[sec] = clip_middle(txt, MAX_SECTION_CHARS)

    # Fallback: config_evidence.json
    if not sections:
        cfg_evidence = read_json(project_dir / "config_evidence.json", default={}) or {}
        core = cfg_evidence.get("core_sections", {}) or {}
        if isinstance(core, dict):
            for sec, txt in core.items():
                txt = clean_text(txt)
                if txt:
                    sections[sec] = clip_middle(txt, MAX_SECTION_CHARS)

    # Fallback: sections.json from older scripts
    if not sections:
        data = read_json(project_dir / "sections.json", default={}) or {}
        inner = data.get("sections", {}) if isinstance(data, dict) else {}
        if isinstance(inner, dict):
            for sec, txt in inner.items():
                txt = clean_text(txt)
                if txt:
                    sections[sec] = clip_middle(txt, MAX_SECTION_CHARS)

    return sections


def load_article_text(project_dir: Path) -> Tuple[str, str]:
    candidates = [
        "article_text.txt",
        "article_markdown.md",
        "Article_Text.txt",
        "COMBINED.txt",
        "full_text.txt",
        "section_files/all_sections_combined.txt",
    ]

    for rel in candidates:
        p = project_dir / rel
        txt = read_text(p)
        if txt:
            return clip_middle(txt, MAX_ARTICLE_CHARS), str(p)

    return "", ""


def load_non_gold_reference_counts(project_dir: Path) -> Dict[str, int]:
    """Return non-gold reference counts only. Do NOT read references_gold.json."""
    candidates = [
        "references.json",
        "references_resolved.json",
    ]

    for rel in candidates:
        p = project_dir / rel
        if not p.exists():
            continue

        data = read_json(p, default={}) or {}
        if not data:
            continue

        n_refs = (
            data.get("n_referenced_works")
            or data.get("n_references")
            or data.get("n_total")
            or len(data.get("references", []) or [])
            or len(data.get("works", []) or [])
            or len(data.get("dois", []) or [])
            or 0
        )

        n_with_doi = (
            data.get("n_with_doi")
            or len(data.get("dois", []) or [])
            or 0
        )

        return {
            "n_references": normalize_int(n_refs),
            "n_with_doi": normalize_int(n_with_doi),
        }

    return {"n_references": 0, "n_with_doi": 0}


def build_article_payload(project_dir: Path) -> Dict[str, Any]:
    metadata = load_article_metadata(project_dir)
    sections = load_sections(project_dir)
    article_text, article_text_path = load_article_text(project_dir)
    reference_counts = load_non_gold_reference_counts(project_dir)

    abstract = sections.get("abstract", "") or normalize_spaces(metadata.get("abstract", ""))
    methods = sections.get("methods", "")

    high_signal_parts = []
    if abstract:
        high_signal_parts.append(f"ABSTRACT:\n{abstract}")
    if methods:
        high_signal_parts.append(f"METHODS / SEARCH STRATEGY SECTION:\n{methods}")
    if article_text:
        high_signal_parts.append(f"FULL TEXT EXCERPT:\n{article_text}")

    composed_text = clip_middle("\n\n".join(high_signal_parts), MAX_ARTICLE_CHARS)

    return {
        "project_name": project_dir.name,
        "metadata": metadata,
        "sections": sections,
        "article_text": composed_text,
        "article_text_path": article_text_path,
        "reference_counts": reference_counts,
        "has_abstract": bool(abstract),
        "has_methods": bool(methods),
    }


def shrink_payload_for_retry(payload: Dict[str, Any]) -> Dict[str, Any]:
    smaller = dict(payload)
    smaller["article_text"] = clip_middle(
        clean_text(payload.get("article_text", "")),
        REPAIR_MAX_ARTICLE_CHARS,
    )
    return smaller


# ============================================================
# FINAL FLAT CONFIG SCHEMA
# ============================================================

def flat_config_schema() -> Dict[str, Any]:
    """Final single-level config returned by the LLM and saved as config.json."""
    return {
        "title": "",
        "review_type": "",
        "objective": "",
        "research_questions": [],

        "n_references": 0,
        "n_with_doi": 0,

        "base_reporting_framework": "PRISMA 2020 + PRISMA-S",

        "concepts_keywords": [],

        "prisma_eligibility_inclusion": [],
        "prisma_eligibility_exclusion": [],

        "prisma_information_sources_databases": [],
        "prisma_information_sources_other_sources": [],
        "prisma_date_last_searched": "",
        "prisma_date_coverage_start": "",
        "prisma_date_coverage_end": "",

        "prisma_title_abstract_include_keywords": [],
        "prisma_title_abstract_exclude_keywords": [],
        "prisma_title_abstract_must_have_keywords": [],
        "prisma_title_abstract_uncertain_keywords": [],

        "prisma_full_text_include_keywords": [],
        "prisma_full_text_exclude_keywords": [],
        "prisma_full_text_must_have_keywords": [],
        "prisma_full_text_uncertain_keywords": [],

        "prisma_quality_assessment_tool": "",
        "prisma_quality_assessment_keywords": [],
        "prisma_quality_assessment_questions": [],
        "prisma_risk_of_bias_reported": False,
        "prisma_flow_reported": False,

        "prisma_s_databases_searched": [],
        "prisma_s_platforms_or_interfaces": [],
        "prisma_s_database_coverage_dates": [],
        "prisma_s_supplementary_search_methods": [],

        "prisma_s_citation_searching": "",
        "prisma_s_contact_with_authors": "",
        "prisma_s_website_searching": "",
        "prisma_s_trial_registry_searching": "",
        "prisma_s_grey_literature_searching": "",

        "prisma_s_full_search_strings": [],
        "prisma_s_search_queries_by_database": [],
        "prisma_s_search_keywords": [],
        "prisma_s_controlled_vocabulary_terms": [],
        "prisma_s_free_text_terms": [],
        "prisma_s_boolean_operators_used": [],

        "prisma_s_language_limits": "",
        "prisma_s_date_limits": "",
        "prisma_s_publication_type_limits": "",
        "prisma_s_species_or_population_limits": "",
        "prisma_s_other_limits": [],

        "prisma_s_date_search_conducted": "",
        "prisma_s_date_search_updated": "",
        "prisma_s_deduplication_method": "",
        "prisma_s_search_records_exported": "",
        "prisma_s_search_documentation_notes": "",

        "generated_databases_to_search": [],
        "generated_date_start": "",
        "generated_date_end": "",
        "generated_language": "",

        "generated_search_queries": [],
        "generated_search_keywords": [],

        "generated_abstract_include_keywords": [],
        "generated_abstract_exclude_keywords": [],

        "generated_full_text_include_keywords": [],
        "generated_full_text_exclude_keywords": [],

        "generated_quality_assessment_keywords": [],
        "generated_quality_assessment_questions": [],
    }


LIST_FIELDS = {
    "research_questions",
    "concepts_keywords",
    "prisma_eligibility_inclusion",
    "prisma_eligibility_exclusion",
    "prisma_information_sources_databases",
    "prisma_information_sources_other_sources",
    "prisma_title_abstract_include_keywords",
    "prisma_title_abstract_exclude_keywords",
    "prisma_title_abstract_must_have_keywords",
    "prisma_title_abstract_uncertain_keywords",
    "prisma_full_text_include_keywords",
    "prisma_full_text_exclude_keywords",
    "prisma_full_text_must_have_keywords",
    "prisma_full_text_uncertain_keywords",
    "prisma_quality_assessment_keywords",
    "prisma_quality_assessment_questions",
    "prisma_s_databases_searched",
    "prisma_s_platforms_or_interfaces",
    "prisma_s_database_coverage_dates",
    "prisma_s_supplementary_search_methods",
    "prisma_s_full_search_strings",
    "prisma_s_search_queries_by_database",
    "prisma_s_search_keywords",
    "prisma_s_controlled_vocabulary_terms",
    "prisma_s_free_text_terms",
    "prisma_s_boolean_operators_used",
    "prisma_s_other_limits",
    "generated_databases_to_search",
    "generated_search_queries",
    "generated_search_keywords",
    "generated_abstract_include_keywords",
    "generated_abstract_exclude_keywords",
    "generated_full_text_include_keywords",
    "generated_full_text_exclude_keywords",
    "generated_quality_assessment_keywords",
    "generated_quality_assessment_questions",
}

STRING_FIELDS = {
    "title",
    "review_type",
    "objective",
    "base_reporting_framework",
    "prisma_date_last_searched",
    "prisma_date_coverage_start",
    "prisma_date_coverage_end",
    "prisma_quality_assessment_tool",
    "prisma_s_citation_searching",
    "prisma_s_contact_with_authors",
    "prisma_s_website_searching",
    "prisma_s_trial_registry_searching",
    "prisma_s_grey_literature_searching",
    "prisma_s_language_limits",
    "prisma_s_date_limits",
    "prisma_s_publication_type_limits",
    "prisma_s_species_or_population_limits",
    "prisma_s_date_search_conducted",
    "prisma_s_date_search_updated",
    "prisma_s_deduplication_method",
    "prisma_s_search_records_exported",
    "prisma_s_search_documentation_notes",
    "generated_date_start",
    "generated_date_end",
    "generated_language",
}

INT_FIELDS = {"n_references", "n_with_doi"}
BOOL_FIELDS = {"prisma_risk_of_bias_reported", "prisma_flow_reported"}


# ============================================================
# PROMPT
# ============================================================

def build_prompt(payload: Dict[str, Any], retry_note: str = "") -> List[Dict[str, str]]:
    metadata = payload.get("metadata", {}) or {}
    ref_counts = payload.get("reference_counts", {}) or {}

    compact_input = {
        "project_name": payload.get("project_name", ""),
        "metadata": {
            "title": normalize_spaces(metadata.get("title", "")),
            "authors": normalize_spaces(metadata.get("authors", "")),
            "year": normalize_spaces(metadata.get("year", "") or metadata.get("publication_year", "")),
            "journal": normalize_spaces(metadata.get("journal", "")),
            "doi": normalize_spaces(metadata.get("doi", "")),
            "abstract": normalize_spaces(metadata.get("abstract", "")),
        },
        "reference_counts_only_no_dois": {
            "n_references": normalize_int(ref_counts.get("n_references", 0)),
            "n_with_doi": normalize_int(ref_counts.get("n_with_doi", 0)),
        },
        "article_text": payload.get("article_text", ""),
    }

    schema = flat_config_schema()

    system = (
        "You extract a clean, flat PRISMA/PRISMA-S configuration from a review article. "
        "Return JSON only. Do not include markdown or commentary. "
        "The output must be a single-level JSON object with exactly the requested keys. "
        "Do not create nested objects. Do not create concept_1_terms or concept_2_terms. "
        "Use one flat list called concepts_keywords for all concept/topic keywords. "
        "Author-reported PRISMA and PRISMA-S fields must be copied verbatim or left empty. "
        "Generated fields must be conservative operational keywords/queries derived from title, abstract, objective, and methods. "
        "Never use references as an answer key. Reference counts may be copied only as counts."
    )

    schema_json = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    input_json = json.dumps(compact_input, ensure_ascii=False, separators=(",", ":"))
    retry_block = f"\nIMPORTANT RETRY NOTE: {retry_note}\n" if retry_note else ""

    user = f"""
Return JSON only using exactly this flat shape:

{schema_json}

Rules:
1. Use a single-level JSON object only. No nested dictionaries.
2. Keep every key in the schema. Do not add new keys.
3. title: copy article title from metadata/article.
4. review_type: copy explicit wording if present, such as survey, systematic review, scoping review, meta-analysis, literature review. Do not invent "narrative" unless stated.
5. objective: copy the aim/objective/purpose as closely as possible from the article.
6. research_questions: copy explicit research questions if stated; otherwise [].
7. n_references and n_with_doi: copy the counts from reference_counts_only_no_dois.
8. base_reporting_framework must be "PRISMA 2020 + PRISMA-S".
9. concepts_keywords: one flat list of all core concept/topic keywords. Do NOT output concept_1_terms, concept_2_terms, concept_3_terms, concept_4_terms, or nested concepts.
10. prisma_* fields are author-reported PRISMA fields only. If not reported, leave empty/false.
11. prisma_s_* fields are author-reported PRISMA-S/search-method fields only. If not reported, leave empty.
12. For reported databases, search dates, search strings, eligibility criteria, language limits, quality/risk-of-bias methods, and PRISMA flow: only copy if explicitly stated.
13. generated_* fields are pipeline-generated fallback fields. Generate useful conservative search queries, search keywords, abstract-screening keywords, full-text-screening keywords, and quality-assessment keywords/questions from the title/abstract/objective/methods.
14. generated_databases_to_search should be broad and practical, e.g. OpenAlex, Crossref, Semantic Scholar, Europe PMC, PubMed, unless a domain clearly suggests otherwise.
15. generated_quality_assessment_questions should be short ranking questions, for example whether the article reports search sources, search strings, eligibility criteria, screening process, synthesis structure, taxonomy, comparison, limitations, and reproducibility details.
16. Boolean fields must be true/false only.
17. List fields must be arrays of strings only.
18. Do not output prose outside JSON.
19. The JSON must be complete, parseable, and end with the final closing brace.
{retry_block}

ARTICLE INPUT:
{input_json}
""".strip()

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ============================================================
# LLM CALL
# ============================================================

def call_llm(settings: Dict[str, Any], messages: List[Dict[str, str]]) -> str:
    api_key = settings.get("api_key", "")
    if not api_key:
        raise RuntimeError("Missing API key. Set LLM_API_KEY or DEEPSEEK_API_KEY.")

    base_url = settings["base_url"].rstrip("/")
    url = f"{base_url}/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload: Dict[str, Any] = {
        "model": settings["model"],
        "messages": messages,
        "temperature": settings["temperature"],
        "top_p": settings["top_p"],
        "response_format": {"type": "json_object"},
    }

    if settings.get("max_tokens"):
        payload["max_tokens"] = settings["max_tokens"]

    if settings.get("reasoning_effort"):
        payload["reasoning_effort"] = settings["reasoning_effort"]

    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )

            # Some OpenAI-compatible endpoints reject reasoning_effort.
            if r.status_code in {400, 422} and "reasoning_effort" in payload:
                payload.pop("reasoning_effort", None)
                r = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=REQUEST_TIMEOUT,
                )

            # Some endpoints reject response_format.
            if r.status_code in {400, 422} and "response_format" in payload:
                payload.pop("response_format", None)
                r = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=REQUEST_TIMEOUT,
                )

            r.raise_for_status()
            data = r.json()
            choices = data.get("choices", [])
            if not choices:
                raise RuntimeError(f"No choices in response: {data}")

            content = choices[0].get("message", {}).get("content", "")
            if not content:
                raise RuntimeError(f"Empty model output: {data}")

            return content

        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP * attempt)
            else:
                raise

    raise last_error or RuntimeError("Unknown LLM error.")


# ============================================================
# CONFIG NORMALIZATION / VALIDATION
# ============================================================

def normalize_flat_config(raw: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    """Force the model output into the exact flat schema."""
    schema = flat_config_schema()
    metadata = payload.get("metadata", {}) or {}
    ref_counts = payload.get("reference_counts", {}) or {}

    config: Dict[str, Any] = {}

    for key, default_value in schema.items():
        value = raw.get(key, default_value)

        if key in LIST_FIELDS:
            config[key] = normalize_string_list(value)
        elif key in STRING_FIELDS:
            config[key] = normalize_spaces(value)
        elif key in INT_FIELDS:
            config[key] = normalize_int(value)
        elif key in BOOL_FIELDS:
            config[key] = normalize_bool(value)
        else:
            config[key] = value

    # Enforce fields that should come from local non-gold metadata/counts.
    if not config["title"]:
        config["title"] = normalize_spaces(metadata.get("title", ""))

    config["n_references"] = normalize_int(ref_counts.get("n_references", config["n_references"]))
    config["n_with_doi"] = normalize_int(ref_counts.get("n_with_doi", config["n_with_doi"]))
    config["base_reporting_framework"] = "PRISMA 2020 + PRISMA-S"

    # Safety: remove any accidental concept placeholder terms.
    config["concepts_keywords"] = [
        x for x in config["concepts_keywords"]
        if x.lower() not in {"concept_1", "concept_2", "concept_3", "concept_4"}
    ]

    return config


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    schema = flat_config_schema()

    missing_keys = [k for k in schema if k not in config]
    extra_keys = [k for k in config if k not in schema]

    type_warnings = []

    for key in LIST_FIELDS:
        if key in config and not isinstance(config[key], list):
            type_warnings.append(f"{key} should be a list")

    for key in STRING_FIELDS:
        if key in config and not isinstance(config[key], str):
            type_warnings.append(f"{key} should be a string")

    for key in INT_FIELDS:
        if key in config and not isinstance(config[key], int):
            type_warnings.append(f"{key} should be an integer")

    for key in BOOL_FIELDS:
        if key in config and not isinstance(config[key], bool):
            type_warnings.append(f"{key} should be a boolean")

    useful_generated_fields = [
        "generated_search_queries",
        "generated_search_keywords",
        "generated_abstract_include_keywords",
        "generated_full_text_include_keywords",
        "generated_quality_assessment_questions",
    ]

    empty_useful_generated_fields = [
        k for k in useful_generated_fields
        if not config.get(k)
    ]

    warnings = []
    if extra_keys:
        warnings.append("Model returned extra keys; they were removed during normalization.")
    if missing_keys:
        warnings.append("Some schema keys were missing; defaults were inserted.")
    if empty_useful_generated_fields:
        warnings.append(
            "Some generated pipeline fields are empty: "
            + ", ".join(empty_useful_generated_fields)
        )

    reported_prisma_nonempty = [
        k for k, v in config.items()
        if k.startswith("prisma_") and k not in {"prisma_risk_of_bias_reported", "prisma_flow_reported"} and not is_empty_for_report(v)
    ]

    reported_prisma_s_nonempty = [
        k for k, v in config.items()
        if k.startswith("prisma_s_") and not is_empty_for_report(v)
    ]

    return {
        "schema_validation_passed": not missing_keys and not type_warnings,
        "n_fields": len(schema),
        "n_missing_keys": len(missing_keys),
        "missing_keys": missing_keys,
        "n_extra_keys_removed": len(extra_keys),
        "extra_keys_removed": extra_keys,
        "type_warnings": type_warnings,
        "reported_field_summary": {
            "n_nonempty_prisma_fields": len(reported_prisma_nonempty),
            "n_nonempty_prisma_s_fields": len(reported_prisma_s_nonempty),
            "nonempty_prisma_fields": reported_prisma_nonempty,
            "nonempty_prisma_s_fields": reported_prisma_s_nonempty,
        },
        "gold_reference_contamination_check": {
            "gold_references_read": False,
            "gold_references_sent_to_model": False,
            "passed": True,
        },
        "warnings": warnings,
    }


def is_empty_for_report(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, str):
        return x.strip() == ""
    if isinstance(x, list):
        return len(x) == 0
    if isinstance(x, dict):
        return len(x) == 0
    if isinstance(x, bool):
        return not x
    if isinstance(x, (int, float)):
        return x == 0
    return False


# ============================================================
# PROJECT PROCESSING
# ============================================================

def process_project(
    project_dir: Path,
    settings: Dict[str, Any],
    force: bool = False,
) -> Dict[str, Any]:
    config_path = project_dir / "config.json"

    if config_path.exists() and not force:
        return {
            "project": project_dir.name,
            "status": "skipped_existing_config",
            "config_path": str(config_path),
        }

    payload = build_article_payload(project_dir)

    if not payload.get("article_text") and not payload.get("metadata"):
        return {
            "project": project_dir.name,
            "status": "failed_no_input",
            "error": "No article text or metadata found.",
        }

    messages = build_prompt(payload)

    try:
        raw_response = call_llm(settings=settings, messages=messages)
        retry_warnings: List[str] = []
        try:
            raw_extracted = extract_json_object(raw_response)
            payload_for_config = payload
        except ValueError as parse_error:
            retry_warnings.append(
                f"First model output was not complete JSON ({parse_error}); retried with shorter prompt."
            )
            smaller_payload = shrink_payload_for_retry(payload)
            retry_messages = build_prompt(
                smaller_payload,
                retry_note=(
                    "The previous response was not parseable JSON. Return one complete JSON object only. "
                    "Keep values concise and close the final brace."
                ),
            )
            raw_response = call_llm(settings=settings, messages=retry_messages)
            raw_extracted = extract_json_object(raw_response)
            payload_for_config = smaller_payload

        config = normalize_flat_config(raw_extracted, payload_for_config)
        quality = validate_config(config)
        if retry_warnings:
            quality.setdefault("warnings", [])
            quality["warnings"] = retry_warnings + list(quality.get("warnings", []))

        write_json(config_path, config)

        return {
            "project": project_dir.name,
            "status": "success",
            "config_path": str(config_path),
            "schema_validation_passed": quality.get("schema_validation_passed", False),
            "warnings": quality.get("warnings", []),
            "reported_field_summary": quality.get("reported_field_summary", {}),
        }

    except Exception as e:
        return {
            "project": project_dir.name,
            "status": "failed",
            "error": f"{type(e).__name__}: {e}",
        }


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    global MAX_ARTICLE_CHARS, REPAIR_MAX_ARTICLE_CHARS, REQUEST_TIMEOUT

    load_dotenv()
    settings = resolve_llm_settings()

    parser = argparse.ArgumentParser(
        description="Generate clean flat PRISMA/PRISMA-S configuration files."
    )

    parser.add_argument(
        "--projects-dir",
        default=str(PROJECTS_DIR),
        help="Path to Projects directory.",
    )

    parser.add_argument(
        "--project",
        default="",
        help="Optional single project folder name or path.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing config.json files.",
    )

    parser.add_argument(
        "--max-article-chars",
        type=int,
        default=MAX_ARTICLE_CHARS,
        help="Maximum article characters sent to the model.",
    )

    parser.add_argument(
        "--repair-max-article-chars",
        type=int,
        default=REPAIR_MAX_ARTICLE_CHARS,
        help="Maximum article characters sent on JSON-repair retry.",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="Override model output max_tokens. Use 0 for environment/default.",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=REQUEST_TIMEOUT,
        help="LLM request read timeout in seconds.",
    )

    args = parser.parse_args()

    MAX_ARTICLE_CHARS = max(5_000, int(args.max_article_chars))
    REPAIR_MAX_ARTICLE_CHARS = max(5_000, int(args.repair_max_article_chars))
    REQUEST_TIMEOUT = max(30, int(args.timeout))
    if args.max_tokens > 0:
        settings["max_tokens"] = int(args.max_tokens)

    if not settings.get("api_key"):
        print(
            "\nERROR: Missing API key.\n"
            "Set LLM_API_KEY or DEEPSEEK_API_KEY in .env or environment variables.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    projects_dir = Path(args.projects_dir)
    projects = discover_projects(projects_dir, args.project)

    print("=" * 80)
    print("STEP 9A: GENERATE FLAT CONFIGURATION")
    print("=" * 80)
    print(f"Projects directory : {projects_dir.resolve()}")
    print(f"Projects found     : {len(projects)}")
    print(f"Provider           : {settings.get('provider')}")
    print(f"Base URL           : {settings.get('base_url')}")
    print(f"Model              : {settings.get('model')}")
    print(f"Temperature        : {settings.get('temperature')}")
    print(f"Top P              : {settings.get('top_p')}")
    print(f"Max tokens         : {settings.get('max_tokens')}")
    print(f"Max article chars  : {MAX_ARTICLE_CHARS}")
    print(f"Repair chars       : {REPAIR_MAX_ARTICLE_CHARS}")
    print(f"Request timeout    : {REQUEST_TIMEOUT}s")
    print(f"Reasoning effort   : {settings.get('reasoning_effort')}")
    print(f"Force rebuild      : {args.force or FORCE_REBUILD}")
    print("Schema             : flat PRISMA 2020 + PRISMA-S")
    print("Concepts           : concepts_keywords only")
    print("Gold references    : NEVER READ IN THIS STEP")
    print("=" * 80)

    results = []

    for i, project_dir in enumerate(projects, start=1):
        print(f"\n[{i}/{len(projects)}] Processing: {project_dir.name}")

        result = process_project(
            project_dir=project_dir,
            settings=settings,
            force=args.force or FORCE_REBUILD,
        )

        results.append(result)
        print(f"Status: {result.get('status')}")

        if result.get("schema_validation_passed") is not None:
            print(f"Schema validation passed: {result.get('schema_validation_passed')}")

        summary = result.get("reported_field_summary", {}) or {}
        if summary:
            print(
                f"Non-empty PRISMA fields: {summary.get('n_nonempty_prisma_fields', 0)}, "
                f"Non-empty PRISMA-S fields: {summary.get('n_nonempty_prisma_s_fields', 0)}"
            )

        for warning in result.get("warnings", []) or []:
            print(f"WARNING: {warning}")

        if result.get("error"):
            print(f"ERROR: {result.get('error')}")

    n_success = sum(1 for r in results if r.get("status") == "success")
    n_failed = sum(1 for r in results if r.get("status") == "failed")
    n_skipped = sum(1 for r in results if r.get("status") == "skipped_existing_config")

    print("\n" + "=" * 80)
    print("CONFIGURATION GENERATION COMPLETE")
    print("=" * 80)
    print(f"Success : {n_success}")
    print(f"Failed  : {n_failed}")
    print(f"Skipped : {n_skipped}")


if __name__ == "__main__":
    main()
