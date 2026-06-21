#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================================
# Step19-GenerateDiagrams.py
# ============================================================
#
# Purpose:
#   Generate local diagrams/tables/JSON summaries for one or more
#   Step16 projects.
#
# Supports:
#   python Step19-GenerateDiagrams.py 1
#   python Step19-GenerateDiagrams.py --project-index 1
#   python Step19-GenerateDiagrams.py 1 --max-articles 50
#   python Step19-GenerateDiagrams.py --all-projects --max-articles 50
#
# Optional DOI metadata:
#   python Step19-GenerateDiagrams.py 1 --fetch-doi-metadata --mailto your@email.com
#
# Input expected:
#   Projects/<project>/Extractions/Step16_Qwen__Qwen2.5-3B-Instruct/article_json/*.json
#
# Output:
#   Projects/<project>/ReviewAssets/
#       figures/*.png
#       tables/*.csv
#       json/*.json
#
# ============================================================

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import textwrap
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
except Exception as e:
    raise SystemExit(
        "\nERROR: matplotlib is not available in this Python environment.\n"
        "Install it inside the active conda env:\n\n"
        "  conda activate aitools\n"
        "  pip install matplotlib\n\n"
        f"Original error: {repr(e)}\n"
    )

try:
    import requests
except Exception:
    requests = None


# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_PROJECTS_DIR = "Projects"
DEFAULT_MODEL_FOLDER_NAME = "Qwen__Qwen2.5-3B-Instruct"
DEFAULT_TOP_N = 200
DEFAULT_ABSTRACT_SCREENING_POOL = 1000

FIG_DPI = 220
MAX_LABEL_LEN = 55


# ============================================================
# BASIC HELPERS
# ============================================================

def clean_text(x: Any) -> str:
    if x is None:
        return ""
    s = str(x)
    if s.lower().strip() in {"nan", "none", "null", "na", "n/a"}:
        return ""
    s = s.replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def safe_model_name(model_folder_name: str) -> str:
    x = clean_text(model_folder_name)
    x = x.replace("/", "__")
    x = re.sub(r"[^A-Za-z0-9_.-]+", "_", x)
    return x.strip("_") or "model"


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


def read_json(path: Path, default: Any = None) -> Any:
    if default is None:
        default = {}
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return default


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if fields is None:
        fields = []
        seen = set()
        for row in rows:
            for k in row.keys():
                if k not in seen:
                    seen.add(k)
                    fields.append(k)

    with path.open("w", encoding="utf-8", newline="", errors="ignore") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0

    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            return sum(1 for _ in csv.DictReader(f))
    except Exception:
        return 0


def count_first_available_csv(project_dir: Path, relative_paths: List[str]) -> Tuple[int, str]:
    for rel in relative_paths:
        path = project_dir / rel
        n = count_csv_rows(path)
        if n > 0:
            return n, rel
    return 0, ""


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


def shorten_label(x: Any, max_len: int = MAX_LABEL_LEN) -> str:
    s = clean_text(x)
    if len(s) <= max_len:
        return s
    return s[: max_len - 3].rstrip() + "..."


def flatten_text_values(x: Any) -> List[str]:
    out: List[str] = []

    if x is None:
        return out

    if isinstance(x, str):
        s = clean_text(x)
        if s:
            out.append(s)
        return out

    if isinstance(x, (int, float, bool)):
        s = clean_text(x)
        if s:
            out.append(s)
        return out

    if isinstance(x, list):
        for item in x:
            out.extend(flatten_text_values(item))
        return out

    if isinstance(x, dict):
        for v in x.values():
            out.extend(flatten_text_values(v))
        return out

    s = clean_text(x)
    if s:
        out.append(s)

    return out


def unique_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []

    for item in items:
        s = clean_text(item)
        if not s:
            continue

        k = s.lower()
        if k in seen:
            continue

        seen.add(k)
        out.append(s)

    return out


def split_semicolon_or_comma(items: List[str]) -> List[str]:
    out = []
    for item in items:
        s = clean_text(item)
        if not s:
            continue

        parts = re.split(r";|\||\n", s)

        for p in parts:
            p = clean_text(p)
            if p:
                out.append(p)

    return unique_keep_order(out)


# ============================================================
# PROJECT DISCOVERY
# ============================================================

def discover_projects(projects_dir: Path) -> List[Path]:
    if not projects_dir.exists():
        return []

    projects = []

    for p in sorted(projects_dir.iterdir()):
        if not p.is_dir():
            continue

        if p.name.lower() in {"rankings", "search", "evaluation", "__pycache__", "extra"}:
            continue

        if (p / "config.json").exists():
            projects.append(p)

    return projects


def select_projects(args: argparse.Namespace) -> List[Path]:
    projects_dir = Path(args.projects_dir)
    projects = discover_projects(projects_dir)

    project_index = args.project_index

    if project_index <= 0 and args.positional_project:
        project_index = safe_int(args.positional_project, 0)

    if args.all_projects:
        return projects

    if project_index > 0:
        idx = project_index - 1
        if idx < 0 or idx >= len(projects):
            raise SystemExit(f"ERROR: invalid project index {project_index}. Found {len(projects)} projects.")
        return [projects[idx]]

    if args.project_name:
        exact = [p for p in projects if p.name == args.project_name]
        if exact:
            return [exact[0]]

        partial = [p for p in projects if args.project_name.lower() in p.name.lower()]
        if partial:
            return [partial[0]]

        raise SystemExit(f"ERROR: project not found: {args.project_name}")

    raise SystemExit(
        "ERROR: provide project index/name.\n\n"
        "Examples:\n"
        "  python Step19-GenerateDiagrams.py 1\n"
        "  python Step19-GenerateDiagrams.py --project-index 1\n"
        "  python Step19-GenerateDiagrams.py --all-projects\n"
    )


# ============================================================
# LOAD STEP16 ARTICLE CARDS
# ============================================================

def article_json_dir(project_dir: Path, model_folder_name: str) -> Path:
    return project_dir / "Extractions" / f"Step16_{safe_model_name(model_folder_name)}" / "article_json"


def load_article_cards(project_dir: Path, model_folder_name: str, max_articles: int = 0) -> List[Dict[str, Any]]:
    d = article_json_dir(project_dir, model_folder_name)
    files = sorted(d.glob("global_*_article_*.json"))

    cards = []

    for f in files:
        rec = read_json(f, {})
        if not rec:
            continue

        rec["_source_json"] = str(f)
        cards.append(rec)

    cards.sort(
        key=lambda r: (
            safe_int(r.get("article_index", 999999), 999999),
            safe_int(r.get("global_index", 999999), 999999),
        )
    )

    if max_articles > 0:
        cards = cards[:max_articles]

    return cards


def get_final_extraction(card: Dict[str, Any]) -> Dict[str, Any]:
    x = card.get("final_extraction", {})
    return x if isinstance(x, dict) else {}


def get_final_screening(card: Dict[str, Any]) -> Dict[str, Any]:
    x = card.get("final_screening", {})
    return x if isinstance(x, dict) else {}


def get_final_card(card: Dict[str, Any]) -> Dict[str, Any]:
    x = card.get("final_synthesis_card", {})
    return x if isinstance(x, dict) else {}


def get_methods(card: Dict[str, Any]) -> List[str]:
    ext = get_final_extraction(card)
    vals = flatten_text_values(ext.get("methods_models_tools", []))
    return split_semicolon_or_comma(vals)


def get_datasets(card: Dict[str, Any]) -> List[str]:
    ext = get_final_extraction(card)
    vals = flatten_text_values(ext.get("datasets_or_data_sources", []))
    return split_semicolon_or_comma(vals)


def get_metrics(card: Dict[str, Any]) -> List[str]:
    ext = get_final_extraction(card)
    vals = flatten_text_values(ext.get("evaluation_metrics", []))
    return split_semicolon_or_comma(vals)


def get_findings(card: Dict[str, Any]) -> List[str]:
    ext = get_final_extraction(card)
    vals = flatten_text_values(ext.get("main_findings", []))
    return unique_keep_order(vals)


def get_limitations(card: Dict[str, Any]) -> List[str]:
    ext = get_final_extraction(card)
    vals = flatten_text_values(ext.get("limitations_or_future_work", []))
    return unique_keep_order(vals)


# ============================================================
# CONFIG EXTRACTION
# ============================================================

def config_values(config: Dict[str, Any], keys: List[str]) -> List[str]:
    vals = []
    for key in keys:
        vals.extend(flatten_text_values(config.get(key, [])))
    return unique_keep_order(vals)


def extract_config_keywords(config: Dict[str, Any]) -> List[str]:
    keys = [
        "concepts_keywords",
        "keywords",
        "main_concepts",
        "review_concepts",
        "generated_search_keywords",
        "prisma_s_search_keywords",
        "prisma_s_free_text_terms",
        "prisma_s_controlled_vocabulary_terms",
        "generated_abstract_include_keywords",
        "generated_full_text_include_keywords",
        "prisma_title_abstract_include_keywords",
        "prisma_full_text_include_keywords",
        "prisma_title_abstract_must_have_keywords",
        "prisma_full_text_must_have_keywords",
    ]

    vals = config_values(config, keys)

    cleaned = []
    for v in vals:
        parts = re.split(r";|,|\||\n", v)
        for p in parts:
            p = clean_text(p)
            if len(p) >= 3:
                cleaned.append(p)

    return unique_keep_order(cleaned)


def extract_search_databases(config: Dict[str, Any]) -> List[str]:
    keys = [
        "databases",
        "database",
        "search_databases",
        "sources",
        "source_databases",
        "prisma_information_sources_databases",
        "prisma_s_databases_searched",
        "prisma_s_databases",
        "prisma_s_information_sources",
        "generated_databases",
        "generated_databases_to_search",
        "generated_sources",
        "search_sources",
    ]

    vals = config_values(config, keys)

    cleaned = []
    for v in vals:
        parts = re.split(r";|,|\||\n", v)
        for p in parts:
            p = clean_text(p)
            if p:
                cleaned.append(p)

    return unique_keep_order(cleaned)


def extract_research_questions(config: Dict[str, Any]) -> List[str]:
    keys = [
        "research_questions",
        "review_questions",
        "rqs",
        "generated_research_questions",
        "prisma_research_questions",
    ]

    rqs = config_values(config, keys)

    if not rqs:
        rqs = [
            "RQ1: What study topics, methods, or article types are represented?",
            "RQ2: What evidence, findings, datasets, tools, or outcomes are reported?",
            "RQ3: What limitations, gaps, or quality issues are identified?",
        ]

    return rqs


def extract_title_objective(config: Dict[str, Any]) -> Dict[str, str]:
    title = "; ".join(config_values(config, ["title", "review_title"]))[:500]
    objective = "; ".join(config_values(config, ["objective", "aim", "review_objective"]))[:1000]
    review_type = "; ".join(config_values(config, ["review_type", "type"]))[:300]

    return {
        "title": title,
        "objective": objective,
        "review_type": review_type,
    }


# ============================================================
# DOI / OPENALEX
# ============================================================

def openalex_cache_path(project_dir: Path) -> Path:
    return project_dir / "ReviewAssets" / "json" / "doi_openalex_cache.json"


def load_openalex_cache(project_dir: Path) -> Dict[str, Any]:
    return read_json(openalex_cache_path(project_dir), {})


def save_openalex_cache(project_dir: Path, cache: Dict[str, Any]) -> None:
    write_json(openalex_cache_path(project_dir), cache)


def fetch_openalex_by_doi(doi: str, mailto: str = "", sleep_seconds: float = 0.15) -> Optional[Dict[str, Any]]:
    if requests is None:
        return None

    doi = normalize_doi(doi)
    if not doi:
        return None

    url = f"https://api.openalex.org/works/https://doi.org/{doi}"
    params = {}
    if mailto:
        params["mailto"] = mailto

    try:
        r = requests.get(url, params=params, timeout=25)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        time.sleep(max(0.0, sleep_seconds))
        return r.json()
    except Exception:
        return None


def enrich_cards_with_openalex(
    project_dir: Path,
    cards: List[Dict[str, Any]],
    fetch: bool,
    mailto: str,
    max_api_calls: int,
) -> Dict[str, Any]:

    cache = load_openalex_cache(project_dir)
    calls = 0

    for card in cards:
        doi = normalize_doi(card.get("doi", ""))

        if not doi:
            continue

        if doi in cache:
            continue

        if not fetch:
            continue

        if max_api_calls > 0 and calls >= max_api_calls:
            break

        print(f"  OpenAlex DOI fetch: {doi}", flush=True)
        meta = fetch_openalex_by_doi(doi, mailto=mailto)
        cache[doi] = meta if isinstance(meta, dict) else {}
        calls += 1

    save_openalex_cache(project_dir, cache)
    return cache


def extract_year(card: Dict[str, Any], cache: Dict[str, Any]) -> int:
    year = safe_int(card.get("year", ""), 0)

    if year > 1500:
        return year

    doi = normalize_doi(card.get("doi", ""))

    if doi and isinstance(cache.get(doi), dict):
        y = safe_int(cache[doi].get("publication_year", ""), 0)
        if y > 1500:
            return y

    return 0


def extract_country_counts(cards: List[Dict[str, Any]], cache: Dict[str, Any]) -> Counter:
    counts = Counter()

    for card in cards:
        doi = normalize_doi(card.get("doi", ""))
        if not doi:
            continue

        meta = cache.get(doi, {})
        if not isinstance(meta, dict) or not meta:
            continue

        article_countries = set()

        authorships = meta.get("authorships", [])
        if not isinstance(authorships, list):
            continue

        for auth in authorships:
            if not isinstance(auth, dict):
                continue

            institutions = auth.get("institutions", [])
            if not isinstance(institutions, list):
                continue

            for inst in institutions:
                if not isinstance(inst, dict):
                    continue

                country = clean_text(inst.get("country_code", "")) or clean_text(inst.get("country", ""))
                if country:
                    article_countries.add(country)

        for country in article_countries:
            counts[country] += 1

    return counts


def extract_source_counts_from_openalex(cards: List[Dict[str, Any]], cache: Dict[str, Any]) -> Counter:
    counts = Counter()

    for card in cards:
        journal = clean_text(card.get("journal", ""))
        if journal:
            counts[journal] += 1
            continue

        doi = normalize_doi(card.get("doi", ""))
        if doi and isinstance(cache.get(doi), dict):
            meta = cache[doi]
            primary_location = meta.get("primary_location", {})
            if isinstance(primary_location, dict):
                source = primary_location.get("source", {})
                if isinstance(source, dict):
                    name = clean_text(source.get("display_name", ""))
                    if name:
                        counts[name] += 1

    return counts


# ============================================================
# QUALITY HEURISTICS
# ============================================================

def classify_quality(card: Dict[str, Any]) -> Dict[str, Any]:
    ext = get_final_extraction(card)

    methods = get_methods(card)
    datasets = get_datasets(card)
    metrics = get_metrics(card)
    findings = get_findings(card)
    limitations = get_limitations(card)

    aim = clean_text(ext.get("aim_or_purpose", ""))
    sample = clean_text(ext.get("sample_size_or_scale", ""))
    validation = clean_text(ext.get("validation_or_evaluation", ""))
    relevance = clean_text(ext.get("relevance_to_review", ""))

    bad = {"", "not_reported", "uncertain", "not reported"}

    criteria = {
        "aim_reported": aim.lower() not in bad,
        "methods_reported": len(methods) > 0,
        "data_sources_reported": len(datasets) > 0,
        "sample_size_reported": sample.lower() not in bad,
        "validation_reported": validation.lower() not in bad,
        "metrics_reported": len(metrics) > 0,
        "main_findings_reported": len(findings) > 0,
        "limitations_reported": len(limitations) > 0,
        "relevance_reported": relevance.lower() not in bad,
    }

    score = sum(1 for v in criteria.values() if v)
    max_score = len(criteria)
    pct = score / max_score if max_score else 0.0

    if pct >= 0.75:
        category = "high"
    elif pct >= 0.45:
        category = "medium"
    else:
        category = "low"

    return {
        "criteria": criteria,
        "quality_score": score,
        "quality_max": max_score,
        "quality_percent": round(pct * 100, 2),
        "quality_category": category,
    }


# ============================================================
# RQ MAPPING
# ============================================================

def map_card_to_rqs(card: Dict[str, Any], rqs: List[str]) -> List[str]:
    article_parts = [
        clean_text(card.get("title", "")),
        clean_text(card.get("one_sentence_summary", "")),
    ]

    article_parts.extend(flatten_text_values(card.get("final_key_points", [])))

    ext = get_final_extraction(card)
    for key in [
        "aim_or_purpose",
        "study_domain_or_context",
        "methods_models_tools",
        "datasets_or_data_sources",
        "outcomes_or_targets",
        "validation_or_evaluation",
        "evaluation_metrics",
        "main_findings",
        "limitations_or_future_work",
        "relevance_to_review",
        "quality_evidence",
    ]:
        article_parts.extend(flatten_text_values(ext.get(key, [])))

    article_text = " ".join(article_parts).lower()
    article_terms = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]{2,}", article_text))

    stop = {
        "what", "which", "how", "are", "the", "and", "for", "with", "from",
        "this", "that", "were", "was", "review", "study", "studies",
        "research", "question", "reported", "identified", "does", "use",
        "using", "based", "between", "into", "large", "small",
    }

    mapped = []

    for i, rq in enumerate(rqs, start=1):
        rq_terms = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]{2,}", rq.lower()))
        rq_terms = {t for t in rq_terms if t not in stop}

        overlap = len(article_terms & rq_terms)

        if overlap >= 2:
            mapped.append(f"RQ{i}")

    if not mapped:
        mapped.append("RQ1")

    return mapped


# ============================================================
# PLOTS
# ============================================================

def save_bar_plot(
    counts: Counter,
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str = "Count",
    top_n: int = 20,
    horizontal: bool = False,
) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)

    items = [(k, v) for k, v in counts.items() if clean_text(k) and v > 0]
    items.sort(key=lambda x: (-x[1], str(x[0])))

    if top_n > 0:
        items = items[:top_n]

    if horizontal:
        fig_h = max(4.8, min(16, 0.45 * max(1, len(items)) + 2.5))
        plt.figure(figsize=(11, fig_h))
    else:
        plt.figure(figsize=(10, 5.8))

    if not items:
        plt.text(0.5, 0.5, "No data available", ha="center", va="center")
        plt.axis("off")
    else:
        labels = [shorten_label(k, 60 if horizontal else 28) for k, _ in items]
        values = [v for _, v in items]

        if horizontal:
            y_pos = list(range(len(labels)))
            plt.barh(y_pos, values)
            plt.yticks(y_pos, labels)
            plt.gca().invert_yaxis()
            plt.xlabel(ylabel)
            plt.ylabel(xlabel)

            for i, v in enumerate(values):
                plt.text(v, i, f" {v}", va="center", fontsize=8)
        else:
            plt.bar(labels, values)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.xticks(rotation=45, ha="right")

            for i, v in enumerate(values):
                plt.text(i, v, str(v), ha="center", va="bottom", fontsize=8)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


def save_publications_per_year_plot(year_counts: Counter, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    years = sorted([y for y in year_counts if isinstance(y, int) and y > 1500])
    values = [year_counts[y] for y in years]

    plt.figure(figsize=(10, 5.5))

    if not years:
        plt.text(0.5, 0.5, "No year data available", ha="center", va="center")
        plt.axis("off")
    else:
        plt.plot(years, values, marker="o")
        plt.xlabel("Publication year")
        plt.ylabel("Article count")
        plt.xticks(years, rotation=45)

        for x, y in zip(years, values):
            plt.text(x, y, str(y), ha="center", va="bottom", fontsize=8)

    plt.title("Publications per year")
    plt.tight_layout()
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


def save_prisma_flow(counts: Dict[str, int], path: Path, project_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    boxes = [
        ("Ranked records available", counts.get("records_ranked_or_available", 0)),
        ("Articles selected for diagram set", counts.get("selected_for_assets", 0)),
        ("Full text available", counts.get("fulltext_available", 0)),
        ("Abstract / metadata only", counts.get("abstract_only", 0)),
        ("Included", counts.get("included", 0)),
        ("Excluded", counts.get("excluded", 0)),
        ("Uncertain", counts.get("uncertain", 0)),
    ]

    plt.figure(figsize=(10, 12))
    ax = plt.gca()
    ax.axis("off")

    y_positions = [0.92, 0.78, 0.64, 0.50, 0.34, 0.20, 0.06]
    x = 0.5
    box_w = 0.74
    box_h = 0.085

    for idx, ((label, value), y) in enumerate(zip(boxes, y_positions)):
        rect = plt.Rectangle(
            (x - box_w / 2, y - box_h / 2),
            box_w,
            box_h,
            fill=False,
            linewidth=1.6,
        )
        ax.add_patch(rect)

        ax.text(
            x,
            y,
            f"{label}\nN = {value}",
            ha="center",
            va="center",
            fontsize=11,
        )

        if idx < len(y_positions) - 1:
            y_next = y_positions[idx + 1]
            ax.annotate(
                "",
                xy=(x, y_next + box_h / 2),
                xytext=(x, y - box_h / 2),
                arrowprops=dict(arrowstyle="->", linewidth=1.3),
            )

    ax.text(
        0.5,
        0.985,
        f"PRISMA-style automated screening flow\n{project_name}",
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


def build_figure1_flow_counts(
    project_dir: Path,
    args: argparse.Namespace,
    cards: List[Dict[str, Any]],
    all_available_cards: List[Dict[str, Any]],
    decision_counts: Counter,
    config_keywords: List[str],
    search_databases: List[str],
    project_info: Dict[str, str],
) -> Dict[str, Any]:
    downloaded_records, downloaded_source = count_first_available_csv(
        project_dir,
        [
            "Search/articles.csv",
            "Search/articles_clean_doi_only.csv",
            "Rankings/final_abstract_rank.csv",
            "Rankings/rank_strategy_3.csv",
        ],
    )

    ranked_candidates, ranked_source = count_first_available_csv(
        project_dir,
        [
            "Rankings/final_abstract_rank.csv",
            "Rankings/rank_strategy_3.csv",
            "Rankings/rank_strategy_1.csv",
        ],
    )

    abstract_screened, abstract_source = count_first_available_csv(
        project_dir,
        [
            "Rankings/full_rank_strategy_3.csv",
            "Rankings/full_rank_strategy_1.csv",
            "Rankings/llm_rank_top1000.csv",
            "Rankings/bigembed_chunk_fulltext_rank_top1000.csv",
        ],
    )

    if abstract_screened <= 0:
        abstract_screened = min(DEFAULT_ABSTRACT_SCREENING_POOL, ranked_candidates) if ranked_candidates > 0 else DEFAULT_ABSTRACT_SCREENING_POOL
        abstract_source = f"default abstract-screening pool ({DEFAULT_ABSTRACT_SCREENING_POOL})"

    llm_cards_available = len(all_available_cards)
    llm_cards_used = len(cards)

    return {
        "project": project_dir.name,
        "project_title": project_info.get("title", ""),
        "objective": project_info.get("objective", ""),
        "downloaded_records": downloaded_records,
        "downloaded_records_source": downloaded_source,
        "ranked_candidates": ranked_candidates,
        "ranked_candidates_source": ranked_source,
        "abstract_screened_top_pool": abstract_screened,
        "abstract_screened_top_pool_source": abstract_source,
        "llm_cards_available": llm_cards_available,
        "llm_cards_used": llm_cards_used,
        "included": decision_counts.get("include", 0),
        "excluded": decision_counts.get("exclude", 0),
        "uncertain": decision_counts.get("uncertain", 0),
        "missing_decision": decision_counts.get("missing", 0),
        "keywords": config_keywords[:18],
        "search_databases": search_databases[:10],
        "model": args.model_folder_name,
    }


def wrapped_lines(text: Any, width: int = 58, max_lines: int = 5) -> str:
    s = clean_text(text)
    if not s:
        return ""

    lines = textwrap.wrap(s, width=width)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(".") + "..."
    return "\n".join(lines)


def format_list_for_panel(items: List[str], max_items: int = 8) -> str:
    cleaned = unique_keep_order([clean_text(x) for x in items if clean_text(x)])
    if not cleaned:
        return "Not reported"

    shown = cleaned[:max_items]
    suffix = "" if len(cleaned) <= max_items else f"\n+ {len(cleaned) - max_items} more"
    return "\n".join(f"- {shorten_label(x, 46)}" for x in shown) + suffix


def draw_flow_box(
    ax: Any,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    facecolor: str,
    edgecolor: str = "#243447",
    title_color: str = "#101820",
    title_fontsize: float = 12,
    body_fontsize: float = 10,
) -> None:
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.6,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(
        x,
        y + h * 0.18,
        title,
        ha="center",
        va="center",
        fontsize=title_fontsize,
        fontweight="bold",
        color=title_color,
    )
    ax.text(
        x,
        y - h * 0.18,
        body,
        ha="center",
        va="center",
        fontsize=body_fontsize,
        color="#18212b",
        linespacing=1.2,
    )


def draw_down_arrow(ax: Any, x: float, y1: float, y2: float) -> None:
    ax.annotate(
        "",
        xy=(x, y2),
        xytext=(x, y1),
        arrowprops=dict(arrowstyle="-|>", linewidth=1.8, color="#243447"),
    )


def save_figure1_review_flow(flow: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 10.5), facecolor="#f6f3ec")
    ax = plt.gca()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    title = wrapped_lines(flow.get("project_title", flow.get("project", "")), width=88, max_lines=2)
    if not title:
        title = flow.get("project", "Review project")

    ax.text(
        0.5,
        0.965,
        "Figure 1. Automated Literature Review Selection Flow",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="#111827",
    )
    ax.text(
        0.5,
        0.925,
        title,
        ha="center",
        va="center",
        fontsize=11,
        color="#374151",
    )

    center_x = 0.32
    box_w = 0.43
    box_h = 0.115
    ys = [0.79, 0.62, 0.45, 0.28, 0.11]

    downloaded_n = flow.get("downloaded_records", 0)
    ranked_n = flow.get("ranked_candidates", 0)
    abstract_n = flow.get("abstract_screened_top_pool", 0)
    llm_available = flow.get("llm_cards_available", 0)
    llm_used = flow.get("llm_cards_used", 0)

    boxes = [
        (
            "Database retrieval",
            f"Records downloaded from databases\nN = {downloaded_n:,}",
            "#dbeafe",
        ),
        (
            "Cleaning + abstract/title ranking",
            f"Unique candidate records ranked\nN = {ranked_n:,}",
            "#e0f2fe",
        ),
        (
            "Abstract screening pool",
            f"Top-ranked records retained for full-text stage\nN = {abstract_n:,}",
            "#dcfce7",
        ),
        (
            "LLM evidence-card assessment",
            f"Top 200 targeted; cards available N = {llm_available:,}\nCards used in this figure N = {llm_used:,}",
            "#fef3c7",
        ),
        (
            "Final screening decision",
            (
                f"Included N = {flow.get('included', 0):,}   "
                f"Excluded N = {flow.get('excluded', 0):,}   "
                f"Uncertain N = {flow.get('uncertain', 0):,}"
            ),
            "#fee2e2",
        ),
    ]

    for idx, ((box_title, body, color), y) in enumerate(zip(boxes, ys)):
        draw_flow_box(ax, center_x, y, box_w, box_h, box_title, body, color)
        if idx < len(ys) - 1:
            draw_down_arrow(ax, center_x, y - box_h / 2 - 0.005, ys[idx + 1] + box_h / 2 + 0.005)

    side_x = 0.78
    side_w = 0.31
    draw_flow_box(
        ax,
        side_x,
        0.70,
        side_w,
        0.20,
        "Search sources",
        format_list_for_panel(flow.get("search_databases", []), max_items=7),
        "#f8fafc",
        edgecolor="#64748b",
        title_fontsize=11,
        body_fontsize=9,
    )
    draw_flow_box(
        ax,
        side_x,
        0.43,
        side_w,
        0.27,
        "Configuration keywords",
        format_list_for_panel(flow.get("keywords", []), max_items=7),
        "#f8fafc",
        edgecolor="#64748b",
        title_fontsize=11,
        body_fontsize=8.6,
    )
    draw_flow_box(
        ax,
        side_x,
        0.16,
        side_w,
        0.21,
        "Review objective",
        wrapped_lines(flow.get("objective", ""), width=38, max_lines=6) or "Not reported",
        "#f8fafc",
        edgecolor="#64748b",
        title_fontsize=11,
        body_fontsize=8.8,
    )

    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


def save_quality_criteria_plot(criteria_counts: Counter, total: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    items = list(criteria_counts.items())
    items.sort(key=lambda x: (-x[1], x[0]))

    plt.figure(figsize=(11, max(5, 0.48 * max(1, len(items)) + 2.5)))

    if not items or total <= 0:
        plt.text(0.5, 0.5, "No quality criteria data available", ha="center", va="center")
        plt.axis("off")
    else:
        labels = [shorten_label(k.replace("_", " "), 50) for k, _ in items]
        values = [round(v / total * 100, 1) for _, v in items]

        y_pos = list(range(len(labels)))
        plt.barh(y_pos, values)
        plt.yticks(y_pos, labels)
        plt.gca().invert_yaxis()
        plt.xlabel("Articles meeting criterion (%)")

        for i, v in enumerate(values):
            plt.text(v, i, f" {v:.1f}%", va="center", fontsize=8)

    plt.title("Quality criteria pass rate")
    plt.tight_layout()
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


# ============================================================
# TABLE BUILDERS
# ============================================================

def build_study_characteristics_table(cards: List[Dict[str, Any]], cache: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []

    for card in cards:
        ext = get_final_extraction(card)
        screening = get_final_screening(card)

        rows.append({
            "article_index": card.get("article_index", ""),
            "global_index": card.get("global_index", ""),
            "input_rank": card.get("input_rank", ""),
            "title": card.get("title", ""),
            "year": extract_year(card, cache) or card.get("year", ""),
            "doi": card.get("doi", ""),
            "journal": card.get("journal", ""),
            "decision": card.get("final_decision", ""),
            "confidence": card.get("decision_confidence", ""),
            "has_fulltext": card.get("has_fulltext", ""),
            "chunk_count": card.get("chunk_count", ""),
            "article_type": ext.get("article_type", ""),
            "aim_or_purpose": ext.get("aim_or_purpose", ""),
            "study_domain_or_context": ext.get("study_domain_or_context", ""),
            "sample_size_or_scale": ext.get("sample_size_or_scale", ""),
            "validation_or_evaluation": ext.get("validation_or_evaluation", ""),
            "one_sentence_summary": card.get("one_sentence_summary", ""),
            "screening_reason": screening.get("reason", ""),
            "source_json": card.get("_source_json", ""),
        })

    return rows


def build_methods_datasets_table(cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []

    for card in cards:
        rows.append({
            "article_index": card.get("article_index", ""),
            "title": card.get("title", ""),
            "doi": card.get("doi", ""),
            "decision": card.get("final_decision", ""),
            "methods_models_tools": "; ".join(get_methods(card)),
            "datasets_or_data_sources": "; ".join(get_datasets(card)),
            "evaluation_metrics": "; ".join(get_metrics(card)),
            "main_findings": "; ".join(get_findings(card)),
            "limitations_or_future_work": "; ".join(get_limitations(card)),
        })

    return rows


def build_quality_table(cards: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Counter, Counter]:
    rows = []
    category_counts = Counter()
    criteria_counts = Counter()

    for card in cards:
        q = classify_quality(card)

        category_counts[q["quality_category"]] += 1

        row = {
            "article_index": card.get("article_index", ""),
            "title": card.get("title", ""),
            "doi": card.get("doi", ""),
            "decision": card.get("final_decision", ""),
            "quality_score": q["quality_score"],
            "quality_max": q["quality_max"],
            "quality_percent": q["quality_percent"],
            "quality_category": q["quality_category"],
        }

        for crit, passed in q["criteria"].items():
            row[crit] = "yes" if passed else "no"
            if passed:
                criteria_counts[crit] += 1

        rows.append(row)

    return rows, category_counts, criteria_counts


def build_rq_mapping_table(cards: List[Dict[str, Any]], rqs: List[str]) -> Tuple[List[Dict[str, Any]], Counter]:
    rows = []
    rq_counts = Counter()

    for card in cards:
        mapped = map_card_to_rqs(card, rqs)

        for rq in mapped:
            rq_counts[rq] += 1

        rows.append({
            "article_index": card.get("article_index", ""),
            "title": card.get("title", ""),
            "doi": card.get("doi", ""),
            "decision": card.get("final_decision", ""),
            "mapped_research_questions": "; ".join(mapped),
            "one_sentence_summary": card.get("one_sentence_summary", ""),
        })

    return rows, rq_counts


def counter_to_rows(counter: Counter, key_name: str, value_name: str = "count") -> List[Dict[str, Any]]:
    return [{key_name: k, value_name: v} for k, v in counter.most_common()]


# ============================================================
# MAIN PROJECT PROCESSING
# ============================================================

def process_project(project_dir: Path, args: argparse.Namespace) -> None:
    print()
    print("=" * 140)
    print(f"GENERATING DIAGRAMS: {project_dir.name}")
    print("=" * 140)

    config = read_json(project_dir / "config.json", {})
    project_info = extract_title_objective(config)
    config_keywords = extract_config_keywords(config)
    search_databases = extract_search_databases(config)
    rqs = extract_research_questions(config)

    all_available_cards = load_article_cards(
        project_dir=project_dir,
        model_folder_name=args.model_folder_name,
        max_articles=0,
    )

    cards = load_article_cards(
        project_dir=project_dir,
        model_folder_name=args.model_folder_name,
        max_articles=args.max_articles,
    )

    if not cards:
        print("No article JSON files found.")
        print("Expected:", article_json_dir(project_dir, args.model_folder_name))
        return

    decision_counts = Counter(clean_text(c.get("final_decision", "")).lower() or "missing" for c in cards)

    included = [c for c in cards if clean_text(c.get("final_decision", "")).lower() == "include"]
    excluded = [c for c in cards if clean_text(c.get("final_decision", "")).lower() == "exclude"]
    uncertain = [c for c in cards if clean_text(c.get("final_decision", "")).lower() == "uncertain"]

    if args.included_only:
        evidence_cards = included
    else:
        evidence_cards = included + uncertain

    if not evidence_cards:
        evidence_cards = cards

    assets_dir = project_dir / "ReviewAssets"
    figures_dir = assets_dir / "figures"
    tables_dir = assets_dir / "tables"
    json_dir = assets_dir / "json"

    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    print("Project title        :", project_info.get("title", "")[:160])
    print("All Step16 cards     :", len(all_available_cards))
    print("Cards used now       :", len(cards))
    print("Max articles setting :", args.max_articles if args.max_articles > 0 else "all available")
    print("Included             :", len(included))
    print("Excluded             :", len(excluded))
    print("Uncertain            :", len(uncertain))
    print("Evidence cards       :", len(evidence_cards))
    print("Config keywords      :", len(config_keywords))
    print("Search databases     :", len(search_databases))
    print("Research questions   :", len(rqs))
    print("Assets dir           :", assets_dir)

    figure1_flow = build_figure1_flow_counts(
        project_dir=project_dir,
        args=args,
        cards=cards,
        all_available_cards=all_available_cards,
        decision_counts=decision_counts,
        config_keywords=config_keywords,
        search_databases=search_databases,
        project_info=project_info,
    )

    figure1_path = figures_dir / "figure_1_review_flow.png"
    figure1_json_path = json_dir / "figure_1_review_flow.json"
    figure1_csv_path = tables_dir / "figure_1_review_flow.csv"

    save_figure1_review_flow(figure1_flow, figure1_path)
    write_json(figure1_json_path, figure1_flow)
    write_csv(figure1_csv_path, [figure1_flow])

    manifest = {
        "project": project_dir.name,
        "assets_dir": str(assets_dir),
        "figures": [str(figure1_path)],
        "tables": [str(figure1_csv_path)],
        "json": [str(figure1_json_path)],
    }
    write_json(assets_dir / "manifest.json", manifest)

    print()
    print("Saved Figure 1:")
    print(" ", figure1_path)
    print("Saved Figure 1 data:")
    print(" ", figure1_json_path)
    print(" ", figure1_csv_path)
    print()
    print("=" * 140)
    print("DONE:", project_dir.name)
    print("=" * 140)
    return

    cache = enrich_cards_with_openalex(
        project_dir=project_dir,
        cards=cards,
        fetch=args.fetch_doi_metadata,
        mailto=args.mailto,
        max_api_calls=args.max_api_calls,
    )

    # Counts.
    year_counts = Counter()
    fulltext_counts = Counter()
    chunk_counts = Counter()
    method_counts = Counter()
    dataset_counts = Counter()
    metric_counts = Counter()
    journal_counts = extract_source_counts_from_openalex(cards, cache)
    country_counts = extract_country_counts(evidence_cards, cache)

    for card in cards:
        year = extract_year(card, cache)
        if year > 1500:
            year_counts[year] += 1

        fulltext_counts["full_text" if safe_int(card.get("has_fulltext", 0)) == 1 else "abstract_or_metadata_only"] += 1

        cc = safe_int(card.get("chunk_count", 0), 0)
        if cc > 0:
            chunk_counts[str(cc)] += 1

    for card in evidence_cards:
        for item in get_methods(card):
            method_counts[item] += 1
        for item in get_datasets(card):
            dataset_counts[item] += 1
        for item in get_metrics(card):
            metric_counts[item] += 1

    keyword_counts = Counter()
    for kw in config_keywords:
        keyword_counts[kw] += 1

    database_counts = Counter()
    for db in search_databases:
        database_counts[db] += 1

    quality_rows, quality_category_counts, quality_criteria_counts = build_quality_table(evidence_cards)
    study_rows = build_study_characteristics_table(cards, cache)
    methods_rows = build_methods_datasets_table(evidence_cards)
    rq_rows, rq_counts = build_rq_mapping_table(evidence_cards, rqs)

    summary_counts = {
        "project": project_dir.name,
        "project_title": project_info.get("title", ""),
        "objective": project_info.get("objective", ""),
        "review_type": project_info.get("review_type", ""),
        "model": args.model_folder_name,
        "all_step16_cards_available": len(all_available_cards),
        "cards_used_for_assets": len(cards),
        "max_articles_argument": args.max_articles,
        "included": len(included),
        "excluded": len(excluded),
        "uncertain": len(uncertain),
        "evidence_cards_used": len(evidence_cards),
        "fulltext_available": fulltext_counts.get("full_text", 0),
        "abstract_or_metadata_only": fulltext_counts.get("abstract_or_metadata_only", 0),
        "config_keywords": len(config_keywords),
        "search_databases": len(search_databases),
        "research_questions": len(rqs),
        "years_with_data": len(year_counts),
        "methods_detected": len(method_counts),
        "datasets_detected": len(dataset_counts),
        "metrics_detected": len(metric_counts),
        "countries_detected": len(country_counts),
    }

    # Save tables.
    write_csv(tables_dir / "study_characteristics.csv", study_rows)
    write_csv(tables_dir / "methods_datasets.csv", methods_rows)
    write_csv(tables_dir / "quality_assessment.csv", quality_rows)
    write_csv(tables_dir / "rq_mapping.csv", rq_rows)
    write_csv(tables_dir / "summary_counts.csv", [summary_counts])
    write_csv(tables_dir / "config_keywords.csv", counter_to_rows(keyword_counts, "keyword"))
    write_csv(tables_dir / "search_databases.csv", counter_to_rows(database_counts, "database"))
    write_csv(tables_dir / "author_countries.csv", counter_to_rows(country_counts, "country_code_or_name"))
    write_csv(tables_dir / "publication_years.csv", [{"year": k, "count": v} for k, v in sorted(year_counts.items())])
    write_csv(tables_dir / "screening_decisions.csv", counter_to_rows(decision_counts, "decision"))
    write_csv(tables_dir / "journal_sources.csv", counter_to_rows(journal_counts, "journal_or_source"))

    # Save JSON.
    write_json(json_dir / "summary_counts.json", summary_counts)
    write_json(json_dir / "config_keywords.json", config_keywords)
    write_json(json_dir / "search_databases.json", search_databases)
    write_json(json_dir / "research_questions.json", rqs)
    write_json(json_dir / "decision_counts.json", dict(decision_counts))
    write_json(json_dir / "year_counts.json", {str(k): v for k, v in sorted(year_counts.items())})
    write_json(json_dir / "fulltext_counts.json", dict(fulltext_counts))
    write_json(json_dir / "chunk_counts.json", dict(chunk_counts))
    write_json(json_dir / "method_counts.json", dict(method_counts.most_common()))
    write_json(json_dir / "dataset_counts.json", dict(dataset_counts.most_common()))
    write_json(json_dir / "metric_counts.json", dict(metric_counts.most_common()))
    write_json(json_dir / "journal_counts.json", dict(journal_counts.most_common()))
    write_json(json_dir / "country_counts.json", dict(country_counts.most_common()))
    write_json(json_dir / "quality_category_counts.json", dict(quality_category_counts))
    write_json(json_dir / "quality_criteria_counts.json", dict(quality_criteria_counts))
    write_json(json_dir / "rq_counts.json", dict(rq_counts))

    # Save plots.
    save_publications_per_year_plot(
        year_counts,
        figures_dir / "publications_per_year.png",
    )

    save_bar_plot(
        decision_counts,
        figures_dir / "screening_decisions.png",
        title="Screening decisions",
        xlabel="Decision",
        ylabel="Article count",
        top_n=10,
        horizontal=False,
    )

    save_bar_plot(
        fulltext_counts,
        figures_dir / "fulltext_availability.png",
        title="Full-text availability",
        xlabel="Source type",
        ylabel="Article count",
        top_n=10,
        horizontal=False,
    )

    save_bar_plot(
        chunk_counts,
        figures_dir / "chunk_count_distribution.png",
        title="Chunk count distribution",
        xlabel="Chunks per article",
        ylabel="Article count",
        top_n=30,
        horizontal=False,
    )

    save_prisma_flow(
        {
            "records_ranked_or_available": max(args.top_n, len(all_available_cards)),
            "selected_for_assets": len(cards),
            "fulltext_available": fulltext_counts.get("full_text", 0),
            "abstract_only": fulltext_counts.get("abstract_or_metadata_only", 0),
            "included": decision_counts.get("include", 0),
            "excluded": decision_counts.get("exclude", 0),
            "uncertain": decision_counts.get("uncertain", 0),
        },
        figures_dir / "prisma_flow.png",
        project_dir.name,
    )

    save_bar_plot(
        quality_category_counts,
        figures_dir / "quality_distribution.png",
        title="Quality assessment distribution",
        xlabel="Quality category",
        ylabel="Article count",
        top_n=10,
        horizontal=False,
    )

    save_quality_criteria_plot(
        quality_criteria_counts,
        total=len(evidence_cards),
        path=figures_dir / "quality_criteria_pass_rate.png",
    )

    save_bar_plot(
        method_counts,
        figures_dir / "top_methods_tools.png",
        title="Top methods, models, and tools",
        xlabel="Method / model / tool",
        ylabel="Article count",
        top_n=args.top_n_terms,
        horizontal=True,
    )

    save_bar_plot(
        dataset_counts,
        figures_dir / "top_datasets_sources.png",
        title="Top datasets and data sources",
        xlabel="Dataset / data source",
        ylabel="Article count",
        top_n=args.top_n_terms,
        horizontal=True,
    )

    save_bar_plot(
        metric_counts,
        figures_dir / "top_evaluation_metrics.png",
        title="Top evaluation metrics",
        xlabel="Metric",
        ylabel="Article count",
        top_n=args.top_n_terms,
        horizontal=True,
    )

    save_bar_plot(
        journal_counts,
        figures_dir / "top_journals_sources.png",
        title="Top journals and sources",
        xlabel="Journal / source",
        ylabel="Article count",
        top_n=args.top_n_terms,
        horizontal=True,
    )

    save_bar_plot(
        rq_counts,
        figures_dir / "research_question_coverage.png",
        title="Research question coverage",
        xlabel="Research question",
        ylabel="Article count",
        top_n=50,
        horizontal=False,
    )

    save_bar_plot(
        keyword_counts,
        figures_dir / "config_keywords.png",
        title="Configuration keywords and search terms",
        xlabel="Keyword / search term",
        ylabel="Count in config",
        top_n=args.top_n_keywords,
        horizontal=True,
    )

    save_bar_plot(
        database_counts,
        figures_dir / "search_databases.png",
        title="Search databases / information sources",
        xlabel="Database / source",
        ylabel="Count in config",
        top_n=args.top_n_terms,
        horizontal=True,
    )

    save_bar_plot(
        country_counts,
        figures_dir / "author_countries.png",
        title="Author affiliation countries",
        xlabel="Country",
        ylabel="Article count",
        top_n=args.top_n_countries,
        horizontal=True,
    )

    manifest = {
        "project": project_dir.name,
        "assets_dir": str(assets_dir),
        "figures": sorted(str(p) for p in figures_dir.glob("*.png")),
        "tables": sorted(str(p) for p in tables_dir.glob("*.csv")),
        "json": sorted(str(p) for p in json_dir.glob("*.json")),
    }

    write_json(assets_dir / "manifest.json", manifest)

    print()
    print("Saved figures:")
    for p in manifest["figures"]:
        print(" ", p)

    print()
    print("Saved tables:")
    for p in manifest["tables"]:
        print(" ", p)

    print()
    print("=" * 140)
    print("DONE:", project_dir.name)
    print("=" * 140)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # This fixes your error:
    # python Step19-GenerateDiagrams.py 1
    parser.add_argument("positional_project", nargs="?", default="", help="Optional project index, e.g. 1")

    parser.add_argument("--projects-dir", default=DEFAULT_PROJECTS_DIR)
    parser.add_argument("--model-folder-name", default=DEFAULT_MODEL_FOLDER_NAME)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)

    parser.add_argument("--project-index", type=int, default=0)
    parser.add_argument("--project-name", default="")
    parser.add_argument("--all-projects", action="store_true")

    # This fixes your concern about not wanting all top 200.
    parser.add_argument(
        "--max-articles",
        type=int,
        default=0,
        help="Maximum Step16 article cards to use. 0 means all available Step16 cards.",
    )

    parser.add_argument("--included-only", action="store_true")
    parser.add_argument("--top-n-terms", type=int, default=20)
    parser.add_argument("--top-n-keywords", type=int, default=30)
    parser.add_argument("--top-n-countries", type=int, default=20)

    parser.add_argument("--fetch-doi-metadata", action="store_true")
    parser.add_argument("--mailto", default="")
    parser.add_argument("--max-api-calls", type=int, default=0, help="0 means no max API limit.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    projects = select_projects(args)

    print("=" * 140)
    print("STEP19 GENERATE DIAGRAMS")
    print("=" * 140)
    print("Projects dir       :", Path(args.projects_dir).resolve())
    print("Model folder       :", args.model_folder_name)
    print("Projects selected  :", len(projects))
    print("Max articles       :", args.max_articles if args.max_articles > 0 else "all available")
    print("Fetch DOI metadata :", args.fetch_doi_metadata)
    print("Mailto             :", args.mailto)
    print("Max API calls      :", args.max_api_calls)
    print("=" * 140)

    if args.fetch_doi_metadata and requests is None:
        print("WARNING: requests is not installed. DOI metadata enrichment skipped.")
        print("Install with: pip install requests")

    for i, project_dir in enumerate(projects, start=1):
        print(f"\n[{i}/{len(projects)}] {project_dir.name}")
        process_project(project_dir, args)


if __name__ == "__main__":
    main()
