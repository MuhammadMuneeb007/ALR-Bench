#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================================
# Step21-BuildFinalArticlePackage.py
# ============================================================
#
# Purpose:
#   Build the final article-writing package for one project.
#
# By default it DOES NOT call DeepSeek. Use --generate 1 to send the
# generated prompts to an OpenAI-compatible LLM endpoint such as DeepSeek.
# Raw LLM responses are saved under FinalArticle/LLMResponses/.
#
# Output:
#   Projects/<project>/FinalArticle/
#       Main.tex
#       Introduction.tex
#       Methodology.tex
#       MainFindings.tex
#       Results.tex
#       Discussion.tex
#       Conclusion.tex
#       Abstract.tex
#       reference.bib
#       project_context.json
#       article_evidence_cards.json
#       article_evidence_cards.tex
#       DeepSeek_Methodology_Prompt.txt
#       DeepSeek_MainFindings_Prompt.txt
#       DeepSeek_Results_Prompt.txt
#       DeepSeek_Discussion_Prompt.txt
#       DeepSeek_Introduction_Prompt.txt
#       DeepSeek_Conclusion_Prompt.txt
#       DeepSeek_Abstract_Prompt.txt
#
# Usage:
#   python Step21-BuildFinalArticlePackage.py 1
#   python Step21-BuildFinalArticlePackage.py 1 --max-cards 80
#   python Step21-BuildFinalArticlePackage.py --project-index 1 --included-only
#
# ============================================================

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Tuple, Optional

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable


# ============================================================
# Defaults
# ============================================================

PROJECTS_DIR = Path("Projects")
DATABASE_DIR = Path("DataBase_Strategy3_Top1000")
MODEL_NAME = "Qwen__Qwen2.5-3B-Instruct"
RANKING_FILE = "full_rank_strategy_10.csv"
TOP_N = 200
LOADED_ENV_FILES: List[str] = []


# ============================================================
# Helpers
# ============================================================

def clean(x: Any) -> str:
    if x is None:
        return ""
    x = str(x).replace("\r", " ").replace("\n", " ")
    x = re.sub(r"\s+", " ", x).strip()
    if x.lower() in {"nan", "none", "null", "na", "n/a"}:
        return ""
    return x


def latex_escape(x: Any) -> str:
    s = clean(x)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for a, b in repl.items():
        s = s.replace(a, b)
    return s


def safe_model_name(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", x.replace("/", "__")).strip("_")


def safe_int(x: Any, default: int = 0) -> int:
    try:
        s = clean(x)
        if not s:
            return default
        return int(float(s))
    except Exception:
        return default


def progress_write(message: str) -> None:
    try:
        tqdm.write(str(message))
    except Exception:
        print(str(message), flush=True)


def run_with_heartbeat(label: str, fn, interval_seconds: int = 15):
    interval = safe_int(interval_seconds, 0)
    if interval <= 0:
        return fn()

    stop_event = threading.Event()
    started = time.time()

    def heartbeat() -> None:
        while not stop_event.wait(interval):
            elapsed = time.time() - started
            progress_write(f"[waiting] {label} | elapsed={elapsed:.1f}s")

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        return fn()
    finally:
        stop_event.set()
        thread.join(timeout=0.2)


def read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return {}


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", errors="ignore")


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def flatten(x: Any) -> List[str]:
    out = []
    if x is None:
        return out
    if isinstance(x, str):
        s = clean(x)
        if s:
            out.append(s)
    elif isinstance(x, (int, float, bool)):
        out.append(str(x))
    elif isinstance(x, list):
        for i in x:
            out.extend(flatten(i))
    elif isinstance(x, dict):
        for v in x.values():
            out.extend(flatten(v))
    return out


def unique(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        x = clean(x)
        if not x:
            continue
        k = x.lower()
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out


def split_terms(items: List[str]) -> List[str]:
    out = []
    for x in items:
        for p in re.split(r";|,|\||\n", clean(x)):
            p = clean(p)
            if p:
                out.append(p)
    return unique(out)


def config_values(config: Dict[str, Any], keys: List[str]) -> List[str]:
    vals = []
    for k in keys:
        vals.extend(flatten(config.get(k)))
    return unique(vals)


def normalize_doi(x: Any) -> str:
    doi = clean(x).lower()
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
    return doi.strip().strip(".,;:()[]{}<>")


# ============================================================
# Project discovery
# ============================================================

def discover_projects() -> List[Path]:
    return sorted([
        p for p in PROJECTS_DIR.iterdir()
        if p.is_dir()
        and (p / "config.json").exists()
        and p.name.lower() not in {"rankings", "search", "evaluation", "__pycache__", "extra"}
    ])


def select_project(args: argparse.Namespace) -> Tuple[Path, int]:
    projects = discover_projects()

    project_index = args.project_index
    if project_index <= 0 and args.project:
        project_index = safe_int(args.project)

    if project_index > 0:
        idx = project_index - 1
        if idx < 0 or idx >= len(projects):
            raise SystemExit(f"ERROR: invalid project index {project_index}. Found {len(projects)} projects.")
        return projects[idx], project_index

    if args.project_name:
        exact = [p for p in projects if p.name == args.project_name]
        if exact:
            return exact[0], projects.index(exact[0]) + 1
        partial = [p for p in projects if args.project_name.lower() in p.name.lower()]
        if partial:
            return partial[0], projects.index(partial[0]) + 1
        raise SystemExit(f"ERROR: project not found: {args.project_name}")

    raise SystemExit("Use: python Step21-BuildFinalArticlePackage.py 1")


# ============================================================
# Config stats
# ============================================================

def get_config_package(project_dir: Path) -> Dict[str, Any]:
    config = read_json(project_dir / "config.json")

    title = config_values(config, ["title", "review_title"])
    review_type = config_values(config, ["review_type", "type"])
    objective = config_values(config, ["objective", "aim", "review_objective"])

    rqs = config_values(config, [
        "research_questions",
        "review_questions",
        "rqs",
        "generated_research_questions",
        "prisma_research_questions",
    ])

    keyword_raw = config_values(config, [
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
    ])

    keywords = split_terms(keyword_raw)

    source_raw = config_values(config, [
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
    ])

    sources = split_terms(source_raw)

    inclusion = config_values(config, [
        "prisma_eligibility_inclusion",
        "inclusion_criteria",
        "eligibility_inclusion",
        "generated_abstract_include_keywords",
        "generated_full_text_include_keywords",
        "prisma_title_abstract_include_keywords",
        "prisma_full_text_include_keywords",
    ])

    exclusion = config_values(config, [
        "prisma_eligibility_exclusion",
        "exclusion_criteria",
        "eligibility_exclusion",
        "generated_abstract_exclude_keywords",
        "generated_full_text_exclude_keywords",
        "prisma_title_abstract_exclude_keywords",
        "prisma_full_text_exclude_keywords",
    ])

    quality = config_values(config, [
        "prisma_quality_assessment_questions",
        "generated_quality_assessment_questions",
        "quality_assessment_questions",
        "prisma_quality_assessment_keywords",
        "generated_quality_assessment_keywords",
        "quality_assessment_keywords",
    ])

    date_start = clean(config.get("prisma_date_coverage_start") or config.get("generated_date_start"))
    date_end = clean(config.get("prisma_date_coverage_end") or config.get("generated_date_end"))
    language = clean(config.get("prisma_s_language_limits") or config.get("generated_language"))

    return {
        "title": title,
        "review_type": review_type,
        "objective": objective,
        "research_questions": rqs,
        "keywords": keywords,
        "sources": sources,
        "inclusion": inclusion,
        "exclusion": exclusion,
        "quality_questions": quality,
        "date_start": date_start,
        "date_end": date_end,
        "language": language,
        "raw_config": config,
    }


# ============================================================
# Ranking and full-text stats
# ============================================================

def get_ranking_package(project_dir: Path) -> Dict[str, Any]:
    rows = read_csv_rows(project_dir / "Rankings" / RANKING_FILE)

    source_counts = Counter()
    journal_counts = Counter()
    year_counts = Counter()

    title_count = 0
    abstract_count = 0
    doi_count = 0
    year_count = 0

    for r in rows:
        title = clean(r.get("title") or r.get("Title") or r.get("article_title") or r.get("display_name"))
        abstract = clean(r.get("abstract") or r.get("Abstract") or r.get("summary") or r.get("description"))
        doi = normalize_doi(r.get("doi") or r.get("DOI") or r.get("article_doi") or r.get("work_doi"))
        year = clean(r.get("year") or r.get("publication_year") or r.get("published_year") or r.get("date"))
        source = clean(r.get("source_db") or r.get("database") or r.get("source_database") or r.get("source"))
        journal = clean(r.get("journal") or r.get("Journal") or r.get("container_title") or r.get("venue"))

        if title:
            title_count += 1
        if abstract:
            abstract_count += 1
        if doi:
            doi_count += 1
        if year:
            year_count += 1
            y = safe_int(year, 0)
            if y > 1500:
                year_counts[y] += 1
        if source:
            source_counts[source] += 1
        if journal:
            journal_counts[journal] += 1

    return {
        "ranking_file": str(project_dir / "Rankings" / RANKING_FILE),
        "ranking_records": len(rows),
        "top_n_for_llm": min(TOP_N, len(rows)),
        "records_with_title": title_count,
        "records_with_abstract": abstract_count,
        "records_with_doi": doi_count,
        "records_with_year": year_count,
        "source_counts": dict(source_counts.most_common()),
        "journal_counts_top20": dict(journal_counts.most_common(20)),
        "year_counts": {str(k): v for k, v in sorted(year_counts.items())},
    }


def find_fulltext_root(project_name: str) -> Optional[Path]:
    if not DATABASE_DIR.exists():
        return None

    exact = DATABASE_DIR / project_name
    if exact.exists():
        return exact

    norm_project = re.sub(r"[^a-z0-9]+", " ", project_name.lower()).strip()

    for p in sorted(DATABASE_DIR.iterdir()):
        if not p.is_dir():
            continue
        norm_p = re.sub(r"[^a-z0-9]+", " ", p.name.lower()).strip()
        if norm_p.startswith(norm_project[:40]) or norm_project[:50] in norm_p:
            return p

    return None


def get_fulltext_package(project_dir: Path) -> Dict[str, Any]:
    root = find_fulltext_root(project_dir.name)

    if root is None:
        return {
            "fulltext_root": "",
            "pdf_files": 0,
            "text_like_files": 0,
            "txt_files": 0,
            "xml_tei_files": 0,
            "json_files": 0,
            "markdown_files": 0,
            "all_files": 0,
        }

    pdf = txt = xml = js = md = all_files = 0

    for p in root.rglob("*"):
        if not p.is_file():
            continue

        all_files += 1
        ext = p.suffix.lower()

        if ext == ".pdf":
            pdf += 1
        elif ext == ".txt":
            txt += 1
        elif ext in {".xml", ".tei"}:
            xml += 1
        elif ext == ".json":
            js += 1
        elif ext in {".md", ".markdown"}:
            md += 1

    return {
        "fulltext_root": str(root),
        "pdf_files": pdf,
        "text_like_files": txt + xml + js + md,
        "txt_files": txt,
        "xml_tei_files": xml,
        "json_files": js,
        "markdown_files": md,
        "all_files": all_files,
    }


# ============================================================
# Cards and citation keys
# ============================================================

def article_json_dir(project_dir: Path) -> Path:
    return project_dir / "Extractions" / f"Step16_{safe_model_name(MODEL_NAME)}" / "article_json"


def load_cards(project_dir: Path) -> List[Dict[str, Any]]:
    d = article_json_dir(project_dir)
    files = sorted(d.glob("*.json")) if d.exists() else []

    cards = []
    for f in files:
        card = read_json(f)
        if not card:
            continue
        card["_source_json"] = str(f)
        cards.append(card)

    cards.sort(key=lambda c: safe_int(c.get("article_index"), 999999))
    return cards


def make_cite_key(card: Dict[str, Any], used: set) -> str:
    year = clean(card.get("year"))
    if not year:
        year = "nd"
    year = re.sub(r"[^0-9]", "", year)[:4] or "nd"

    title = clean(card.get("title"))
    words = re.findall(r"[A-Za-z0-9]+", title.lower())
    stop = {"the", "and", "for", "with", "from", "this", "that", "using", "based", "review", "survey"}
    words = [w for w in words if w not in stop]

    base = "art"
    if words:
        base = words[0][:12]

    key = f"{base}{year}"
    key = re.sub(r"[^A-Za-z0-9_:-]", "", key)

    if not key:
        key = f"article{card.get('article_index', '')}"

    original = key
    i = 2
    while key in used:
        key = f"{original}_{i}"
        i += 1

    used.add(key)
    return key


def get_final_extraction(card: Dict[str, Any]) -> Dict[str, Any]:
    ext = card.get("final_extraction", {})
    return ext if isinstance(ext, dict) else {}


def compact_card(card: Dict[str, Any]) -> Dict[str, Any]:
    ext = get_final_extraction(card)

    return {
        "article_index": card.get("article_index", ""),
        "global_index": card.get("global_index", ""),
        "input_rank": card.get("input_rank", ""),
        "cite_key": card.get("cite_key", ""),
        "latex_citation": f"\\cite{{{card.get('cite_key', '')}}}",
        "title": card.get("title", ""),
        "year": card.get("year", ""),
        "doi": card.get("doi", ""),
        "journal": card.get("journal", ""),
        "decision": card.get("final_decision", ""),
        "confidence": card.get("decision_confidence", ""),
        "has_fulltext": card.get("has_fulltext", ""),
        "chunk_count": card.get("chunk_count", ""),
        "one_sentence_summary": card.get("one_sentence_summary", ""),
        "article_type": ext.get("article_type", ""),
        "aim_or_purpose": ext.get("aim_or_purpose", ""),
        "study_domain_or_context": ext.get("study_domain_or_context", ""),
        "datasets_or_data_sources": ext.get("datasets_or_data_sources", []),
        "sample_size_or_scale": ext.get("sample_size_or_scale", ""),
        "methods_models_tools": ext.get("methods_models_tools", []),
        "outcomes_or_targets": ext.get("outcomes_or_targets", []),
        "validation_or_evaluation": ext.get("validation_or_evaluation", ""),
        "evaluation_metrics": ext.get("evaluation_metrics", []),
        "main_numerical_results": ext.get("main_numerical_results", []),
        "main_findings": ext.get("main_findings", []),
        "limitations_or_future_work": ext.get("limitations_or_future_work", []),
        "quality_evidence": ext.get("quality_evidence", []),
    }


def build_cards_package(project_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    cards = load_cards(project_dir)

    used = set()
    for c in cards:
        c["cite_key"] = make_cite_key(c, used)

    decision_counts = Counter(clean(c.get("final_decision")).lower() or "missing" for c in cards)
    fulltext_used = sum(1 for c in cards if safe_int(c.get("has_fulltext")) == 1)
    no_fulltext = len(cards) - fulltext_used
    chunk_counts = Counter(str(c.get("chunk_count") or "missing") for c in cards)

    selected = []
    for c in cards:
        decision = clean(c.get("final_decision")).lower()

        if args.included_only:
            if decision == "include":
                selected.append(c)
        else:
            if decision in {"include", "uncertain"}:
                selected.append(c)

    if not selected:
        selected = cards

    if args.max_cards > 0:
        selected = selected[:args.max_cards]

    compact = [compact_card(c) for c in selected]

    return {
        "all_cards": cards,
        "selected_cards": selected,
        "compact_cards": compact,
        "stats": {
            "article_cards_generated": len(cards),
            "cards_used_for_writing_package": len(selected),
            "decision_counts": dict(decision_counts),
            "cards_using_fulltext": fulltext_used,
            "cards_using_abstract_metadata_only": no_fulltext,
            "chunk_count_distribution": dict(chunk_counts),
        }
    }


# ============================================================
# BibTeX and LaTeX
# ============================================================

def bibtex_escape(x: Any) -> str:
    s = clean(x)
    s = s.replace("{", "").replace("}", "")
    return s


def bibtex_user_agent(mailto: str = "") -> str:
    ua = "Project34-ALR DOI-to-BibTeX builder"
    mailto = clean(mailto)
    if mailto:
        ua += f" (mailto:{mailto})"
    return ua


def fetch_url_text(url: str, accept: str, mailto: str = "", timeout: int = 25) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": accept,
            "User-Agent": bibtex_user_agent(mailto),
        },
    )

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        charset = resp.headers.get_content_charset() or "utf-8"
        return raw.decode(charset, errors="ignore").strip()


def looks_like_bibtex(text: str) -> bool:
    s = text.strip()
    return bool(re.search(r"@\w+\s*\{", s)) and "title" in s.lower()


def replace_bibtex_key(entry: str, new_key: str) -> str:
    new_key = clean(new_key)
    if not new_key:
        return entry

    return re.sub(
        r"(@\w+\s*\{)\s*[^,\s]+(\s*,)",
        rf"\g<1>{new_key}\g<2>",
        entry.strip(),
        count=1,
    )


def normalize_bibtex_months(entry: str) -> str:
    month_names = {
        "jan": "January",
        "january": "January",
        "feb": "February",
        "february": "February",
        "mar": "March",
        "march": "March",
        "apr": "April",
        "april": "April",
        "may": "May",
        "jun": "June",
        "june": "June",
        "jul": "July",
        "july": "July",
        "aug": "August",
        "august": "August",
        "sep": "September",
        "sept": "September",
        "september": "September",
        "oct": "October",
        "october": "October",
        "nov": "November",
        "november": "November",
        "dec": "December",
        "december": "December",
    }

    def repl(match: re.Match) -> str:
        value = clean(match.group(1) or match.group(2) or match.group(3)).strip("{}\"").lower()
        month = month_names.get(value)
        if not month:
            return match.group(0)
        return f"month = {{{month}}}"

    return re.sub(
        r"month\s*=\s*(?:\{([^{}]+)\}|\"([^\"]+)\"|([A-Za-z]+))",
        repl,
        entry,
        flags=re.I,
    )


def clean_fetched_bibtex(entry: str, cite_key: str) -> str:
    entry = replace_bibtex_key(entry, cite_key)
    entry = normalize_bibtex_months(entry)
    return entry.strip()


def load_bibtex_cache(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    data = read_json(path)
    if isinstance(data, dict) and isinstance(data.get("items"), dict):
        return data["items"]
    if isinstance(data, dict):
        return data
    return {}


def save_bibtex_cache(path: Optional[Path], cache: Dict[str, Dict[str, Any]]) -> None:
    if not path:
        return
    payload = {
        "cache_version": 1,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_items": len(cache),
        "items": cache,
    }
    write_json(path, payload)


def fetch_bibtex_from_doi(doi: str, cite_key: str, mailto: str = "") -> Tuple[str, str]:
    doi = normalize_doi(doi)
    if not doi:
        return "", "missing_doi"

    encoded_for_crossref = urllib.parse.quote(doi, safe="")
    encoded_for_doi_org = urllib.parse.quote(doi, safe="/")

    attempts = [
        (
            "crossref_transform",
            f"https://api.crossref.org/works/{encoded_for_crossref}/transform/application/x-bibtex",
        ),
        (
            "doi_content_negotiation",
            f"https://doi.org/{encoded_for_doi_org}",
        ),
    ]

    for source, url in attempts:
        try:
            bib = fetch_url_text(url, accept="application/x-bibtex", mailto=mailto)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
            continue

        if looks_like_bibtex(bib):
            return clean_fetched_bibtex(bib, cite_key), source

    return "", "doi_lookup_failed"


def build_metadata_bib_entry(card: Dict[str, Any]) -> str:
    key = card.get("cite_key", "")
    title = bibtex_escape(card.get("title", ""))
    year = clean(card.get("year", ""))
    journal = bibtex_escape(card.get("journal", ""))
    doi = normalize_doi(card.get("doi", ""))

    if not title:
        title = f"Article {card.get('article_index', '')}"

    fields = [
        f"  title = {{{title}}}",
    ]

    if journal:
        fields.append(f"  journal = {{{journal}}}")

    if year:
        fields.append(f"  year = {{{year}}}")

    if doi:
        fields.append(f"  doi = {{{doi}}}")
        fields.append(f"  url = {{https://doi.org/{doi}}}")

    fields.append("  note = {Fallback reference generated from article metadata; DOI BibTeX lookup was unavailable}")

    return "@article{" + key + ",\n" + ",\n".join(fields) + "\n}"


def build_bib(
    cards: List[Dict[str, Any]],
    mailto: str = "",
    sleep_seconds: float = 0.1,
    show_progress: bool = True,
    heartbeat_seconds: int = 15,
    cache_path: Optional[Path] = None,
) -> Tuple[str, Dict[str, Any]]:
    entries = []
    run_cache: Dict[str, Tuple[str, str]] = {}
    persistent_cache = load_bibtex_cache(cache_path)
    cache_dirty = False
    stats = Counter()

    card_items = list(enumerate(cards, start=1))
    if show_progress:
        progress_write(f"Resolving BibTeX from DOI metadata: {len(card_items)} selected article cards")
        if cache_path:
            progress_write(f"Using BibTeX DOI cache: {cache_path}")
        iterator = tqdm(
            card_items,
            total=len(card_items),
            desc="BibTeX DOI",
            unit="ref",
            dynamic_ncols=True,
        )
    else:
        iterator = card_items

    for idx, c in iterator:
        doi = normalize_doi(c.get("doi", ""))
        key = c.get("cite_key", "")

        bib = ""
        source = "missing_doi"

        if hasattr(iterator, "set_postfix_str"):
            short_doi = doi[:42] + "..." if len(doi) > 45 else doi
            iterator.set_postfix_str(f"{idx}/{len(card_items)} doi={short_doi or 'missing'}")

        if doi:
            if doi in run_cache:
                bib, source = run_cache[doi]
                stats["cache_hit_run"] += 1
            elif doi in persistent_cache and clean(persistent_cache[doi].get("bibtex", "")):
                cached = persistent_cache[doi]
                bib = clean_fetched_bibtex(cached.get("bibtex", ""), key)
                source = clean(cached.get("source", "persistent_cache"))
                run_cache[doi] = (bib, source)
                stats["cache_hit_persistent"] += 1
            else:
                bib, source = run_with_heartbeat(
                    f"BibTeX DOI {idx}/{len(card_items)} {doi}",
                    lambda doi=doi, key=key: fetch_bibtex_from_doi(doi, key, mailto=mailto),
                    heartbeat_seconds,
                )
                run_cache[doi] = (bib, source)
                if bib:
                    persistent_cache[doi] = {
                        "doi": doi,
                        "source": source,
                        "bibtex": clean_fetched_bibtex(bib, "__CITE_KEY__"),
                        "cached_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    cache_dirty = True
                    stats["cache_saved"] += 1
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)

        if bib:
            entries.append(bib)
            stats[f"resolved_{source}"] += 1
        else:
            entries.append(build_metadata_bib_entry(c))
            stats[f"fallback_{source}"] += 1

    if cache_dirty:
        save_bibtex_cache(cache_path, persistent_cache)
    if cache_path:
        stats["persistent_cache_items"] = len(persistent_cache)

    header = [
        "% Auto-generated BibTeX.",
        "% DOI metadata was used when available; fallback entries are marked with a note field.",
        f"% Total entries: {len(cards)}",
    ]

    for key, value in sorted(stats.items()):
        header.append(f"% {key}: {value}")

    progress_write(f"BibTeX resolution complete: {dict(stats)}")
    return "\n".join(header) + "\n\n" + "\n\n".join(entries) + "\n", dict(stats)


def build_evidence_cards_tex(cards: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append("% Auto-generated article evidence cards.")
    lines.append("% Use this as grounding material for DeepSeek-generated sections.")
    lines.append("")

    for c in cards:
        lines.append(r"\paragraph{" + latex_escape(f"Article {c.get('article_index', '')}: {c.get('title', '')}") + "}")
        lines.append("")
        lines.append(r"\textbf{Citation:} " + f"\\cite{{{c.get('cite_key', '')}}}\\\\")
        lines.append(r"\textbf{Decision:} " + latex_escape(c.get("final_decision", "")) + r"\\")
        lines.append(r"\textbf{Summary:} " + latex_escape(c.get("one_sentence_summary", "")) + r"\\")
        ext = get_final_extraction(c)

        for label, key in [
            ("Aim/purpose", "aim_or_purpose"),
            ("Methods/tools", "methods_models_tools"),
            ("Datasets/data sources", "datasets_or_data_sources"),
            ("Validation/evaluation", "validation_or_evaluation"),
            ("Evaluation metrics", "evaluation_metrics"),
            ("Main findings", "main_findings"),
            ("Limitations", "limitations_or_future_work"),
            ("Quality evidence", "quality_evidence"),
        ]:
            vals = flatten(ext.get(key))
            if vals:
                lines.append(r"\textbf{" + latex_escape(label) + ":} " + latex_escape("; ".join(vals[:8])) + r"\\")

        lines.append("")

    return "\n".join(lines)


# ============================================================
# Deterministic paper tables and flowchart
# ============================================================

def count_csv_rows(path: Path) -> int:
    return len(read_csv_rows(path))


def first_existing_csv_count(project_dir: Path, relative_paths: List[str]) -> Tuple[int, str]:
    for rel in relative_paths:
        path = project_dir / rel
        n = count_csv_rows(path)
        if n > 0:
            return n, rel
    return 0, ""


def split_source_value(x: Any) -> List[str]:
    parts = []
    for item in flatten(x):
        for p in re.split(r";|,|\||\n", clean(item)):
            p = canonical_source_name(p)
            if p:
                parts.append(p)
    return parts


def canonical_source_name(x: Any) -> str:
    s = clean(x)
    key = re.sub(r"[^a-z0-9]+", "", s.lower())
    mapping = {
        "openalex": "OpenAlex",
        "crossref": "Crossref",
        "semanticscholar": "Semantic Scholar",
        "semanticsscholar": "Semantic Scholar",
        "europepmc": "Europe PMC",
        "pubmed": "PubMed",
    }
    return mapping.get(key, s)


def source_counts_from_cards(cards: List[Dict[str, Any]]) -> Counter:
    counts = Counter()
    for card in cards:
        for source in split_source_value(card.get("source_db", "")):
            counts[source] += 1
    return counts


def source_counts_from_csv_rows(rows: List[Dict[str, Any]]) -> Counter:
    counts = Counter()
    keys = ["source_db", "source_database", "database", "source", "query_database"]
    for row in rows:
        value = ""
        for key in keys:
            if clean(row.get(key)):
                value = row.get(key)
                break
        for source in split_source_value(value):
            counts[source] += 1
    return counts


def latex_short(x: Any, max_len: int = 140) -> str:
    s = latex_escape(x)
    if len(s) <= max_len:
        return s
    return s[: max_len - 3].rstrip() + "..."


def short_raw(x: Any, max_len: int = 160) -> str:
    s = clean(x)
    if len(s) <= max_len:
        return s
    return s[: max_len - 3].rstrip() + "..."


def latex_list(items: Any, max_items: int = 4, max_len: int = 160) -> str:
    vals = unique(flatten(items))
    if not vals:
        return "N/A"
    shown = vals[:max_items]
    text = "; ".join(shown)
    if len(vals) > max_items:
        text += f"; +{len(vals) - max_items} more"
    return short_raw(text, max_len=max_len)


def latex_table(
    caption: str,
    label: str,
    headers: List[str],
    rows: List[List[Any]],
    column_spec: str,
) -> str:
    lines = [
        r"\begin{longtable}{" + column_spec + "}",
        r"\caption{" + latex_escape(caption) + r"}\label{" + label + r"}\\",
        r"\toprule",
        " & ".join(latex_escape(h) for h in headers) + r" \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        " & ".join(latex_escape(h) for h in headers) + r" \\",
        r"\midrule",
        r"\endhead",
    ]

    if not rows:
        lines.append(r"\multicolumn{" + str(len(headers)) + r"}{l}{No data available.}\\")
    else:
        for row in rows:
            vals = [latex_cell(v, 240) for v in row]
            lines.append(" & ".join(vals) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{longtable}",
        "",
    ])
    return "\n".join(lines)


def raw_latex(x: Any) -> str:
    return "__RAW_LATEX__" + str(x)


def latex_cell(x: Any, max_len: int = 240) -> str:
    s = str(x)
    if s.startswith("__RAW_LATEX__"):
        return s.replace("__RAW_LATEX__", "", 1)
    return latex_short(x, max_len)


def write_table_asset(path: Path, caption: str, label: str, headers: List[str], rows: List[List[Any]], column_spec: str) -> None:
    write_text(path, latex_table(caption, label, headers, rows, column_spec))


def write_csv_asset(path: Path, headers: List[str], rows: List[List[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="", errors="ignore") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow([str(x).replace("__RAW_LATEX__", "") for x in row])


def field_has_value(x: Any) -> bool:
    if x is None:
        return False
    if isinstance(x, str):
        s = clean(x)
        return bool(s) and s.lower() not in {"not reported", "not applicable", "n/a", "none", "null"}
    if isinstance(x, list):
        return any(field_has_value(v) for v in x)
    if isinstance(x, dict):
        return any(field_has_value(v) for v in x.values())
    return True


def build_selection_summary_rows(project_dir: Path, ranking_pkg: Dict[str, Any], cards_pkg: Dict[str, Any]) -> Tuple[List[List[Any]], Dict[str, Any]]:
    downloaded, downloaded_source = first_existing_csv_count(project_dir, ["Search/articles.csv"])
    doi_clean, doi_clean_source = first_existing_csv_count(project_dir, ["Search/articles_clean_doi_only.csv"])
    abstract_ranked, abstract_ranked_source = first_existing_csv_count(project_dir, [
        "Rankings/final_abstract_rank.csv",
        "Rankings/rank_strategy_3.csv",
        "Rankings/rank_strategy_1.csv",
    ])

    top_pool = safe_int(ranking_pkg.get("ranking_records"), 0)
    top_pool_source = clean(ranking_pkg.get("ranking_file", ""))
    if top_pool_source:
        top_pool_source = re.split(r"[\\/]", top_pool_source)[-1]
    top_n_for_llm = safe_int(ranking_pkg.get("top_n_for_llm"), TOP_N)
    stats = cards_pkg.get("stats", {})
    decisions = stats.get("decision_counts", {})

    rows = [
        ["Database retrieval", "Records downloaded from configured bibliographic databases", downloaded, downloaded_source or "Search/articles.csv"],
        ["DOI/metadata cleaning", "Records retained after DOI/metadata cleaning", doi_clean, doi_clean_source or "Search/articles_clean_doi_only.csv"],
        ["Abstract/title ranking", "Candidate records ranked using abstract/title metadata", abstract_ranked, abstract_ranked_source or "Ranking CSV"],
        ["Top-ranked screening pool", "Records retained for full-text-aware ranking / article-card stage", top_pool, top_pool_source],
        ["LLM card target", "Top records targeted for LLM evidence-card generation", top_n_for_llm, "TOP_N"],
        ["LLM cards generated", "Article cards available for synthesis", stats.get("article_cards_generated", 0), "Step16 article_json"],
        ["Cards used for writing", "Cards selected for this article package", stats.get("cards_used_for_writing_package", 0), "Step18.1 selected cards"],
        ["Included", "Cards with final screening decision = include", decisions.get("include", 0), "Step16 final_decision"],
        ["Excluded", "Cards with final screening decision = exclude", decisions.get("exclude", 0), "Step16 final_decision"],
        ["Uncertain", "Cards with final screening decision = uncertain", decisions.get("uncertain", 0), "Step16 final_decision"],
        ["Full text available", "Cards processed with full text available", stats.get("cards_using_fulltext", 0), "Step16 has_fulltext"],
        ["Abstract/metadata only", "Cards processed without full text", stats.get("cards_using_abstract_metadata_only", 0), "Step16 has_fulltext"],
    ]

    flow = {
        "downloaded": downloaded,
        "doi_clean": doi_clean,
        "abstract_ranked": abstract_ranked,
        "top_pool": top_pool,
        "top_n_for_llm": top_n_for_llm,
        "cards_generated": stats.get("article_cards_generated", 0),
        "cards_used": stats.get("cards_used_for_writing_package", 0),
        "included": decisions.get("include", 0),
        "excluded": decisions.get("exclude", 0),
        "uncertain": decisions.get("uncertain", 0),
    }
    return rows, flow


def build_database_rows(project_dir: Path, config_sources: List[str], cards: List[Dict[str, Any]]) -> List[List[Any]]:
    search_rows = read_csv_rows(project_dir / "Search" / "articles.csv")
    search_counts = source_counts_from_csv_rows(search_rows)
    card_counts = source_counts_from_cards(cards)
    config_sources_norm = [canonical_source_name(x) for x in config_sources]
    all_sources = unique(config_sources_norm + list(search_counts.keys()) + list(card_counts.keys()))
    return [
        [source, search_counts.get(source, 0), card_counts.get(source, 0)]
        for source in all_sources
    ]


def build_year_rows(cards: List[Dict[str, Any]]) -> List[List[Any]]:
    counts = Counter()
    for card in cards:
        year = safe_int(card.get("year"), 0)
        if year > 1500:
            counts[year] += 1
    return [[year, counts[year]] for year in sorted(counts)]


def build_included_study_rows(cards: List[Dict[str, Any]], max_rows: int) -> List[List[Any]]:
    rows = []
    for card in cards[:max_rows]:
        ext = get_final_extraction(card)
        rows.append([
            card.get("input_rank", ""),
            card.get("cite_key", ""),
            card.get("title", ""),
            card.get("year", ""),
            card.get("journal", ""),
            card.get("doi", ""),
            card.get("source_db", ""),
            ext.get("article_type", ""),
            "Yes" if safe_int(card.get("has_fulltext"), 0) == 1 else "No",
        ])
    return rows


def build_evidence_summary_rows(cards: List[Dict[str, Any]], max_rows: int) -> List[List[Any]]:
    rows = []
    for card in cards[:max_rows]:
        ext = get_final_extraction(card)
        rows.append([
            raw_latex(f"\\cite{{{card.get('cite_key', '')}}}"),
            ext.get("aim_or_purpose", ""),
            ext.get("study_domain_or_context", ""),
            latex_list(ext.get("methods_models_tools", []), max_items=4),
            latex_list(ext.get("datasets_or_data_sources", []), max_items=3),
            latex_list(ext.get("outcomes_or_targets", []), max_items=3),
            latex_list(ext.get("evaluation_metrics", []), max_items=3),
            latex_list(ext.get("main_findings", []), max_items=3),
        ])
    return rows


THEME_STOP_PHRASES = {
    "not reported",
    "not applicable",
    "n/a",
    "none",
    "null",
    "large language model",
    "large language models",
    "llm",
    "llms",
    "model",
    "models",
    "framework",
    "approach",
    "method",
    "methods",
    "system",
    "study",
    "review",
    "survey",
}


def split_theme_candidates(x: Any) -> List[str]:
    out = []
    for value in flatten(x):
        for part in re.split(r";|\||\n|,|\band\b|\busing\b|\bwith\b", clean(value), flags=re.I):
            p = clean(part)
            if p:
                out.append(p)
    return out


def normalize_theme_label(x: Any) -> str:
    s = clean(x)
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"[^A-Za-z0-9+/ -]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" -")
    if not s:
        return ""

    words = [w for w in s.split() if len(w) > 1]
    if not words:
        return ""
    s = " ".join(words[:6])
    key = s.lower().strip()
    if key in THEME_STOP_PHRASES:
        return ""
    if len(key) < 4:
        return ""
    return s[:1].upper() + s[1:]


def candidate_theme_labels(card: Dict[str, Any]) -> List[str]:
    ext = get_final_extraction(card)
    candidates = []
    for field in [
        ext.get("outcomes_or_targets", []),
        ext.get("methods_models_tools", []),
        ext.get("study_domain_or_context", ""),
        ext.get("main_findings", []),
        card.get("one_sentence_summary", ""),
    ]:
        for item in split_theme_candidates(field):
            label = normalize_theme_label(item)
            if label:
                candidates.append(label)
    return unique(candidates)


def build_theme_synthesis_groups(
    cards: List[Dict[str, Any]],
    max_themes: int = 8,
    max_studies_per_theme: int = 10,
) -> List[Dict[str, Any]]:
    per_card_labels = []
    counts = Counter()

    for card in cards:
        labels = candidate_theme_labels(card)
        if not labels:
            labels = ["Other included evidence"]
        per_card_labels.append((card, labels))
        for label in labels[:4]:
            counts[label] += 1

    chosen = {label for label, _ in counts.most_common(max_themes)}
    groups: Dict[str, Dict[str, Any]] = {}

    for card, labels in per_card_labels:
        theme = next((label for label in labels if label in chosen), "")
        if not theme:
            theme = "Other included evidence"

        if theme not in groups:
            groups[theme] = {
                "theme": theme,
                "n_studies": 0,
                "citations": [],
                "methods_or_tools": [],
                "outcomes": [],
                "main_findings": [],
                "limitations_or_gaps": [],
            }

        ext = get_final_extraction(card)
        group = groups[theme]
        group["n_studies"] += 1
        if card.get("cite_key"):
            group["citations"].append(f"\\cite{{{card.get('cite_key')}}}")
        group["methods_or_tools"].extend(flatten(ext.get("methods_models_tools", []))[:3])
        group["outcomes"].extend(flatten(ext.get("outcomes_or_targets", []))[:3])
        group["main_findings"].extend(flatten(ext.get("main_findings", []))[:3])
        group["limitations_or_gaps"].extend(flatten(ext.get("limitations_or_future_work", []))[:2])

    sorted_groups = sorted(groups.values(), key=lambda g: (-safe_int(g.get("n_studies")), clean(g.get("theme")).lower()))
    for group in sorted_groups:
        group["citations"] = unique(group["citations"])[:max_studies_per_theme]
        group["methods_or_tools"] = unique(group["methods_or_tools"])[:8]
        group["outcomes"] = unique(group["outcomes"])[:8]
        group["main_findings"] = unique(group["main_findings"])[:8]
        group["limitations_or_gaps"] = unique(group["limitations_or_gaps"])[:6]
    return sorted_groups


def build_theme_synthesis_rows(groups: List[Dict[str, Any]]) -> List[List[Any]]:
    rows = []
    for group in groups:
        rows.append([
            group.get("theme", ""),
            group.get("n_studies", 0),
            latex_list(group.get("main_findings", []), max_items=3, max_len=220),
            latex_list(group.get("methods_or_tools", []), max_items=4, max_len=180),
            latex_list(group.get("outcomes", []), max_items=3, max_len=160),
            raw_latex("; ".join(group.get("citations", [])[:8]) or "N/A"),
        ])
    return rows


def build_completeness_rows(cards: List[Dict[str, Any]]) -> List[List[Any]]:
    checks = [
        ("DOI present", lambda c, e: field_has_value(c.get("doi"))),
        ("Full text available", lambda c, e: safe_int(c.get("has_fulltext"), 0) == 1),
        ("Article type extracted", lambda c, e: field_has_value(e.get("article_type"))),
        ("Aim/purpose extracted", lambda c, e: field_has_value(e.get("aim_or_purpose"))),
        ("Domain/context extracted", lambda c, e: field_has_value(e.get("study_domain_or_context"))),
        ("Methods/tools extracted", lambda c, e: field_has_value(e.get("methods_models_tools"))),
        ("Datasets/data sources extracted", lambda c, e: field_has_value(e.get("datasets_or_data_sources"))),
        ("Outcomes/targets extracted", lambda c, e: field_has_value(e.get("outcomes_or_targets"))),
        ("Evaluation metrics extracted", lambda c, e: field_has_value(e.get("evaluation_metrics"))),
        ("Main findings extracted", lambda c, e: field_has_value(e.get("main_findings"))),
        ("Limitations/future work extracted", lambda c, e: field_has_value(e.get("limitations_or_future_work"))),
        ("Quality evidence extracted", lambda c, e: field_has_value(e.get("quality_evidence"))),
    ]
    total = max(len(cards), 1)
    rows = []
    for label, fn in checks:
        n = sum(1 for card in cards if fn(card, get_final_extraction(card)))
        rows.append([label, n, f"{n / total * 100:.1f}%"])
    return rows


def build_tikz_prisma(flow: Dict[str, Any]) -> str:
    return rf"""
\begin{{figure}}[htbp]
\centering
\begin{{tikzpicture}}[
  node distance=0.9cm,
  box/.style={{rectangle, rounded corners, draw=black, very thick, align=center, text width=0.78\linewidth, minimum height=1.05cm}},
  arrow/.style={{-{{Latex[length=3mm]}}, very thick}}
]
\node[box, fill=blue!8] (retrieval) {{Database retrieval\\Records downloaded: $N={flow.get("downloaded", 0)}$}};
\node[box, fill=cyan!8, below=of retrieval] (cleaning) {{DOI/metadata cleaning\\Records retained: $N={flow.get("doi_clean", 0)}$}};
\node[box, fill=green!8, below=of cleaning] (abstract) {{Abstract/title ranking\\Candidate records ranked: $N={flow.get("abstract_ranked", 0)}$}};
\node[box, fill=yellow!15, below=of abstract] (pool) {{Top-ranked screening pool\\Records retained: $N={flow.get("top_pool", 0)}$}};
\node[box, fill=orange!12, below=of pool] (llm) {{LLM evidence-card generation\\Target top records: $N={flow.get("top_n_for_llm", 0)}$; cards generated: $N={flow.get("cards_generated", 0)}$}};
\node[box, fill=red!8, below=of llm] (decision) {{Final screening decisions\\Included: $N={flow.get("included", 0)}$; Excluded: $N={flow.get("excluded", 0)}$; Uncertain: $N={flow.get("uncertain", 0)}$}};
\draw[arrow] (retrieval) -- (cleaning);
\draw[arrow] (cleaning) -- (abstract);
\draw[arrow] (abstract) -- (pool);
\draw[arrow] (pool) -- (llm);
\draw[arrow] (llm) -- (decision);
\end{{tikzpicture}}
\caption{{Automated review selection flow from database retrieval to LLM-assisted screening.}}
\label{{fig:automated-review-flow}}
\end{{figure}}
""".strip() + "\n"


def build_generated_tables_tex(figure_tex: str, table_texts: List[str]) -> str:
    return (
r"""
\subsection{Automatically generated review assets}

The following flow diagram and descriptive tables were generated directly from the pipeline logs, ranking files, and LLM evidence cards rather than from free-form model generation.

Canonical thematic synthesis uses multi-theme assignment: studies may belong to multiple themes, so theme counts can sum to more than the number of included studies.
""".strip()
        + "\n\n"
        + str(figure_tex or "").strip()
        + "\n\n"
        + "\n\n".join(str(x or "").strip() for x in table_texts if str(x or "").strip())
        + "\n"
    )


def build_paper_assets(
    project_dir: Path,
    out_dir: Path,
    config_pkg: Dict[str, Any],
    ranking_pkg: Dict[str, Any],
    cards_pkg: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    tables_dir = out_dir / "Tables"
    figures_dir = out_dir / "Figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    for old_name in [
        "table_4_included_studies.tex",
        "table_4_included_studies.csv",
        "table_5_evidence_extraction_summary.tex",
        "table_5_evidence_extraction_summary.csv",
        "table_6_quality_completeness.tex",
        "table_6_quality_completeness.csv",
        "table_4_evidence_extraction_summary.tex",
        "table_4_evidence_extraction_summary.csv",
        "table_4_theme_synthesis.tex",
        "table_4_theme_synthesis.csv",
        "table_4_theme_synthesis.json",
        "table_4_evidence_card_completeness.tex",
        "table_4_evidence_card_completeness.csv",
    ]:
        old_path = tables_dir / old_name
        if old_path.exists():
            old_path.unlink()

    selected_cards = cards_pkg["selected_cards"]
    all_cards = cards_pkg["all_cards"]
    max_rows = max(args.max_table_rows, 1)

    selection_rows, flow = build_selection_summary_rows(project_dir, ranking_pkg, cards_pkg)
    database_rows = build_database_rows(project_dir, config_pkg.get("sources", []), all_cards)
    year_rows = build_year_rows(selected_cards)
    completeness_rows = build_completeness_rows(selected_cards)

    write_table_asset(
        tables_dir / "table_1_selection_summary.tex",
        "Automated search, ranking, and screening summary.",
        "tab:selection-summary",
        ["Review stage", "What was counted", "Count", "Source"],
        selection_rows,
        "p{0.20\\linewidth}p{0.42\\linewidth}rp{0.22\\linewidth}",
    )
    write_csv_asset(tables_dir / "table_1_selection_summary.csv", ["Review stage", "What was counted", "Count", "Source"], selection_rows)

    write_table_asset(
        tables_dir / "table_2_database_counts.tex",
        "Information sources represented in retrieved records and LLM evidence cards.",
        "tab:database-counts",
        ["Information source", "Retrieved records", "Evidence cards"],
        database_rows,
        "p{0.46\\linewidth}rr",
    )
    write_csv_asset(tables_dir / "table_2_database_counts.csv", ["Information source", "Retrieved records", "Evidence cards"], database_rows)

    write_table_asset(
        tables_dir / "table_3_publication_years.tex",
        "Publication year distribution among selected evidence cards.",
        "tab:publication-years",
        ["Publication year", "Evidence cards"],
        year_rows,
        "rr",
    )
    write_csv_asset(tables_dir / "table_3_publication_years.csv", ["Publication year", "Evidence cards"], year_rows)

    write_table_asset(
        tables_dir / "table_4_evidence_card_completeness.tex",
        "Evidence-card completeness and extraction audit across selected articles.",
        "tab:evidence-card-completeness",
        ["Evidence item", "Cards with item", "Coverage"],
        completeness_rows,
        "p{0.58\\linewidth}rr",
    )
    write_csv_asset(tables_dir / "table_4_evidence_card_completeness.csv", ["Evidence item", "Cards with item", "Coverage"], completeness_rows)

    figure_tex = build_tikz_prisma(flow)
    table_texts = [
        read_text(tables_dir / "table_1_selection_summary.tex"),
        read_text(tables_dir / "table_2_database_counts.tex"),
        read_text(tables_dir / "table_3_publication_years.tex"),
        read_text(tables_dir / "table_4_evidence_card_completeness.tex"),
    ]
    write_text(figures_dir / "figure_1_prisma_flow.tex", figure_tex)
    write_text(out_dir / "GeneratedTables.tex", build_generated_tables_tex(figure_tex, table_texts))

    assets = {
        "tables_dir": str(tables_dir),
        "figures_dir": str(figures_dir),
        "generated_tables_tex": str(out_dir / "GeneratedTables.tex"),
        "figure_1_prisma_flow_tex": str(figures_dir / "figure_1_prisma_flow.tex"),
        "table_count": 4,
        "max_table_rows": max_rows,
        "selection_flow_counts": flow,
        "tables": [
            str(tables_dir / "table_1_selection_summary.tex"),
            str(tables_dir / "table_2_database_counts.tex"),
            str(tables_dir / "table_3_publication_years.tex"),
            str(tables_dir / "table_4_evidence_card_completeness.tex"),
        ],
    }
    write_json(out_dir / "paper_assets_manifest.json", assets)
    return assets


def build_main_tex(project_title: str, body: str = "") -> str:
    title = latex_escape(project_title or "Automated Literature Review")
    body = str(body or "").strip()
    if not body:
        body = "% Article body has not been assembled yet.\n"

    return rf"""
\documentclass[11pt]{{article}}

\usepackage[margin=1in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{array}}
\usepackage{{graphicx}}
\usepackage{{hyperref}}
\usepackage{{natbib}}
\usepackage{{tikz}}
\usetikzlibrary{{arrows.meta,positioning,shapes.geometric}}

\title{{{title}}}
\author{{Automated LLM-assisted Review Pipeline}}
\date{{}}

\begin{{document}}

\maketitle

{body}

\bibliographystyle{{plainnat}}
\bibliography{{reference}}

\end{{document}}
""".strip() + "\n"


def build_main_body_from_files(out_dir: Path) -> str:
    order = [
        "Abstract.tex",
        "Introduction.tex",
        "Methodology.tex",
        "Results.tex",
        "GeneratedTables.tex",
        "MainFindings.tex",
        "Discussion.tex",
        "Conclusion.tex",
    ]
    chunks = []
    for name in order:
        text = read_text(out_dir / name).strip()
        if text:
            chunks.append(text)
    return "\n\n".join(chunks).strip() + "\n"


def placeholder_section(title: str, note: str) -> str:
    return f"% {note}\n\\section{{{latex_escape(title)}}}\n\n% Paste generated content here.\n"


def write_text_if_missing(path: Path, text: str) -> None:
    existing = read_text(path).strip()
    if existing:
        return
    write_text(path, text)


def context_to_text(context: Dict[str, Any]) -> str:
    return json.dumps(context, ensure_ascii=False, indent=2)


# ============================================================
# Optional LLM generation
# ============================================================

SECTION_GENERATION_PLAN = [
    ("methodology", "DeepSeek_Methodology_Prompt.txt", "Methodology.tex"),
    ("results", "DeepSeek_Results_Prompt.txt", "Results.tex"),
    ("main_findings", "DeepSeek_MainFindings_Prompt.txt", "MainFindings.tex"),
    ("discussion", "DeepSeek_Discussion_Prompt.txt", "Discussion.tex"),
    ("introduction", "DeepSeek_Introduction_Prompt.txt", "Introduction.tex"),
    ("conclusion", "DeepSeek_Conclusion_Prompt.txt", "Conclusion.tex"),
    ("abstract", "DeepSeek_Abstract_Prompt.txt", "Abstract.tex"),
]


def env_or_default(name: str, default: str = "") -> str:
    return clean(os.getenv(name, default))


def parse_dotenv_line(line: str) -> Tuple[str, str]:
    s = line.strip()
    if not s or s.startswith("#"):
        return "", ""
    if s.startswith("export "):
        s = s[len("export "):].strip()
    if "=" not in s:
        return "", ""

    key, value = s.split("=", 1)
    key = key.strip()
    value = value.strip()

    if not key:
        return "", ""

    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        value = value[1:-1]
    else:
        # Allow inline comments after a separating space.
        value = re.sub(r"\s+#.*$", "", value).strip()

    value = value.replace(r"\n", "\n").replace(r"\t", "\t")
    return key, value


def load_env_file(path: Path, override: bool = False) -> bool:
    if not path.exists() or not path.is_file():
        return False

    loaded_any = False
    try:
        lines = path.read_text(encoding="utf-8-sig", errors="ignore").splitlines()
    except Exception:
        return False

    for line in lines:
        key, value = parse_dotenv_line(line)
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
            loaded_any = True

    if loaded_any:
        LOADED_ENV_FILES.append(str(path.resolve()))
    return loaded_any


def load_default_env_files() -> None:
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    candidates = []
    for p in [
        cwd / ".env",
        script_dir / ".env",
        script_dir.parent / ".env",
    ]:
        if p not in candidates:
            candidates.append(p)

    for path in candidates:
        load_env_file(path, override=False)


def env_float(name: str, default: float) -> float:
    try:
        value = env_or_default(name, "")
        return float(value) if value else default
    except Exception:
        return default


def env_int(name: str, default: int) -> int:
    try:
        value = env_or_default(name, "")
        return int(float(value)) if value else default
    except Exception:
        return default


def mask_secret(x: str) -> str:
    x = clean(x)
    if not x:
        return ""
    if len(x) <= 8:
        return "***"
    return x[:4] + "..." + x[-4:]


def llm_chat_url(base_url: str) -> str:
    base = clean(base_url).rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def extract_llm_content(response: Dict[str, Any]) -> str:
    choices = response.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    content = message.get("content", "")
    if isinstance(content, list):
        return "\n".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content).strip()
    return str(content or "").strip()


def call_llm_chat_completion(prompt: str, args: argparse.Namespace, include_reasoning_effort: bool = True) -> Dict[str, Any]:
    api_key = clean(args.llm_api_key)
    if not api_key:
        raise RuntimeError("Missing LLM API key. Set LLM_API_KEY or pass --llm-api-key.")

    payload: Dict[str, Any] = {
        "model": args.llm_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a careful academic writing assistant. "
                    "Return only valid LaTeX for the requested section. "
                    "Do not invent unsupported facts, counts, citations, or references."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": args.llm_temperature,
        "top_p": args.llm_top_p,
        "max_tokens": args.llm_max_tokens,
    }

    if include_reasoning_effort and clean(args.llm_reasoning_effort):
        payload["reasoning_effort"] = clean(args.llm_reasoning_effort)

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        llm_chat_url(args.llm_base_url),
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=args.llm_timeout) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
            return json.loads(raw)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"LLM HTTP {e.code}: {body[:2000]}") from e
    except Exception as e:
        raise RuntimeError(f"LLM request failed: {repr(e)}") from e


def call_llm_with_retry(prompt: str, args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    meta = {
        "provider": args.llm_provider,
        "base_url": args.llm_base_url,
        "model": args.llm_model,
        "temperature": args.llm_temperature,
        "top_p": args.llm_top_p,
        "max_tokens": args.llm_max_tokens,
        "reasoning_effort": args.llm_reasoning_effort,
        "api_key": mask_secret(args.llm_api_key),
        "retried_without_reasoning_effort": False,
    }

    try:
        return call_llm_chat_completion(prompt, args, include_reasoning_effort=True), meta
    except RuntimeError as e:
        if clean(args.llm_reasoning_effort):
            meta["first_error"] = str(e)
            meta["retried_without_reasoning_effort"] = True
            response = call_llm_chat_completion(prompt, args, include_reasoning_effort=False)
            return response, meta
        raise


def load_valid_cite_keys(out_dir: Path) -> set:
    keys = set()
    for card in read_json(out_dir / "article_evidence_cards.json") or []:
        if isinstance(card, dict) and clean(card.get("cite_key", "")):
            keys.add(clean(card.get("cite_key", "")))
    bib_text = read_text(out_dir / "reference.bib")
    for key in re.findall(r"@\w+\s*\{\s*([^,\s]+)", bib_text):
        if clean(key):
            keys.add(clean(key))
    return keys


def sanitize_latex_citations(content: str, valid_keys: set) -> Tuple[str, Dict[str, Any]]:
    if not valid_keys:
        return content, {"checked": False, "reason": "no_valid_keys_loaded"}

    invalid = []
    checked = 0

    def repl(match: re.Match) -> str:
        nonlocal checked
        checked += 1
        command = match.group(1)
        body = match.group(2)
        keys = [clean(k) for k in body.split(",") if clean(k)]
        kept = [k for k in keys if k in valid_keys]
        bad = [k for k in keys if k and k not in valid_keys]
        invalid.extend(bad)
        if not kept:
            return ""
        return f"{command}" + "{" + ",".join(kept) + "}"

    pattern = r"(\\cite\w*\*?(?:\[[^\]]*\]){0,2})\{([^{}]+)\}"
    cleaned = re.sub(pattern, repl, content)
    return cleaned, {
        "checked": True,
        "n_cite_commands": checked,
        "invalid_cite_keys": unique(invalid),
        "n_invalid_cite_keys": len(unique(invalid)),
    }


def validate_main_tex_citations(out_dir: Path) -> Dict[str, Any]:
    main_text = read_text(out_dir / "Main.tex")
    valid_keys = load_valid_cite_keys(out_dir)
    used = []
    for group in re.findall(r"\\cite\w*\*?(?:\[[^\]]*\]){0,2}\{([^{}]+)\}", main_text):
        used.extend([clean(k) for k in group.split(",") if clean(k)])
    invalid = [k for k in used if k not in valid_keys]
    report = {
        "valid_reference_keys": len(valid_keys),
        "citation_keys_used": len(unique(used)),
        "cite_commands": len(re.findall(r"\\cite\w*\*?(?:\[[^\]]*\]){0,2}\{", main_text)),
        "invalid_citation_keys": unique(invalid),
        "n_invalid_citation_keys": len(unique(invalid)),
    }
    write_json(out_dir / "citation_validation.json", report)
    write_json(out_dir / "Reports" / "FINAL_citation_report.json", report)
    return report


def write_generated_section(path: Path, content: str, section_name: str, valid_cite_keys: Optional[set] = None) -> Dict[str, Any]:
    content = clean_text_for_latex_output(content)
    if not content:
        content = f"% LLM returned empty content for {section_name}.\n"
    validation = {"checked": False}
    if valid_cite_keys is not None:
        content, validation = sanitize_latex_citations(content, valid_cite_keys)
    write_text(path, content.strip() + "\n")
    return validation


def clean_text_for_latex_output(text: str) -> str:
    s = str(text or "").strip()
    s = re.sub(r"^```(?:latex|tex)?\s*", "", s, flags=re.I)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def generate_article_sections(out_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    responses_dir = out_dir / "LLMResponses"
    responses_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "generate": args.generate,
        "provider": args.llm_provider,
        "base_url": args.llm_base_url,
        "model": args.llm_model,
        "temperature": args.llm_temperature,
        "top_p": args.llm_top_p,
        "max_tokens": args.llm_max_tokens,
        "reasoning_effort": args.llm_reasoning_effort,
        "api_key": mask_secret(args.llm_api_key),
        "request_total": len(SECTION_GENERATION_PLAN),
        "sections": [],
    }

    if safe_int(args.generate, 0) != 1:
        summary["status"] = "skipped_generate_0"
        write_json(responses_dir / "generation_summary.json", summary)
        return summary

    request_total = len(SECTION_GENERATION_PLAN)
    print()
    print("=" * 120)
    print("LLM ARTICLE SECTION GENERATION")
    print("=" * 120)
    print(f"Provider          : {args.llm_provider}")
    print(f"Base URL          : {args.llm_base_url}")
    print(f"Model             : {args.llm_model}")
    print(f"Requests to send  : {request_total}")
    print(f"Temperature       : {args.llm_temperature}")
    print(f"Top-p             : {args.llm_top_p}")
    print(f"Max tokens        : {args.llm_max_tokens}")
    print(f"Reasoning effort  : {args.llm_reasoning_effort or 'not sent'}")
    print(f"Progress interval : {args.progress_interval}s")
    print("=" * 120)

    valid_cite_keys = load_valid_cite_keys(out_dir)
    progress_write(f"Valid citation keys loaded: {len(valid_cite_keys)}")

    section_iter = tqdm(
        list(enumerate(SECTION_GENERATION_PLAN, start=1)),
        total=request_total,
        desc="DeepSeek sections",
        unit="request",
        dynamic_ncols=True,
    )

    for request_index, (section_name, prompt_file, output_file) in section_iter:
        prompt_path = out_dir / prompt_file
        output_path = out_dir / output_file
        prompt = read_text(prompt_path)

        msg = (
            f"[request {request_index}/{request_total}] "
            f"{section_name} -> {output_file} | prompt_chars={len(prompt)}"
        )
        if hasattr(section_iter, "set_description"):
            section_iter.set_description(f"DeepSeek {request_index}/{request_total} {section_name}")
        progress_write(msg)

        started = time.time()
        section_record = {
            "request_index": request_index,
            "request_total": request_total,
            "section": section_name,
            "prompt_file": str(prompt_path),
            "output_file": str(output_path),
            "status": "started",
            "prompt_chars": len(prompt),
        }

        try:
            response, meta = run_with_heartbeat(
                f"LLM request {request_index}/{request_total} {section_name}",
                lambda prompt=prompt: call_llm_with_retry(prompt, args),
                args.progress_interval,
            )
            content = extract_llm_content(response)
            citation_validation = write_generated_section(output_path, content, section_name, valid_cite_keys)
            write_json(responses_dir / f"{section_name}_response.json", response)
            write_json(responses_dir / f"{section_name}_request_meta.json", meta)

            section_record.update({
                "status": "ok",
                "elapsed_seconds": round(time.time() - started, 2),
                "output_chars": len(content),
                "output_words_est": len(re.findall(r"\w+", content)),
                "retried_without_reasoning_effort": meta.get("retried_without_reasoning_effort", False),
                "citation_validation": citation_validation,
            })
            ok_msg = (
                f"[done {request_index}/{request_total}] {section_name} | "
                f"{section_record['elapsed_seconds']}s | output_chars={section_record['output_chars']} | "
                f"invalid_cites={citation_validation.get('n_invalid_cite_keys', 0)}"
            )
            progress_write(ok_msg)
        except Exception as e:
            section_record.update({
                "status": "failed",
                "elapsed_seconds": round(time.time() - started, 2),
                "error": str(e),
            })
            fail_msg = f"[failed {request_index}/{request_total}] {section_name} | {section_record['elapsed_seconds']}s | {e}"
            progress_write(fail_msg)
            write_json(responses_dir / f"{section_name}_error.json", section_record)
            if not args.continue_on_generation_error:
                summary["sections"].append(section_record)
                summary["status"] = "failed"
                write_json(responses_dir / "generation_summary.json", summary)
                raise

        summary["sections"].append(section_record)
        summary["requests_completed"] = len(summary["sections"])
        write_json(responses_dir / "generation_summary.json", summary)

    summary["status"] = "complete"
    summary["requests_completed"] = len(summary["sections"])
    write_json(responses_dir / "generation_summary.json", summary)
    return summary


# ============================================================
# REVIEW-STYLE SYNTHESIS OVERRIDES
# ============================================================
# These definitions intentionally override selected functions above.
# This preserves the original Step21 feature set while changing the generation
# architecture from article-card summaries into theme-based review synthesis.

REVIEW_THEME_SPECS = [
    {"id":"architecture","title":"Agent architecture and construction","keywords":["architecture","framework","module","profile","profiling","memory","planning","action","agent construction","cognitive architecture","role","environment"],"questions":["What architectural modules are repeatedly proposed?","How do studies decompose agents into roles, memory, planning, and actions?","Which frameworks are conceptual surveys and which are implemented systems?"]},
    {"id":"memory","title":"Memory mechanisms and context management","keywords":["memory","long-term memory","short-term memory","retrieval","vector database","reflection","experience","working memory","context window","knowledge"],"questions":["How is memory represented and retrieved?","What limitations are associated with context windows?","What evidence supports hybrid memory over prompt-only memory?"]},
    {"id":"planning","title":"Planning, reasoning, and feedback","keywords":["planning","reasoning","chain-of-thought","cot","tree-of-thought","tot","graph-of-thought","pddl","planner","feedback","reflection","self-correction"],"questions":["Which planning strategies are used?","How do feedback and reflection change agent performance?","Where do planning failures remain?"]},
    {"id":"tool_use","title":"Tool use and task automation","keywords":["tool","tool use","api","function","plugin","task automation","workflow","action space","environment","tool selection","tool usage"],"questions":["How do agents select and sequence tools?","What are the typical task automation settings?","What kinds of tool-use failures are reported?"]},
    {"id":"multi_agent","title":"Multi-agent coordination and collaboration","keywords":["multi-agent","multiagent","collaboration","coordination","debate","communication","role-playing","team","society","agent society"],"questions":["How are multiple agents coordinated?","What benefits are claimed for debate or role specialization?","What coordination risks are reported?"]},
    {"id":"simulation_social","title":"Social simulation and human-behaviour modelling","keywords":["social","simulation","human behavior","human behaviour","user behavior","user behaviour","society","econom","recommendation","movie","social network","population","agent-based modeling"],"questions":["What social or behavioural phenomena are simulated?","How are profiles and interactions constructed?","How is simulation validity evaluated?"]},
    {"id":"embodied_robotics","title":"Embodied, robotic, and autonomous-driving agents","keywords":["robot","robotics","embodied","minecraft","starcraft","driving","autonomous driving","navigation","visual","environment","behavior tree"],"questions":["What embodied environments are used?","How are language, perception, and action connected?","What benchmarks and task outcomes are reported?"]},
    {"id":"software_enterprise","title":"Software engineering, enterprise, and operational agents","keywords":["software","code","programming","testing","enterprise","cloud","root cause","eda","database","sql","devops","workflow"],"questions":["What software or enterprise tasks are targeted?","How do agents interact with tools, code, or databases?","What evaluation criteria are used in operational settings?"]},
    {"id":"domain_apps","title":"Domain-specific applications in science, healthcare, finance, and education","keywords":["healthcare","medicine","medical","science","scientific","finance","trading","education","chemistry","biology","6g","wireless","domain"],"questions":["Which domain-specific applications are represented?","What domain constraints are emphasized?","What validation gaps are common?"]},
    {"id":"evaluation","title":"Evaluation strategies, benchmarks, and metrics","keywords":["evaluation","benchmark","metric","accuracy","success rate","f1","precision","recall","bleu","human evaluation","agentbench","taskbench","promptbench"],"questions":["Which evaluation strategies recur?","What metrics are used for agent capability?","Where are benchmarks insufficient or incomparable?"]},
    {"id":"safety","title":"Safety, robustness, trustworthiness, and limitations","keywords":["safety","security","privacy","trust","trustworthy","robust","hallucination","bias","fairness","alignment","risk","limitation","failure","adversarial"],"questions":["What safety and robustness problems are reported?","Which limitations appear across domains?","What future work is repeatedly proposed?"]},
]

GENERIC_REVIEW_THEME_SPECS = [
    {"id": "methods_approaches", "title": "Methods, approaches, and technical strategies", "keywords": ["method", "approach", "model", "framework", "algorithm", "tool", "technique", "strategy", "architecture"]},
    {"id": "application_contexts", "title": "Application domains and study contexts", "keywords": ["domain", "application", "context", "setting", "case study", "industry", "clinical", "education", "software", "science"]},
    {"id": "data_sources", "title": "Datasets, data sources, and experimental material", "keywords": ["dataset", "data", "corpus", "benchmark", "sample", "source", "records", "participants", "cases"]},
    {"id": "outcomes_targets", "title": "Outcomes, targets, and reported effects", "keywords": ["outcome", "target", "performance", "effect", "impact", "prediction", "classification", "detection", "improvement"]},
    {"id": "evaluation_metrics", "title": "Evaluation methods, metrics, and validation practices", "keywords": ["evaluation", "metric", "validation", "accuracy", "precision", "recall", "f1", "benchmark", "human evaluation"]},
    {"id": "implementation_tools", "title": "Implementation tools, systems, and resources", "keywords": ["tool", "software", "platform", "implementation", "prototype", "library", "pipeline", "system"]},
    {"id": "limitations_gaps", "title": "Limitations, gaps, and future research needs", "keywords": ["limitation", "gap", "challenge", "future work", "risk", "bias", "robustness", "generalization", "validity"]},
    {"id": "synthesis_implications", "title": "Cross-study implications and synthesis patterns", "keywords": ["finding", "implication", "trend", "pattern", "taxonomy", "theme", "comparison", "synthesis"]},
]


def review_short(x: Any, max_len: int = 360) -> str:
    s = clean(x)
    if len(s) <= max_len:
        return s
    return s[:max_len - 3].rstrip() + "..."


def build_review_truth(project_dir: Path, ranking_pkg: Dict[str, Any], cards_pkg: Dict[str, Any]) -> Dict[str, Any]:
    downloaded, downloaded_source = first_existing_csv_count(project_dir, ["Search/articles.csv"])
    doi_clean, doi_clean_source = first_existing_csv_count(project_dir, ["Search/articles_clean_doi_only.csv"])
    abstract_ranked, abstract_ranked_source = first_existing_csv_count(project_dir, ["Rankings/final_abstract_rank.csv", "Rankings/rank_strategy_3.csv", "Rankings/rank_strategy_1.csv"])
    stats = cards_pkg.get("stats", {})
    decisions = stats.get("decision_counts", {})
    return {
        "records_downloaded": downloaded,
        "records_downloaded_source": downloaded_source or "Search/articles.csv",
        "records_after_doi_metadata_cleaning": doi_clean,
        "records_after_doi_metadata_cleaning_source": doi_clean_source or "Search/articles_clean_doi_only.csv",
        "records_ranked_abstract_title": abstract_ranked,
        "records_ranked_abstract_title_source": abstract_ranked_source or "Rankings/final_abstract_rank.csv",
        "top_ranked_screening_pool": safe_int(ranking_pkg.get("ranking_records"), 0),
        "top_ranked_screening_pool_source": re.split(r"[\\/]", clean(ranking_pkg.get("ranking_file", RANKING_FILE)))[-1],
        "llm_screening_target_top_n": safe_int(ranking_pkg.get("top_n_for_llm"), TOP_N),
        "article_cards_generated": stats.get("article_cards_generated", 0),
        "cards_used_for_synthesis": stats.get("cards_used_for_writing_package", 0),
        "included": decisions.get("include", 0),
        "excluded": decisions.get("exclude", 0),
        "uncertain": decisions.get("uncertain", 0),
        "missing_decision": decisions.get("missing", 0),
        "cards_with_fulltext": stats.get("cards_using_fulltext", 0),
        "cards_without_fulltext": stats.get("cards_using_abstract_metadata_only", 0),
        "chunk_count_distribution": stats.get("chunk_count_distribution", {}),
        "wording_rules": [
            "Use records_downloaded for database retrieval.",
            "Use top_ranked_screening_pool for full_rank_strategy_10.csv.",
            "Never say top_ranked_screening_pool records were retrieved.",
            "Use evidence-card completeness and extraction audit instead of quality assessment unless formal scores exist.",
        ],
    }


def compact_cards_for_theme_discovery(cards: List[Dict[str, Any]], max_cards: int = 90) -> List[Dict[str, Any]]:
    compact = []
    for card in cards[:max_cards]:
        ext = get_final_extraction(card)
        compact.append({
            "cite_key": card.get("cite_key", ""),
            "title": review_short(card.get("title", ""), 180),
            "year": card.get("year", ""),
            "summary": review_short(card.get("one_sentence_summary", ""), 260),
            "article_type": review_short(ext.get("article_type", ""), 80),
            "aim": review_short(ext.get("aim_or_purpose", ""), 260),
            "context": review_short(ext.get("study_domain_or_context", ""), 220),
            "methods_or_tools": unique(flatten(ext.get("methods_models_tools", [])))[:8],
            "datasets_or_sources": unique(flatten(ext.get("datasets_or_data_sources", [])))[:5],
            "outcomes_or_targets": unique(flatten(ext.get("outcomes_or_targets", [])))[:6],
            "main_findings": unique(flatten(ext.get("main_findings", [])))[:6],
            "limitations_or_gaps": unique(flatten(ext.get("limitations_or_future_work", [])))[:4],
        })
    return compact


def canonical_theme_from_spec(spec: Dict[str, Any], idx: int) -> Dict[str, Any]:
    return {
        "theme_id": clean(spec.get("id", f"theme_{idx}")),
        "theme_name": clean(spec.get("title", f"Theme {idx}")),
        "theme_definition": "Evidence related to " + clean(spec.get("title", f"theme {idx}")).lower() + ".",
        "include_keywords": unique([clean(x) for x in spec.get("keywords", [])]),
        "exclude_keywords": [],
    }


def deterministic_review_theme_schema(max_themes: int) -> Dict[str, Any]:
    max_themes = max(1, safe_int(max_themes, 8))
    specs = GENERIC_REVIEW_THEME_SPECS[:max_themes]
    themes = [canonical_theme_from_spec(spec, i + 1) for i, spec in enumerate(specs)]
    return {
        "schema_version": 1,
        "source": "deterministic_fallback",
        "n_themes": len(themes),
        "multi_theme_assignment_allowed": True,
        "multi_assignment_disclaimer": "Studies may belong to multiple themes; therefore theme counts can sum to more than the number of included studies.",
        "themes": themes,
    }


def extract_json_object(text: str) -> Dict[str, Any]:
    s = str(text or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
    s = re.sub(r"\s*```$", "", s)
    start = s.find("{")
    end = s.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        return json.loads(s[start:end + 1])
    except Exception:
        return {}


def normalize_review_theme_schema(schema: Dict[str, Any], max_themes: int) -> Dict[str, Any]:
    fallback = deterministic_review_theme_schema(max_themes)
    themes = schema.get("themes", []) if isinstance(schema, dict) else []
    if not isinstance(themes, list) or not themes:
        return fallback

    normalized = []
    used = set()
    for i, theme in enumerate(themes[:max(1, safe_int(max_themes, 8))], start=1):
        if not isinstance(theme, dict):
            continue
        name = clean(theme.get("theme_name") or theme.get("title") or theme.get("name"))
        if not name:
            continue
        theme_id = clean(theme.get("theme_id") or re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or f"theme_{i}")
        base = theme_id
        n = 2
        while theme_id in used:
            theme_id = f"{base}_{n}"
            n += 1
        used.add(theme_id)
        normalized.append({
            "theme_id": theme_id,
            "theme_name": name,
            "theme_definition": clean(theme.get("theme_definition") or theme.get("definition") or f"Evidence related to {name.lower()}."),
            "include_keywords": unique(flatten(theme.get("include_keywords") or theme.get("keywords") or []))[:25],
            "exclude_keywords": unique(flatten(theme.get("exclude_keywords") or []))[:25],
        })

    if not normalized:
        return fallback
    return {
        "schema_version": 1,
        "source": clean(schema.get("source", "llm_theme_discovery")) or "llm_theme_discovery",
        "n_themes": len(normalized),
        "multi_theme_assignment_allowed": bool(schema.get("multi_theme_assignment_allowed", True)),
        "multi_assignment_disclaimer": "Studies may belong to multiple themes; therefore theme counts can sum to more than the number of included studies.",
        "themes": normalized,
    }


def discover_review_theme_schema(config_pkg: Dict[str, Any], cards: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    fallback = deterministic_review_theme_schema(args.max_themes)
    if safe_int(args.generate, 0) != 1:
        return fallback

    prompt = f"""
Return ONLY one JSON object defining the canonical theme schema for a literature review.
Do not write prose.
The number of themes must be at most {safe_int(args.max_themes, 8)}.
Use approximately {safe_int(args.max_themes, 8)} themes only if supported by the evidence.
Themes must be review-level synthesis themes, not one theme per article.
Do not include a generic "Other included evidence" theme unless unavoidable.

Required JSON shape:
{{
  "source": "llm_theme_discovery",
  "multi_theme_assignment_allowed": true,
  "themes": [
    {{
      "theme_id": "short_snake_case",
      "theme_name": "Human readable theme name",
      "theme_definition": "One sentence definition",
      "include_keywords": ["keyword", "phrase"],
      "exclude_keywords": ["keyword", "phrase"]
    }}
  ]
}}

Review configuration:
{json.dumps({
    "title": config_pkg.get("title", []),
    "objective": config_pkg.get("objective", []),
    "research_questions": config_pkg.get("research_questions", []),
    "keywords": config_pkg.get("keywords", []),
    "inclusion": config_pkg.get("inclusion", []),
    "exclusion": config_pkg.get("exclusion", []),
}, ensure_ascii=False, indent=2)}

Compact article evidence cards:
{json.dumps(compact_cards_for_theme_discovery(cards), ensure_ascii=False, indent=2)}
""".strip()
    try:
        response, _ = run_with_heartbeat(
            "LLM theme discovery",
            lambda: call_llm_with_retry(prompt, args),
            args.progress_interval,
        )
        schema = extract_json_object(extract_llm_content(response))
        return normalize_review_theme_schema(schema, args.max_themes)
    except Exception as e:
        progress_write(f"Theme discovery failed; using deterministic fallback: {e}")
        return fallback


def load_or_build_review_theme_schema(out_dir: Path, config_pkg: Dict[str, Any], cards: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    path = out_dir / "review_theme_schema.json"
    if path.exists() and not args.force:
        schema = normalize_review_theme_schema(read_json(path), args.max_themes)
    else:
        schema = discover_review_theme_schema(config_pkg, cards, args)
    write_json(path, schema)
    return schema


def score_text_for_theme(text: str, theme: Dict[str, Any]) -> int:
    text = text.lower()
    score = 0
    for kw in flatten(theme.get("include_keywords", [])):
        k = clean(kw).lower()
        if k and k in text:
            score += 3 if " " in k else 1
    for kw in flatten(theme.get("exclude_keywords", [])):
        k = clean(kw).lower()
        if k and k in text:
            score -= 3 if " " in k else 1
    return score


def build_article_theme_assignments(cards: List[Dict[str, Any]], schema: Dict[str, Any], max_themes_per_article: int = 3) -> List[Dict[str, Any]]:
    themes = schema.get("themes", [])
    assignments = []
    for card in cards:
        text = card_text_for_review_theme(card)
        scored = []
        for theme in themes:
            score = score_text_for_theme(text, theme)
            if score > 0:
                scored.append((theme, score))
        scored.sort(key=lambda x: (-x[1], clean(x[0].get("theme_name", "")).lower()))
        if not scored and themes:
            scored = [(themes[0], 1)]
        selected = scored[:max_themes_per_article if schema.get("multi_theme_assignment_allowed", True) else 1]
        assignments.append({
            "article_index": card.get("article_index", ""),
            "input_rank": card.get("input_rank", ""),
            "cite_key": card.get("cite_key", ""),
            "title": card.get("title", ""),
            "assigned_themes": [
                {
                    "theme_id": theme.get("theme_id", ""),
                    "theme_name": theme.get("theme_name", ""),
                    "score": score,
                }
                for theme, score in selected
            ],
        })
    return assignments


def build_theme_evidence_packs_from_assignments(
    cards: List[Dict[str, Any]],
    schema: Dict[str, Any],
    assignments: List[Dict[str, Any]],
    max_articles_per_theme: int = 25,
) -> List[Dict[str, Any]]:
    cards_by_key = {clean(c.get("cite_key", "")): c for c in cards}
    themes_by_id = {t.get("theme_id"): t for t in schema.get("themes", [])}
    buckets: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {theme_id: [] for theme_id in themes_by_id}

    for assignment in assignments:
        card = cards_by_key.get(clean(assignment.get("cite_key", "")))
        if not card:
            continue
        for item in assignment.get("assigned_themes", []):
            buckets.setdefault(item.get("theme_id", ""), []).append((safe_int(item.get("score"), 1), card))

    packs = []
    for theme_id, scored_cards in buckets.items():
        theme = themes_by_id.get(theme_id, {"theme_id": theme_id, "theme_name": theme_id})
        scored_cards.sort(key=lambda x: (-x[0], safe_int(x[1].get("input_rank"), 999999)))
        selected = [c for _, c in scored_cards[:max_articles_per_theme]]
        methods, datasets, metrics, findings, limitations, numerical, citations, titles = [], [], [], [], [], [], [], []
        for c in selected:
            ext = get_final_extraction(c)
            if clean(c.get("cite_key", "")):
                citations.append(clean(c.get("cite_key", "")))
            titles.append(clean(c.get("title", "")))
            methods.extend(flatten(ext.get("methods_models_tools", []))[:6])
            datasets.extend(flatten(ext.get("datasets_or_data_sources", []))[:5])
            metrics.extend(flatten(ext.get("evaluation_metrics", []))[:5])
            findings.extend(flatten(ext.get("main_findings", []))[:8])
            limitations.extend(flatten(ext.get("limitations_or_future_work", []))[:5])
            numerical.extend(flatten(ext.get("main_numerical_results", []))[:5])
        packs.append({
            "theme_id": theme_id,
            "theme_name": theme.get("theme_name", ""),
            "theme_definition": theme.get("theme_definition", ""),
            "n_assigned_articles": len(scored_cards),
            "n_articles_in_pack": len(selected),
            "supporting_citation_keys": unique(citations)[:15],
            "representative_titles": unique(titles)[:10],
            "recurring_methods_or_tools": unique(methods)[:18],
            "recurring_datasets_or_sources": unique(datasets)[:14],
            "recurring_metrics_or_outcomes": unique(metrics)[:14],
            "recurring_numerical_results": unique(numerical)[:12],
            "recurring_findings": unique(findings)[:22],
            "recurring_limitations_or_gaps": unique(limitations)[:18],
            "article_evidence": [compact_article_for_review_theme(c) for c in selected[:14]],
        })
    packs.sort(key=lambda p: (-safe_int(p.get("n_assigned_articles"), 0), clean(p.get("theme_name", ""))))
    return packs


def deterministic_theme_brief(pack: Dict[str, Any]) -> Dict[str, Any]:
    ckeys = pack.get("supporting_citation_keys", [])[:6]
    cite = f"\\cite{{{','.join(ckeys[:5])}}}" if ckeys else ""
    methods = pack.get("recurring_methods_or_tools", [])[:5]
    findings = pack.get("recurring_findings", [])[:5]
    gaps = pack.get("recurring_limitations_or_gaps", [])[:4]
    points = []
    if methods:
        points.append("Recurring methods/tools include " + "; ".join(methods) + ".")
    points.extend(findings[:4])
    if gaps:
        points.append("Common gaps or limitations include " + "; ".join(gaps) + ".")
    return {
        "theme_id": pack.get("theme_id"),
        "theme_name": pack.get("theme_name"),
        "n_articles": pack.get("n_assigned_articles"),
        "synthesis_brief": f"{pack.get('theme_name')} is supported by {pack.get('n_assigned_articles')} assigned article-theme links. {cite}",
        "synthesis_points": points,
        "limitations_or_gaps": gaps,
        "supporting_citation_keys": ckeys,
        "source": "deterministic_fallback",
    }


def synthesize_theme_brief_with_llm(pack: Dict[str, Any], schema: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    prompt = f"""
Return ONLY one JSON object for a theme synthesis brief.
Do not invent citation keys. Use only supporting_citation_keys.
Use at most 5 citations in any one sentence.
Do not list articles one by one.

Required JSON shape:
{{
  "theme_id": "{pack.get('theme_id')}",
  "theme_name": "{pack.get('theme_name')}",
  "n_articles": {safe_int(pack.get('n_assigned_articles'), 0)},
  "synthesis_brief": "One review-style paragraph with citations.",
  "synthesis_points": ["cross-study pattern", "contrast", "evaluation trend"],
  "limitations_or_gaps": ["gap"],
  "supporting_citation_keys": []
}}

Multi-theme disclaimer: {schema.get('multi_assignment_disclaimer', '')}

Theme evidence pack:
{json.dumps(pack, ensure_ascii=False, indent=2)}
""".strip()
    response, _ = run_with_heartbeat(
        f"LLM theme synthesis {pack.get('theme_id')}",
        lambda: call_llm_with_retry(prompt, args),
        args.progress_interval,
    )
    obj = extract_json_object(extract_llm_content(response))
    if not obj:
        return deterministic_theme_brief(pack)
    obj["theme_id"] = clean(obj.get("theme_id") or pack.get("theme_id"))
    obj["theme_name"] = clean(obj.get("theme_name") or pack.get("theme_name"))
    obj["n_articles"] = safe_int(obj.get("n_articles"), safe_int(pack.get("n_assigned_articles"), 0))
    obj["supporting_citation_keys"] = [k for k in unique(flatten(obj.get("supporting_citation_keys", []))) if k in set(pack.get("supporting_citation_keys", []))]
    obj["source"] = "llm_theme_synthesis"
    return obj


def load_or_build_theme_synthesis_briefs(out_dir: Path, schema: Dict[str, Any], packs: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    path = out_dir / "theme_synthesis_briefs.json"
    if path.exists() and not args.force:
        data = read_json(path)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("briefs"), list):
            return data["briefs"]

    briefs = []
    for pack in tqdm(packs, desc="Theme synthesis briefs", unit="theme", dynamic_ncols=True):
        if safe_int(args.generate, 0) == 1:
            try:
                briefs.append(synthesize_theme_brief_with_llm(pack, schema, args))
            except Exception as e:
                progress_write(f"Theme synthesis failed for {pack.get('theme_id')}; using fallback: {e}")
                briefs.append(deterministic_theme_brief(pack))
        else:
            briefs.append(deterministic_theme_brief(pack))
    write_json(path, briefs)
    return briefs


def summarize_theme_assignments(assignments: List[Dict[str, Any]], schema: Dict[str, Any]) -> Dict[str, Any]:
    counts = Counter()
    for assignment in assignments:
        for item in assignment.get("assigned_themes", []):
            if clean(item.get("theme_id", "")):
                counts[clean(item.get("theme_id", ""))] += 1
    names = {t.get("theme_id"): t.get("theme_name") for t in schema.get("themes", [])}
    return {
        "n_themes": schema.get("n_themes", len(schema.get("themes", []))),
        "multi_theme_assignment_allowed": schema.get("multi_theme_assignment_allowed", True),
        "multi_assignment_disclaimer": schema.get("multi_assignment_disclaimer", "Studies may belong to multiple themes; therefore theme counts can sum to more than the number of included studies."),
        "theme_counts": [
            {
                "theme_id": theme_id,
                "theme_name": names.get(theme_id, theme_id),
                "assigned_article_theme_links": count,
            }
            for theme_id, count in counts.most_common()
        ],
    }


def card_text_for_review_theme(card: Dict[str, Any]) -> str:
    ext = get_final_extraction(card)
    parts = [
        card.get("title", ""), card.get("journal", ""), card.get("one_sentence_summary", ""),
        ext.get("article_type", ""), ext.get("aim_or_purpose", ""), ext.get("study_domain_or_context", ""),
        ext.get("datasets_or_data_sources", []), ext.get("methods_models_tools", []), ext.get("outcomes_or_targets", []),
        ext.get("validation_or_evaluation", ""), ext.get("evaluation_metrics", []), ext.get("main_numerical_results", []),
        ext.get("important_tables_figures_taxonomies", []), ext.get("main_findings", []), ext.get("limitations_or_future_work", []),
        ext.get("quality_evidence", []),
    ]
    return " ".join(flatten(parts)).lower()


def score_card_for_review_theme(card: Dict[str, Any], theme: Dict[str, Any]) -> int:
    txt = card_text_for_review_theme(card)
    score = 0
    for kw in theme.get("keywords", []):
        k = kw.lower()
        if k in txt:
            score += 3 if " " in k else 1
    return score


def assign_review_themes(card: Dict[str, Any], max_themes: int = 3) -> List[Tuple[str, int]]:
    scored = []
    for theme in REVIEW_THEME_SPECS:
        score = score_card_for_review_theme(card, theme)
        if score > 0:
            scored.append((theme["id"], score))
    scored.sort(key=lambda x: x[1], reverse=True)
    if not scored:
        return [("other", 1)]
    return scored[:max_themes]


def compact_article_for_review_theme(card: Dict[str, Any]) -> Dict[str, Any]:
    ext = get_final_extraction(card)
    key = clean(card.get("cite_key", ""))
    return {
        "cite_key": key,
        "latex_citation": f"\\cite{{{key}}}" if key else "",
        "title": card.get("title", ""),
        "year": card.get("year", ""),
        "article_type": ext.get("article_type", ""),
        "aim": review_short(ext.get("aim_or_purpose", ""), 360),
        "context": review_short(ext.get("study_domain_or_context", ""), 260),
        "methods_or_tools": unique(flatten(ext.get("methods_models_tools", [])))[:10],
        "datasets_or_sources": unique(flatten(ext.get("datasets_or_data_sources", [])))[:8],
        "outcomes_or_targets": unique(flatten(ext.get("outcomes_or_targets", [])))[:8],
        "metrics": unique(flatten(ext.get("evaluation_metrics", [])))[:8],
        "numerical_results": unique(flatten(ext.get("main_numerical_results", [])))[:8],
        "main_findings": unique(flatten(ext.get("main_findings", [])))[:10],
        "limitations_or_gaps": unique(flatten(ext.get("limitations_or_future_work", [])))[:8],
    }


def build_review_theme_evidence_packs(cards: List[Dict[str, Any]], max_articles_per_theme: int = 25) -> List[Dict[str, Any]]:
    theme_map = {t["id"]: dict(t) for t in REVIEW_THEME_SPECS}
    theme_map["other"] = {"id": "other", "title": "Other included evidence", "keywords": [], "questions": ["What residual evidence does not fit the main thematic groups?"]}
    buckets: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
    for card in cards:
        for theme_id, score in assign_review_themes(card):
            buckets.setdefault(theme_id, []).append((score, card))
    packs = []
    for theme_id, scored_cards in buckets.items():
        theme = theme_map[theme_id]
        scored_cards.sort(key=lambda x: (-x[0], safe_int(x[1].get("input_rank"), 999999)))
        selected = [c for _, c in scored_cards[:max_articles_per_theme]]
        methods, datasets, metrics, findings, limitations, numerical, citations, titles = [], [], [], [], [], [], [], []
        for c in selected:
            ext = get_final_extraction(c)
            if clean(c.get("cite_key", "")):
                citations.append(clean(c.get("cite_key", "")))
            titles.append(clean(c.get("title", "")))
            methods.extend(flatten(ext.get("methods_models_tools", []))[:6])
            datasets.extend(flatten(ext.get("datasets_or_data_sources", []))[:5])
            metrics.extend(flatten(ext.get("evaluation_metrics", []))[:5])
            findings.extend(flatten(ext.get("main_findings", []))[:8])
            limitations.extend(flatten(ext.get("limitations_or_future_work", []))[:5])
            numerical.extend(flatten(ext.get("main_numerical_results", []))[:5])
        packs.append({
            "theme_id": theme_id,
            "theme_title": theme["title"],
            "theme_review_questions": theme.get("questions", []),
            "n_assigned_articles": len(scored_cards),
            "n_articles_in_pack": len(selected),
            "supporting_citation_keys": unique(citations)[:12],
            "representative_titles": unique(titles)[:10],
            "recurring_methods_or_tools": unique(methods)[:18],
            "recurring_datasets_or_sources": unique(datasets)[:14],
            "recurring_metrics_or_outcomes": unique(metrics)[:14],
            "recurring_numerical_results": unique(numerical)[:12],
            "recurring_findings": unique(findings)[:22],
            "recurring_limitations_or_gaps": unique(limitations)[:18],
            "article_evidence": [compact_article_for_review_theme(c) for c in selected[:14]],
        })
    packs.sort(key=lambda p: (-safe_int(p.get("n_assigned_articles"), 0), clean(p.get("theme_title", ""))))
    return packs


def build_deterministic_review_theme_briefs(packs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    briefs = []
    for pack in packs:
        ckeys = pack.get("supporting_citation_keys", [])[:6]
        cite = f"\\cite{{{','.join(ckeys[:5])}}}" if ckeys else ""
        findings = pack.get("recurring_findings", [])[:5]
        methods = pack.get("recurring_methods_or_tools", [])[:5]
        gaps = pack.get("recurring_limitations_or_gaps", [])[:4]
        points = []
        if methods:
            points.append("Recurring methods/tools include " + "; ".join(methods) + ".")
        points.extend(findings[:4])
        if gaps:
            points.append("Common gaps or limitations include " + "; ".join(gaps) + ".")
        claim = f"The theme '{pack.get('theme_title')}' contains {pack.get('n_assigned_articles')} assigned articles and captures a recurring pattern across the included evidence. {cite}"
        briefs.append({"theme_id": pack.get("theme_id"), "theme_title": pack.get("theme_title"), "n_articles": pack.get("n_assigned_articles"), "review_style_claim": claim, "synthesis_points": points, "contrasts_or_subthemes": [], "limitations_or_gaps": gaps, "supporting_citation_keys": ckeys})
    return briefs


def build_review_style_section_contexts(context: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    cfg = context.get("configuration", {})
    truth = context.get("canonical_truth", {})
    theme_schema = context.get("review_theme_schema", {})
    theme_assignments = context.get("article_theme_assignments", [])
    theme_packs = context.get("theme_evidence_packs", [])
    theme_briefs = context.get("theme_synthesis_briefs", [])
    valid_citations = [{"cite_key": c.get("cite_key", ""), "latex": f"\\cite{{{c.get('cite_key', '')}}}", "title": c.get("title", ""), "year": c.get("year", "")} for c in context.get("article_evidence_cards", []) if clean(c.get("cite_key", ""))]
    project_brief = {"project_index": context.get("project_index"), "project_name": context.get("project_name"), "model_name": context.get("model_name"), "title": cfg.get("title", []), "review_type": cfg.get("review_type", []), "objective": cfg.get("objective", []), "research_questions": cfg.get("research_questions", [])}
    citation_rules = {"use_only_exact_latex_citations_from_valid_citations": True, "never_invent_citation_keys": True, "max_citations_per_cite_command": 5, "valid_citations": valid_citations}
    return {
        "introduction": {"purpose": "Write only the Introduction as a real review introduction, not a pipeline report.", "project": project_brief, "keywords": cfg.get("keywords", []), "scope": {"sources": cfg.get("sources", []), "date_start": cfg.get("date_start", ""), "date_end": cfg.get("date_end", ""), "language": cfg.get("language", "")}},
        "methodology": {"purpose": "Write only the Methods section using canonical counts. Do not call the top-ranked pool retrieved records.", "project": project_brief, "canonical_truth": truth, "search_configuration": {"keywords": cfg.get("keywords", []), "sources": cfg.get("sources", []), "inclusion": cfg.get("inclusion", []), "exclusion": cfg.get("exclusion", []), "date_start": cfg.get("date_start", ""), "date_end": cfg.get("date_end", ""), "language": cfg.get("language", "")}, "fulltext": context.get("fulltext", {}), "llm_article_cards": context.get("llm_article_cards", {})},
        "results": {"purpose": "Write only Results for selection flow, metadata coverage, full-text availability, decisions, extraction completeness, and canonical review themes.", "project": project_brief, "canonical_truth": truth, "review_theme_schema": theme_schema, "theme_assignment_summary": summarize_theme_assignments(theme_assignments, theme_schema), "fulltext": context.get("fulltext", {}), "llm_article_cards": context.get("llm_article_cards", {}), "paper_assets": context.get("paper_assets", {})},
        "main_findings": {"purpose": "Write Main findings as a synthesis across canonical themes. Do not list articles one-by-one.", "project": project_brief, "citation_rules": citation_rules, "canonical_truth": truth, "review_theme_schema": theme_schema, "theme_synthesis_briefs": theme_briefs, "theme_evidence_packs": theme_packs},
        "discussion": {"purpose": "Interpret the theme synthesis and limitations. Do not re-report all article details.", "project": project_brief, "citation_rules": citation_rules, "canonical_truth": truth, "review_theme_schema": theme_schema, "theme_synthesis_briefs": theme_briefs, "theme_evidence_packs": theme_packs, "llm_article_cards": context.get("llm_article_cards", {})},
        "conclusion": {"purpose": "Write the conclusion from the canonical theme briefs.", "project": project_brief, "review_theme_schema": theme_schema, "theme_synthesis_briefs": theme_briefs},
        "abstract": {"purpose": "Write abstract last using canonical counts and canonical theme count.", "project": project_brief, "canonical_truth": truth, "review_theme_schema": theme_schema, "theme_synthesis_briefs": theme_briefs},
    }


def build_section_contexts(context: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return build_review_style_section_contexts(context)


REVIEW_STYLE_POLICY = r"""
REVIEW-STYLE ARTICLE GENERATION POLICY:
- Write like a literature review, not like a pipeline log and not like a list of article cards.
- Synthesize across studies: identify patterns, contrasts, mechanisms, evidence gaps, and evaluation trends.
- Do not summarize one article after another.
- Do not invent citation keys; use only the supplied citation keys.
- Use at most 3--5 citation keys in any one \cite{} command.
- Do not invent numbers, databases, inclusion criteria, datasets, tools, results, or limitations.
- Use canonical_truth for all counts.
- Use review_theme_schema for all theme names and the theme count.
- Never invent additional themes or change the number of themes.
- State that studies may belong to multiple themes when reporting theme counts.
- Never write that 1,000 records were retrieved. 1,000 is the top-ranked screening pool.
- Use "evidence-card completeness and extraction audit" instead of "quality assessment" unless formal quality scores are provided.
- Use LaTeX only. Escape ampersands as \&.
""".strip()


def build_methodology_prompt(context: Dict[str, Any]) -> str:
    return f"""You are writing the Methods section of a systematic/scoping review.

{REVIEW_STYLE_POLICY}

Write only:
\\section{{Methods}}

Required subsections:
\\subsection{{Review design and objective}}
\\subsection{{Search sources and keyword strategy}}
\\subsection{{Eligibility criteria}}
\\subsection{{Article retrieval, cleaning, and ranking}}
\\subsection{{Full-text acquisition and LLM-assisted evidence-card extraction}}
\\subsection{{Evidence-card completeness and synthesis preparation}}

Use the canonical truth exactly. Do not call the top-ranked screening pool retrieved records.

COMPACT METHODS CONTEXT:
{context_to_text(context)}""".strip()


def build_results_prompt(context: Dict[str, Any]) -> str:
    return f"""You are writing the Results section of a review article.

{REVIEW_STYLE_POLICY}

Write only:
\\section{{Results}}

Results must report selection flow, metadata coverage, full-text availability, screening decisions, extraction completeness, and canonical review themes.
Report canonical theme counts/schema context only; put scientific interpretation in Main findings.
Use exact counts from canonical_truth.

Required subsections:
\\subsection{{Study selection flow}}
\\subsection{{Candidate metadata and ranking coverage}}
\\subsection{{Full-text availability}}
\\subsection{{LLM screening decisions}}
\\subsection{{Evidence-card completeness and extraction audit}}
\\subsection{{Canonical review themes}}

COMPACT RESULTS CONTEXT:
{context_to_text(context)}""".strip()


def build_main_findings_prompt(context: Dict[str, Any]) -> str:
    return f"""You are writing the Main findings / Thematic synthesis section of a literature review.

{REVIEW_STYLE_POLICY}

This is the most important section. Make it read like a real review article.

Write only:
\\section{{Main findings}}

Instructions:
- Start with a compact longtable summarizing the strongest themes.
- Then write conceptual synthesis paragraphs.
- Do not list every article.
- Use theme_synthesis_briefs as the main source.
- Use the theme_evidence_packs only to add detail and representative citations.
- Use exactly the theme names in review_theme_schema.
- Discuss agreements, differences, recurring mechanisms, evaluation practices, and gaps.

Required subsections:
\\subsection{{Overview of canonical themes}}
\\subsection{{Theme-level evidence patterns}}
\\subsection{{Methods, tools, and data sources}}
\\subsection{{Outcomes, metrics, and evaluation practices}}
\\subsection{{Cross-cutting limitations and open problems}}

COMPACT MAIN FINDINGS CONTEXT:
{context_to_text(context)}""".strip()


def build_discussion_prompt(context: Dict[str, Any]) -> str:
    return f"""You are writing the Discussion section of a review article.

{REVIEW_STYLE_POLICY}

Write only:
\\section{{Discussion}}

Interpret what the theme synthesis means. Do not repeat all article details.

Required subsections:
\\subsection{{Interpretation of the evidence}}
\\subsection{{Strengths of the evidence base}}
\\subsection{{Gaps and limitations in the literature}}
\\subsection{{Implications for future research}}
\\subsection{{Limitations of the automated review process}}

COMPACT DISCUSSION CONTEXT:
{context_to_text(context)}""".strip()


def build_introduction_prompt(context: Dict[str, Any]) -> str:
    return f"""You are writing the Introduction of a review article.

{REVIEW_STYLE_POLICY}

Write only:
\\section{{Introduction}}

Make it sound like a real review introduction. Do not focus on the pipeline.

Required subsections:
\\subsection{{Background}}
\\subsection{{Motivation and gap}}
\\subsection{{Review objective}}
\\subsection{{Research questions}}
\\subsection{{Contributions}}

COMPACT INTRODUCTION CONTEXT:
{context_to_text(context)}""".strip()


def build_conclusion_prompt(context: Dict[str, Any]) -> str:
    return f"""You are writing the Conclusion of a review article.

{REVIEW_STYLE_POLICY}

Write only:
\\section{{Conclusion}}

Required subsections:
\\subsection{{Overall conclusion}}
\\subsection{{Most important evidence patterns}}
\\subsection{{Practical and research implications}}
\\subsection{{Future work}}

COMPACT CONCLUSION CONTEXT:
{context_to_text(context)}""".strip()


def build_abstract_prompt(context: Dict[str, Any]) -> str:
    return f"""You are writing the Abstract of a review article.

{REVIEW_STYLE_POLICY}

Write only:
\\begin{{abstract}}
...
\\end{{abstract}}

Use 180--250 words. Include the correct canonical counts. Do not say 1,000 records were retrieved.

COMPACT ABSTRACT CONTEXT:
{context_to_text(context)}""".strip()


def sanitize_latex_citations(content: str, valid_keys: set) -> Tuple[str, Dict[str, Any]]:
    if not valid_keys:
        return content, {"checked": False, "reason": "no_valid_keys_loaded"}
    invalid = []
    checked = 0
    citation_walls_trimmed = 0
    def repl(match: re.Match) -> str:
        nonlocal checked, citation_walls_trimmed
        checked += 1
        command = match.group(1)
        body = match.group(2)
        keys = [clean(k) for k in body.split(",") if clean(k)]
        kept = []
        for k in keys:
            if k in valid_keys and k not in kept:
                kept.append(k)
            elif k and k not in valid_keys:
                invalid.append(k)
        if len(kept) > 5:
            citation_walls_trimmed += 1
            kept = kept[:5]
        if not kept:
            return ""
        return f"{command}" + "{" + ",".join(kept) + "}"
    pattern = r"(\\cite\w*\*?(?:\[[^\]]*\]){0,2})\{([^{}]+)\}"
    cleaned = re.sub(pattern, repl, content)
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
    cleaned = re.sub(r" {2,}", " ", cleaned)
    return cleaned, {"checked": True, "n_cite_commands": checked, "invalid_cite_keys": unique(invalid), "n_invalid_cite_keys": len(unique(invalid)), "citation_walls_trimmed": citation_walls_trimmed}


def validate_count_wording(out_dir: Path, truth: Dict[str, Any]) -> Dict[str, Any]:
    text = read_text(out_dir / "Main.tex")
    warnings = []
    if re.search(r"\b1,?000\s+records\s+were\s+retrieved", text, flags=re.I):
        warnings.append("Text says 1,000 records were retrieved; should be top-ranked screening pool.")
    if re.search(r"quality assessment was performed|quality assessment results|quality assessment summary", text, flags=re.I):
        warnings.append("Text claims quality assessment was performed; use evidence-card completeness unless formal scores exist.")
    report = {"warnings": warnings, "n_warnings": len(warnings)}
    write_json(out_dir / "count_wording_validation.json", report)
    write_json(out_dir / "Reports" / "FINAL_count_wording_report.json", report)
    return report


def collect_allowed_numbers(*objects: Any) -> set:
    allowed = set()
    def visit(x: Any) -> None:
        if isinstance(x, dict):
            for v in x.values():
                visit(v)
        elif isinstance(x, list):
            for v in x:
                visit(v)
        elif isinstance(x, (int, float)):
            allowed.add(str(int(x)) if float(x).is_integer() else str(x))
        elif isinstance(x, str):
            for n in re.findall(r"\b\d{1,3}(?:,\d{3})+|\b\d+(?:\.\d+)?%?", x):
                allowed.add(n.strip("%").replace(",", ""))
    for obj in objects:
        visit(obj)
    return allowed


def validate_numeric_provenance(
    out_dir: Path,
    truth: Dict[str, Any],
    schema: Dict[str, Any],
    assignments: List[Dict[str, Any]],
    briefs: List[Dict[str, Any]],
    compact_cards: List[Dict[str, Any]],
    theme_evidence_packs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    main_text = read_text(out_dir / "Main.tex")
    seen = unique([n.strip("%").replace(",", "") for n in re.findall(r"\b\d{1,3}(?:,\d{3})+|\b\d+(?:\.\d+)?%?", main_text)])
    allowed = collect_allowed_numbers(truth, schema, assignments, briefs, compact_cards, theme_evidence_packs)
    # Common section/table/page numbers are structural LaTeX output, not evidence claims.
    structural = {str(i) for i in range(0, 21)}
    unknown = [n for n in seen if n not in allowed and n not in structural]
    report = {
        "numbers_in_main_tex": seen,
        "allowed_numbers": sorted(allowed, key=lambda x: (len(x), x)),
        "unknown_numbers": unknown,
        "unknown_numbers_count": len(unknown),
    }
    write_json(out_dir / "Reports" / "FINAL_numeric_provenance_report.json", report)
    return report


def validate_theme_consistency(out_dir: Path, schema: Dict[str, Any]) -> Dict[str, Any]:
    main_text = read_text(out_dir / "Main.tex")
    generated_tables = read_text(out_dir / "GeneratedTables.tex")
    warnings = []
    if "tab:theme-synthesis" in generated_tables or "table_4_theme_synthesis" in generated_tables:
        warnings.append("Old heuristic theme synthesis table is still present in GeneratedTables.tex.")
    if re.search(r"Other included evidence\s*(?:=|&)\s*139", generated_tables + "\n" + main_text, flags=re.I):
        warnings.append("Old 'Other included evidence = 139' pattern is still present.")
    n_themes = safe_int(schema.get("n_themes"), len(schema.get("themes", [])))
    for n in re.findall(r"\b(\d+)\s+(?:identified\s+)?themes\b", main_text, flags=re.I):
        if safe_int(n, -1) != n_themes:
            warnings.append(f"Theme count mismatch: text says {n} themes but schema has {n_themes}.")
    if "studies may belong to multiple themes" not in main_text.lower():
        warnings.append("Multi-assignment disclaimer is missing from Main.tex.")
    report = {
        "schema_theme_count": n_themes,
        "warnings": warnings,
        "n_warnings": len(warnings),
    }
    write_json(out_dir / "Reports" / "FINAL_theme_consistency_report.json", report)
    return report


def write_compile_script(out_dir: Path) -> None:
    script = """#!/usr/bin/env bash
set -e
pdflatex -interaction=nonstopmode Main.tex || true
bibtex Main || true
pdflatex -interaction=nonstopmode Main.tex || true
pdflatex -interaction=nonstopmode Main.tex || true
echo \"Done. Check Main.pdf, Main.log, citation_validation.json, and count_wording_validation.json.\"
"""
    path = out_dir / "compile_latex.sh"
    write_text(path, script)
    try:
        os.chmod(path, 0o755)
    except Exception:
        pass


def run_latex_compile(out_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    commands = [
        ["pdflatex", "-interaction=nonstopmode", "Main.tex"],
        ["bibtex", "Main"],
        ["pdflatex", "-interaction=nonstopmode", "Main.tex"],
        ["pdflatex", "-interaction=nonstopmode", "Main.tex"],
    ]
    results = []
    for i, cmd in enumerate(commands, start=1):
        progress_write(f"[compile {i}/{len(commands)}] {' '.join(cmd)}")
        started = time.time()
        try:
            completed = subprocess.run(
                cmd,
                cwd=str(out_dir),
                text=True,
                capture_output=True,
                timeout=max(60, safe_int(args.llm_timeout, 600)),
            )
            results.append({
                "command": cmd,
                "returncode": completed.returncode,
                "elapsed_seconds": round(time.time() - started, 2),
                "stdout_tail": completed.stdout[-4000:],
                "stderr_tail": completed.stderr[-4000:],
            })
        except Exception as e:
            results.append({
                "command": cmd,
                "returncode": -1,
                "elapsed_seconds": round(time.time() - started, 2),
                "error": str(e),
            })
    status = "complete" if (out_dir / "Main.pdf").exists() else "failed"
    return {"status": status, "commands": results, "pdf": str(out_dir / "Main.pdf")}


def build_package(project_dir: Path, project_index: int, args: argparse.Namespace) -> None:
    out_dir = project_dir / "FinalArticle"
    out_dir.mkdir(parents=True, exist_ok=True)
    config_pkg = get_config_package(project_dir)
    ranking_pkg = get_ranking_package(project_dir)
    fulltext_pkg = get_fulltext_package(project_dir)
    cards_pkg = build_cards_package(project_dir, args)
    progress_write(f"Preparing references: resolving BibTeX for {len(cards_pkg['selected_cards'])} selected cards before LLM generation")
    bib_text, bib_stats = build_bib(cards_pkg["selected_cards"], mailto=args.mailto, heartbeat_seconds=args.progress_interval, cache_path=out_dir / "bibtex_doi_cache.json")
    title = config_pkg["title"][0] if config_pkg["title"] else project_dir.name
    context = {
        "project_index": project_index,
        "project_name": project_dir.name,
        "model_name": MODEL_NAME,
        "configuration": {"title": config_pkg["title"], "review_type": config_pkg["review_type"], "objective": config_pkg["objective"], "research_questions": config_pkg["research_questions"], "keywords": config_pkg["keywords"], "sources": config_pkg["sources"], "inclusion": config_pkg["inclusion"], "exclusion": config_pkg["exclusion"], "quality_questions": config_pkg["quality_questions"], "date_start": config_pkg["date_start"], "date_end": config_pkg["date_end"], "language": config_pkg["language"]},
        "ranking_and_metadata": ranking_pkg,
        "fulltext": fulltext_pkg,
        "llm_article_cards": cards_pkg["stats"],
        "bibtex_generation": bib_stats,
        "article_evidence_cards": cards_pkg["compact_cards"],
    }
    canonical_truth = build_review_truth(project_dir, ranking_pkg, cards_pkg)
    review_theme_schema = load_or_build_review_theme_schema(out_dir, config_pkg, cards_pkg["selected_cards"], args)
    article_theme_assignments = build_article_theme_assignments(cards_pkg["selected_cards"], review_theme_schema)
    theme_evidence_packs = build_theme_evidence_packs_from_assignments(
        cards_pkg["selected_cards"],
        review_theme_schema,
        article_theme_assignments,
        max_articles_per_theme=25,
    )
    theme_synthesis_briefs = load_or_build_theme_synthesis_briefs(out_dir, review_theme_schema, theme_evidence_packs, args)
    paper_assets = build_paper_assets(project_dir=project_dir, out_dir=out_dir, config_pkg=config_pkg, ranking_pkg=ranking_pkg, cards_pkg=cards_pkg, args=args)
    context["paper_assets"] = paper_assets
    context["canonical_truth"] = canonical_truth
    context["review_theme_schema"] = review_theme_schema
    context["article_theme_assignments"] = article_theme_assignments
    context["theme_evidence_packs"] = theme_evidence_packs
    context["theme_synthesis_briefs"] = theme_synthesis_briefs
    section_contexts = build_section_contexts(context)
    section_contexts_dir = out_dir / "section_contexts"
    old_methodlog = out_dir / "Methodlog.tex"
    if old_methodlog.exists():
        old_methodlog.unlink()
    write_json(out_dir / "project_context.json", context)
    write_json(out_dir / "project_truth.json", canonical_truth)
    write_json(out_dir / "review_theme_schema.json", review_theme_schema)
    write_json(out_dir / "article_theme_assignments.json", article_theme_assignments)
    write_json(out_dir / "theme_evidence_packs.json", theme_evidence_packs)
    write_json(out_dir / "theme_synthesis_briefs.json", theme_synthesis_briefs)
    write_json(out_dir / "article_evidence_cards.json", cards_pkg["compact_cards"])
    for section_name, section_context in section_contexts.items():
        write_json(section_contexts_dir / f"{section_name}_context.json", section_context)
    write_text(out_dir / "reference.bib", bib_text)
    write_text(out_dir / "article_evidence_cards.tex", build_evidence_cards_tex(cards_pkg["selected_cards"]))
    write_text(out_dir / "Main.tex", build_main_tex(title))
    write_text_if_missing(out_dir / "Introduction.tex", "% Paste DeepSeek-generated introduction here.\n\\section{Introduction}\n")
    write_text_if_missing(out_dir / "Methodology.tex", "% Paste DeepSeek-generated methodology here.\n\\section{Methods}\n")
    write_text_if_missing(out_dir / "MainFindings.tex", "% Paste DeepSeek-generated main findings here.\n\\section{Main findings}\n")
    write_text_if_missing(out_dir / "Results.tex", "% Paste DeepSeek-generated results here.\n\\section{Results}\n")
    write_text_if_missing(out_dir / "Discussion.tex", "% Paste DeepSeek-generated discussion here.\n\\section{Discussion}\n")
    write_text_if_missing(out_dir / "Conclusion.tex", "\\section{Conclusion}\n\n% Paste generated conclusion here.\n")
    write_text_if_missing(out_dir / "Abstract.tex", "\\begin{abstract}\n% Generate this last after Methods, Results, Main Findings, and Discussion are ready.\n\\end{abstract}\n")
    write_text(out_dir / "DeepSeek_Introduction_Prompt.txt", build_introduction_prompt(section_contexts["introduction"]))
    write_text(out_dir / "DeepSeek_Methodology_Prompt.txt", build_methodology_prompt(section_contexts["methodology"]))
    write_text(out_dir / "DeepSeek_Results_Prompt.txt", build_results_prompt(section_contexts["results"]))
    write_text(out_dir / "DeepSeek_MainFindings_Prompt.txt", build_main_findings_prompt(section_contexts["main_findings"]))
    write_text(out_dir / "DeepSeek_Discussion_Prompt.txt", build_discussion_prompt(section_contexts["discussion"]))
    write_text(out_dir / "DeepSeek_Conclusion_Prompt.txt", build_conclusion_prompt(section_contexts["conclusion"]))
    write_text(out_dir / "DeepSeek_Abstract_Prompt.txt", build_abstract_prompt(section_contexts["abstract"]))
    generation_summary = generate_article_sections(out_dir, args)
    write_text(out_dir / "Main.tex", build_main_tex(title, build_main_body_from_files(out_dir)))
    reports_dir = out_dir / "Reports"
    citation_report = validate_main_tex_citations(out_dir)
    wording_report = validate_count_wording(out_dir, canonical_truth)
    numeric_report = validate_numeric_provenance(
        out_dir,
        canonical_truth,
        review_theme_schema,
        article_theme_assignments,
        theme_synthesis_briefs,
        cards_pkg["compact_cards"],
        theme_evidence_packs,
    )
    theme_report = validate_theme_consistency(out_dir, review_theme_schema)
    write_compile_script(out_dir)
    compile_report = run_latex_compile(out_dir, args) if safe_int(args.compile, 1) == 1 else {"status": "skipped_compile_0"}
    write_json(reports_dir / "FINAL_compile_report.json", compile_report)
    print("=" * 120)
    print("FINAL REVIEW-STYLE ARTICLE PACKAGE CREATED")
    print("=" * 120)
    print(f"Project            : {project_index} | {project_dir.name}")
    print(f"Output folder      : {out_dir}")
    print(f"Article cards total: {cards_pkg['stats']['article_cards_generated']}")
    print(f"Cards used package : {cards_pkg['stats']['cards_used_for_writing_package']}")
    print(f"Decisions          : {cards_pkg['stats']['decision_counts']}")
    print(f"Canonical truth    : {out_dir / 'project_truth.json'}")
    print(f"Theme schema       : {out_dir / 'review_theme_schema.json'}")
    print(f"Theme assignments  : {out_dir / 'article_theme_assignments.json'}")
    print(f"Theme packs        : {out_dir / 'theme_evidence_packs.json'}")
    print(f"Theme briefs       : {out_dir / 'theme_synthesis_briefs.json'}")
    print(f"LLM generation     : {generation_summary.get('status', 'unknown')}")
    print(f"Citation validation: invalid={citation_report.get('n_invalid_citation_keys', 0)} | report={reports_dir / 'FINAL_citation_report.json'}")
    print(f"Wording validation : warnings={wording_report.get('n_warnings', 0)} | report={reports_dir / 'FINAL_count_wording_report.json'}")
    print(f"Numeric validation : unknown={numeric_report.get('unknown_numbers_count', 0)} | report={reports_dir / 'FINAL_numeric_provenance_report.json'}")
    print(f"Theme validation   : warnings={theme_report.get('n_warnings', 0)} | report={reports_dir / 'FINAL_theme_consistency_report.json'}")
    print(f"Compile status     : {compile_report.get('status')}")
    print(f"Compile script     : {out_dir / 'compile_latex.sh'}")
    print("=" * 120)

# ============================================================
# CLI
# ============================================================

def main():
    load_default_env_files()

    parser = argparse.ArgumentParser()
    parser.add_argument("project", nargs="?", default="", help="Project index, e.g. 1")
    parser.add_argument("--project-index", type=int, default=0)
    parser.add_argument("--project-name", default="")
    parser.add_argument("--max-cards", type=int, default=80)
    parser.add_argument("--max-table-rows", type=int, default=30, help="Maximum article rows shown in long paper tables.")
    parser.add_argument("--max-themes", type=int, default=8, help="Maximum canonical review themes to discover/use.")
    parser.add_argument("--included-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="Regenerate cached theme schema/brief outputs instead of reusing them.")
    parser.add_argument("--compile", type=int, choices=[0, 1], default=1, help="1 compiles Main.tex to Main.pdf at the end; 0 only writes files.")
    parser.add_argument("--mailto", default="", help="Optional email for polite DOI/Crossref BibTeX requests.")
    parser.add_argument("--generate", type=int, choices=[0, 1], default=0, help="0 builds prompts/assets only; 1 sends prompts to the LLM and writes section .tex files.")
    parser.add_argument("--llm-provider", default=env_or_default("LLM_PROVIDER", "deepseek"))
    parser.add_argument("--llm-base-url", default=env_or_default("LLM_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--llm-model", default=env_or_default("LLM_MODEL", "deepseek-chat"))
    parser.add_argument("--llm-api-key", default=env_or_default("LLM_API_KEY", ""))
    parser.add_argument("--llm-temperature", type=float, default=env_float("LLM_TEMPERATURE", 0.2))
    parser.add_argument("--llm-top-p", type=float, default=env_float("LLM_TOP_P", 0.95))
    parser.add_argument("--llm-max-tokens", type=int, default=env_int("LLM_MAX_TOKENS", 100000))
    parser.add_argument("--llm-reasoning-effort", default=env_or_default("LLM_REASONING_EFFORT", ""))
    parser.add_argument("--llm-timeout", type=int, default=env_int("LLM_TIMEOUT", 600))
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=env_int("PROGRESS_INTERVAL", 15),
        help="Seconds between heartbeat messages during long DOI/LLM requests. Use 0 to disable.",
    )
    parser.add_argument("--continue-on-generation-error", action="store_true")
    args = parser.parse_args()

    if LOADED_ENV_FILES:
        print("Loaded .env file(s):")
        for p in LOADED_ENV_FILES:
            print(f"  {p}")
    if safe_int(args.generate, 0) == 1:
        print("LLM config:")
        print(f"  provider : {args.llm_provider}")
        print(f"  base_url : {args.llm_base_url}")
        print(f"  model    : {args.llm_model}")
        print(f"  api_key  : {mask_secret(args.llm_api_key)}")

    project_dir, project_index = select_project(args)
    build_package(project_dir, project_index, args)



main()
