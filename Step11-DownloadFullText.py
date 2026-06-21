#!/usr/bin/env python3
"""
Step 11: Comprehensive Article Download (35 Sources) with Sci-Hub Integration

INPUT:  Output/allarticles2.csv
OUTPUT: DataBase_Strategy3_Top1000/Temp_{loop}/document.pdf per article
        Output/download_log.csv
        Output/decisions_log.csv (appended)

PIPELINE ORDER (tries each source until PDF is obtained):
─────────────────────────────────────────────────────────
01. PubMed Central   – NCBI OA API (free, fast, authoritative)
02. Europe PMC       – European mirror of PMC with extra OA content
03. Unpaywall        – Legal OA PDF finder via DOI
04. CrossRef         – Publisher PDF links via CrossRef API
05. Semantic Scholar  – AI-indexed OA links
06. CORE API         – Open access aggregator
07. scidownl         – Sci-Hub package wrapper
07b. Sci-Hub Direct  – Direct HTTP requests to Sci-Hub mirrors (NO GUI)
08. Elsevier API     – Elsevier full-text API (requires key)
09. Nature           – Direct Nature.com PDF scraper
10. PLOS ONE         – Direct PLOS printable PDF
11. PLOS Genetics    – Direct PLOS Genetics PDF
12. Frontiers        – Frontiersin.org PDF
13. bioRxiv          – bioRxiv preprint PDF (v1/v2/v3)
14. medRxiv          – medRxiv preprint PDF (v1/v2/v3)
15. arXiv            – arXiv preprint PDF
16. Wiley OA         – Wiley online library OA PDF
17. SpringerOpen     – SpringerOpen/BioMedCentral PDF
18. MDPI             – MDPI open access PDF
19. Taylor & Francis – T&F OA PDF
20. OUP              – Oxford University Press OA
21. Cambridge Core   – Cambridge OA
22. Hindawi          – Hindawi OA publisher (10.1155)
23. JCI              – J. Clinical Investigation OA (10.1172)
24. eLife            – eLife OA journal (10.7554)
25. PNAS             – Proc. Natl. Acad. Sci. OA (10.1073)
26. Blood / ASH      – ASH hematology journals (10.1182)
27. Genetics (GSA)   – Genetics Society journals (10.1534)
28. Karger           – Karger OA content (10.1159)
29. De Gruyter       – De Gruyter OA content (10.1515)
30. LWW              – Wolters Kluwer / LWW OA (10.1097)
31. DOAJ             – Directory of Open Access Journals API
32. Internet Archive – Wayback Machine / Scholar archive
33. PLOS (all)       – All PLOS journals catch-all
34. DOI Direct       – Direct DOI URL with PDF accept header
35. Open Access Btn  – openaccessbutton.org API
"""

import os
import sys
import re
import time
import shutil
import json
import hashlib
import logging
import subprocess
import requests
import traceback
import threading
import pandas as pd
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus, urlparse, unquote
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

# ============================================================
# CONFIGURATION
# ============================================================

DB_DIR = "DataBase_Strategy3_Top1000"
PROJECTS_DIR = "Projects"
RERANK_OUTPUT_DIR = "_rerank_outputs"
BEST_STRATEGY_FILE = "rerank_strategy_best_per_project.csv"

# ============================================================
# RANKING SELECTION FOR FULL-TEXT DOWNLOAD
# ============================================================

SELECTED_RANK_STRATEGY_NUM = 3
SELECTED_RANK_STRATEGY_NAME = "TitleObjectiveSBERT"
FULLTEXT_TOP_K_PER_PROJECT = 1000

# Elsevier API key (get free at https://dev.elsevier.com/)
ELSEVIER_API_KEY = "af1c032dae8dec3f90ac653a2fd3bb31"

# CORE API key (get free at https://core.ac.uk/services/api)
CORE_API_KEY = "pwjBGEh2t03dCAi4WlcJyRkTLHZmYfbq"

# Unpaywall registered email
UNPAYWALL_EMAIL = "muneebsiddique007@gmail.com"

# Request timeouts (seconds)
TIMEOUT_SHORT = 15
TIMEOUT_LONG = 45

# Delay between source attempts within one paper (seconds)
DELAY_BETWEEN_SOURCES = 1

# Delay between papers (be polite to servers) — used only in
# sequential fallback; parallel mode relies on natural stagger
DELAY_BETWEEN_PAPERS = 2

# Number of parallel download threads (set 1 for sequential)
MAX_WORKERS = 5

# How often to save intermediate results (every N completions)
SAVE_EVERY = 50

# Minimum valid PDF size in bytes.
# Keep this at 0 so valid smaller PDFs are not rejected.
MIN_PDF_SIZE = 0

# This step is PDF-first: download and validate PDFs only.
EXTRACT_FULLTEXT = False

# PDF validation / extraction safeguards
PDF_PROBE_TIMEOUT = 20
PDF_TEXT_EXTRACT_TIMEOUT = 90

# Sci-Hub mirrors (prioritized working mirrors)
SCI_HUB_MIRRORS = [
    "https://sci-hub.red",
    "https://sci-hub.box",
    "https://sci-hub.ee",
    "https://sci-hub.ru",
    "https://sci-hub.st",
    "https://sci-hub.su",
]

# HTTP headers — mimic a real browser
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/135.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

METHOD_LABELS = {
    "pmc_oa": "PubMed Central OA API",
    "europe_pmc": "Europe PMC",
    "unpaywall": "Unpaywall",
    "crossref": "CrossRef",
    "semantic_scholar": "Semantic Scholar",
    "core": "CORE",
    "scidownl": "SciDownl / Sci-Hub",
    "scihub_direct": "Sci-Hub Direct",
    "elsevier": "Elsevier",
    "nature": "Nature",
    "plos_one": "PLOS ONE",
    "plos_genetics": "PLOS Genetics",
    "frontiers": "Frontiers",
    "biorxiv": "bioRxiv",
    "medrxiv": "medRxiv",
    "arxiv": "arXiv",
    "wiley": "Wiley",
    "springer_bmc": "SpringerOpen / BMC",
    "mdpi": "MDPI",
    "taylor_francis": "Taylor & Francis",
    "oup": "Oxford University Press",
    "cambridge": "Cambridge Core",
    "hindawi": "Hindawi",
    "jci": "JCI",
    "elife": "eLife",
    "pnas": "PNAS",
    "blood_ash": "Blood / ASH",
    "genetics_gsa": "Genetics (GSA)",
    "karger": "Karger",
    "degruyter": "De Gruyter",
    "lww": "LWW / Wolters Kluwer",
    "doaj": "DOAJ",
    "internet_archive": "Internet Archive",
    "plos_all": "PLOS (all journals)",
    "doi_direct": "DOI Direct",
    "oa_button": "Open Access Button",
    "previously_downloaded": "Previously Downloaded",
    "already_exists": "Already Exists",
    "failed": "Failed",
}

PROVENANCE_FIELDS = [
    "source_database",
    "source_query",
    "query_origin",
    "rendered_query",
    "url",
    "openalex_id",
    "pmid",
    "pmcid",
    "semantic_scholar_id",
    "authors",
    "journal",
    "year",
    "article_type",
    "rank",
    "rank_strategy_3_score",
    "score_backend",
]

# ============================================================
# LOGGING SETUP
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("Step11Download")
logging.getLogger("pypdf").setLevel(logging.ERROR)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def clean(value) -> str:
    """Strip and normalise a field; return empty string for nan/None."""
    s = str(value).strip()
    return "" if s.lower() in ("nan", "none", "", "n/a") else s


def extract_provenance(row_data: Dict[str, Any]) -> Dict[str, str]:
    return {field: clean(row_data.get(field, "")) for field in PROVENANCE_FIELDS}


def read_existing_metadata(item: Dict[str, Any]) -> Dict[str, Any]:
    try:
        path = item.get("metadata_json_path", "")
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh) or {}
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}


def known_original_download_method(method: Any) -> str:
    method = clean(method)
    if method in {"previously_downloaded", "already_exists", "failed"}:
        return ""
    return method


def get_existing_original_download_method(item: Dict[str, Any]) -> str:
    metadata = read_existing_metadata(item)
    return (
        known_original_download_method(metadata.get("original_download_method", ""))
        or known_original_download_method(metadata.get("method", ""))
    )


def is_valid_pdf_bytes(content: bytes) -> bool:
    """Return True if the byte content looks like a real PDF."""
    return len(content) >= 4 and content[:4] == b"%PDF"


def is_valid_pdf_file(path: str) -> bool:
    """Check if a file on disk is a valid PDF."""
    try:
        with open(path, "rb") as f:
            header = f.read(4)
        return header == b"%PDF" and os.path.getsize(path) > 0
    except Exception:
        return False


def save_pdf_bytes(dest_path: str, content: bytes) -> bool:
    """Write PDF bytes to disk if content is valid. Return True on success."""
    if not is_valid_pdf_bytes(content):
        return False
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(content)
    log.info(f"    Saved {len(content) // 1024} KB -> {dest_path}")
    return True


def is_usable_pdf_file(path: str, require_text_probe: bool = False) -> bool:
    """
    Stronger PDF validation than a simple %PDF header check.
    """
    if not is_valid_pdf_file(path):
        return False

    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        if len(reader.pages) <= 0:
            return False
        _ = reader.pages[0]
    except Exception:
        return False

    if require_text_probe:
        ok, _ = _run_pypdf_helper(path, "probe", PDF_PROBE_TIMEOUT)
        return ok

    return True


def _run_pypdf_helper(pdf_path: str, mode: str, timeout: int) -> tuple[bool, str]:
    """Run pypdf in a subprocess to avoid hangs."""
    helper_code = r"""
import sys
import warnings
warnings.filterwarnings("ignore")
from pypdf import PdfReader

path = sys.argv[1]
mode = sys.argv[2]

reader = PdfReader(path)
page_count = len(reader.pages)
if page_count <= 0:
    raise RuntimeError("PDF has no pages")

if mode == "probe":
    text = reader.pages[0].extract_text() or ""
    print(f"{page_count}\t{len(text.strip())}")
else:
    parts = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        if txt:
            parts.append(txt)
    print("\n\n".join(parts))
"""

    try:
        proc = subprocess.run(
            [sys.executable, "-c", helper_code, pdf_path, mode],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"Timed out after {timeout}s"
    except Exception as e:
        return False, str(e)

    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip()
        return False, msg or f"helper exited with code {proc.returncode}"

    return True, proc.stdout


def quarantine_bad_pdf(path: str, reason: str) -> str:
    """Move a broken PDF aside so other download methods can try again."""
    if not os.path.exists(path):
        return ""

    stem, ext = os.path.splitext(path)
    bad_path = (
        f"{stem}.bad_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        f"{ext or '.pdf'}"
    )
    try:
        shutil.move(path, bad_path)
        log.warning(f"    Bad PDF moved aside: {bad_path} ({reason})")
        return bad_path
    except Exception as e:
        try:
            os.remove(path)
            log.warning(f"    Bad PDF deleted: {path} ({reason}; {e})")
        except Exception:
            pass
        return ""


def get_pdf_from_url(url: str, session: requests.Session = None) -> bytes:
    """Fetch a URL and return content if it's a valid PDF, else empty bytes."""
    try:
        requester = session or requests
        resp = requester.get(
            url, headers=HEADERS, timeout=TIMEOUT_LONG,
            allow_redirects=True, stream=True
        )
        if resp.status_code == 200:
            content = resp.content
            if is_valid_pdf_bytes(content):
                return content
    except Exception:
        pass
    return b""


# ============================================================
# DOI RESOLUTION HELPERS
# ============================================================

def doi_from_title_crossref(title: str) -> str:
    """Try CrossRef API to find DOI by title."""
    if not title:
        return ""
    try:
        url = "https://api.crossref.org/works"
        params = {
            "query.bibliographic": title,
            "rows": 1,
            "mailto": UNPAYWALL_EMAIL,
        }
        resp = requests.get(url, params=params,
                            headers=HEADERS, timeout=TIMEOUT_SHORT)
        if resp.status_code == 200:
            items = resp.json().get("message", {}).get("items", [])
            if items:
                doi = items[0].get("DOI", "")
                if doi:
                    log.info(f"    CrossRef title -> DOI {doi}")
                    return doi
    except Exception:
        pass
    return ""


def normalize_doi(doi: str) -> str:
    doi = clean(doi).lower()
    for pref in (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "doi.org/",
        "doi:",
    ):
        if doi.startswith(pref):
            doi = doi[len(pref):]
    doi = doi.strip().strip(".,;:)>]} \t\r\n'\"").lstrip("<([{ \t\r\n'\"")
    doi = re.sub(r"\s+", "", doi)
    return doi


def safe_slug(text: str, max_len: int = 80) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", clean(text)).strip("_")
    return (s or "x")[:max_len]


def short_hash(text: str, n: int = 8) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:n]


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def build_project_key(project_name: str) -> str:
    return f"{safe_slug(project_name, 60)}_{short_hash(project_name)}"


def build_citation_text(row: Dict[str, Any]) -> str:
    authors = clean(row.get("authors", ""))
    title = clean(row.get("title", ""))
    journal = clean(row.get("journal", ""))
    year = clean(row.get("year", ""))
    doi = normalize_doi(clean(row.get("doi", "")))

    parts = []
    if authors:
        parts.append(authors)
    if year:
        parts.append(f"({year})")
    if title:
        parts.append(title)
    if journal:
        parts.append(journal)
    if doi:
        parts.append(f"doi:{doi}")
    return " | ".join(parts)


def build_bibtex(row: Dict[str, Any]) -> str:
    authors = clean(row.get("authors", ""))
    title = clean(row.get("title", ""))
    journal = clean(row.get("journal", ""))
    year = clean(row.get("year", ""))
    doi = normalize_doi(clean(row.get("doi", "")))
    key_base = doi.replace("/", "_") if doi else (title[:50] or "ref")
    cite_key = safe_slug(key_base, 64)

    lines = [f"@article{{{cite_key},"]
    if authors:
        lines.append(f"  author = {{{authors}}},")
    if title:
        lines.append(f"  title = {{{title}}},")
    if journal:
        lines.append(f"  journal = {{{journal}}},")
    if year:
        lines.append(f"  year = {{{year}}},")
    if doi:
        lines.append(f"  doi = {{{doi}}},")
        lines.append(f"  url = {{https://doi.org/{doi}}},")
    lines.append("}")
    return "\n".join(lines)


def write_article_artifacts(
    item: Dict[str, Any],
    method: str,
    success: bool,
    refresh_fulltext: bool = False,
) -> None:
    item_dir = item["item_dir"]
    ensure_dir(item_dir)

    existing_metadata = read_existing_metadata(item)
    existing_original_method = (
        known_original_download_method(existing_metadata.get("original_download_method", ""))
        or known_original_download_method(existing_metadata.get("method", ""))
    )
    method_this_run = method
    original_download_method = (
        existing_original_method
        if method == "previously_downloaded" and existing_original_method
        else known_original_download_method(method)
    )
    method_for_metadata = original_download_method or method

    metadata = dict(item["row_data"])
    metadata.update({
        "project_name": item["project_name"],
        "project_key": item["project_key"],
        "rank_in_project": item["rank_in_project"],
        "method": method_for_metadata,
        "method_this_run": method_this_run,
        "resume_status": "previously_downloaded" if method == "previously_downloaded" else "",
        "original_download_method": original_download_method,
        "original_method_preserved": bool(method == "previously_downloaded" and existing_original_method),
        "success": bool(success),
        "downloaded_at": datetime.now().isoformat(timespec="seconds"),
        "paths": {
            "pdf": item["dest_pdf"],
            "full_text": item["fulltext_path"],
            "citation_text": item["citation_txt_path"],
            "citation_bibtex": item["citation_bib_path"],
        },
    })

    with open(item["metadata_json_path"], "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)

    with open(item["citation_txt_path"], "w", encoding="utf-8") as fh:
        fh.write(build_citation_text(item["row_data"]))
        fh.write("\n")

    with open(item["citation_bib_path"], "w", encoding="utf-8") as fh:
        fh.write(build_bibtex(item["row_data"]))
        fh.write("\n")


def choose_strategy_csv(project_dir: str, strategy_name: str) -> str:
    candidates = [
        os.path.join(project_dir, f"{safe_slug(strategy_name, 40)}.csv"),
        os.path.join(project_dir, f"{strategy_name}.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    all_csv = [
        os.path.join(project_dir, x)
        for x in os.listdir(project_dir)
        if x.lower().endswith(".csv")
    ]
    all_csv = sorted(all_csv)
    return all_csv[0] if all_csv else ""


def _project_summary_path(project_db_dir: str) -> str:
    return os.path.join(project_db_dir, "project_candidates_selected.csv")


def load_ranked_work_items(config: dict) -> List[Dict[str, Any]]:
    projects_root = config.get("ranking", {}).get("projects_dir", PROJECTS_DIR)
    selected_strategy_num = int(
        config.get("ranking", {}).get(
            "selected_strategy_num",
            SELECTED_RANK_STRATEGY_NUM,
        )
    )

    selected_strategy_name = clean(
        config.get("ranking", {}).get(
            "selected_strategy_name",
            SELECTED_RANK_STRATEGY_NAME,
        )
    )

    top_k = int(
        config.get("ranking", {}).get(
            "top_k_per_project",
            FULLTEXT_TOP_K_PER_PROJECT,
        )
    )

    if not os.path.isdir(projects_root):
        log.warning(f"Projects directory not found: {projects_root}")
        return []

    items: List[Dict[str, Any]] = []
    project_dirs = []

    for name in sorted(os.listdir(projects_root)):
        project_dir = os.path.join(projects_root, name)
        rankings_dir = os.path.join(project_dir, "Rankings")

        if not os.path.isdir(project_dir):
            continue
        if name.lower() == "rankings":
            continue
        if not os.path.isdir(rankings_dir):
            continue

        project_dirs.append(project_dir)

    log.info("=" * 100)
    log.info("FULL-TEXT DOWNLOAD SELECTION")
    log.info("=" * 100)
    log.info(f"Projects root       : {os.path.abspath(projects_root)}")
    log.info(f"Selected strategy   : {selected_strategy_num} - {selected_strategy_name}")
    log.info(f"Top K per project   : {top_k}")
    log.info(f"Projects found      : {len(project_dirs)}")

    for project_dir in project_dirs:
        project_name = os.path.basename(project_dir)
        rankings_dir = os.path.join(project_dir, "Rankings")
        strategy_csv = os.path.join(rankings_dir, f"rank_strategy_{selected_strategy_num}.csv")

        if not os.path.exists(strategy_csv):
            log.warning(f"Missing ranking file: {strategy_csv}")
            continue

        try:
            ranked_df = pd.read_csv(strategy_csv).fillna("")
        except Exception as e:
            log.warning(f"Could not read {strategy_csv}: {e}")
            continue

        if ranked_df.empty:
            log.warning(f"Empty ranking file: {strategy_csv}")
            continue

        selected = ranked_df.head(top_k).copy()

        project_key = build_project_key(project_name)
        project_db_dir = os.path.join(DB_DIR, project_key)
        ensure_dir(project_db_dir)

        summary_rows = []
        seen_keys = set()
        kept_rows = []

        for _, row in selected.iterrows():
            row_data = {k: clean(v) for k, v in row.to_dict().items()}
            doi = normalize_doi(row_data.get("doi", ""))
            title = row_data.get("title", "")

            article_key = f"doi::{doi}" if doi else f"title::{title.lower().strip()}"
            if not article_key or article_key in seen_keys:
                continue

            seen_keys.add(article_key)
            kept_rows.append(row_data)

        log.info(
            f"Project: {project_name} | "
            f"ranked={len(ranked_df)} | "
            f"selected_top_k={len(selected)} | "
            f"after_dedup={len(kept_rows)}"
        )

        for idx, row_data in enumerate(kept_rows):
            doi = normalize_doi(row_data.get("doi", ""))
            title = row_data.get("title", "")
            provenance = extract_provenance(row_data)
            item_dir = os.path.join(project_db_dir, f"Temp_{idx}")
            ensure_dir(item_dir)

            summary_rows.append({
                "rank_in_project": idx,
                "title": title,
                "doi": doi,
                "strategy_num": selected_strategy_num,
                "strategy": selected_strategy_name,
                "project_name": project_name,
                "project_id": project_name,
                "project_key": project_key,
                "source_ranked_csv": strategy_csv,
                **provenance,
            })

            items.append({
                "global_index": len(items),
                "loop": idx,
                "doi": doi,
                "title": title,
                "project_name": project_name,
                "project_id": project_name,
                "project_key": project_key,
                "strategy": selected_strategy_name,
                "strategy_num": selected_strategy_num,
                "source_ranked_csv": strategy_csv,
                "rank_in_project": idx,
                **provenance,
                "row_data": row_data,
                "project_dir": project_dir,
                "project_db_dir": project_db_dir,
                "item_dir": item_dir,
                "dest_pdf": os.path.join(item_dir, "document.pdf"),
                "fulltext_path": os.path.join(item_dir, "fulltext.txt"),
                "citation_txt_path": os.path.join(item_dir, "citation.txt"),
                "citation_bib_path": os.path.join(item_dir, "citation.bib"),
                "metadata_json_path": os.path.join(item_dir, "metadata.json"),
            })

        if summary_rows:
            pd.DataFrame(summary_rows).to_csv(_project_summary_path(project_db_dir), index=False)

    log.info("=" * 100)
    log.info(f"Total selected articles for download: {len(items)}")
    log.info("=" * 100)

    return items


def load_legacy_work_items(start_idx: int = 0, end_idx: int = None) -> List[Dict[str, Any]]:
    """Load articles from allarticles2.csv with index range support."""
    if not os.path.exists("Output/allarticles2.csv"):
        return []

    df = pd.read_csv("Output/allarticles2.csv").fillna("")
    
    # Apply index range if specified
    if end_idx is not None:
        df = df.iloc[start_idx:end_idx]
    elif start_idx > 0:
        df = df.iloc[start_idx:]
    
    items: List[Dict[str, Any]] = []

    for loop, row in df.iterrows():
        doi = normalize_doi(clean(row.get("doi", "")))
        title = clean(row.get("title", ""))
        item_dir = os.path.join(DB_DIR, f"Temp_{loop}")
        ensure_dir(item_dir)
        row_data = {k: clean(v) for k, v in row.to_dict().items()}

        items.append({
            "global_index": len(items),
            "loop": loop,
            "doi": doi,
            "title": title,
            "project_name": "legacy",
            "project_id": "legacy",
            "project_key": "legacy",
            "strategy": "legacy_csv",
            "source_ranked_csv": "Output/allarticles2.csv",
            "rank_in_project": loop,
            "row_data": row_data,
            "project_dir": "",
            "project_db_dir": DB_DIR,
            "item_dir": item_dir,
            "dest_pdf": os.path.join(item_dir, "document.pdf"),
            "fulltext_path": os.path.join(item_dir, "fulltext.txt"),
            "citation_txt_path": os.path.join(item_dir, "citation.txt"),
            "citation_bib_path": os.path.join(item_dir, "citation.bib"),
            "metadata_json_path": os.path.join(item_dir, "metadata.json"),
        })

    return items


# ============================================================
# SOURCE 01 — PubMed Central (NCBI OA API)
# ============================================================

def source_pmc_oa(doi: str, dest_path: str) -> bool:
    """Download via NCBI PMC OA API. Needs a DOI that maps to a PMC article."""
    if not doi:
        return False
    log.info("    [01] PubMed Central OA API...")
    try:
        conv_url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
        params = {"ids": doi, "format": "json",
                  "tool": "pipeline", "email": UNPAYWALL_EMAIL}
        resp = requests.get(conv_url, params=params,
                            headers=HEADERS, timeout=TIMEOUT_SHORT)
        if resp.status_code != 200:
            return False
        records = resp.json().get("records", [])
        if not records:
            return False
        pmcid = records[0].get("pmcid", "")
        if not pmcid:
            return False

        pmc_num = pmcid.replace("PMC", "").strip()

        oa_url = (f"https://www.ncbi.nlm.nih.gov/pmc/utils/"
                  f"oa/oa.fcgi?id=PMC{pmc_num}")
        resp = requests.get(oa_url, headers=HEADERS,
                            timeout=TIMEOUT_SHORT)
        if resp.status_code == 200 and "<link" in resp.text:
            import xml.etree.ElementTree as ET
            try:
                root = ET.fromstring(resp.text)
            except ET.ParseError:
                root = None
            if root:
                for link in root.iter("link"):
                    fmt = link.get("format", "")
                    href = link.get("href", "")
                    if fmt == "pdf" and href:
                        if href.startswith("ftp://"):
                            href = href.replace("ftp://", "https://")
                        content = get_pdf_from_url(href)
                        if content:
                            return save_pdf_bytes(dest_path, content)

        for suffix in ["/pdf/", "/pdf"]:
            url = (f"https://www.ncbi.nlm.nih.gov/pmc/articles/"
                   f"PMC{pmc_num}{suffix}")
            content = get_pdf_from_url(url)
            if content:
                return save_pdf_bytes(dest_path, content)
    except Exception as e:
        log.debug(f"      PMC OA error: {e}")
    return False


# ============================================================
# SOURCE 02 — Europe PMC
# ============================================================

def source_europe_pmc(doi: str, title: str, dest_path: str) -> bool:
    """Download via Europe PMC REST API."""
    log.info("    [02] Europe PMC...")
    try:
        query = f"DOI:{quote_plus(doi)}" if doi else quote_plus(title)
        api_url = (
            f"https://www.ebi.ac.uk/europepmc/webservices/rest/search"
            f"?query={query}&resultType=core&format=json&pageSize=1"
        )
        resp = requests.get(api_url, headers=HEADERS,
                            timeout=TIMEOUT_SHORT)
        if resp.status_code != 200:
            return False
        results = resp.json().get("resultList", {}).get("result", [])
        if not results:
            return False

        article = results[0]
        is_oa = article.get("isOpenAccess", "N")
        pmcid = article.get("pmcid", "")

        if is_oa == "Y" and pmcid:
            pmc_num = str(pmcid).replace("PMC", "").strip()
            for url in [
                f"https://europepmc.org/articles/PMC{pmc_num}/pdf/render",
                f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_num}/pdf/",
            ]:
                content = get_pdf_from_url(url)
                if content:
                    return save_pdf_bytes(dest_path, content)

        ftl_list = article.get("fullTextUrlList", {}).get(
            "fullTextUrl", [])
        for ftl in ftl_list:
            if ftl.get("documentStyle") == "pdf":
                url = ftl.get("url", "")
                if url:
                    content = get_pdf_from_url(url)
                    if content:
                        return save_pdf_bytes(dest_path, content)
    except Exception as e:
        log.debug(f"      Europe PMC error: {e}")
    return False


# ============================================================
# SOURCE 03 — Unpaywall
# ============================================================

def source_unpaywall(doi: str, dest_path: str) -> bool:
    """Find legal free PDF via Unpaywall."""
    if not doi:
        return False
    log.info("    [03] Unpaywall...")
    try:
        uw_url = (f"https://api.unpaywall.org/v2/"
                  f"{quote_plus(doi)}?email={UNPAYWALL_EMAIL}")
        resp = requests.get(uw_url, headers=HEADERS,
                            timeout=TIMEOUT_SHORT)
        if resp.status_code != 200:
            return False
        data = resp.json()
        if not data.get("is_oa", False):
            return False

        pdf_urls = []
        best = data.get("best_oa_location") or {}
        if best.get("url_for_pdf"):
            pdf_urls.append(best["url_for_pdf"])
        if best.get("url"):
            pdf_urls.append(best["url"])
        for loc in data.get("oa_locations", []):
            if loc.get("url_for_pdf"):
                pdf_urls.append(loc["url_for_pdf"])
            elif loc.get("url"):
                pdf_urls.append(loc["url"])

        for pdf_url in pdf_urls:
            content = get_pdf_from_url(pdf_url)
            if content:
                return save_pdf_bytes(dest_path, content)
    except Exception as e:
        log.debug(f"      Unpaywall error: {e}")
    return False


# ============================================================
# SOURCE 04 — CrossRef
# ============================================================

def source_crossref(doi: str, dest_path: str) -> bool:
    """Try CrossRef API for publisher PDF links."""
    if not doi:
        return False
    log.info("    [04] CrossRef...")
    try:
        api_url = f"https://api.crossref.org/works/{quote_plus(doi)}"
        resp = requests.get(api_url, headers=HEADERS,
                            timeout=TIMEOUT_SHORT)
        if resp.status_code != 200:
            return False
        links = resp.json().get("message", {}).get("link", [])
        for link in links:
            ct = link.get("content-type", "")
            url = link.get("URL", "")
            if ct in ("application/pdf", "unspecified") and url:
                content = get_pdf_from_url(url)
                if content:
                    return save_pdf_bytes(dest_path, content)

        doi_url = f"https://doi.org/{doi}"
        resp2 = requests.get(
            doi_url,
            headers={**HEADERS, "Accept": "application/pdf"},
            timeout=TIMEOUT_LONG, allow_redirects=True
        )
        if resp2.status_code == 200 and is_valid_pdf_bytes(resp2.content):
            return save_pdf_bytes(dest_path, resp2.content)
    except Exception as e:
        log.debug(f"      CrossRef error: {e}")
    return False


# ============================================================
# SOURCE 05 — Semantic Scholar
# ============================================================

def source_semantic_scholar(doi: str, dest_path: str) -> bool:
    """Semantic Scholar OA PDF links."""
    if not doi:
        return False
    log.info("    [05] Semantic Scholar...")
    try:
        api_url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
        params = {"fields": "openAccessPdf,title"}
        resp = requests.get(api_url, params=params,
                            headers=HEADERS, timeout=TIMEOUT_SHORT)
        if resp.status_code != 200:
            return False
        oa_pdf = resp.json().get("openAccessPdf", {}) or {}
        pdf_url = oa_pdf.get("url", "")
        if pdf_url:
            content = get_pdf_from_url(pdf_url)
            if content:
                return save_pdf_bytes(dest_path, content)
    except Exception as e:
        log.debug(f"      Semantic Scholar error: {e}")
    return False


# ============================================================
# SOURCE 06 — CORE API
# ============================================================

def source_core(doi: str, dest_path: str) -> bool:
    """CORE aggregator for OA papers."""
    if not CORE_API_KEY or not doi:
        return False
    log.info("    [06] CORE...")
    try:
        headers = {**HEADERS, "Authorization": f"Bearer {CORE_API_KEY}"}
        api_url = "https://api.core.ac.uk/v3/search/works"
        params = {"q": f'doi:"{doi}"', "limit": 1}
        resp = requests.get(api_url, params=params,
                            headers=headers, timeout=TIMEOUT_SHORT)
        if resp.status_code != 200:
            return False
        results = resp.json().get("results", [])
        if not results:
            return False
        article = results[0]
        pdf_url = (article.get("downloadUrl", "") or
                   article.get("sourceFulltextUrls", [""])[0])
        if pdf_url:
            content = get_pdf_from_url(pdf_url)
            if content:
                return save_pdf_bytes(dest_path, content)
    except Exception as e:
        log.debug(f"      CORE error: {e}")
    return False


# ============================================================
# SOURCE 07 — scidownl (Sci-Hub wrapper)
# ============================================================

def source_scidownl(doi: str, title: str, dest_path: str) -> bool:
    """Use scidownl Python package to download from Sci-Hub."""
    log.info("    [07] scidownl (Sci-Hub)...")
    try:
        from scidownl import scihub_download
    except ImportError:
        log.debug("      scidownl not installed, skipping")
        return False

    dest_dir = os.path.dirname(dest_path)
    attempts = []
    if doi:
        attempts.append(("doi", doi))
    if title:
        attempts.append(("title", title))

    for id_type, identifier in attempts:
        try:
            scihub_download(identifier, paper_type=id_type,
                            out=dest_dir)
            time.sleep(2)
            for f in os.listdir(dest_dir):
                fp = os.path.join(dest_dir, f)
                if f.endswith(".pdf") and is_valid_pdf_file(fp):
                    if fp != dest_path:
                        shutil.move(fp, dest_path)
                    return True
        except Exception as e:
            log.debug(f"      scidownl error ({id_type}): {e}")
        time.sleep(DELAY_BETWEEN_SOURCES)
    return False


# ============================================================
# SOURCE 07b — Sci-Hub Direct (NO GUI, HTTP only)
# ============================================================

def source_scihub_direct(doi: str, title: str, dest_path: str) -> bool:
    """
    Download from Sci-Hub using direct HTTP requests.
    NO GUI, NO browser automation - pure requests.
    """
    log.info("    [07b] Sci-Hub Direct...")
    
    if not doi and not title:
        return False
    
    # Clean DOI
    doi_clean = clean(doi)
    if doi_clean:
        # Remove any URL prefix
        if "doi.org/" in doi_clean:
            doi_clean = doi_clean.split("doi.org/")[-1]
    
    # Build list of identifiers to try
    identifiers = []
    if doi_clean:
        identifiers.append(doi_clean)
    if title:
        identifiers.append(title)
    
    if not identifiers:
        return False
    
    for mirror in SCI_HUB_MIRRORS:
        # Check mirror reachability
        try:
            test = requests.get(mirror, headers=HEADERS, timeout=8, allow_redirects=True)
            if test.status_code >= 500:
                log.debug(f"      Mirror unreachable ({test.status_code}): {mirror}")
                continue
        except Exception:
            log.debug(f"      Mirror unreachable: {mirror}")
            continue
        
        for identifier in identifiers:
            # Try different URL formats
            urls_to_try = [
                f"{mirror}/{identifier}",
                f"{mirror}/doi:{identifier}",
                f"{mirror}/{quote_plus(identifier)}",
            ]
            
            for url in urls_to_try:
                try:
                    resp = requests.get(
                        url,
                        headers=HEADERS,
                        timeout=TIMEOUT_LONG,
                        allow_redirects=True
                    )
                    
                    if resp.status_code != 200:
                        continue
                    
                    # Check if response is directly a PDF
                    if is_valid_pdf_bytes(resp.content):
                        log.info(f"      Direct PDF from {mirror}")
                        return save_pdf_bytes(dest_path, resp.content)
                    
                    # Extract PDF URL from HTML
                    content = resp.text
                    
                    # Pattern 1: PDF links in HTML
                    pdf_patterns = [
                        r'(https?://[^\s"\'<>]+\.pdf[^"\']*)',
                        r'location\.href\s*=\s*["\']([^"\']*)["\']',
                        r'src=["\']([^"\']+\.pdf[^"\']*)["\']',
                        r'href=["\']([^"\']+\.pdf[^"\']*)["\']',
                        r'<embed[^>]+src=["\']([^"\']+)["\']',
                        r'<iframe[^>]+src=["\']([^"\']+)["\']',
                    ]
                    
                    for pattern in pdf_patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        for match in matches:
                            pdf_link = match
                            # Fix escaped URLs
                            pdf_link = pdf_link.replace('\\/', '/')
                            pdf_link = re.sub(r'\\(.)', r'\1', pdf_link)
                            
                            # Handle relative URLs
                            if pdf_link.startswith('//'):
                                pdf_link = 'https:' + pdf_link
                            elif pdf_link.startswith('/'):
                                pdf_link = mirror + pdf_link
                            elif not pdf_link.startswith('http'):
                                pdf_link = mirror + '/' + pdf_link.lstrip('/')
                            
                            # Decode URL
                            pdf_link = unquote(pdf_link)
                            
                            if '.pdf' in pdf_link.lower():
                                log.debug(f"      Found PDF link: {pdf_link}")
                                pdf_resp = requests.get(
                                    pdf_link,
                                    headers=HEADERS,
                                    timeout=TIMEOUT_LONG,
                                    allow_redirects=True
                                )
                                if pdf_resp.status_code == 200 and is_valid_pdf_bytes(pdf_resp.content):
                                    log.info(f"      PDF downloaded from {mirror}")
                                    return save_pdf_bytes(dest_path, pdf_resp.content)
                
                except Exception as e:
                    log.debug(f"      Sci-Hub error ({mirror}): {e}")
                    continue
                
                time.sleep(0.5)
            
            time.sleep(0.5)
        
        log.debug(f"      Mirror {mirror} failed, trying next...")
    
    return False


# ============================================================
# SOURCE 08 — Elsevier API
# ============================================================

def source_elsevier(doi: str, dest_path: str) -> bool:
    """Download via Elsevier full-text API."""
    if not doi or not ELSEVIER_API_KEY:
        return False
    log.info("    [08] Elsevier API...")
    try:
        for url in [
            f"http://api.elsevier.com/content/article/doi:{doi}?view=FULL",
            f"https://api.elsevier.com/content/article/doi/{doi}",
        ]:
            headers = {
                **HEADERS,
                "X-ELS-APIKEY": ELSEVIER_API_KEY,
                "Accept": "application/pdf",
            }
            resp = requests.get(url, headers=headers,
                                timeout=TIMEOUT_LONG, stream=True)
            if resp.status_code == 200 and is_valid_pdf_bytes(resp.content):
                return save_pdf_bytes(dest_path, resp.content)
    except Exception as e:
        log.debug(f"      Elsevier error: {e}")
    return False


# ============================================================
# SOURCE 09 — Nature.com
# ============================================================

def source_nature(doi: str, dest_path: str) -> bool:
    """Scrape Nature.com for OA PDF."""
    if not doi or "10.1038" not in doi:
        return False
    log.info("    [09] Nature.com...")
    try:
        article_id = doi.split("/")[-1]
        for url in [
            f"https://www.nature.com/articles/{article_id}.pdf",
            f"https://www.nature.com/articles/{article_id}/pdf",
        ]:
            content = get_pdf_from_url(url)
            if content:
                return save_pdf_bytes(dest_path, content)

        page_url = f"https://doi.org/{doi}"
        resp = requests.get(page_url, headers=HEADERS,
                            timeout=TIMEOUT_LONG, allow_redirects=True)
        if resp.status_code == 200:
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(resp.content, "html.parser")
                for a_tag in soup.find_all("a", href=True):
                    href = a_tag["href"]
                    if ".pdf" in href and "articles" in href:
                        full_url = (f"https://www.nature.com{href}"
                                    if href.startswith("/") else href)
                        content = get_pdf_from_url(full_url)
                        if content:
                            return save_pdf_bytes(dest_path, content)
            except ImportError:
                pass
    except Exception as e:
        log.debug(f"      Nature error: {e}")
    return False


# ============================================================
# SOURCE 10 — PLOS ONE
# ============================================================

def source_plos_one(doi: str, dest_path: str) -> bool:
    """Download from PLOS ONE printable PDF."""
    if not doi or "10.1371" not in doi:
        return False
    log.info("    [10] PLOS ONE...")
    try:
        url = (f"https://journals.plos.org/plosone/article/file"
               f"?id={doi}&type=printable")
        resp = requests.get(url, headers=HEADERS,
                            timeout=TIMEOUT_LONG, stream=True)
        if resp.status_code == 200 and is_valid_pdf_bytes(resp.content):
            return save_pdf_bytes(dest_path, resp.content)
    except Exception as e:
        log.debug(f"      PLOS ONE error: {e}")
    return False


# ============================================================
# SOURCE 11 — PLOS Genetics
# ============================================================

def source_plos_genetics(doi: str, dest_path: str) -> bool:
    """Download from PLOS Genetics."""
    if not doi or "10.1371" not in doi:
        return False
    log.info("    [11] PLOS Genetics...")
    try:
        url = (f"https://journals.plos.org/plosgenetics/article/file"
               f"?id={doi}&type=printable")
        resp = requests.get(url, headers=HEADERS,
                            timeout=TIMEOUT_LONG, stream=True)
        if resp.status_code == 200 and is_valid_pdf_bytes(resp.content):
            return save_pdf_bytes(dest_path, resp.content)
    except Exception as e:
        log.debug(f"      PLOS Genetics error: {e}")
    return False


# ============================================================
# SOURCE 12 — Frontiers
# ============================================================

def source_frontiers(doi: str, dest_path: str) -> bool:
    """Download from Frontiers journals."""
    if not doi or "10.3389" not in doi:
        return False
    log.info("    [12] Frontiers...")
    try:
        for url in [
            f"https://www.frontiersin.org/articles/{doi}/pdf",
            f"https://www.frontiersin.org/article/{doi}/pdf",
        ]:
            resp = requests.get(url, headers=HEADERS,
                                timeout=TIMEOUT_LONG, stream=True)
            if resp.status_code == 200 and is_valid_pdf_bytes(resp.content):
                return save_pdf_bytes(dest_path, resp.content)
    except Exception as e:
        log.debug(f"      Frontiers error: {e}")
    return False


# ============================================================
# SOURCE 13 — bioRxiv
# ============================================================

def source_biorxiv(doi: str, dest_path: str) -> bool:
    """Download preprint PDF from bioRxiv."""
    if not doi or ("10.1101" not in doi
                   and "biorxiv" not in doi.lower()):
        return False
    log.info("    [13] bioRxiv...")
    try:
        for version in ["v3", "v2", "v1"]:
            url = (f"https://www.biorxiv.org/content/"
                   f"{doi}{version}.full.pdf")
            resp = requests.get(url, headers=HEADERS,
                                timeout=TIMEOUT_LONG, stream=True)
            if resp.status_code == 200 and is_valid_pdf_bytes(resp.content):
                return save_pdf_bytes(dest_path, resp.content)
    except Exception as e:
        log.debug(f"      bioRxiv error: {e}")
    return False


# ============================================================
# SOURCE 14 — medRxiv
# ============================================================

def source_medrxiv(doi: str, dest_path: str) -> bool:
    """Download preprint PDF from medRxiv."""
    if not doi or ("10.1101" not in doi
                   and "medrxiv" not in doi.lower()):
        return False
    log.info("    [14] medRxiv...")
    try:
        for version in ["v3", "v2", "v1"]:
            url = (f"https://www.medrxiv.org/content/"
                   f"{doi}{version}.full.pdf")
            resp = requests.get(url, headers=HEADERS,
                                timeout=TIMEOUT_LONG, stream=True)
            if resp.status_code == 200 and is_valid_pdf_bytes(resp.content):
                return save_pdf_bytes(dest_path, resp.content)
    except Exception as e:
        log.debug(f"      medRxiv error: {e}")
    return False


# ============================================================
# SOURCE 15 — arXiv
# ============================================================

def source_arxiv(doi: str, dest_path: str) -> bool:
    """Download from arXiv."""
    arxiv_id = None
    if doi and "arxiv" in doi.lower():
        m = re.search(r"arxiv[:/.]?(\d{4}\.\d{4,5})",
                      doi, re.IGNORECASE)
        if m:
            arxiv_id = m.group(1)
    if not arxiv_id:
        return False
    log.info("    [15] arXiv...")
    try:
        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        resp = requests.get(url, headers=HEADERS,
                            timeout=TIMEOUT_LONG, stream=True)
        if resp.status_code == 200 and is_valid_pdf_bytes(resp.content):
            return save_pdf_bytes(dest_path, resp.content)
    except Exception as e:
        log.debug(f"      arXiv error: {e}")
    return False


# ============================================================
# SOURCE 16 — Wiley Open Access
# ============================================================

def source_wiley(doi: str, dest_path: str) -> bool:
    """Download OA content from Wiley Online Library."""
    if not doi or "10.1002" not in doi:
        return False
    log.info("    [16] Wiley OA...")
    try:
        for url in [
            f"https://onlinelibrary.wiley.com/doi/pdfdirect/{doi}",
            f"https://onlinelibrary.wiley.com/doi/pdf/{doi}?download=true",
        ]:
            resp = requests.get(url, headers=HEADERS,
                                timeout=TIMEOUT_LONG,
                                allow_redirects=True, stream=True)
            if resp.status_code == 200 and is_valid_pdf_bytes(resp.content):
                return save_pdf_bytes(dest_path, resp.content)
    except Exception as e:
        log.debug(f"      Wiley error: {e}")
    return False


# ============================================================
# SOURCE 17 — SpringerOpen / BioMedCentral
# ============================================================

def source_springer_bmc(doi: str, dest_path: str) -> bool:
    """Download from SpringerOpen or BMC."""
    if not doi:
        return False
    if "10.1186" not in doi and "10.1007" not in doi:
        return False
    log.info("    [17] SpringerOpen/BMC...")
    try:
        urls = [f"https://link.springer.com/content/pdf/{doi}.pdf"]
        if "10.1186" in doi:
            urls.insert(0,
                f"https://bmcbioinformatics.biomedcentral.com/track/pdf/{doi}")
        for url in urls:
            content = get_pdf_from_url(url)
            if content:
                return save_pdf_bytes(dest_path, content)
    except Exception as e:
        log.debug(f"      SpringerOpen/BMC error: {e}")
    return False


# ============================================================
# SOURCE 18 — MDPI
# ============================================================

def source_mdpi(doi: str, dest_path: str) -> bool:
    """Download from MDPI (fully OA)."""
    if not doi or "10.3390" not in doi:
        return False
    log.info("    [18] MDPI...")
    try:
        url = (f"https://www.mdpi.com/"
               f"{doi.replace('10.3390/', '')}/pdf")
        resp = requests.get(url, headers=HEADERS,
                            timeout=TIMEOUT_LONG,
                            allow_redirects=True, stream=True)
        if resp.status_code == 200 and is_valid_pdf_bytes(resp.content):
            return save_pdf_bytes(dest_path, resp.content)
        doi_url = f"https://doi.org/{doi}"
        resp2 = requests.get(
            doi_url,
            headers={**HEADERS, "Accept": "application/pdf"},
            timeout=TIMEOUT_LONG, allow_redirects=True)
        if resp2.status_code == 200 and is_valid_pdf_bytes(resp2.content):
            return save_pdf_bytes(dest_path, resp2.content)
    except Exception as e:
        log.debug(f"      MDPI error: {e}")
    return False


# ============================================================
# SOURCE 19 — Taylor & Francis Open Access
# ============================================================

def source_taylor_francis(doi: str, dest_path: str) -> bool:
    """Download from Taylor & Francis OA."""
    if not doi or "10.1080" not in doi:
        return False
    log.info("    [19] Taylor & Francis OA...")
    try:
        url = (f"https://www.tandfonline.com/doi/pdf/"
               f"{doi}?download=true")
        resp = requests.get(url, headers=HEADERS,
                            timeout=TIMEOUT_LONG,
                            allow_redirects=True, stream=True)
        if resp.status_code == 200 and is_valid_pdf_bytes(resp.content):
            return save_pdf_bytes(dest_path, resp.content)
    except Exception as e:
        log.debug(f"      T&F error: {e}")
    return False


# ============================================================
# SOURCE 20 — Oxford University Press OA
# ============================================================

def source_oup(doi: str, dest_path: str) -> bool:
    """Download from OUP OA content."""
    if not doi or "10.1093" not in doi:
        return False
    log.info("    [20] OUP...")
    try:
        doi_clean = doi.replace("10.1093/", "")
        for url in [
            f"https://academic.oup.com/{doi_clean}.pdf",
            f"https://watermark.silverchair.com/doi/{doi}.pdf",
        ]:
            content = get_pdf_from_url(url)
            if content:
                return save_pdf_bytes(dest_path, content)
        resp = requests.get(
            f"https://doi.org/{doi}",
            headers={**HEADERS, "Accept": "application/pdf"},
            timeout=TIMEOUT_LONG, allow_redirects=True)
        if resp.status_code == 200 and is_valid_pdf_bytes(resp.content):
            return save_pdf_bytes(dest_path, resp.content)
    except Exception as e:
        log.debug(f"      OUP error: {e}")
    return False


# ============================================================
# SOURCE 21 — Cambridge Core OA
# ============================================================

def source_cambridge(doi: str, dest_path: str) -> bool:
    """Download from Cambridge Core OA."""
    if not doi or ("10.1017" not in doi and "10.1042" not in doi):
        return False
    log.info("    [21] Cambridge Core...")
    try:
        article_id = doi.split("/")[-1]
        for url in [
            f"https://www.cambridge.org/core/services/"
            f"aop-cambridge-core/content/view/{article_id}",
            f"https://www.cambridge.org/core/product/"
            f"{article_id}/download/pdf",
        ]:
            content = get_pdf_from_url(url)
            if content:
                return save_pdf_bytes(dest_path, content)
    except Exception as e:
        log.debug(f"      Cambridge error: {e}")
    return False


# ============================================================
# SOURCE 22 — Hindawi
# ============================================================

def source_hindawi(doi: str, dest_path: str) -> bool:
    """Download from Hindawi (fully OA)."""
    if not doi or "10.1155" not in doi:
        return False
    log.info("    [22] Hindawi...")
    try:
        article_id = doi.split("/")[-1]
        for url in [
            f"https://downloads.hindawi.com/journals/{article_id}.pdf",
            f"https://www.hindawi.com/journals/{article_id}/",
        ]:
            content = get_pdf_from_url(url)
            if content:
                return save_pdf_bytes(dest_path, content)
        doi_url = f"https://doi.org/{doi}"
        resp = requests.get(doi_url, headers=HEADERS,
                            timeout=TIMEOUT_LONG, allow_redirects=True)
        if resp.status_code == 200:
            pdf_url = resp.url.rstrip("/") + "/pdf"
            content = get_pdf_from_url(pdf_url)
            if content:
                return save_pdf_bytes(dest_path, content)
    except Exception as e:
        log.debug(f"      Hindawi error: {e}")
    return False


# ============================================================
# SOURCE 23 — JCI (Journal of Clinical Investigation)
# ============================================================

def source_jci(doi: str, dest_path: str) -> bool:
    """Download from JCI (fully OA)."""
    if not doi or "10.1172" not in doi:
        return False
    log.info("    [23] JCI...")
    try:
        article_id = doi.split("/")[-1]
        for url in [
            f"https://www.jci.org/articles/view/{article_id}/pdf",
            f"https://dm5migu4zj3pb.cloudfront.net/manuscripts/"
            f"{article_id}/pdf",
        ]:
            content = get_pdf_from_url(url)
            if content:
                return save_pdf_bytes(dest_path, content)
    except Exception as e:
        log.debug(f"      JCI error: {e}")
    return False


# ============================================================
# SOURCE 24 — eLife
# ============================================================

def source_elife(doi: str, dest_path: str) -> bool:
    """Download from eLife (fully OA)."""
    if not doi or "10.7554" not in doi:
        return False
    log.info("    [24] eLife...")
    try:
        article_id = (doi.split("eLife.")[-1]
                      if "eLife." in doi
                      else doi.split("/")[-1])
        url = (f"https://elifesciences.org/articles/"
               f"{article_id}/download/pdf")
        content = get_pdf_from_url(url)
        if content:
            return save_pdf_bytes(dest_path, content)
        doi_url = f"https://doi.org/{doi}"
        resp = requests.get(doi_url, headers=HEADERS,
                            timeout=TIMEOUT_LONG, allow_redirects=True)
        if resp.status_code == 200 and "elifesciences.org" in resp.url:
            pdf_url = resp.url.rstrip("/") + "/download/pdf"
            content = get_pdf_from_url(pdf_url)
            if content:
                return save_pdf_bytes(dest_path, content)
    except Exception as e:
        log.debug(f"      eLife error: {e}")
    return False


# ============================================================
# SOURCE 25 — PNAS
# ============================================================

def source_pnas(doi: str, dest_path: str) -> bool:
    """Download from PNAS (OA after 6 months)."""
    if not doi or "10.1073" not in doi:
        return False
    log.info("    [25] PNAS...")
    try:
        for url in [
            f"https://www.pnas.org/doi/pdf/{doi}",
            f"https://www.pnas.org/doi/epdf/{doi}",
            f"https://www.pnas.org/doi/pdf/{doi}?download=true",
        ]:
            content = get_pdf_from_url(url)
            if content:
                return save_pdf_bytes(dest_path, content)
    except Exception as e:
        log.debug(f"      PNAS error: {e}")
    return False


# ============================================================
# SOURCE 26 — Blood / ASH
# ============================================================

def source_blood_ash(doi: str, dest_path: str) -> bool:
    """Download from Blood / ASH journals."""
    if not doi or "10.1182" not in doi:
        return False
    log.info("    [26] Blood / ASH...")
    try:
        for url in [
            f"https://ashpublications.org/blood/article-pdf/doi/{doi}",
            f"https://ashpublications.org/bloodadvances/"
            f"article-pdf/doi/{doi}",
        ]:
            content = get_pdf_from_url(url)
            if content:
                return save_pdf_bytes(dest_path, content)
        resp = requests.get(
            f"https://doi.org/{doi}",
            headers={**HEADERS, "Accept": "application/pdf"},
            timeout=TIMEOUT_LONG, allow_redirects=True)
        if resp.status_code == 200 and is_valid_pdf_bytes(resp.content):
            return save_pdf_bytes(dest_path, resp.content)
    except Exception as e:
        log.debug(f"      Blood/ASH error: {e}")
    return False


# ============================================================
# SOURCE 27 — Genetics (GSA)
# ============================================================

def source_genetics_gsa(doi: str, dest_path: str) -> bool:
    """Download from Genetics Society journals."""
    if not doi:
        return False
    if "10.1534" not in doi and "genetics" not in doi.lower():
        return False
    log.info("    [27] Genetics (GSA)...")
    try:
        for url in [
            f"https://academic.oup.com/genetics/article-pdf/doi/{doi}",
            f"https://www.genetics.org/content/genetics/"
            f"doi/{doi}.full.pdf",
        ]:
            content = get_pdf_from_url(url)
            if content:
                return save_pdf_bytes(dest_path, content)
        resp = requests.get(
            f"https://doi.org/{doi}",
            headers={**HEADERS, "Accept": "application/pdf"},
            timeout=TIMEOUT_LONG, allow_redirects=True)
        if resp.status_code == 200 and is_valid_pdf_bytes(resp.content):
            return save_pdf_bytes(dest_path, resp.content)
    except Exception as e:
        log.debug(f"      Genetics GSA error: {e}")
    return False


# ============================================================
# SOURCE 28 — Karger
# ============================================================

def source_karger(doi: str, dest_path: str) -> bool:
    """Download from Karger Publishers."""
    if not doi or "10.1159" not in doi:
        return False
    log.info("    [28] Karger...")
    try:
        article_id = doi.split("/")[-1]
        for url in [
            f"https://karger.com/Article/Pdf/{article_id}",
            f"https://www.karger.com/Article/FullText/{article_id}",
        ]:
            content = get_pdf_from_url(url)
            if content:
                return save_pdf_bytes(dest_path, content)
        resp = requests.get(
            f"https://doi.org/{doi}",
            headers={**HEADERS, "Accept": "application/pdf"},
            timeout=TIMEOUT_LONG, allow_redirects=True)
        if resp.status_code == 200 and is_valid_pdf_bytes(resp.content):
            return save_pdf_bytes(dest_path, resp.content)
    except Exception as e:
        log.debug(f"      Karger error: {e}")
    return False


# ============================================================
# SOURCE 29 — De Gruyter
# ============================================================

def source_degruyter(doi: str, dest_path: str) -> bool:
    """Download from De Gruyter."""
    if not doi or "10.1515" not in doi:
        return False
    log.info("    [29] De Gruyter...")
    try:
        for url in [
            f"https://www.degruyter.com/document/doi/{doi}/pdf",
            f"https://www.degruyter.com/downloadpdf/journals/"
            f"{doi.split('/')[-1]}.pdf",
        ]:
            content = get_pdf_from_url(url)
            if content:
                return save_pdf_bytes(dest_path, content)
    except Exception as e:
        log.debug(f"      De Gruyter error: {e}")
    return False


# ============================================================
# SOURCE 30 — LWW / Wolters Kluwer
# ============================================================

def source_lww(doi: str, dest_path: str) -> bool:
    """Download from LWW / Wolters Kluwer."""
    if not doi or "10.1097" not in doi:
        return False
    log.info("    [30] LWW / Wolters Kluwer...")
    try:
        for url in [
            f"https://journals.lww.com/doi/pdf/{doi}",
            f"https://journals.lww.com/{doi.split('/')[-1]}.pdf",
        ]:
            content = get_pdf_from_url(url)
            if content:
                return save_pdf_bytes(dest_path, content)
        resp = requests.get(
            f"https://doi.org/{doi}",
            headers={**HEADERS, "Accept": "application/pdf"},
            timeout=TIMEOUT_LONG, allow_redirects=True)
        if resp.status_code == 200 and is_valid_pdf_bytes(resp.content):
            return save_pdf_bytes(dest_path, resp.content)
    except Exception as e:
        log.debug(f"      LWW error: {e}")
    return False


# ============================================================
# SOURCE 31 — DOAJ
# ============================================================

def source_doaj(doi: str, dest_path: str) -> bool:
    """Search DOAJ for OA article."""
    if not doi:
        return False
    log.info("    [31] DOAJ...")
    try:
        api_url = (f"https://doaj.org/api/search/articles/"
                   f"doi:{quote_plus(doi)}")
        resp = requests.get(api_url, headers=HEADERS,
                            timeout=TIMEOUT_SHORT)
        if resp.status_code != 200:
            return False
        results = resp.json().get("results", [])
        if not results:
            return False
        bibjson = results[0].get("bibjson", {})
        links = bibjson.get("link", [])
        for link in links:
            url = link.get("url", "")
            if url:
                content = get_pdf_from_url(url)
                if content:
                    return save_pdf_bytes(dest_path, content)
                if link.get("type") == "fulltext":
                    pdf_url = url.rstrip("/") + "/pdf"
                    content = get_pdf_from_url(pdf_url)
                    if content:
                        return save_pdf_bytes(dest_path, content)
    except Exception as e:
        log.debug(f"      DOAJ error: {e}")
    return False


# ============================================================
# SOURCE 32 — Internet Archive
# ============================================================

def source_internet_archive(doi: str, dest_path: str) -> bool:
    """Search Internet Archive / Wayback Machine."""
    if not doi:
        return False
    log.info("    [32] Internet Archive...")
    try:
        target_url = f"https://doi.org/{doi}"
        wb_url = f"https://web.archive.org/web/2/{target_url}"
        resp = requests.get(wb_url, headers=HEADERS,
                            timeout=TIMEOUT_LONG, allow_redirects=True)
        if resp.status_code == 200:
            if is_valid_pdf_bytes(resp.content):
                return save_pdf_bytes(dest_path, resp.content)
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(resp.content, "html.parser")
                for a_tag in soup.find_all("a", href=True):
                    href = a_tag["href"]
                    if ".pdf" in href.lower():
                        if not href.startswith("http"):
                            href = "https://web.archive.org" + href
                        content = get_pdf_from_url(href)
                        if content:
                            return save_pdf_bytes(dest_path, content)
            except ImportError:
                pass
    except Exception as e:
        log.debug(f"      Internet Archive error: {e}")
    return False


# ============================================================
# SOURCE 33 — PLOS (all journals catch-all)
# ============================================================

def source_plos_all(doi: str, dest_path: str) -> bool:
    """Download from any PLOS journal."""
    if not doi or "10.1371" not in doi:
        return False
    log.info("    [33] PLOS (all journals)...")
    try:
        journals = [
            "plosbiology", "plosmedicine", "plospathogens",
            "ploscompbiol", "plosntds",
        ]
        for journal in journals:
            url = (f"https://journals.plos.org/{journal}/article/file"
                   f"?id={doi}&type=printable")
            content = get_pdf_from_url(url)
            if content:
                return save_pdf_bytes(dest_path, content)
    except Exception as e:
        log.debug(f"      PLOS all error: {e}")
    return False


# ============================================================
# SOURCE 34 — DOI Direct (PDF accept header)
# ============================================================

def source_doi_direct(doi: str, dest_path: str) -> bool:
    """Try DOI URL directly with PDF accept header."""
    if not doi:
        return False
    log.info("    [34] DOI direct...")
    try:
        doi_url = f"https://doi.org/{doi}"
        resp = requests.get(
            doi_url,
            headers={**HEADERS, "Accept": "application/pdf"},
            timeout=TIMEOUT_LONG, allow_redirects=True)
        if resp.status_code == 200 and is_valid_pdf_bytes(resp.content):
            return save_pdf_bytes(dest_path, resp.content)
        if resp.status_code == 200:
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(resp.content, "html.parser")
                for a_tag in soup.find_all("a", href=True):
                    href = a_tag["href"]
                    if ".pdf" in href.lower():
                        if href.startswith("//"):
                            href = "https:" + href
                        elif href.startswith("/"):
                            parsed = urlparse(resp.url)
                            href = f"{parsed.scheme}://{parsed.netloc}{href}"
                        content = get_pdf_from_url(href)
                        if content:
                            return save_pdf_bytes(dest_path, content)
            except ImportError:
                pass
    except Exception as e:
        log.debug(f"      DOI direct error: {e}")
    return False


# ============================================================
# SOURCE 35 — Open Access Button
# ============================================================

def source_oa_button(doi: str, dest_path: str) -> bool:
    """Open Access Button API."""
    if not doi:
        return False
    log.info("    [35] Open Access Button...")
    try:
        api_url = (f"https://api.openaccessbutton.org/find"
                   f"?id={quote_plus(doi)}")
        resp = requests.get(api_url, headers=HEADERS,
                            timeout=TIMEOUT_SHORT)
        if resp.status_code != 200:
            return False
        data = resp.json()
        url_str = (data.get("url", "") or
                   data.get("data", {}).get("url", ""))
        if url_str:
            content = get_pdf_from_url(url_str)
            if content:
                return save_pdf_bytes(dest_path, content)
    except Exception as e:
        log.debug(f"      OA Button error: {e}")
    return False


# ============================================================
# MASTER DOWNLOAD FUNCTION
# Tries all 35 sources in order, stops on first success.
# ============================================================

def download_article(
    doi: str,
    title: str,
    dest_path: str,
    progress_label: str = "",
) -> str:
    """
    Try all methods in order. Returns method name if success,
    'failed' if all methods fail.
    """
    dest_dir = os.path.dirname(dest_path)
    ensure_dir(dest_dir)
    prefix = f"    {progress_label} " if progress_label else "    "

    # Skip if already downloaded
    if os.path.exists(dest_path):
        if is_usable_pdf_file(dest_path, require_text_probe=False):
            log.info(f"{prefix}Using existing validated PDF.")
            return "already_exists"
        quarantine_bad_pdf(dest_path, "existing PDF failed validation")

    # If no DOI provided, try to resolve from title
    if not doi and title:
        doi = doi_from_title_crossref(title)
        time.sleep(0.5)

    # Ordered list of (method_name, callable)
    methods = [
        ("pmc_oa",
         lambda: source_pmc_oa(doi, dest_path)),
        ("europe_pmc",
         lambda: source_europe_pmc(doi, title, dest_path)),
        ("unpaywall",
         lambda: source_unpaywall(doi, dest_path)),
        ("crossref",
         lambda: source_crossref(doi, dest_path)),
        ("semantic_scholar",
         lambda: source_semantic_scholar(doi, dest_path)),
        ("core",
         lambda: source_core(doi, dest_path)),
        ("scidownl",
         lambda: source_scidownl(doi, title, dest_path)),
        ("scihub_direct",
         lambda: source_scihub_direct(doi, title, dest_path)),
        ("elsevier",
         lambda: source_elsevier(doi, dest_path)),
        ("nature",
         lambda: source_nature(doi, dest_path)),
        ("plos_one",
         lambda: source_plos_one(doi, dest_path)),
        ("plos_genetics",
         lambda: source_plos_genetics(doi, dest_path)),
        ("frontiers",
         lambda: source_frontiers(doi, dest_path)),
        ("biorxiv",
         lambda: source_biorxiv(doi, dest_path)),
        ("medrxiv",
         lambda: source_medrxiv(doi, dest_path)),
        ("arxiv",
         lambda: source_arxiv(doi, dest_path)),
        ("wiley",
         lambda: source_wiley(doi, dest_path)),
        ("springer_bmc",
         lambda: source_springer_bmc(doi, dest_path)),
        ("mdpi",
         lambda: source_mdpi(doi, dest_path)),
        ("taylor_francis",
         lambda: source_taylor_francis(doi, dest_path)),
        ("oup",
         lambda: source_oup(doi, dest_path)),
        ("cambridge",
         lambda: source_cambridge(doi, dest_path)),
        ("hindawi",
         lambda: source_hindawi(doi, dest_path)),
        ("jci",
         lambda: source_jci(doi, dest_path)),
        ("elife",
         lambda: source_elife(doi, dest_path)),
        ("pnas",
         lambda: source_pnas(doi, dest_path)),
        ("blood_ash",
         lambda: source_blood_ash(doi, dest_path)),
        ("genetics_gsa",
         lambda: source_genetics_gsa(doi, dest_path)),
        ("karger",
         lambda: source_karger(doi, dest_path)),
        ("degruyter",
         lambda: source_degruyter(doi, dest_path)),
        ("lww",
         lambda: source_lww(doi, dest_path)),
        ("doaj",
         lambda: source_doaj(doi, dest_path)),
        ("internet_archive",
         lambda: source_internet_archive(doi, dest_path)),
        ("plos_all",
         lambda: source_plos_all(doi, dest_path)),
        ("doi_direct",
         lambda: source_doi_direct(doi, dest_path)),
        ("oa_button",
         lambda: source_oa_button(doi, dest_path)),
    ]

    for method_name, method_fn in methods:
        try:
            log.info(f"{prefix}Trying source: {method_name}")
            if method_fn():
                if is_usable_pdf_file(dest_path, require_text_probe=False):
                    return method_name
                quarantine_bad_pdf(
                    dest_path,
                    f"{method_name} produced an unusable PDF",
                )
                log.warning(
                    f"{prefix}Source {method_name} returned a bad PDF; "
                    "trying the next source."
                )
        except Exception as e:
            log.debug(f"    [{method_name}] exception: {e}")
        time.sleep(DELAY_BETWEEN_SOURCES)

    return "failed"


# ============================================================
# PARALLEL WORKER
# ============================================================

_progress_lock = threading.Lock()
_results_lock = threading.Lock()
_progress_done = 0
_progress_success = 0
_progress_failed = 0


def _download_worker(item: Dict[str, Any], total: int) -> dict:
    """Worker function executed by each thread."""
    global _progress_done, _progress_success, _progress_failed

    idx = int(item["global_index"]) + 1
    doi = item["doi"]
    title = item["title"]
    project_name = item["project_name"]
    rank_in_project = item["rank_in_project"]
    short_title = (title[:57] + "...") if len(title) > 60 else title
    progress_label = f"[{idx}/{total}] [{project_name}] Temp_{rank_in_project}"

    log.info(f"  {progress_label}: {short_title}")

    try:
        method = download_article(
            doi,
            title,
            item["dest_pdf"],
            progress_label=progress_label,
        )
    except Exception as e:
        log.error(f"    [{idx}] Unexpected error: {e}")
        log.debug(traceback.format_exc())
        method = "failed"

    success = method != "failed"
    try:
        write_article_artifacts(item, method=method, success=success)
    except Exception as e:
        log.debug(f"    Artifact write error: {e}")

    with _progress_lock:
        _progress_done += 1
        if success:
            _progress_success += 1
        else:
            _progress_failed += 1
        done = _progress_done
        successes = _progress_success
        failures = _progress_failed
        remaining = max(total - done, 0)

    log.info(
        f"    [{idx}/{total}] -> {method}  "
        f"(done {done}/{total}, success {successes}, "
        f"failed {failures}, remaining {remaining})"
    )

    return {
        "loop": item["loop"],
        "doi": doi,
        "title": title,
        "project_name": project_name,
        "project_key": item["project_key"],
        "project_id": item["project_id"],
        "strategy": item["strategy"],
        "rank_in_project": rank_in_project,
        "item_dir": item["item_dir"],
        "pdf_path": item["dest_pdf"],
        "fulltext_path": item["fulltext_path"],
        "metadata_json_path": item["metadata_json_path"],
        **extract_provenance(item),
        "method": method,
        "method_this_run": method,
        "resume_status": "",
        "original_download_method": known_original_download_method(method),
        "success": success,
    }


def _save_results(log_records: list, decisions: list):
    """Write current results to disk (thread-safe snapshot)."""
    with _results_lock:
        records_copy = list(log_records)
        decisions_copy = list(decisions)

    if not records_copy:
        return

    log_df = pd.DataFrame(records_copy)
    log_df.to_csv("Output/download_log.csv", index=False)
    save_source_stats(log_df)

    if "project_key" in log_df.columns:
        for project_key, sub in log_df.groupby("project_key", sort=True):
            if not project_key:
                continue
            project_log_path = os.path.join(DB_DIR, str(project_key), "download_log.csv")
            ensure_dir(os.path.dirname(project_log_path))
            sub.sort_values(["rank_in_project", "loop"], ascending=[True, True]).to_csv(
                project_log_path,
                index=False,
            )

    dec_df = pd.DataFrame(decisions_copy)
    dec_path = "Output/decisions_log.csv"
    if os.path.exists(dec_path):
        try:
            existing = pd.read_csv(dec_path)
            existing = existing[existing["step"] != "step11_download"]
            dec_df = pd.concat([existing, dec_df], ignore_index=True)
        except Exception:
            pass
    dec_df.to_csv(dec_path, index=False)


def save_source_stats(log_df: pd.DataFrame):
    """Save statistics about which sources succeeded."""
    if log_df.empty:
        return
    stats = log_df[log_df["method"] != "failed"].groupby("method").size().reset_index(name="count")
    stats["pct"] = 100 * stats["count"] / stats["count"].sum()
    stats = stats.sort_values("count", ascending=False)
    stats.to_csv("Output/download_source_stats.csv", index=False)


# ============================================================
# PARSE COMMAND LINE ARGUMENTS
# ============================================================

def parse_args():
    """Parse command line arguments for index range."""
    import argparse
    parser = argparse.ArgumentParser(description="Download full text articles for DOIs")
    parser.add_argument("--index_start", type=int, default=0,
                        help="Start index for processing (0-based)")
    parser.add_argument("--index_end", type=int, default=None,
                        help="End index for processing (exclusive)")
    return parser.parse_args()


# ============================================================
# PIPELINE ENTRY POINT — run(config)
# ============================================================

def run(config: dict):
    """Main entry point called by the pipeline runner."""
    global DB_DIR, _progress_done, _progress_success, _progress_failed

    # Parse command line arguments
    args = parse_args()
    index_start = args.index_start
    index_end = args.index_end

    # Use database directory from config if available
    db_dir = config.get("output", {}).get(
        "database_directory", "DataBase")
    DB_DIR = db_dir
    ensure_dir(DB_DIR)

    ranked_items = load_ranked_work_items(config)
    if ranked_items:
        if index_end is not None:
            all_items = ranked_items[index_start:index_end]
        elif index_start > 0:
            all_items = ranked_items[index_start:]
        else:
            all_items = ranked_items
        mode = "ranked"
    else:
        # Load with index range
        all_items = load_legacy_work_items(index_start, index_end)
        mode = "legacy"

    if not all_items:
        raise FileNotFoundError(
            "No input candidates found. Expected ranked outputs under "
            f"{RERANK_OUTPUT_DIR} or legacy Output/allarticles2.csv"
        )

    log_records = []
    decisions = []
    _progress_done = 0
    _progress_success = 0
    _progress_failed = 0

    total = len(all_items)
    log.info(f"\n  Step 11 mode: {mode}")
    log.info(f"  Index range: {index_start} to {index_end if index_end is not None else 'end'}")
    log.info(f"  Total articles in this range: {total}")
    log.info(f"  Database directory: {DB_DIR}\n")
    log.info(f"  Full-text extraction enabled: {EXTRACT_FULLTEXT}\n")
    log.info(f"  Parallelism: ThreadPoolExecutor (multithreading, not multiprocessing)")
    log.info(f"  Sci-Hub: HTTP-only mode (NO GUI)\n")

    start_time = time.time()

    # Pre-filter: separate already-done from pending
    already_done_items: List[Dict[str, Any]] = []
    work_items: List[Dict[str, Any]] = []
    n_corrupt_existing = 0

    for item in all_items:
        dest_pdf = item["dest_pdf"]

        if not os.path.exists(dest_pdf):
            work_items.append(item)
            continue

        if is_usable_pdf_file(dest_pdf, require_text_probe=False):
            already_done_items.append(item)
            try:
                write_article_artifacts(
                    item,
                    method="previously_downloaded",
                    success=True,
                    refresh_fulltext=False,
                )
            except Exception as e:
                log.debug(f"    Artifact refresh error for existing PDF: {e}")
        else:
            n_corrupt_existing += 1
            log.warning(
                f"    Existing PDF failed validation; will re-download: "
                f"{dest_pdf}"
            )
            quarantine_bad_pdf(dest_pdf, "resume scan failed validation")
            work_items.append(item)

    n_already_done = len(already_done_items)
    n_pending = len(work_items)

    log.info(f"  Already downloaded (valid PDF): {n_already_done}")
    log.info(f"  Corrupt existing PDFs to retry:   {n_corrupt_existing}")
    log.info(f"  Pending / failed (will attempt): {n_pending}\n")

    # Add already-done articles to log
    for item in already_done_items:
        doi_val = item["doi"]
        title_val = item["title"]
        existing_method = get_existing_original_download_method(item)
        log_records.append({
            "loop": item["loop"],
            "doi": doi_val,
            "title": title_val,
            "project_name": item["project_name"],
            "project_key": item["project_key"],
            "project_id": item["project_id"],
            "strategy": item["strategy"],
            "rank_in_project": item["rank_in_project"],
            "item_dir": item["item_dir"],
            "pdf_path": item["dest_pdf"],
            "fulltext_path": item["fulltext_path"],
            "metadata_json_path": item["metadata_json_path"],
            **extract_provenance(item),
            "method": existing_method or "previously_downloaded",
            "method_this_run": "previously_downloaded",
            "resume_status": "previously_downloaded",
            "original_download_method": existing_method,
            "success": True,
        })

    if n_pending == 0:
        log.info("  All articles already downloaded. Step 11 complete.")
        _save_results(log_records, decisions)
        return

    workers = min(MAX_WORKERS, n_pending)
    log.info(f"  Launching {workers} download threads...\n")

    interrupted = False
    try:
        with ThreadPoolExecutor(max_workers=workers,
                                thread_name_prefix="dl") as executor:
            futures = {}
            for item in work_items:
                fut = executor.submit(
                    _download_worker, item, n_pending)
                futures[fut] = item

            for fut in as_completed(futures):
                try:
                    result = fut.result()
                except Exception as e:
                    item = futures[fut]
                    idx = int(item["global_index"]) + 1
                    log.error(f"    [{idx}] Thread error: {e}")
                    result = {
                        "loop": item["loop"],
                        "doi": item["doi"],
                        "title": item["title"],
                        "project_name": item["project_name"],
                        "project_key": item["project_key"],
                        "project_id": item["project_id"],
                        "strategy": item["strategy"],
                        "rank_in_project": item["rank_in_project"],
                        "item_dir": item["item_dir"],
                        "pdf_path": item["dest_pdf"],
                        "fulltext_path": item["fulltext_path"],
                        "metadata_json_path": item["metadata_json_path"],
                        **extract_provenance(item),
                        "method": "failed",
                        "method_this_run": "failed",
                        "resume_status": "",
                        "original_download_method": "",
                        "success": False,
                    }

                    try:
                        write_article_artifacts(item, method="failed", success=False)
                    except Exception:
                        pass

                with _results_lock:
                    log_records.append(result)
                    decisions.append({
                        "step": "step11_download",
                        "project_name": result.get("project_name", ""),
                        "project_key": result.get("project_key", ""),
                        "rank_in_project": result.get("rank_in_project", ""),
                        "loop": result["loop"],
                        "title": result["title"],
                        "decision": ("downloaded"
                                     if result["success"]
                                     else "failed"),
                        "reason": f"method={result['method']}",
                    })

                if len(log_records) % SAVE_EVERY == 0:
                    _save_results(log_records, decisions)
                    log.info(
                        f"  ... intermediate save ({len(log_records)}/{total})"
                    )

    except KeyboardInterrupt:
        log.warning("\n  Interrupted by user. Saving progress so far...")
        interrupted = True

    # Final save
    _save_results(log_records, decisions)

    # Summary
    elapsed = time.time() - start_time
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)

    log_df = pd.DataFrame(log_records)
    this_run_df = log_df[log_df["method"] != "previously_downloaded"]
    n_attempted = n_pending
    n_success = int(this_run_df["success"].sum()) if n_attempted else 0
    n_failed = n_pending - n_success

    log.info(f"\n{'=' * 55}")
    log.info(f"  DOWNLOAD SUMMARY  ({mins}m {secs}s)")
    log.info(f"{'=' * 55}")
    log.info(f"  Threads used:                    {workers}")
    log.info(f"  Total articles in range:         {total}")
    log.info(f"  Already had valid PDF (skipped): {n_already_done}")
    log.info(f"  Attempted this run:              {n_pending}")
    log.info(f"  Successfully downloaded:         {n_success}")
    log.info(f"  Failed / no usable PDF:          {n_failed}")
    if n_attempted > 0:
        log.info(f"  Success rate (this run):         "
                 f"{100 * n_success / max(n_pending, 1):.1f}%")
    total_have_pdf = n_already_done + n_success
    log.info(f"  Total with valid PDF now:        "
             f"{total_have_pdf}/{total}")
    log.info(f"\n  Downloads by method (this run):")
    if n_attempted > 0:
        source_stats_df = this_run_df[this_run_df["method"] != "failed"].groupby("method").size().reset_index(name="count")
        if not source_stats_df.empty:
            source_stats_df["pct"] = 100 * source_stats_df["count"] / source_stats_df["count"].sum()
            source_stats_df = source_stats_df.sort_values("count", ascending=False)
            log.info(
                source_stats_df[["method", "count", "pct"]]
                .to_string(index=False)
            )
        else:
            log.info("  (none)")
    log.info("\n  Saved source stats:              Output/download_source_stats.csv")

    if interrupted:
        remaining = max(n_pending - n_success, 0)
        processed = len(log_df[log_df["method"] != "previously_downloaded"])
        log.warning(f"\n  ** Interrupted — {processed}/{n_pending} "
                    f"pending articles processed this run **")
        log.warning(f"  ** {remaining} still pending — re-run to continue **")
        log.info(f"\nStep 11 interrupted — re-run to resume.")
        raise KeyboardInterrupt("Step 11 interrupted by user")

    if n_failed > 0:
        log.info(f"\n  {n_failed} articles could not be downloaded "
                 f"(all 35 sources exhausted). Re-run to retry.")

    total_have_pdf_final = n_already_done + n_success
    if total_have_pdf_final < total:
        pct = 100 * total_have_pdf_final / max(total, 1)
        if n_success > 0:
            log.warning(
                f"\n  {total_have_pdf_final}/{total} articles "
                f"have valid PDFs ({pct:.0f}%). "
                f"{n_success} new this run. "
                f"Step will NOT be marked complete — re-run to "
                f"retry the remaining {total - total_have_pdf_final}."
            )
            raise RuntimeError(
                f"Step 11 incomplete: {total_have_pdf_final}/{total} "
                f"PDFs obtained ({pct:.0f}%). "
                f"{n_success} newly downloaded. "
                f"Re-run to retry {total - total_have_pdf_final} "
                f"failed articles."
            )
        else:
            log.warning(
                f"\n  {total_have_pdf_final}/{total} articles "
                f"have valid PDFs ({pct:.0f}%). "
                f"No new downloads this run — remaining "
                f"{total - total_have_pdf_final} appear unavailable. "
                f"Proceeding to next step."
            )

    log.info(f"\nStep 11 complete.")


def _default_config() -> dict:
    """Return a minimal config for direct script execution."""
    return {
        "output": {
            "database_directory": DB_DIR,
        },
        "ranking": {
            "projects_dir": PROJECTS_DIR,
            "selected_strategy_num": 3,
            "selected_strategy_name": "TitleObjectiveSBERT",
            "top_k_per_project": 1000,
        },
    }


def main() -> None:
    """Allow this script to be run directly from an IDE or terminal."""
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)

    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        config_path = Path(sys.argv[1]).expanduser()
        if not config_path.is_absolute():
            config_path = (script_dir / config_path).resolve()
        with open(config_path, "r", encoding="utf-8") as fh:
            config = json.load(fh)
        log.info(f"Using config: {config_path}")
    else:
        config = _default_config()
        log.info("Using built-in default config for direct execution.")

    run(config)


if __name__ == "__main__":
    main()
