# Step1-DownloadResearchArticleForSystemEvaluation.py
from __future__ import annotations

import csv
import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote_plus

import requests

# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = Path("benchmark_reviews")
EMAIL_FOR_UNPAYWALL = "muneebsiddique007@gmail.com"
# Deliberate per-topic sampling cap for the benchmark dataset.
MAX_PER_TOPIC = 3
REQUEST_TIMEOUT = 30
MIN_PDF_SIZE = 10000
MIN_YEAR = 2018
# A review with too few indexed references can't serve as a recall benchmark
# target, so it is excluded from the dataset.
MIN_REFERENCED_WORKS = 10

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}

# ============================================================
# TOPIC DEFINITIONS
# query = broad retrieval
# required_terms = must appear in title/abstract
# global document-type exclusions are applied via REVIEW_EXCLUDE_TERMS
# ============================================================

TOPIC_QUERIES = [
    {
        "topic": "polygenic_risk",
        "query": 'polygenic risk score report communication review',
        "required_terms": [
            "polygenic risk score", "polygenic score", "pgs", "prs",
            "report", "communication", "risk communication", "visual aid"
        ],
    },
    {
        "topic": "genomics",
        "query": 'genomics clinical implementation review',
        "required_terms": [
            "genomics", "genomic", "clinical implementation",
            "genetic testing", "precision medicine"
        ],
    },
    {
        "topic": "bioinformatics",
        "query": 'bioinformatics benchmarking systematic review',
        "required_terms": [
            "bioinformatics", "benchmark", "benchmarking",
            "computational biology", "omics"
        ],
    },
    {
        "topic": "llm_evaluation",
        "query": 'large language model evaluation systematic review',
        "required_terms": [
            "large language model", "llm", "language model",
            "evaluation", "assessment", "benchmark"
        ],
    },
    {
        "topic": "code_generation",
        "query": 'code generation model systematic review',
        "required_terms": [
            "code generation", "program synthesis",
            "source code", "software engineering", "coding"
        ],
    },
    {
        "topic": "machine_learning",
        "query": 'machine learning benchmarking systematic review',
        "required_terms": [
            "machine learning", "benchmark", "benchmarking",
            "evaluation", "prediction model"
        ],
    },
    {
        "topic": "public_health",
        "query": 'digital health intervention systematic review',
        "required_terms": [
            "digital health", "intervention", "public health",
            "mobile health", "mhealth", "telehealth"
        ],
    },
    {
        "topic": "psychology",
        "query": 'mental health prediction model systematic review',
        "required_terms": [
            "mental health", "prediction model",
            "psychology", "depression", "anxiety"
        ],
    },
    {
        "topic": "education",
        "query": 'learning analytics systematic review',
        "required_terms": [
            "learning analytics", "education", "educational data mining",
            "student performance", "higher education"
        ],
    },
    {
        "topic": "social_science",
        "query": 'misinformation detection systematic review',
        "required_terms": [
            "misinformation", "disinformation", "fake news",
            "detection", "social media"
        ],
    },
]

# ============================================================
# GLOBAL FILTERS
# ============================================================

REVIEW_INCLUDE_TERMS = [
    "systematic review",
    "scoping review",
    "literature review",
    "narrative review",
    "umbrella review",
    "meta-analysis",
    "metaanalysis",
    "evidence synthesis",
]

REVIEW_EXCLUDE_TERMS = [
    "guideline",
    "practice guideline",
    "clinical guideline",
    "recommendation",
    "recommendations",
    "consensus statement",
    "position statement",
    "statement",
    "tutorial",
    "practical tutorial",
    "protocol",
    "study protocol",
    "checklist",
    "reporting guideline",
    "reporting statement",
    "prisma 2020",
    "prisma",
    "probast",
    "explanation and elaboration",
    "editorial",
    "commentary",
    "letter to the editor",
]

# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class Paper:
    topic: str
    title: str
    year: Optional[int]
    journal: str
    doi: str
    openalex_id: str
    authors: str
    abstract: str
    type_crossref: str
    is_oa: bool
    oa_pdf_url: str
    landing_page_url: str
    referenced_works: List[str]
    source: str = "openalex"
    download_status: str = "not_attempted"
    pdf_path: str = ""

# ============================================================
# HELPERS
# ============================================================

def safe_slug(text: str, max_len: int = 80) -> str:
    text = re.sub(r"\s+", "_", text.strip().lower())
    text = re.sub(r"[^a-z0-9_]+", "", text)
    return text[:max_len].strip("_") or "paper"

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def is_valid_pdf_bytes(content: bytes) -> bool:
    return len(content) >= MIN_PDF_SIZE and content[:4] == b"%PDF"

def save_pdf(path: Path, content: bytes) -> bool:
    if not is_valid_pdf_bytes(content):
        return False
    path.write_bytes(content)
    return True

def request_json(url: str, params: Optional[dict] = None) -> dict:
    r = requests.get(url, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()

def request_bytes(url: str, headers: Optional[dict] = None) -> bytes:
    h = HEADERS.copy()
    if headers:
        h.update(headers)
    r = requests.get(url, headers=h, timeout=REQUEST_TIMEOUT, allow_redirects=True)
    if r.status_code == 200:
        return r.content
    return b""

def invert_openalex_abstract(inv_idx: dict) -> str:
    if not inv_idx:
        return ""
    try:
        max_pos = max(max(v) for v in inv_idx.values()) + 1
        arr = [""] * max_pos
        for word, positions in inv_idx.items():
            for pos in positions:
                arr[pos] = word
        return " ".join(arr).strip()
    except Exception:
        return ""

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()

def contains_any(text: str, terms: List[str]) -> bool:
    text = normalize_text(text)
    return any(term.lower() in text for term in terms)

def count_matches(text: str, terms: List[str]) -> int:
    text = normalize_text(text)
    return sum(1 for term in terms if term.lower() in text)

def looks_like_review(title: str, abstract: str) -> bool:
    combined = normalize_text(f"{title} {abstract}")
    return contains_any(combined, REVIEW_INCLUDE_TERMS)

def looks_like_bad_doc_type(title: str, abstract: str) -> bool:
    combined = normalize_text(f"{title} {abstract}")
    return contains_any(combined, REVIEW_EXCLUDE_TERMS)

def passes_topic_filter(title: str, abstract: str, topic_def: Dict) -> bool:
    combined = normalize_text(f"{title} {abstract}")
    req_hits = count_matches(combined, topic_def["required_terms"])

    # Require at least 1 topic-specific match
    if req_hits < 1:
        return False
    return True

# ============================================================
# SEARCH
# ============================================================

def search_openalex_reviews(query: str, max_results: int = 50) -> List[Paper]:
    url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "per-page": min(max_results, 100),
        "mailto": EMAIL_FOR_UNPAYWALL,
        "filter": ",".join([
            "language:en",
            "type:article",
            f"from_publication_date:{MIN_YEAR}-01-01"
        ]),
        "sort": "relevance_score:desc",
        "select": (
            "id,doi,title,publication_year,primary_location,open_access,"
            "authorships,abstract_inverted_index,referenced_works,type"
        ),
    }

    data = request_json(url, params=params)
    results = data.get("results", [])
    papers: List[Paper] = []

    for item in results:
        title = item.get("title") or ""
        abstract = invert_openalex_abstract(item.get("abstract_inverted_index") or {})
        doi = (item.get("doi") or "").replace("https://doi.org/", "")
        primary_loc = item.get("primary_location") or {}
        source = (primary_loc.get("source") or {}).get("display_name", "")
        oa = item.get("open_access") or {}
        oa_url = oa.get("oa_url") or ""
        landing = primary_loc.get("landing_page_url") or ""
        authorships = item.get("authorships") or []
        authors = "; ".join(
            a.get("author", {}).get("display_name", "")
            for a in authorships[:10]
            if a.get("author", {}).get("display_name")
        )

        papers.append(
            Paper(
                topic="",
                title=title,
                year=item.get("publication_year"),
                journal=source,
                doi=doi,
                openalex_id=item.get("id", ""),
                authors=authors,
                abstract=abstract,
                type_crossref=item.get("type", ""),
                is_oa=bool(oa.get("is_oa")),
                oa_pdf_url=oa_url if oa_url and ".pdf" in oa_url.lower() else "",
                landing_page_url=landing,
                referenced_works=item.get("referenced_works") or [],
            )
        )

    return papers

# ============================================================
# DOWNLOADERS (LEGAL OA ONLY)
# ============================================================

def try_openalex_oa(paper: Paper) -> Optional[bytes]:
    if paper.oa_pdf_url:
        content = request_bytes(paper.oa_pdf_url)
        if is_valid_pdf_bytes(content):
            return content
    return None

def try_unpaywall(doi: str) -> Optional[bytes]:
    if not doi:
        return None
    url = f"https://api.unpaywall.org/v2/{quote_plus(doi)}"
    params = {"email": EMAIL_FOR_UNPAYWALL}
    try:
        data = request_json(url, params=params)
    except Exception:
        return None

    pdf_urls = []
    best = data.get("best_oa_location") or {}
    if best.get("url_for_pdf"):
        pdf_urls.append(best["url_for_pdf"])
    if best.get("url"):
        pdf_urls.append(best["url"])

    for loc in data.get("oa_locations", []) or []:
        if loc.get("url_for_pdf"):
            pdf_urls.append(loc["url_for_pdf"])
        elif loc.get("url"):
            pdf_urls.append(loc["url"])

    seen = set()
    for pdf_url in pdf_urls:
        if not pdf_url or pdf_url in seen:
            continue
        seen.add(pdf_url)
        try:
            content = request_bytes(pdf_url)
            if is_valid_pdf_bytes(content):
                return content
        except Exception:
            pass
    return None

def try_europe_pmc(doi: str, title: str) -> Optional[bytes]:
    query = f"DOI:{doi}" if doi else title
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    params = {
        "query": query,
        "format": "json",
        "pageSize": 1,
        "resultType": "core"
    }

    try:
        data = request_json(url, params=params)
        results = data.get("resultList", {}).get("result", [])
        if not results:
            return None
        art = results[0]
        pmcid = art.get("pmcid", "")
        if pmcid:
            pmc_num = pmcid.replace("PMC", "")
            for pdf_url in [
                f"https://europepmc.org/articles/PMC{pmc_num}/pdf",
                f"https://europepmc.org/articles/PMC{pmc_num}/pdf/render",
                f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_num}/pdf/"
            ]:
                try:
                    content = request_bytes(pdf_url)
                    if is_valid_pdf_bytes(content):
                        return content
                except Exception:
                    pass

        for ftl in art.get("fullTextUrlList", {}).get("fullTextUrl", []) or []:
            if ftl.get("documentStyle") == "pdf" and ftl.get("url"):
                try:
                    content = request_bytes(ftl["url"])
                    if is_valid_pdf_bytes(content):
                        return content
                except Exception:
                    pass
    except Exception:
        return None

    return None

def try_doi_direct(doi: str) -> Optional[bytes]:
    if not doi:
        return None
    url = f"https://doi.org/{doi}"
    try:
        content = request_bytes(url, headers={"Accept": "application/pdf"})
        if is_valid_pdf_bytes(content):
            return content
    except Exception:
        return None
    return None

def download_pdf_for_paper(paper: Paper) -> Optional[bytes]:
    for fn in [
        lambda: try_openalex_oa(paper),
        lambda: try_unpaywall(paper.doi),
        lambda: try_europe_pmc(paper.doi, paper.title),
        lambda: try_doi_direct(paper.doi),
    ]:
        try:
            content = fn()
            if content and is_valid_pdf_bytes(content):
                return content
        except Exception:
            pass
        time.sleep(0.5)
    return None

# ============================================================
# MAIN
# ============================================================

def main() -> None:
    ensure_dir(OUTPUT_DIR)

    all_rows: List[Dict] = []
    seen_titles = set()
    seen_dois = set()
    saved_count = 0

    for topic_idx, topic_def in enumerate(TOPIC_QUERIES, start=1):
        topic = topic_def["topic"]
        query = topic_def["query"]

        print(f"\n=== [{topic_idx}/{len(TOPIC_QUERIES)}] Searching topic: {topic} ===")
        print(f"Query: {query}")

        try:
            results = search_openalex_reviews(query, max_results=60)
        except Exception as e:
            print(f"Search failed for topic '{topic}': {e}")
            continue

        kept = 0
        for paper in results:
            if kept >= MAX_PER_TOPIC:
                break

            title_norm = normalize_text(paper.title)
            abs_norm = normalize_text(paper.abstract)

            # Dedup
            if paper.doi and paper.doi in seen_dois:
                continue
            if title_norm in seen_titles:
                continue

            # Strict review filter
            if not looks_like_review(paper.title, paper.abstract):
                continue

            # Exclude guidelines/tutorials/reporting papers
            if looks_like_bad_doc_type(paper.title, paper.abstract):
                continue

            # Topic filter
            if not passes_topic_filter(paper.title, paper.abstract, topic_def):
                continue

            # Extra hard exclusion for PRISMA-like junk
            if "prisma" in title_norm or "probast" in title_norm:
                continue

            if len(paper.referenced_works) < MIN_REFERENCED_WORKS:
                print(f"     SKIP (only {len(paper.referenced_works)} referenced_works)")
                continue

            if paper.year and int(paper.year) < MIN_YEAR:
                continue

            if paper.doi:
                seen_dois.add(paper.doi)
            seen_titles.add(title_norm)

            paper.topic = topic
            slug = safe_slug(paper.title)
            folder = OUTPUT_DIR / f"{topic}_{kept+1:02d}_{slug}"
            ensure_dir(folder)

            pdf_path = folder / "paper.pdf"
            meta_path = folder / "metadata.json"

            print(f"  -> {paper.title[:110]}")

            content = download_pdf_for_paper(paper)
            if content:
                saved = save_pdf(pdf_path, content)
                if saved:
                    paper.download_status = "downloaded"
                    paper.pdf_path = str(pdf_path)
                    saved_count += 1
                    print("     PDF downloaded")
                else:
                    paper.download_status = "pdf_invalid"
                    print("     PDF invalid")
            else:
                paper.download_status = "metadata_only"
                print("     No legal OA PDF found")

            meta_path.write_text(
                json.dumps(asdict(paper), indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            all_rows.append(asdict(paper))
            kept += 1
            time.sleep(1.0)

    csv_path = OUTPUT_DIR / "metadata.csv"
    if all_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

    n_with_pdf = sum(1 for r in all_rows if r["download_status"] == "downloaded")
    total_refs = sum(len(r["referenced_works"]) for r in all_rows)
    print("\n=== DONE ===")
    print(f"Reviews kept       : {len(all_rows)}")
    print(f"  PDF downloaded   : {n_with_pdf}")
    print(f"Total referenced_works : {total_refs}")
    print(f"Mean refs/review   : {total_refs / max(len(all_rows), 1):.1f}")
    print(f"Metadata CSV       : {csv_path}")

if __name__ == "__main__":
    main()
