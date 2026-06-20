from __future__ import annotations

import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


# ============================================================
# CONFIG
# ============================================================

BENCHMARK_DIR = Path("benchmark_reviews")
PROJECTS_DIR = Path("Projects")
REQUEST_TIMEOUT = 30

CORE_SECTIONS = [
    "abstract",
    "introduction",
    "methods",
    "results",
    "discussion",
    "conclusion",
    "limitations",
]


# ============================================================
# HELPERS
# ============================================================

def clean_text(x: Any) -> str:
    if x is None:
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def safe_len_list(x: Any) -> int:
    return len(x) if isinstance(x, list) else 0


def mean_or_zero(values: List[float]) -> float:
    return statistics.mean(values) if values else 0.0


def median_or_zero(values: List[float]) -> float:
    return statistics.median(values) if values else 0.0


def min_or_zero(values: List[float]) -> float:
    return min(values) if values else 0.0


def max_or_zero(values: List[float]) -> float:
    return max(values) if values else 0.0


def pct(n: int, d: int) -> float:
    return (100.0 * n / d) if d else 0.0


def slugify(text: str, max_len: int = 80) -> str:
    text = re.sub(r"\s+", "_", clean_text(text).lower())
    text = re.sub(r"[^a-z0-9_]+", "", text)
    return text[:max_len].strip("_") or "untitled_project"


def normalize_doi(doi: Any) -> str:
    doi = clean_text(doi).lower()
    if doi in {"", "none", "null", "nan"}:
        return ""
    for prefix in (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "doi.org/",
        "doi:",
    ):
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
    doi = re.sub(r"\s+", "", doi)
    doi = doi.strip().strip(".,;:)>]} \t\r\n'\"").lstrip("<([{ \t\r\n'\"")
    return "" if doi in {"", "none", "null", "nan"} else doi


def fetch_bibtex_from_crossref(doi: Any) -> str:
    doi = normalize_doi(doi)
    if not doi:
        return ""

    url = f"https://api.crossref.org/works/{doi}/transform/application/x-bibtex"
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200 and response.text.strip().startswith("@"):
            return response.text.strip()
    except Exception:
        return ""
    return ""


def load_local_article_bibtex(project_dir: Optional[Path]) -> str:
    if project_dir is None:
        return ""
    bib_path = project_dir / "references.bib"
    if not bib_path.exists():
        return ""
    return bib_path.read_text(encoding="utf-8", errors="ignore").strip()


def get_article_bibtex(row: Dict[str, Any], project_dir: Optional[Path]) -> str:
    bibtex = fetch_bibtex_from_crossref(row.get("doi", ""))
    if bibtex:
        return bibtex
    return load_local_article_bibtex(project_dir)


def find_downloaded_benchmark_records() -> List[Dict[str, Any]]:
    rows = []

    if not BENCHMARK_DIR.exists():
        return rows

    for folder in sorted(BENCHMARK_DIR.iterdir()):
        if not folder.is_dir():
            continue

        meta_path = folder / "metadata.json"
        pdf_path = folder / "paper.pdf"

        if not meta_path.exists():
            continue

        meta = read_json(meta_path)
        status = clean_text(meta.get("download_status", "")).lower()

        has_pdf = pdf_path.exists() and pdf_path.stat().st_size > 1000

        meta["_benchmark_folder"] = folder.name
        meta["_benchmark_folder_path"] = str(folder)
        meta["_paper_pdf_path"] = str(pdf_path) if has_pdf else ""
        meta["_has_pdf"] = has_pdf

        if status == "downloaded" and has_pdf:
            rows.append(meta)

    return rows


def find_project_dir_for_record(row: Dict[str, Any]) -> Optional[Path]:
    title = clean_text(row.get("title", ""))
    expected = PROJECTS_DIR / slugify(title)

    if expected.exists():
        return expected

    # Fallback: match by title inside article_metadata.json
    if not PROJECTS_DIR.exists():
        return None

    title_norm = clean_text(title).lower()

    for p in sorted(PROJECTS_DIR.iterdir()):
        if not p.is_dir():
            continue
        meta_path = p / "article_metadata.json"
        if not meta_path.exists():
            continue
        meta = read_json(meta_path)
        if clean_text(meta.get("title", "")).lower() == title_norm:
            return p

    return None


def get_pdf_page_count(pdf_path: Path) -> int:
    if not pdf_path.exists():
        return 0

    # Prefer PyMuPDF if installed.
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        n = len(doc)
        doc.close()
        return n
    except Exception:
        pass

    return 0


def load_reference_stats(project_dir: Optional[Path], row: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "n_referenced_works_metadata": safe_len_list(row.get("referenced_works")),
        "n_references_json": 0,
        "n_reference_dois": 0,
        "reference_source": "",
    }

    if project_dir is None:
        return out

    references_json = project_dir / "references.json"
    config_evidence_json = project_dir / "config_evidence.json"

    if references_json.exists():
        ref = read_json(references_json)
        out["n_references_json"] = safe_int(ref.get("n_referenced_works", 0))
        out["n_reference_dois"] = safe_int(ref.get("n_with_doi", 0))
        out["reference_source"] = clean_text(ref.get("source", ""))

    elif config_evidence_json.exists():
        cfg = read_json(config_evidence_json)
        ref = cfg.get("reference_validation_targets", {}) or {}
        out["n_references_json"] = safe_int(ref.get("n_referenced_works", 0))
        out["n_reference_dois"] = safe_int(ref.get("n_with_doi", 0))
        out["reference_source"] = clean_text(ref.get("source", ""))

    return out


def load_preprocessing_stats(project_dir: Optional[Path]) -> Dict[str, Any]:
    out = {
        "markdown_characters": 0,
        "plain_text_characters": 0,
        "plain_text_words": 0,
        "n_extracted_images": 0,
        "n_images_ocr_ok": 0,
        "backend_used_for_markdown": "",
        "image_ocr_enabled": "",
        "image_ocr_method": "",
    }

    if project_dir is None:
        return out

    path = project_dir / "preprocessing_summary.json"
    if not path.exists():
        return out

    payload = read_json(path)
    stats = payload.get("text_statistics", {}) or {}

    out["markdown_characters"] = safe_int(stats.get("markdown_characters", 0))
    out["plain_text_characters"] = safe_int(stats.get("plain_text_characters", 0))
    out["plain_text_words"] = safe_int(stats.get("plain_text_words", 0))
    out["n_extracted_images"] = safe_int(stats.get("n_extracted_images", 0))
    out["n_images_ocr_ok"] = safe_int(stats.get("n_images_ocr_ok", 0))
    out["backend_used_for_markdown"] = clean_text(payload.get("backend_used_for_markdown", ""))
    out["image_ocr_enabled"] = str(payload.get("image_ocr_enabled", ""))
    out["image_ocr_method"] = clean_text(payload.get("image_ocr_method", ""))

    return out


def load_section_stats(project_dir: Optional[Path]) -> Dict[str, Any]:
    out = {
        "sectioning_method": "",
        "n_headings": 0,
        "n_sections_found_all": 0,
        "n_core_sections_found": 0,
        "n_core_sections_usable": 0,
        "section_word_counts": {sec: 0 for sec in CORE_SECTIONS},
        "section_found": {sec: False for sec in CORE_SECTIONS},
        "section_usable": {sec: False for sec in CORE_SECTIONS},
        "n_figures_tei": 0,
        "n_tables_tei": 0,
    }

    if project_dir is None:
        return out

    sectioned_path = project_dir / "sectioned_article.json"
    config_path = project_dir / "config_evidence.json"

    payload = {}
    if sectioned_path.exists():
        payload = read_json(sectioned_path)
    elif config_path.exists():
        payload = read_json(config_path)

    if not payload:
        return out

    out["sectioning_method"] = clean_text(payload.get("sectioning_method", ""))
    out["n_headings"] = safe_int(payload.get("n_headings", 0))

    # Section stats can be stored in different shapes depending on the step/version.
    sections = payload.get("sections", {}) or {}
    section_presence = payload.get("section_presence", {}) or {}

    if sections:
        for sec, info in sections.items():
            if isinstance(info, dict) and bool(info.get("found", False)):
                out["n_sections_found_all"] += 1

        for sec in CORE_SECTIONS:
            info = sections.get(sec, {}) or {}
            found = bool(info.get("found", False))
            wc = safe_int(info.get("word_count", 0))
            usable = found and wc >= 50

            out["section_found"][sec] = found
            out["section_word_counts"][sec] = wc
            out["section_usable"][sec] = usable

    elif section_presence:
        for sec, info in section_presence.items():
            if isinstance(info, dict) and bool(info.get("found", False)):
                out["n_sections_found_all"] += 1

        for sec in CORE_SECTIONS:
            info = section_presence.get(sec, {}) or {}
            found = bool(info.get("found", False))
            wc = safe_int(info.get("word_count", 0))
            usable = bool(info.get("usable_for_scoring", False))

            out["section_found"][sec] = found
            out["section_word_counts"][sec] = wc
            out["section_usable"][sec] = usable

    out["n_core_sections_found"] = sum(1 for sec in CORE_SECTIONS if out["section_found"][sec])
    out["n_core_sections_usable"] = sum(1 for sec in CORE_SECTIONS if out["section_usable"][sec])

    # Figure/table evidence can be in supporting_evidence or figures_tables.
    support = payload.get("supporting_evidence", {}) or {}
    figures_tables = payload.get("figures_tables", {}) or {}

    figures = support.get("figures", figures_tables.get("figures", []))
    tables = support.get("tables", figures_tables.get("tables", []))

    out["n_figures_tei"] = safe_len_list(figures)
    out["n_tables_tei"] = safe_len_list(tables)

    return out


def print_counter(title: str, counter: Counter) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if not counter:
        print("None")
        return

    for key, value in sorted(counter.items(), key=lambda x: str(x[0])):
        print(f"{key}: {value}")


def print_numeric_summary(title: str, values: List[int]) -> None:
    print(f"\n{title}")
    print("-" * len(title))

    if not values:
        print("n=0")
        return

    print(f"n       : {len(values)}")
    print(f"total   : {sum(values)}")
    print(f"mean    : {mean_or_zero(values):.2f}")
    print(f"median  : {median_or_zero(values):.2f}")
    print(f"min     : {min_or_zero(values):.2f}")
    print(f"max     : {max_or_zero(values):.2f}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    rows = find_downloaded_benchmark_records()

    print("=" * 80)
    print("SUPPLEMENTARY MATERIAL 1 - SECTION 1 DATASET GENERATION STATS")
    print("=" * 80)

    print(f"Benchmark directory : {BENCHMARK_DIR.resolve()}")
    print(f"Projects directory  : {PROJECTS_DIR.resolve()}")
    print(f"Downloaded PDF articles included: {len(rows)}")

    if not rows:
        print("\nNo downloaded PDF records found.")
        print("Expected structure:")
        print("  benchmark_reviews/<article_folder>/metadata.json")
        print("  benchmark_reviews/<article_folder>/paper.pdf")
        print("  Projects/<project>/preprocessing_summary.json")
        print("  Projects/<project>/sectioned_article.json")
        print("  Projects/<project>/config_evidence.json")
        print("  Projects/<project>/references.json")
        return

    enriched = []

    for i, row in enumerate(rows, start=1):
        project_dir = find_project_dir_for_record(row)

        pdf_path = Path(clean_text(row.get("_paper_pdf_path", "")))
        if project_dir is not None and (project_dir / "article.pdf").exists():
            pdf_path = project_dir / "article.pdf"

        ref_stats = load_reference_stats(project_dir, row)
        prep_stats = load_preprocessing_stats(project_dir)
        sec_stats = load_section_stats(project_dir)
        page_count = get_pdf_page_count(pdf_path)
        article_bibtex = get_article_bibtex(row, project_dir)

        record = {
            "index": i,
            "title": clean_text(row.get("title", "")),
            "topic": clean_text(row.get("topic", "unknown")) or "unknown",
            "year": clean_text(row.get("year", "")),
            "journal": clean_text(row.get("journal", "")),
            "doi": clean_text(row.get("doi", "")),
            "openalex_id": clean_text(row.get("openalex_id", "")),
            "authors": clean_text(row.get("authors", "")),
            "article_bibtex": article_bibtex,
            "download_status": clean_text(row.get("download_status", "")),
            "benchmark_folder": clean_text(row.get("_benchmark_folder", "")),
            "project_dir": str(project_dir) if project_dir else "",
            "pdf_path": str(pdf_path) if pdf_path else "",
            "pdf_size_bytes": pdf_path.stat().st_size if pdf_path.exists() else 0,
            "pdf_pages": page_count,
            **ref_stats,
            **prep_stats,
            **sec_stats,
        }

        enriched.append(record)

    # ------------------------------------------------------------
    # Overall distributions
    # ------------------------------------------------------------

    topic_counter = Counter(r["topic"] for r in enriched)
    year_counter = Counter(r["year"] for r in enriched if r["year"])
    journal_counter = Counter(r["journal"] for r in enriched if r["journal"])
    backend_counter = Counter(r["backend_used_for_markdown"] for r in enriched if r["backend_used_for_markdown"])
    section_method_counter = Counter(r["sectioning_method"] for r in enriched if r["sectioning_method"])

    print_counter("Topic distribution", topic_counter)
    print_counter("Publication year distribution", year_counter)
    print_counter("Journal distribution", journal_counter)
    print_counter("PDF-to-Markdown backend distribution", backend_counter)
    print_counter("Sectioning method distribution", section_method_counter)

    # ------------------------------------------------------------
    # Numeric summaries
    # ------------------------------------------------------------

    print_numeric_summary("PDF page count", [r["pdf_pages"] for r in enriched if r["pdf_pages"]])
    print_numeric_summary("PDF size bytes", [r["pdf_size_bytes"] for r in enriched if r["pdf_size_bytes"]])
    print_numeric_summary("OpenAlex referenced works per article", [r["n_referenced_works_metadata"] for r in enriched])
    print_numeric_summary("Resolved reference DOIs per article", [r["n_reference_dois"] for r in enriched])
    print_numeric_summary("Markdown characters per article", [r["markdown_characters"] for r in enriched if r["markdown_characters"]])
    print_numeric_summary("Plain-text words per article", [r["plain_text_words"] for r in enriched if r["plain_text_words"]])
    print_numeric_summary("Extracted images per article", [r["n_extracted_images"] for r in enriched])
    print_numeric_summary("OCR-success images per article", [r["n_images_ocr_ok"] for r in enriched])
    print_numeric_summary("GROBID TEI headings per article", [r["n_headings"] for r in enriched])
    print_numeric_summary("Core sections found per article", [r["n_core_sections_found"] for r in enriched])
    print_numeric_summary("Core sections usable per article", [r["n_core_sections_usable"] for r in enriched])
    print_numeric_summary("TEI figures per article", [r["n_figures_tei"] for r in enriched])
    print_numeric_summary("TEI tables per article", [r["n_tables_tei"] for r in enriched])

    # ------------------------------------------------------------
    # Section presence
    # ------------------------------------------------------------

    print("\nCore section detection summary")
    print("------------------------------")
    for sec in CORE_SECTIONS:
        found_n = sum(1 for r in enriched if r["section_found"].get(sec, False))
        usable_n = sum(1 for r in enriched if r["section_usable"].get(sec, False))
        word_counts = [
            r["section_word_counts"].get(sec, 0)
            for r in enriched
            if r["section_word_counts"].get(sec, 0) > 0
        ]

        print(
            f"{sec:13s} found={found_n}/{len(enriched)} "
            f"({pct(found_n, len(enriched)):.1f}%), "
            f"usable={usable_n}/{len(enriched)} "
            f"({pct(usable_n, len(enriched)):.1f}%), "
            f"mean_words={mean_or_zero(word_counts):.1f}"
        )

    # ------------------------------------------------------------
    # Manuscript-ready summary text
    # ------------------------------------------------------------

    n_articles = len(enriched)
    n_topics = len(topic_counter)
    min_years = [safe_int(r["year"], 0) for r in enriched if safe_int(r["year"], 0)]
    year_min = min(min_years) if min_years else "NA"
    year_max = max(min_years) if min_years else "NA"

    total_refs = sum(r["n_referenced_works_metadata"] for r in enriched)
    total_dois = sum(r["n_reference_dois"] for r in enriched)
    total_images = sum(r["n_extracted_images"] for r in enriched)
    total_ocr_images = sum(r["n_images_ocr_ok"] for r in enriched)
    total_figures = sum(r["n_figures_tei"] for r in enriched)
    total_tables = sum(r["n_tables_tei"] for r in enriched)

    mean_refs = mean_or_zero([r["n_referenced_works_metadata"] for r in enriched])
    mean_dois = mean_or_zero([r["n_reference_dois"] for r in enriched])
    mean_words = mean_or_zero([r["plain_text_words"] for r in enriched if r["plain_text_words"]])
    mean_sections = mean_or_zero([r["n_core_sections_found"] for r in enriched])
    mean_images = mean_or_zero([r["n_extracted_images"] for r in enriched])
    mean_figures = mean_or_zero([r["n_figures_tei"] for r in enriched])
    mean_tables = mean_or_zero([r["n_tables_tei"] for r in enriched])

    topic_text = ", ".join(
        f"{topic.replace('_', ' ')} (n={count})"
        for topic, count in sorted(topic_counter.items())
    )

    print("\n" + "=" * 80)
    print("MANUSCRIPT-READY NUMERIC SUMMARY")
    print("=" * 80)

    print(
        f"The evaluation dataset contained {n_articles} downloaded open-access review articles "
        f"across {n_topics} topic groups: {topic_text}. Publication years ranged from "
        f"{year_min} to {year_max}. Across the included articles, OpenAlex metadata contained "
        f"{total_refs} referenced works in total, corresponding to a mean of {mean_refs:.1f} "
        f"referenced works per article. DOI resolution of these reference targets produced "
        f"{total_dois} DOI-level reference targets in total, with a mean of {mean_dois:.1f} "
        f"resolved reference DOIs per article."
    )

    print(
        f"After PDF preprocessing, the articles contained a mean of {mean_words:.1f} plain-text "
        f"words per article. Image extraction identified {total_images} embedded images in total "
        f"({mean_images:.1f} per article), of which {total_ocr_images} produced usable OCR text. "
        f"GROBID TEI parsing detected a mean of {mean_sections:.1f} core sections per article. "
        f"TEI-based supporting-evidence extraction identified {total_figures} figure records "
        f"({mean_figures:.1f} per article) and {total_tables} table records "
        f"({mean_tables:.1f} per article)."
    )

    # ------------------------------------------------------------
    # Article-level table
    # ------------------------------------------------------------

    print("\n" + "=" * 80)
    print("ARTICLE-LEVEL DATASET TABLE")
    print("=" * 80)

    header = [
        "#",
        "Topic",
        "Year",
        "Title",
        "Journal",
        "DOI",
        "Refs",
        "Ref_DOIs",
        "Pages",
        "Words",
        "Images",
        "OCR_OK",
        "Sections",
        "Figures",
        "Tables",
    ]

    print("\t".join(header))

    for r in enriched:
        print(
            "\t".join([
                str(r["index"]),
                r["topic"],
                str(r["year"]),
                r["title"],
                r["journal"],
                r["doi"],
                str(r["n_referenced_works_metadata"]),
                str(r["n_reference_dois"]),
                str(r["pdf_pages"]),
                str(r["plain_text_words"]),
                str(r["n_extracted_images"]),
                str(r["n_images_ocr_ok"]),
                str(r["n_core_sections_found"]),
                str(r["n_figures_tei"]),
                str(r["n_tables_tei"]),
            ])
        )

    # ------------------------------------------------------------
    # Article BibTeX entries
    # ------------------------------------------------------------

    print("\n" + "=" * 80)
    print("ARTICLE BIBTEX ENTRIES")
    print("=" * 80)

    for r in enriched:
        print(f"\n[{r['index']}] {r['title']}")
        print(f"DOI: {r['doi']}")
        if r["article_bibtex"]:
            print(r["article_bibtex"])
        else:
            print("[BibTeX unavailable]")

    # ------------------------------------------------------------
    # Per-article section word counts
    # ------------------------------------------------------------

    print("\n" + "=" * 80)
    print("ARTICLE-LEVEL SECTION WORD COUNTS")
    print("=" * 80)

    print("\t".join(["#", "Title"] + CORE_SECTIONS))

    for r in enriched:
        row = [str(r["index"]), r["title"]]
        for sec in CORE_SECTIONS:
            row.append(str(r["section_word_counts"].get(sec, 0)))
        print("\t".join(row))

    # ------------------------------------------------------------
    # Missing / warning diagnostics
    # ------------------------------------------------------------

    print("\n" + "=" * 80)
    print("DIAGNOSTICS")
    print("=" * 80)

    missing_projects = [r for r in enriched if not r["project_dir"]]
    missing_prep = [r for r in enriched if r["plain_text_words"] == 0]
    missing_sections = [r for r in enriched if r["n_core_sections_found"] == 0]
    missing_reference_dois = [r for r in enriched if r["n_reference_dois"] == 0]

    print(f"Missing project folders              : {len(missing_projects)}")
    print(f"Missing preprocessing/text stats     : {len(missing_prep)}")
    print(f"Articles with zero detected sections : {len(missing_sections)}")
    print(f"Articles with zero reference DOIs    : {len(missing_reference_dois)}")

    if missing_projects:
        print("\nMissing project folders:")
        for r in missing_projects:
            print(f"- {r['title']}")

    if missing_prep:
        print("\nMissing preprocessing stats:")
        for r in missing_prep:
            print(f"- {r['title']}")

    if missing_sections:
        print("\nZero-section articles:")
        for r in missing_sections:
            print(f"- {r['title']}")

    if missing_reference_dois:
        print("\nZero-reference-DOI articles:")
        for r in missing_reference_dois:
            print(f"- {r['title']}")


if __name__ == "__main__":
    main()
