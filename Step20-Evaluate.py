 
# -*- coding: utf-8 -*-

# ============================================================
# Step20-EvaluatePDFs-AllProjects.py
# ============================================================
#
# Purpose:
#   Automatically evaluate generated PDFs against benchmark/reference PDFs
#   for ONE project or ALL projects.
#
# This updates the PDF-vs-PDF evaluator so it can:
#   1. Discover all Projects/<project>/FinalArticle/Main.pdf files.
#   2. Automatically find a matching reference PDF for each project.
#   3. Compare generated PDF vs reference PDF.
#   4. Calculate scores for whole article and detected sections.
#   5. Save per-project reports.
#   6. Save a global summary across all projects.
#
# Expected generated PDF:
#   Projects/<project>/FinalArticle/Main.pdf
#
# Reference PDF lookup priority:
#   1. --actual-pdf, if evaluating one project only
#   2. Project-local files:
#        article.pdf
#        paper.pdf
#        benchmark.pdf
#        original.pdf
#        Actual.pdf
#        reference.pdf
#   3. Project-local recursive PDFs outside FinalArticle
#   4. Benchmark root, default:
#        benchmark_reviews/
#      using metadata.json / metadata.csv / folder-name fuzzy matching.
#
# Output:
#   Projects/<project>/FinalArticle/EvaluationPDF/
#       actual_extracted_text.txt
#       generated_extracted_text.txt
#       extracted_actual_sections.json
#       extracted_generated_sections.json
#       evaluation_metrics_by_section.csv
#       evaluation_metrics_table.csv
#       evaluation_summary.json
#       evaluation_printable_table.txt
#
#   EvaluationPDF_AllProjects/
#       all_projects_pdf_evaluation_summary.csv
#       all_projects_pdf_evaluation_summary.json
#       skipped_projects.csv
#
# Usage:
#
#   Evaluate all projects automatically:
#     python Step20-EvaluatePDFs-AllProjects.py --all-projects
#
#   Evaluate all projects, specify benchmark root:
#     python Step20-EvaluatePDFs-AllProjects.py --all-projects --benchmark-root benchmark_reviews
#
#   Evaluate one project:
#     python Step20-EvaluatePDFs-AllProjects.py 1
#
#   Evaluate one project with explicit reference:
#     python Step20-EvaluatePDFs-AllProjects.py 1 --actual-pdf /path/to/article.pdf
#
#   Evaluate explicit PDFs:
#     python Step20-EvaluatePDFs-AllProjects.py \
#       --actual-pdf /path/to/article.pdf \
#       --generated-pdf /path/to/Main.pdf
#
#   Skip model-backed semantic metrics for a quick debug run:
#     python Step20-EvaluatePDFs-AllProjects.py --all-projects --skip-model-metrics
#
# Notes:
#   All available metrics run by default.
#   Model-backed metrics require optional packages:
#     pip install sentence-transformers bert-score
#
# ============================================================

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ============================================================
# Defaults
# ============================================================

PROJECTS_DIR = Path("Projects")
FINAL_ARTICLE_DIR = "FinalArticle"
EVALUATION_DIR = "EvaluationPDF"
GLOBAL_EVAL_DIR = Path("EvaluationPDF_AllProjects")
BENCHMARK_ROOT = Path("benchmark_reviews")

DEFAULT_SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BERTSCORE_MODEL = "distilbert-base-uncased"

SECTION_ALIASES = {
    "abstract": ["abstract"],
    "introduction": ["introduction", "background"],
    "methods": ["methods", "methodology", "materials and methods"],
    "results": ["results"],
    "main_findings": [
        "main findings",
        "findings",
        "thematic synthesis",
        "main findings thematic synthesis",
        "synthesis",
    ],
    "discussion": ["discussion"],
    "conclusion": ["conclusion", "conclusions"],
}

SECTION_ORDER = [
    "abstract",
    "introduction",
    "methods",
    "results",
    "main_findings",
    "discussion",
    "conclusion",
]

GENERATED_SECTION_FILES = {
    "abstract": ["Abstract.tex"],
    "introduction": ["Introduction.tex"],
    "methods": ["Methodology.tex", "Methods.tex"],
    "results": ["Results.tex"],
    "main_findings": ["MainFindings.tex", "Findings.tex", "Synthesis.tex"],
    "discussion": ["Discussion.tex"],
    "conclusion": ["Conclusion.tex", "Conclusions.tex"],
}

ACTUAL_SECTION_FILES = {
    "abstract": ["abstract.txt"],
    "introduction": ["introduction.txt", "background.txt"],
    "methods": ["methods.txt", "methodology.txt", "materials_and_methods.txt"],
    "results": ["results.txt"],
    "main_findings": ["main_findings.txt", "findings.txt", "synthesis.txt"],
    "discussion": ["discussion.txt"],
    "conclusion": ["conclusion.txt", "conclusions.txt"],
}

METRIC_ORDER = [
    "Exact Match",
    "Levenshtein Similarity",
    "Jaccard Similarity",
    "TF-IDF Cosine Similarity",
    "ROUGE-1",
    "ROUGE-2",
    "ROUGE-L",
    "BLEU-1",
    "BLEU-4",
    "BLEU-CodeNN",
    "CodeBERT Similarity",
    "Hybrid Text-Code Score",
    "METEOR",
    "spaCy Similarity",
    "SBERT Similarity",
    "Sentence Embedding Precision",
    "Sentence Embedding Recall",
    "Sentence Embedding F1",
    "WMD Similarity",
    "Entailment Score",
    "BERTScore Precision",
    "BERTScore Recall",
    "BERTScore F1",
    "Citation DOI Jaccard",
    "Citation Title-Year Jaccard",
    "Keyphrase Coverage",
    "Light Composite Review Score",
    "Heavy Composite Review Score",
    "Composite Review Similarity Score",
]

LIGHT_COMPOSITE_WEIGHTS = [
    ("TF-IDF Cosine Similarity", 0.3125),
    ("ROUGE-L", 0.2500),
    ("ROUGE-1", 0.1875),
    ("Jaccard Similarity", 0.1875),
    ("BLEU-4", 0.0625),
]

HEAVY_COMPOSITE_WEIGHTS = [
    ("TF-IDF Cosine Similarity", 0.2083333333),
    ("ROUGE-L", 0.1666666667),
    ("ROUGE-1", 0.1250),
    ("Jaccard Similarity", 0.1250),
    ("SBERT Similarity", 0.1666666667),
    ("BERTScore F1", 0.1666666667),
    ("BLEU-4", 0.0416666667),
]

OPTIONAL_MODEL_METRICS = {
    "CodeBERT Similarity",
    "Hybrid Text-Code Score",
    "spaCy Similarity",
    "SBERT Similarity",
    "Sentence Embedding Precision",
    "Sentence Embedding Recall",
    "Sentence Embedding F1",
    "WMD Similarity",
    "Entailment Score",
    "BERTScore Precision",
    "BERTScore Recall",
    "BERTScore F1",
}


# ============================================================
# Basic helpers
# ============================================================

def clean_text(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n[ \t]+", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(str(x).strip()))
    except Exception:
        return default


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(str(x).strip())
    except Exception:
        return default


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(text), encoding="utf-8", errors="ignore")


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


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


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
    with path.open("w", encoding="utf-8", errors="ignore", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def slug_text(x: Any) -> str:
    s = str(x or "").lower()
    s = re.sub(r"https?://\S+", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokens_for_match(x: Any) -> List[str]:
    stop = {
        "a", "an", "the", "on", "in", "of", "for", "to", "and", "or", "with",
        "based", "using", "review", "survey", "systematic", "scoping",
        "literature", "large", "language", "model", "models",
    }
    return [t for t in slug_text(x).split() if len(t) > 2 and t not in stop]


def token_jaccard(a: Any, b: Any) -> float:
    sa = set(tokens_for_match(a))
    sb = set(tokens_for_match(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(len(sa | sb), 1)


def sequence_similarity(a: Any, b: Any) -> float:
    aa = slug_text(a)
    bb = slug_text(b)
    if not aa or not bb:
        return 0.0
    return SequenceMatcher(None, aa, bb).ratio()


def round_or_blank(x: Optional[float], digits: int = 4) -> str:
    if x is None:
        return "-"
    try:
        x = float(x)
        if math.isnan(x):
            return "-"
        return f"{x:.{digits}f}"
    except Exception:
        return "-"


# ============================================================
# Project discovery
# ============================================================

def discover_projects(projects_dir: Path) -> List[Path]:
    if not projects_dir.exists():
        return []
    return sorted([
        p for p in projects_dir.iterdir()
        if p.is_dir()
        and (p / "config.json").exists()
        and p.name.lower() not in {"rankings", "search", "evaluation", "__pycache__", "extra"}
    ])


def load_project_title(project_dir: Path) -> str:
    cfg = read_json(project_dir / "config.json", {})
    for key in ["title", "review_title", "objective", "aim"]:
        value = cfg.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list) and value:
            return str(value[0]).strip()
    return project_dir.name


def select_projects(args: argparse.Namespace) -> List[Path]:
    projects = discover_projects(Path(args.projects_dir))
    if args.all_projects or (not args.generated_pdf and not args.positional_project and not args.project_name and args.project_index <= 0):
        return projects

    project_index = args.project_index
    if project_index <= 0 and args.positional_project:
        project_index = safe_int(args.positional_project, 0)

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

    return []


# ============================================================
# PDF extraction
# ============================================================

def extract_pdf_text_pymupdf(pdf_path: Path) -> Tuple[str, Dict[str, Any]]:
    import fitz
    doc = fitz.open(str(pdf_path))
    pages = []
    page_infos = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text") or ""
        pages.append(text)
        page_infos.append({"page": i, "chars": len(text)})
    meta = {"extractor": "pymupdf", "pages": len(doc), "page_infos": page_infos}
    doc.close()
    return clean_text("\n\n".join(pages)), meta


def extract_pdf_text_pypdf(pdf_path: Path) -> Tuple[str, Dict[str, Any]]:
    from pypdf import PdfReader
    reader = PdfReader(str(pdf_path))
    pages = []
    page_infos = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(text)
        page_infos.append({"page": i, "chars": len(text)})
    return clean_text("\n\n".join(pages)), {"extractor": "pypdf", "pages": len(reader.pages), "page_infos": page_infos}


def extract_pdf_text_pdfminer(pdf_path: Path) -> Tuple[str, Dict[str, Any]]:
    from pdfminer.high_level import extract_text
    text = extract_text(str(pdf_path)) or ""
    return clean_text(text), {"extractor": "pdfminer.six", "pages": None, "page_infos": []}


def extract_pdf_text_pdftotext(pdf_path: Path) -> Tuple[str, Dict[str, Any]]:
    proc = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=180,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-1000:])
    return clean_text(proc.stdout), {"extractor": "pdftotext", "pages": None, "page_infos": []}


def extract_pdf_text(pdf_path: Path) -> Tuple[str, Dict[str, Any]]:
    errors = []
    for name, fn in [
        ("pymupdf", extract_pdf_text_pymupdf),
        ("pypdf", extract_pdf_text_pypdf),
        ("pdfminer", extract_pdf_text_pdfminer),
        ("pdftotext", extract_pdf_text_pdftotext),
    ]:
        try:
            text, meta = fn(pdf_path)
            if len(text.strip()) >= 50:
                meta["path"] = str(pdf_path)
                meta["chars"] = len(text)
                meta["words"] = len(tokenize_plain(text))
                meta["errors_before_success"] = errors
                return text, meta
            errors.append({"extractor": name, "error": "too little text extracted"})
        except Exception as e:
            errors.append({"extractor": name, "error": repr(e)})
    return "", {"path": str(pdf_path), "extractor": "failed", "chars": 0, "words": 0, "errors": errors}


# ============================================================
# Section extraction
# ============================================================

def remove_references(text: str) -> str:
    s = clean_text(text)
    patterns = [
        r"\n\s*references\s*\n",
        r"\n\s*bibliography\s*\n",
        r"\n\s*acknowledg(?:e)?ments?\s*\n",
        r"\n\s*funding\s*\n",
        r"\n\s*supplementary materials?\s*\n",
    ]
    positions = []
    for pat in patterns:
        m = re.search(pat, s, flags=re.I)
        if m:
            positions.append(m.start())
    if positions:
        return s[:min(positions)].strip()
    return s


def is_likely_heading(line: str) -> bool:
    l = line.strip()
    if not l or len(l) > 120:
        return False
    low = l.lower().strip(" .:")
    simple = {
        "abstract", "introduction", "background", "methods", "methodology",
        "materials and methods", "results", "main findings", "findings",
        "discussion", "conclusion", "conclusions", "references",
    }
    if low in simple:
        return True
    if re.match(r"^\d+(\.\d+)*\s+[A-Z][A-Za-z0-9 ,:/&()\-]{2,}$", l):
        return True
    if re.match(r"^(Abstract|Introduction|Methods|Methodology|Results|Main findings|Discussion|Conclusion|Conclusions)\b", l, flags=re.I):
        return True
    return False


def remove_pdf_noise(text: str) -> str:
    s = clean_text(text)
    s = re.sub(r"([A-Za-z])-\n([A-Za-z])", r"\1\2", s)
    lines = [line.strip() for line in s.splitlines()]
    out = []
    buffer = ""
    for line in lines:
        if not line:
            if buffer:
                out.append(buffer.strip())
                buffer = ""
            out.append("")
            continue
        if is_likely_heading(line):
            if buffer:
                out.append(buffer.strip())
                buffer = ""
            out.append(line)
            continue
        if not buffer:
            buffer = line
        else:
            if buffer.endswith((".", ":", ";", "?", "!", ")")):
                out.append(buffer.strip())
                buffer = line
            else:
                buffer += " " + line
    if buffer:
        out.append(buffer.strip())
    s = "\n".join(out)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def canonical_section_name(heading: str) -> Optional[str]:
    h = heading.lower().strip()
    h = re.sub(r"^\d+(\.\d+)*\s+", "", h)
    h = re.sub(r"[^a-z0-9 ]+", " ", h)
    h = re.sub(r"\s+", " ", h).strip()
    for canon, aliases in SECTION_ALIASES.items():
        for alias in aliases:
            a = alias.lower()
            if h == a or h.startswith(a + " "):
                return canon
    return None


def extract_sections_from_pdf_text(text: str) -> Dict[str, str]:
    s = remove_pdf_noise(text)
    lines = s.splitlines()
    headings = []
    for i, line in enumerate(lines):
        canon = canonical_section_name(line)
        if canon:
            headings.append((i, canon, line.strip()))

    sections = {"whole_article": remove_references(s)}
    if not headings:
        return sections

    filtered = []
    last_i = -999
    last_c = ""
    for i, canon, raw in headings:
        if canon == last_c and i - last_i <= 2:
            continue
        filtered.append((i, canon, raw))
        last_i, last_c = i, canon

    for idx, (start_i, canon, raw) in enumerate(filtered):
        end_i = len(lines)
        if idx + 1 < len(filtered):
            end_i = filtered[idx + 1][0]
        body = clean_text("\n".join(lines[start_i + 1:end_i]))
        if body:
            if canon in sections:
                sections[canon] += "\n\n" + body
            else:
                sections[canon] = body

    if not sections.get("main_findings"):
        parts = [sections.get("results", ""), sections.get("discussion", "")]
        combined = "\n\n".join([p for p in parts if p]).strip()
        if combined:
            sections["main_findings"] = combined

    return sections


def strip_latex_to_text(text: str) -> str:
    """Convert generated LaTeX section files into plain text for scoring."""
    s = str(text or "")
    s = re.sub(r"(?<!\\)%.*", " ", s)
    s = re.sub(r"\\begin\{[^}]+\}|\\end\{[^}]+\}", " ", s)
    s = re.sub(r"\\(?:section|subsection|subsubsection|paragraph)\*?\{([^{}]*)\}", r"\1\n", s)
    s = re.sub(r"\\(?:caption|label|ref|autoref|url|href)\*?(?:\[[^\]]*\])?\{[^{}]*\}", " ", s)
    s = re.sub(r"\\cite\w*\*?(?:\[[^\]]*\])?(?:\[[^\]]*\])?\{([^{}]*)\}", r" \1 ", s)
    s = re.sub(r"\\textbf\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\textit\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\emph\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?", " ", s)
    s = s.replace("~", " ")
    s = s.replace("\\&", "&").replace("\\%", "%").replace("\\_", "_")
    s = re.sub(r"[{}$]", " ", s)
    s = re.sub(r"\s*&\s*", " ", s)
    s = re.sub(r"\s*\\\\\s*", "\n", s)
    return clean_text(s)


def build_whole_article_from_sections(sections: Dict[str, str]) -> str:
    ordered = [sections.get(section, "") for section in SECTION_ORDER]
    whole = "\n\n".join([x for x in ordered if x and x.strip()]).strip()
    if whole:
        return whole
    return sections.get("whole_article", "")


def load_generated_sections_from_tex(final_article_dir: Path) -> Tuple[Dict[str, str], Dict[str, Any]]:
    sections: Dict[str, str] = {}
    used_files: Dict[str, str] = {}

    for section, names in GENERATED_SECTION_FILES.items():
        for name in names:
            path = final_article_dir / name
            if path.exists() and path.is_file():
                text = strip_latex_to_text(read_text(path))
                if text:
                    sections[section] = text
                    used_files[section] = str(path)
                break

    if sections:
        sections["whole_article"] = build_whole_article_from_sections(sections)

    return sections, {
        "source": "generated_tex_section_files" if sections else "not_found",
        "files": used_files,
        "sections": sorted([k for k in sections if k != "whole_article"]),
    }


def load_actual_sections_from_section_files(project_dir: Path) -> Tuple[Dict[str, str], Dict[str, Any]]:
    section_dir = project_dir / "section_files"
    sections: Dict[str, str] = {}
    used_files: Dict[str, str] = {}

    if not section_dir.exists():
        return {}, {"source": "not_found", "section_dir": str(section_dir)}

    combined_path = section_dir / "all_sections_combined.txt"
    if combined_path.exists():
        combined_text = read_text(combined_path)
        parsed = extract_sections_from_pdf_text(combined_text)
        for key, value in parsed.items():
            if value and key not in sections:
                sections[key] = value
        used_files["combined"] = str(combined_path)

    for section, names in ACTUAL_SECTION_FILES.items():
        for name in names:
            path = section_dir / name
            if path.exists() and path.is_file():
                text = remove_references(read_text(path))
                if text:
                    sections[section] = clean_text(text)
                    used_files[section] = str(path)
                break

    if sections:
        sections["whole_article"] = build_whole_article_from_sections(sections)

    return sections, {
        "source": "actual_extracted_section_files" if sections else "not_found",
        "section_dir": str(section_dir),
        "files": used_files,
        "sections": sorted([k for k in sections if k != "whole_article"]),
    }


def infer_project_dir_from_generated_pdf(generated_pdf: Path, final_article_dir: str) -> Optional[Path]:
    try:
        if generated_pdf.parent.name.lower() == final_article_dir.lower():
            return generated_pdf.parent.parent
    except Exception:
        pass
    return None


# ============================================================
# Metric text normalization
# ============================================================

def normalize_for_metrics(text: str) -> str:
    s = clean_text(text)
    s = re.sub(r"https?://\S+", " ", s)
    s = s.lower()
    s = re.sub(r"\[[0-9,\s\-]+\]", " ", s)
    s = re.sub(r"\([a-z][a-z\-]+ et al\.?,? \d{4}[a-z]?\)", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", normalize_for_metrics(text))


def tokenize_plain(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", clean_text(text))


def is_empty_or_bad(text: str) -> bool:
    return len(tokenize(text)) < 25


def split_sentences(text: str, max_sentences: int = 160) -> List[str]:
    s = clean_text(text)
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])|\n+", s)
    sentences = []
    for item in raw:
        item = clean_text(item)
        if 8 <= len(tokenize(item)) <= 80:
            sentences.append(item)
        if len(sentences) >= max_sentences:
            break
    return sentences


def normalize_doi(value: Any) -> str:
    s = str(value or "").strip().lower()
    s = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi:\s*", "", s)
    s = s.strip(" .;,)]}>\"'")
    return s


def extract_dois_from_text(text: str) -> List[str]:
    pattern = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", flags=re.I)
    dois = []
    for match in pattern.findall(str(text or "")):
        doi = normalize_doi(match)
        doi = re.sub(r"[),.;:\]\}]+$", "", doi)
        if doi:
            dois.append(doi)
    return sorted(set(dois))


def extract_reference_section_text(text: str) -> str:
    s = clean_text(text)
    patterns = [
        r"\n\s*references\s*\n",
        r"\n\s*bibliography\s*\n",
        r"\n\s*works cited\s*\n",
    ]
    positions = []
    for pat in patterns:
        m = re.search(pat, s, flags=re.I)
        if m:
            positions.append(m.end())
    if not positions:
        return ""
    return s[min(positions):].strip()


def normalize_title_year(title: Any, year: Any) -> str:
    t = slug_text(title)
    y = re.search(r"\b(19|20)\d{2}\b", str(year or ""))
    if not t or not y:
        return ""
    words = [w for w in t.split() if len(w) > 2][:12]
    if len(words) < 3:
        return ""
    return " ".join(words) + "|" + y.group(0)


def extract_bibtex_entries(text: str) -> Tuple[List[str], List[str]]:
    dois: List[str] = []
    title_year: List[str] = []
    for entry in re.split(r"\n\s*@", str(text or "")):
        doi_match = re.search(r"\bdoi\s*=\s*[{\"']?([^,}\"]+)", entry, flags=re.I)
        title_match = re.search(r"\btitle\s*=\s*[{\"']([^}\"]+)", entry, flags=re.I | re.S)
        year_match = re.search(r"\byear\s*=\s*[{\"']?(\d{4})", entry, flags=re.I)
        if doi_match:
            doi = normalize_doi(doi_match.group(1))
            if doi:
                dois.append(doi)
        if title_match and year_match:
            key = normalize_title_year(title_match.group(1), year_match.group(1))
            if key:
                title_year.append(key)
    return sorted(set(dois)), sorted(set(title_year))


def extract_title_year_from_reference_text(text: str) -> List[str]:
    ref_text = extract_reference_section_text(text) or text
    title_year: List[str] = []
    lines = [clean_text(x) for x in ref_text.splitlines()]
    buffer = ""
    for line in lines:
        if not line:
            continue
        if re.match(r"^\s*(?:\[\d+\]|\d+[.)])\s+", line) and buffer:
            key = title_year_key_from_reference_line(buffer)
            if key:
                title_year.append(key)
            buffer = line
        else:
            buffer = (buffer + " " + line).strip()
    if buffer:
        key = title_year_key_from_reference_line(buffer)
        if key:
            title_year.append(key)
    return sorted(set(title_year))


def title_year_key_from_reference_line(line: str) -> str:
    y = re.search(r"\b(19|20)\d{2}\b", line)
    if not y:
        return ""
    before_year = line[: y.start()]
    after_year = line[y.end():]
    candidate = after_year
    if len(tokenize_plain(candidate)) < 4:
        candidate = before_year
    candidate = re.sub(r"^[\s.,;:()\[\]\d]+", " ", candidate)
    candidate = re.split(r"\b(?:journal|proceedings|conference|arxiv|doi|http)\b", candidate, flags=re.I)[0]
    return normalize_title_year(candidate, y.group(0))


def jaccard_sets(a: Iterable[str], b: Iterable[str]) -> Optional[float]:
    sa = set(x for x in a if x)
    sb = set(x for x in b if x)
    if not sa and not sb:
        return None
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(len(sa | sb), 1)


def load_generated_bibliography(final_article_dir: Path) -> Tuple[List[str], List[str]]:
    dois: List[str] = []
    title_year: List[str] = []
    for name in ["reference.bib", "references.bib", "References.bib"]:
        path = final_article_dir / name
        if path.exists():
            d, ty = extract_bibtex_entries(read_text(path))
            dois.extend(d)
            title_year.extend(ty)
    return sorted(set(dois)), sorted(set(title_year))


def build_citation_overlap_scores(
    actual_text: str,
    generated_text: str,
    final_article_dir: Path,
) -> Dict[str, Any]:
    generated_bib_dois, generated_bib_title_year = load_generated_bibliography(final_article_dir)
    actual_ref_text = extract_reference_section_text(actual_text)
    generated_ref_text = extract_reference_section_text(generated_text)

    actual_dois = sorted(set(extract_dois_from_text(actual_ref_text or actual_text)))
    generated_dois = sorted(set(generated_bib_dois + extract_dois_from_text(generated_ref_text or generated_text)))
    actual_title_year = sorted(set(extract_title_year_from_reference_text(actual_ref_text or actual_text)))
    generated_title_year = sorted(set(generated_bib_title_year + extract_title_year_from_reference_text(generated_ref_text or generated_text)))

    doi_jaccard = jaccard_sets(actual_dois, generated_dois)
    title_year_jaccard = jaccard_sets(actual_title_year, generated_title_year)
    return {
        "Citation DOI Jaccard": doi_jaccard,
        "Citation Title-Year Jaccard": title_year_jaccard,
        "actual_dois": actual_dois,
        "generated_dois": generated_dois,
        "actual_title_year": actual_title_year,
        "generated_title_year": generated_title_year,
        "n_actual_dois": len(actual_dois),
        "n_generated_dois": len(generated_dois),
        "n_actual_title_year": len(actual_title_year),
        "n_generated_title_year": len(generated_title_year),
        "doi_overlap": sorted(set(actual_dois) & set(generated_dois)),
        "title_year_overlap": sorted(set(actual_title_year) & set(generated_title_year)),
    }


def extract_candidate_keyphrases(text: str, top_k: int = 30) -> List[str]:
    stop = {
        "the", "and", "for", "with", "that", "this", "from", "were", "was", "are", "has", "have",
        "study", "studies", "paper", "review", "article", "results", "method", "methods", "based",
        "using", "used", "also", "between", "within", "across", "their", "these", "those", "into",
    }
    s = normalize_for_metrics(text)
    words = [w for w in s.split() if len(w) > 2 and w not in stop]
    phrases = []
    for n in [3, 2]:
        for gram in ngrams(words, n):
            phrase = " ".join(gram)
            if not any(x in stop for x in gram):
                phrases.append(phrase)
    counts = Counter(phrases)
    if not counts:
        counts = Counter(words)
    return [phrase for phrase, _ in counts.most_common(top_k)]


def keyphrase_coverage(reference: str, candidate: str, top_k: int = 30) -> Optional[float]:
    phrases = extract_candidate_keyphrases(reference, top_k=top_k)
    if not phrases:
        return None
    candidate_norm = " " + normalize_for_metrics(candidate) + " "
    covered = 0
    for phrase in phrases:
        if f" {phrase} " in candidate_norm:
            covered += 1
    return covered / max(len(phrases), 1)


# ============================================================
# Metrics
# ============================================================

def exact_match(a: str, b: str) -> float:
    aa = normalize_for_metrics(a)
    bb = normalize_for_metrics(b)
    if not aa and not bb:
        return 1.0
    if not aa or not bb:
        return 0.0
    return 1.0 if aa == bb else 0.0


def levenshtein_similarity(a: str, b: str) -> float:
    aa = normalize_for_metrics(a)
    bb = normalize_for_metrics(b)
    if not aa and not bb:
        return 1.0
    if not aa or not bb:
        return 0.0
    return SequenceMatcher(None, aa, bb).ratio()


def jaccard_similarity(a: str, b: str) -> float:
    sa = set(tokenize(a))
    sb = set(tokenize(b))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(len(sa | sb), 1)


def term_cosine_similarity(a: str, b: str) -> float:
    ta = Counter(tokenize(a))
    tb = Counter(tokenize(b))
    if not ta or not tb:
        return 0.0
    vocab = set(ta) | set(tb)
    dot = sum(ta[t] * tb[t] for t in vocab)
    na = math.sqrt(sum(v * v for v in ta.values()))
    nb = math.sqrt(sum(v * v for v in tb.values()))
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (na * nb)


def tfidf_cosine_similarity(a: str, b: str) -> float:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        docs = [normalize_for_metrics(a), normalize_for_metrics(b)]
        if not docs[0] or not docs[1]:
            return 0.0
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=40000)
        mat = vec.fit_transform(docs)
        return float(cosine_similarity(mat[0], mat[1])[0][0])
    except Exception:
        return term_cosine_similarity(a, b)


def ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def rouge_n_f1(reference: str, candidate: str, n: int) -> float:
    ref = Counter(ngrams(tokenize(reference), n))
    cand = Counter(ngrams(tokenize(candidate), n))
    if not ref or not cand:
        return 0.0
    overlap = sum((ref & cand).values())
    precision = overlap / max(sum(cand.values()), 1)
    recall = overlap / max(sum(ref.values()), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def lcs_len(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            if x == y:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l_f1(reference: str, candidate: str) -> float:
    ref = tokenize(reference)
    cand = tokenize(candidate)
    if not ref or not cand:
        return 0.0
    lcs = lcs_len(ref, cand)
    precision = lcs / max(len(cand), 1)
    recall = lcs / max(len(ref), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def bleu_score(reference: str, candidate: str, max_n: int = 4) -> float:
    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)
    if not ref_tokens or not cand_tokens:
        return 0.0
    weights = [1.0 / max_n] * max_n
    precisions = []
    for n in range(1, max_n + 1):
        ref_ng = Counter(ngrams(ref_tokens, n))
        cand_ng = Counter(ngrams(cand_tokens, n))
        if not cand_ng:
            precisions.append(1e-9)
            continue
        overlap = sum((ref_ng & cand_ng).values())
        precisions.append((overlap + 1.0) / (sum(cand_ng.values()) + 1.0))
    log_precision = sum(w * math.log(max(p, 1e-12)) for w, p in zip(weights, precisions))
    bp = 1.0 if len(cand_tokens) > len(ref_tokens) else math.exp(1 - len(ref_tokens) / max(len(cand_tokens), 1))
    return bp * math.exp(log_precision)


def bleu_codenn_score(reference: str, candidate: str) -> float:
    """A CodeNN-style BLEU variant over plain tokens; useful as a stable extra lexical metric."""
    ref_tokens = [t.lower() for t in tokenize_plain(reference)]
    cand_tokens = [t.lower() for t in tokenize_plain(candidate)]
    if not ref_tokens or not cand_tokens:
        return 0.0
    scores = []
    for n in range(1, 5):
        ref_ng = Counter(ngrams(ref_tokens, n))
        cand_ng = Counter(ngrams(cand_tokens, n))
        if not cand_ng:
            scores.append(1e-9)
            continue
        overlap = sum((ref_ng & cand_ng).values())
        scores.append((overlap + 0.1) / (sum(cand_ng.values()) + 0.1))
    log_precision = sum(math.log(max(x, 1e-12)) for x in scores) / 4.0
    bp = 1.0 if len(cand_tokens) > len(ref_tokens) else math.exp(1 - len(ref_tokens) / max(len(cand_tokens), 1))
    return bp * math.exp(log_precision)


def meteor_score(reference: str, candidate: str) -> Optional[float]:
    try:
        from nltk.translate.meteor_score import meteor_score as nltk_meteor_score
        return float(nltk_meteor_score([tokenize(reference)], tokenize(candidate)))
    except Exception:
        return None


class HeavyMetricModels:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.sbert_model = None
        self.codebert_tokenizer = None
        self.codebert_model = None
        self.spacy_nlp = None
        self.entailment_pipeline = None
        self.unavailable_reasons: Dict[str, str] = {}

    def sbert_similarity(self, reference: str, candidate: str) -> Optional[float]:
        try:
            if self.sbert_model is None:
                from sentence_transformers import SentenceTransformer
                self.sbert_model = SentenceTransformer(self.args.sbert_model)
            emb = self.sbert_model.encode([candidate, reference], normalize_embeddings=True)
            return float((emb[0] * emb[1]).sum())
        except Exception as e:
            self.unavailable_reasons["SBERT Similarity"] = repr(e)
            return None

    def sentence_embedding_coverage(self, reference: str, candidate: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        try:
            if self.sbert_model is None:
                from sentence_transformers import SentenceTransformer
                self.sbert_model = SentenceTransformer(self.args.sbert_model)
            ref_sentences = split_sentences(reference, max_sentences=self.args.sentence_metric_max_sentences)
            cand_sentences = split_sentences(candidate, max_sentences=self.args.sentence_metric_max_sentences)
            if not ref_sentences or not cand_sentences:
                raise RuntimeError("Not enough sentences for sentence-level embedding coverage.")
            ref_emb = self.sbert_model.encode(ref_sentences, normalize_embeddings=True, batch_size=self.args.sentence_metric_batch_size)
            cand_emb = self.sbert_model.encode(cand_sentences, normalize_embeddings=True, batch_size=self.args.sentence_metric_batch_size)
            sims = ref_emb @ cand_emb.T
            recall = float(sims.max(axis=1).mean())
            precision = float(sims.max(axis=0).mean())
            if precision + recall <= 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            return precision, recall, f1
        except Exception as e:
            msg = repr(e)
            self.unavailable_reasons["Sentence Embedding Precision"] = msg
            self.unavailable_reasons["Sentence Embedding Recall"] = msg
            self.unavailable_reasons["Sentence Embedding F1"] = msg
            return None, None, None

    def codebert_similarity(self, reference: str, candidate: str) -> Optional[float]:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
            if self.codebert_tokenizer is None or self.codebert_model is None:
                self.codebert_tokenizer = AutoTokenizer.from_pretrained(self.args.codebert_model)
                self.codebert_model = AutoModel.from_pretrained(self.args.codebert_model)
                self.codebert_model.eval()

            texts = [candidate[: self.args.model_metric_chars], reference[: self.args.model_metric_chars]]
            encoded = self.codebert_tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.args.model_metric_tokens,
                return_tensors="pt",
            )
            with torch.no_grad():
                out = self.codebert_model(**encoded)
                emb = out.last_hidden_state[:, 0, :]
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                return float((emb[0] * emb[1]).sum().item())
        except Exception as e:
            self.unavailable_reasons["CodeBERT Similarity"] = repr(e)
            return None

    def spacy_similarity(self, reference: str, candidate: str) -> Optional[float]:
        try:
            import spacy
            if self.spacy_nlp is None:
                errors = []
                for model_name in [self.args.spacy_model, "en_core_web_lg", "en_core_web_md", "en_core_web_sm"]:
                    if not model_name:
                        continue
                    try:
                        self.spacy_nlp = spacy.load(model_name)
                        break
                    except Exception as e:
                        errors.append(f"{model_name}: {e}")
                if self.spacy_nlp is None:
                    raise RuntimeError("; ".join(errors))
            a = self.spacy_nlp(candidate[: self.args.model_metric_chars])
            b = self.spacy_nlp(reference[: self.args.model_metric_chars])
            return float(a.similarity(b))
        except Exception as e:
            self.unavailable_reasons["spaCy Similarity"] = repr(e)
            return None

    def wmd_similarity(self, reference: str, candidate: str) -> Optional[float]:
        try:
            if self.spacy_nlp is None:
                _ = self.spacy_similarity(reference, candidate)
            if self.spacy_nlp is None:
                raise RuntimeError("spaCy vector model is not available.")
            a_doc = self.spacy_nlp(candidate[: self.args.model_metric_chars])
            b_doc = self.spacy_nlp(reference[: self.args.model_metric_chars])
            a_vecs = [t.vector for t in a_doc if t.is_alpha and not t.is_stop and t.has_vector][: self.args.wmd_max_tokens]
            b_vecs = [t.vector for t in b_doc if t.is_alpha and not t.is_stop and t.has_vector][: self.args.wmd_max_tokens]
            if not a_vecs or not b_vecs:
                raise RuntimeError("No usable word vectors found; install a spaCy md/lg model.")
            import numpy as np
            a = np.vstack(a_vecs)
            b = np.vstack(b_vecs)
            a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
            b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
            sims = a @ b.T
            relaxed_distance = 1.0 - ((float(np.max(sims, axis=1).mean()) + float(np.max(sims, axis=0).mean())) / 2.0)
            return max(0.0, min(1.0, 1.0 - relaxed_distance))
        except Exception as e:
            self.unavailable_reasons["WMD Similarity"] = repr(e)
            return None

    def entailment_score(self, reference: str, candidate: str) -> Optional[float]:
        try:
            from transformers import pipeline
            if self.entailment_pipeline is None:
                self.entailment_pipeline = pipeline(
                    "text-classification",
                    model=self.args.entailment_model,
                    tokenizer=self.args.entailment_model,
                    truncation=True,
                    top_k=None,
                )
            premise = candidate[: self.args.model_metric_chars]
            hypothesis = reference[: self.args.model_metric_chars]
            result = self.entailment_pipeline({"text": premise, "text_pair": hypothesis})
            labels = result[0] if result and isinstance(result[0], list) else result
            for item in labels:
                if "entail" in str(item.get("label", "")).lower():
                    return float(item.get("score", 0.0))
            raise RuntimeError("Entailment label not found in model output.")
        except Exception as e:
            self.unavailable_reasons["Entailment Score"] = repr(e)
            return None


def bertscore(reference: str, candidate: str, args: argparse.Namespace) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    try:
        from bert_score import score
        p, r, f1 = score(
            [candidate],
            [reference],
            model_type=args.bertscore_model,
            lang="en",
            verbose=False,
            rescale_with_baseline=args.bertscore_rescale_with_baseline,
        )
        return float(p[0]), float(r[0]), float(f1[0])
    except Exception:
        return None, None, None


def valid_score_value(value: Any) -> bool:
    return isinstance(value, (int, float)) and not math.isnan(float(value))


def fixed_composite_score(
    scores: Dict[str, Optional[float]],
    weights: List[Tuple[str, float]],
    name: str,
    require_all: bool = True,
) -> Tuple[Optional[float], Dict[str, Any]]:
    used = []
    missing = []
    total = 0.0
    for metric, weight in weights:
        value = scores.get(metric)
        if valid_score_value(value):
            total += float(value) * weight
            used.append({"metric": metric, "weight": weight, "score": float(value)})
        else:
            missing.append(metric)

    weight_sum = sum(weight for _, weight in weights)
    provenance = {
        "name": name,
        "formula": "fixed_weight_sum_no_renormalization",
        "required_metrics": [metric for metric, _ in weights],
        "used_metrics": used,
        "missing_metrics": missing,
        "weight_sum": weight_sum,
        "require_all": require_all,
    }

    if require_all and missing:
        provenance["score_available"] = False
        return None, provenance

    if abs(weight_sum - 1.0) > 1e-6:
        provenance["score_available"] = False
        provenance["error"] = f"Composite weights must sum to 1.0, got {weight_sum:.6f}"
        return None, provenance

    provenance["score_available"] = True
    return total, provenance


def compute_pair_metrics(
    reference: str,
    candidate: str,
    args: argparse.Namespace,
    heavy: Optional[HeavyMetricModels] = None,
    citation_scores: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    scores: Dict[str, Optional[float]] = {
        "Exact Match": exact_match(reference, candidate),
        "Levenshtein Similarity": levenshtein_similarity(reference, candidate),
        "Jaccard Similarity": jaccard_similarity(reference, candidate),
        "TF-IDF Cosine Similarity": tfidf_cosine_similarity(reference, candidate),
        "ROUGE-1": rouge_n_f1(reference, candidate, 1),
        "ROUGE-2": rouge_n_f1(reference, candidate, 2),
        "ROUGE-L": rouge_l_f1(reference, candidate),
        "BLEU-1": bleu_score(reference, candidate, 1),
        "BLEU-4": bleu_score(reference, candidate, 4),
        "BLEU-CodeNN": bleu_codenn_score(reference, candidate),
        "CodeBERT Similarity": None,
        "Hybrid Text-Code Score": None,
        "METEOR": meteor_score(reference, candidate),
        "spaCy Similarity": None,
        "SBERT Similarity": None,
        "Sentence Embedding Precision": None,
        "Sentence Embedding Recall": None,
        "Sentence Embedding F1": None,
        "WMD Similarity": None,
        "Entailment Score": None,
        "BERTScore Precision": None,
        "BERTScore Recall": None,
        "BERTScore F1": None,
        "Citation DOI Jaccard": citation_scores.get("Citation DOI Jaccard") if citation_scores else None,
        "Citation Title-Year Jaccard": citation_scores.get("Citation Title-Year Jaccard") if citation_scores else None,
        "Keyphrase Coverage": keyphrase_coverage(reference, candidate, top_k=args.keyphrase_top_k),
        "Light Composite Review Score": None,
        "Heavy Composite Review Score": None,
        "Composite Review Similarity Score": None,
    }
    if heavy is not None:
        scores["SBERT Similarity"] = heavy.sbert_similarity(reference, candidate)
        sentence_p, sentence_r, sentence_f1 = heavy.sentence_embedding_coverage(reference, candidate)
        scores["Sentence Embedding Precision"] = sentence_p
        scores["Sentence Embedding Recall"] = sentence_r
        scores["Sentence Embedding F1"] = sentence_f1
        scores["CodeBERT Similarity"] = heavy.codebert_similarity(reference, candidate)
        scores["spaCy Similarity"] = heavy.spacy_similarity(reference, candidate)
        scores["WMD Similarity"] = heavy.wmd_similarity(reference, candidate)
        scores["Entailment Score"] = heavy.entailment_score(reference, candidate)
        p, r, f1 = bertscore(reference, candidate, args)
        scores["BERTScore Precision"] = p
        scores["BERTScore Recall"] = r
        scores["BERTScore F1"] = f1
        if p is None and r is None and f1 is None:
            heavy.unavailable_reasons["BERTScore Precision"] = "bert-score failed or is not installed."
            heavy.unavailable_reasons["BERTScore Recall"] = "bert-score failed or is not installed."
            heavy.unavailable_reasons["BERTScore F1"] = "bert-score failed or is not installed."

    light_score, light_provenance = fixed_composite_score(
        scores,
        LIGHT_COMPOSITE_WEIGHTS,
        "Light Composite Review Score",
        require_all=True,
    )
    heavy_score, heavy_provenance = fixed_composite_score(
        scores,
        HEAVY_COMPOSITE_WEIGHTS,
        "Heavy Composite Review Score",
        require_all=True,
    )
    scores["Light Composite Review Score"] = light_score
    scores["Heavy Composite Review Score"] = heavy_score
    # Backward-compatible column name. This is now fixed light-only and therefore
    # comparable across runs, regardless of model-backed metric availability.
    scores["Composite Review Similarity Score"] = light_score
    scores["_composite_provenance"] = {
        "primary_composite": "Light Composite Review Score",
        "compatibility_alias": "Composite Review Similarity Score",
        "light": light_provenance,
        "heavy": heavy_provenance,
    }

    codebert = scores.get("CodeBERT Similarity")
    if valid_score_value(codebert) and valid_score_value(light_score):
        scores["Hybrid Text-Code Score"] = 0.70 * float(light_score) + 0.30 * float(codebert)
    else:
        scores["Hybrid Text-Code Score"] = None
        if heavy is not None:
            heavy.unavailable_reasons["Hybrid Text-Code Score"] = "Requires both Light Composite Review Score and CodeBERT Similarity."
    return scores


# ============================================================
# Reference PDF discovery
# ============================================================

def generated_pdf_for_project(project_dir: Path, final_article_dir: str) -> Path:
    return project_dir / final_article_dir / "Main.pdf"


def project_local_reference_candidates(project_dir: Path, final_article_dir: str) -> List[Path]:
    direct_names = [
        "article.pdf",
        "paper.pdf",
        "benchmark.pdf",
        "original.pdf",
        "Actual.pdf",
        "reference.pdf",
        "article(2).pdf",
    ]
    candidates = []
    for name in direct_names:
        p = project_dir / name
        if p.exists() and p.is_file():
            candidates.append(p)

    # Recursive local PDFs, excluding generated article folders and database dumps.
    for p in project_dir.rglob("*.pdf"):
        parts_lower = [x.lower() for x in p.parts]
        if final_article_dir.lower() in parts_lower:
            continue
        if "finalarticle" in parts_lower:
            continue
        if p not in candidates:
            candidates.append(p)
    return candidates


def scan_benchmark_pdfs(benchmark_root: Path) -> List[Dict[str, Any]]:
    if not benchmark_root.exists():
        return []

    rows_by_pdf: Dict[str, Dict[str, Any]] = {}

    metadata_csv = benchmark_root / "metadata.csv"
    for row in read_csv_rows(metadata_csv):
        pdf_path = row.get("pdf_path", "")
        if pdf_path:
            p = Path(pdf_path)
            if not p.is_absolute():
                p = benchmark_root / p if not str(p).startswith(str(benchmark_root)) else p
            if p.exists():
                rows_by_pdf[str(p.resolve())] = {
                    "pdf": p.resolve(),
                    "title": row.get("title", ""),
                    "topic": row.get("topic", ""),
                    "doi": row.get("doi", ""),
                    "source": "metadata.csv",
                }

    for pdf in benchmark_root.rglob("*.pdf"):
        key = str(pdf.resolve())
        meta_path = pdf.parent / "metadata.json"
        meta = read_json(meta_path, {})
        title = meta.get("title", "") if isinstance(meta, dict) else ""
        topic = meta.get("topic", "") if isinstance(meta, dict) else ""
        doi = meta.get("doi", "") if isinstance(meta, dict) else ""
        if key not in rows_by_pdf:
            rows_by_pdf[key] = {
                "pdf": pdf.resolve(),
                "title": title,
                "topic": topic,
                "doi": doi,
                "source": "metadata.json" if meta else "pdf_scan",
            }
        else:
            if title:
                rows_by_pdf[key]["title"] = title
            if topic:
                rows_by_pdf[key]["topic"] = topic
            if doi:
                rows_by_pdf[key]["doi"] = doi

    return list(rows_by_pdf.values())


def reference_match_score(project_dir: Path, project_title: str, candidate: Dict[str, Any]) -> float:
    pdf = Path(candidate["pdf"])
    cand_title = candidate.get("title", "") or pdf.parent.name or pdf.stem
    cand_blob = " ".join([cand_title, candidate.get("topic", ""), pdf.parent.name, pdf.stem])
    project_blob = " ".join([project_title, project_dir.name])

    score = 0.0
    score += 0.55 * token_jaccard(project_blob, cand_blob)
    score += 0.25 * sequence_similarity(project_dir.name, pdf.parent.name)
    score += 0.20 * sequence_similarity(project_title, cand_title)

    # Boost if one normalized string contains the other.
    a = slug_text(project_dir.name).replace(" ", "_")
    b = slug_text(pdf.parent.name).replace(" ", "_")
    if a and (a in b or b in a):
        score += 0.25

    return min(score, 1.0)


def find_reference_pdf_for_project(project_dir: Path, args: argparse.Namespace, benchmark_index: List[Dict[str, Any]]) -> Tuple[Optional[Path], Dict[str, Any]]:
    if args.actual_pdf and len(select_projects(args)) <= 1:
        p = Path(args.actual_pdf).expanduser()
        if p.exists():
            return p.resolve(), {"method": "explicit_actual_pdf", "score": 1.0}

    local = project_local_reference_candidates(project_dir, args.final_article_dir)
    if local:
        return local[0].resolve(), {"method": "project_local", "score": 1.0, "candidates": [str(p) for p in local[:10]]}

    project_title = load_project_title(project_dir)
    best = None
    best_score = -1.0
    scored = []
    for cand in benchmark_index:
        s = reference_match_score(project_dir, project_title, cand)
        scored.append({
            "pdf": str(cand["pdf"]),
            "title": cand.get("title", ""),
            "topic": cand.get("topic", ""),
            "score": s,
            "source": cand.get("source", ""),
        })
        if s > best_score:
            best_score = s
            best = cand

    scored.sort(key=lambda x: x["score"], reverse=True)
    threshold = args.match_threshold
    if best is not None and best_score >= threshold:
        return Path(best["pdf"]).resolve(), {
            "method": "benchmark_root_fuzzy",
            "score": best_score,
            "project_title": project_title,
            "top_candidates": scored[:8],
        }

    return None, {
        "method": "not_found",
        "score": best_score,
        "project_title": project_title,
        "top_candidates": scored[:8],
    }


# ============================================================
# Evaluation core
# ============================================================

def make_section_pairs(actual_sections: Dict[str, str], generated_sections: Dict[str, str]) -> Dict[str, Tuple[str, str]]:
    pairs: Dict[str, Tuple[str, str]] = {}
    pairs["whole_article"] = (
        actual_sections.get("whole_article", ""),
        generated_sections.get("whole_article", ""),
    )

    for section in SECTION_ORDER:
        actual = actual_sections.get(section, "")
        generated = generated_sections.get(section, "")

        if section == "main_findings" and not actual:
            actual = "\n\n".join([
                actual_sections.get("results", ""),
                actual_sections.get("discussion", ""),
            ]).strip()

        if section == "methods" and not generated:
            generated = generated_sections.get("methodology", "")

        pairs[section] = (actual, generated)

    return pairs


def format_printable_table(rows: List[Dict[str, Any]], sections: List[str]) -> str:
    keep_metrics = METRIC_ORDER
    filtered = [r for r in rows if r.get("Metric") in keep_metrics]
    cols = ["Metric"] + sections
    widths = {}
    for c in cols:
        widths[c] = max(len(c), max(len(str(r.get(c, ""))) for r in filtered) if filtered else len(c))
        widths[c] = min(widths[c], 28 if c == "Metric" else 13)

    def trunc(s: Any, w: int) -> str:
        s = str(s)
        if len(s) <= w:
            return s
        return s[:w - 3] + "..."

    line = " | ".join(trunc(c, widths[c]).ljust(widths[c]) for c in cols)
    sep = "-+-".join("-" * widths[c] for c in cols)
    out = [line, sep]
    for r in filtered:
        out.append(" | ".join(trunc(r.get(c, ""), widths[c]).ljust(widths[c]) for c in cols))
    return "\n".join(out)


def format_overall_metrics_table(metric_table_rows: List[Dict[str, Any]]) -> str:
    rows = [{"Metric": row.get("Metric", ""), "whole_article": row.get("whole_article", "")} for row in metric_table_rows]
    widths = {
        "Metric": max(len("Metric"), max(len(str(r["Metric"])) for r in rows) if rows else 0),
        "whole_article": max(len("Overall score"), max(len(str(r["whole_article"])) for r in rows) if rows else 0),
    }
    widths["Metric"] = min(widths["Metric"], 34)

    def trunc(value: Any, width: int) -> str:
        s = str(value)
        return s if len(s) <= width else s[:width - 3] + "..."

    lines = [
        f"{'Metric'.ljust(widths['Metric'])} | {'Overall score'.ljust(widths['whole_article'])}",
        f"{'-' * widths['Metric']}-+-{'-' * widths['whole_article']}",
    ]
    for row in rows:
        lines.append(
            f"{trunc(row['Metric'], widths['Metric']).ljust(widths['Metric'])} | "
            f"{trunc(row['whole_article'], widths['whole_article']).ljust(widths['whole_article'])}"
        )
    return "\n".join(lines)


def build_section_summary_rows(
    project_name: str,
    pairs: Dict[str, Tuple[str, str]],
    score_by_section: Dict[str, Dict[str, Any]],
    digits: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for section in ["whole_article"] + SECTION_ORDER:
        actual, generated = pairs.get(section, ("", ""))
        scores = score_by_section.get(section, {})
        rows.append({
            "project": project_name,
            "section": section,
            "actual_present": "yes" if not is_empty_or_bad(actual) else "no",
            "generated_present": "yes" if not is_empty_or_bad(generated) else "no",
            "actual_words": len(tokenize(actual)),
            "generated_words": len(tokenize(generated)),
            "actual_chars": len(actual),
            "generated_chars": len(generated),
            "composite": round_or_blank(scores.get("Composite Review Similarity Score"), digits),
            "light_composite": round_or_blank(scores.get("Light Composite Review Score"), digits),
            "heavy_composite": round_or_blank(scores.get("Heavy Composite Review Score"), digits),
            "tfidf": round_or_blank(scores.get("TF-IDF Cosine Similarity"), digits),
            "rouge1": round_or_blank(scores.get("ROUGE-1"), digits),
            "rouge2": round_or_blank(scores.get("ROUGE-2"), digits),
            "rougeL": round_or_blank(scores.get("ROUGE-L"), digits),
            "bleu4": round_or_blank(scores.get("BLEU-4"), digits),
            "jaccard": round_or_blank(scores.get("Jaccard Similarity"), digits),
            "sbert": round_or_blank(scores.get("SBERT Similarity"), digits),
            "sentence_f1": round_or_blank(scores.get("Sentence Embedding F1"), digits),
            "bertscore_f1": round_or_blank(scores.get("BERTScore F1"), digits),
            "citation_doi_jaccard": round_or_blank(scores.get("Citation DOI Jaccard"), digits),
            "citation_title_year_jaccard": round_or_blank(scores.get("Citation Title-Year Jaccard"), digits),
            "keyphrase_coverage": round_or_blank(scores.get("Keyphrase Coverage"), digits),
        })
    return rows


def format_section_summary_table(rows: List[Dict[str, Any]]) -> str:
    cols = [
        "section", "actual_words", "generated_words", "actual_present",
        "generated_present", "light_composite", "heavy_composite", "tfidf",
        "rougeL", "rouge1", "bleu4", "jaccard", "keyphrase_coverage",
    ]
    headers = {
        "section": "Section",
        "actual_words": "Actual words",
        "generated_words": "Generated words",
        "actual_present": "Actual?",
        "generated_present": "Generated?",
        "light_composite": "Light composite",
        "heavy_composite": "Heavy composite",
        "tfidf": "TF-IDF",
        "rougeL": "ROUGE-L",
        "rouge1": "ROUGE-1",
        "bleu4": "BLEU-4",
        "jaccard": "Jaccard",
        "keyphrase_coverage": "Keyphrase",
    }
    widths: Dict[str, int] = {}
    for col in cols:
        widths[col] = max(len(headers[col]), max(len(str(row.get(col, ""))) for row in rows) if rows else 0)
        widths[col] = min(widths[col], 22 if col == "section" else 16)

    def trunc(value: Any, width: int) -> str:
        s = str(value)
        return s if len(s) <= width else s[:width - 3] + "..."

    lines = []
    lines.append(" | ".join(trunc(headers[col], widths[col]).ljust(widths[col]) for col in cols))
    lines.append("-+-".join("-" * widths[col] for col in cols))
    for row in rows:
        lines.append(" | ".join(trunc(row.get(col, ""), widths[col]).ljust(widths[col]) for col in cols))
    return "\n".join(lines)


def build_composite_provenance_rows(
    project_name: str,
    score_by_section: Dict[str, Dict[str, Any]],
    digits: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for section in ["whole_article"] + SECTION_ORDER:
        scores = score_by_section.get(section, {})
        provenance = scores.get("_composite_provenance", {})
        for composite_key in ["light", "heavy"]:
            item = provenance.get(composite_key, {})
            used = item.get("used_metrics", [])
            missing = item.get("missing_metrics", [])
            rows.append({
                "project": project_name,
                "section": section,
                "composite": item.get("name", composite_key),
                "score": round_or_blank(scores.get(item.get("name", "")), digits),
                "score_available": item.get("score_available", False),
                "formula": item.get("formula", ""),
                "weight_sum": round_or_blank(item.get("weight_sum"), digits),
                "required_metrics": "; ".join(item.get("required_metrics", [])),
                "used_metrics": "; ".join([x.get("metric", "") for x in used]),
                "missing_metrics": "; ".join(missing),
            })
    return rows


def format_composite_provenance_table(rows: List[Dict[str, Any]]) -> str:
    cols = ["section", "composite", "score", "score_available", "missing_metrics"]
    headers = {
        "section": "Section",
        "composite": "Composite",
        "score": "Score",
        "score_available": "Available?",
        "missing_metrics": "Missing metrics",
    }
    widths: Dict[str, int] = {}
    for col in cols:
        widths[col] = max(len(headers[col]), max(len(str(row.get(col, ""))) for row in rows) if rows else 0)
        widths[col] = min(widths[col], 44 if col == "missing_metrics" else 28)

    def trunc(value: Any, width: int) -> str:
        s = str(value)
        return s if len(s) <= width else s[:width - 3] + "..."

    lines = [
        " | ".join(trunc(headers[col], widths[col]).ljust(widths[col]) for col in cols),
        "-+-".join("-" * widths[col] for col in cols),
    ]
    for row in rows:
        lines.append(" | ".join(trunc(row.get(col, ""), widths[col]).ljust(widths[col]) for col in cols))
    return "\n".join(lines)


def evaluate_pdf_pair(actual_pdf: Path, generated_pdf: Path, out_dir: Path, args: argparse.Namespace, project_name: str = "") -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    actual_text, actual_meta = extract_pdf_text(actual_pdf)
    generated_text, generated_meta = extract_pdf_text(generated_pdf)

    actual_sections = extract_sections_from_pdf_text(actual_text)
    generated_sections = extract_sections_from_pdf_text(generated_text)
    actual_section_source: Dict[str, Any] = {
        "source": "pdf_extraction",
        "sections": sorted([k for k in actual_sections if k != "whole_article"]),
    }
    generated_section_source: Dict[str, Any] = {
        "source": "pdf_extraction",
        "sections": sorted([k for k in generated_sections if k != "whole_article"]),
    }

    project_dir = infer_project_dir_from_generated_pdf(generated_pdf, args.final_article_dir)
    if project_dir and not args.no_section_files:
        file_sections, file_source = load_actual_sections_from_section_files(project_dir)
        if file_sections.get("whole_article") and len(file_sections.get("whole_article", "")) > 100:
            actual_sections = file_sections
            actual_section_source = file_source

    if not args.no_generated_tex_sections:
        tex_sections, tex_source = load_generated_sections_from_tex(generated_pdf.parent)
        if tex_sections.get("whole_article") and len(tex_sections.get("whole_article", "")) > 100:
            generated_sections = tex_sections
            generated_section_source = tex_source

    write_text(out_dir / "actual_extracted_text.txt", actual_text)
    write_text(out_dir / "generated_extracted_text.txt", generated_text)
    write_json(out_dir / "actual_extraction_meta.json", actual_meta)
    write_json(out_dir / "generated_extraction_meta.json", generated_meta)
    write_json(out_dir / "extracted_actual_sections.json", actual_sections)
    write_json(out_dir / "extracted_generated_sections.json", generated_sections)
    write_json(out_dir / "section_source_report.json", {
        "actual": actual_section_source,
        "generated": generated_section_source,
    })

    pairs = make_section_pairs(actual_sections, generated_sections)
    citation_overlap = build_citation_overlap_scores(actual_text, generated_text, generated_pdf.parent)
    write_json(out_dir / "citation_overlap_report.json", citation_overlap)
    heavy_models = None if args.skip_model_metrics else HeavyMetricModels(args)

    row_records: List[Dict[str, Any]] = []
    metric_table_rows: List[Dict[str, Any]] = []
    score_by_section: Dict[str, Dict[str, Any]] = {}
    warnings = []

    for section, (actual, generated) in pairs.items():
        if is_empty_or_bad(actual):
            warnings.append(f"{section}: actual section missing or too short")
        if is_empty_or_bad(generated):
            warnings.append(f"{section}: generated section missing or too short")

        section_citation_scores = citation_overlap if section == "whole_article" else None
        scores = compute_pair_metrics(actual, generated, args, heavy=heavy_models, citation_scores=section_citation_scores)
        score_by_section[section] = scores

        for metric in METRIC_ORDER:
            row_records.append({
                "project": project_name,
                "section": section,
                "metric": metric,
                "score": round_or_blank(scores.get(metric), args.digits),
                "actual_words": len(tokenize(actual)),
                "generated_words": len(tokenize(generated)),
                "actual_chars": len(actual),
                "generated_chars": len(generated),
            })

    section_order = ["whole_article"] + SECTION_ORDER
    for metric in METRIC_ORDER:
        row = {"Metric": metric}
        for section in section_order:
            row[section] = round_or_blank(score_by_section.get(section, {}).get(metric), args.digits)
        metric_table_rows.append(row)

    section_summary_rows = build_section_summary_rows(project_name, pairs, score_by_section, args.digits)
    composite_provenance_rows = build_composite_provenance_rows(project_name, score_by_section, args.digits)

    write_csv(
        out_dir / "evaluation_metrics_by_section.csv",
        row_records,
        fields=["project", "section", "metric", "score", "actual_words", "generated_words", "actual_chars", "generated_chars"],
    )
    write_csv(out_dir / "evaluation_metrics_table.csv", metric_table_rows)
    write_csv(out_dir / "section_comparison_summary.csv", section_summary_rows)
    write_csv(out_dir / "composite_provenance_by_section.csv", composite_provenance_rows)

    printable = format_printable_table(metric_table_rows, section_order)
    write_text(out_dir / "evaluation_printable_table.txt", printable)
    section_printable = format_section_summary_table(section_summary_rows)
    write_text(out_dir / "evaluation_section_summary_table.txt", section_printable)
    overall_printable = format_overall_metrics_table(metric_table_rows)
    write_text(out_dir / "evaluation_overall_metrics_table.txt", overall_printable)
    composite_provenance_printable = format_composite_provenance_table(composite_provenance_rows)
    write_text(out_dir / "composite_provenance_table.txt", composite_provenance_printable)

    metric_unavailable_reasons: Dict[str, str] = {}
    if args.skip_model_metrics:
        for metric in OPTIONAL_MODEL_METRICS:
            metric_unavailable_reasons[metric] = "Model-backed metric skipped because --skip-model-metrics was used."
    elif heavy_models is not None:
        metric_unavailable_reasons.update(heavy_models.unavailable_reasons)
    if all(score_by_section.get(section, {}).get("METEOR") is None for section in section_order):
        metric_unavailable_reasons["METEOR"] = "NLTK METEOR is unavailable or failed. Install/configure nltk data if this metric is required."
    whole_provenance = score_by_section.get("whole_article", {}).get("_composite_provenance", {})
    light_missing = whole_provenance.get("light", {}).get("missing_metrics", [])
    heavy_missing = whole_provenance.get("heavy", {}).get("missing_metrics", [])
    if light_missing:
        metric_unavailable_reasons["Light Composite Review Score"] = "Missing required metrics: " + ", ".join(light_missing)
    if heavy_missing:
        metric_unavailable_reasons["Heavy Composite Review Score"] = "Missing required metrics: " + ", ".join(heavy_missing)
    if citation_overlap.get("Citation DOI Jaccard") is None:
        metric_unavailable_reasons["Citation DOI Jaccard"] = (
            "No DOI-bearing references were found in one or both bibliographies/reference lists."
        )
    if citation_overlap.get("Citation Title-Year Jaccard") is None:
        metric_unavailable_reasons["Citation Title-Year Jaccard"] = (
            "No parseable title-year references were found in one or both bibliographies/reference lists."
        )

    summary = {
        "project": project_name,
        "actual_pdf": str(actual_pdf),
        "generated_pdf": str(generated_pdf),
        "actual_extractor": actual_meta.get("extractor"),
        "generated_extractor": generated_meta.get("extractor"),
        "actual_section_source": actual_section_source,
        "generated_section_source": generated_section_source,
        "citation_overlap": {
            "Citation DOI Jaccard": citation_overlap.get("Citation DOI Jaccard"),
            "Citation Title-Year Jaccard": citation_overlap.get("Citation Title-Year Jaccard"),
            "n_actual_dois": citation_overlap.get("n_actual_dois"),
            "n_generated_dois": citation_overlap.get("n_generated_dois"),
            "n_actual_title_year": citation_overlap.get("n_actual_title_year"),
            "n_generated_title_year": citation_overlap.get("n_generated_title_year"),
            "n_doi_overlap": len(citation_overlap.get("doi_overlap", [])),
            "n_title_year_overlap": len(citation_overlap.get("title_year_overlap", [])),
        },
        "model_backed_metrics_enabled": not args.skip_model_metrics,
        "bertscore_rescale_with_baseline": args.bertscore_rescale_with_baseline,
        "sections_detected_actual": sorted([k for k in actual_sections.keys() if k != "whole_article"]),
        "sections_detected_generated": sorted([k for k in generated_sections.keys() if k != "whole_article"]),
        "warnings": warnings,
        "metric_unavailable_reasons": metric_unavailable_reasons,
        "whole_article_composite_score": score_by_section.get("whole_article", {}).get("Composite Review Similarity Score"),
        "whole_article_light_composite_score": score_by_section.get("whole_article", {}).get("Light Composite Review Score"),
        "whole_article_heavy_composite_score": score_by_section.get("whole_article", {}).get("Heavy Composite Review Score"),
        "composite_policy": {
            "primary_summary_composite": "Light Composite Review Score",
            "compatibility_alias": "Composite Review Similarity Score",
            "light_composite_weights": dict(LIGHT_COMPOSITE_WEIGHTS),
            "heavy_composite_weights": dict(HEAVY_COMPOSITE_WEIGHTS),
            "renormalization": "disabled",
            "heavy_composite_rule": "blank unless all heavy composite metrics are available",
        },
        "composite_provenance_by_section": composite_provenance_rows,
        "section_composite_scores": {
            section: score_by_section.get(section, {}).get("Composite Review Similarity Score")
            for section in section_order
        },
        "section_light_composite_scores": {
            section: score_by_section.get(section, {}).get("Light Composite Review Score")
            for section in section_order
        },
        "section_heavy_composite_scores": {
            section: score_by_section.get(section, {}).get("Heavy Composite Review Score")
            for section in section_order
        },
        "key_metrics_whole_article": {
            "TF-IDF Cosine Similarity": score_by_section.get("whole_article", {}).get("TF-IDF Cosine Similarity"),
            "ROUGE-1": score_by_section.get("whole_article", {}).get("ROUGE-1"),
            "ROUGE-2": score_by_section.get("whole_article", {}).get("ROUGE-2"),
            "ROUGE-L": score_by_section.get("whole_article", {}).get("ROUGE-L"),
            "BLEU-4": score_by_section.get("whole_article", {}).get("BLEU-4"),
            "Jaccard Similarity": score_by_section.get("whole_article", {}).get("Jaccard Similarity"),
            "SBERT Similarity": score_by_section.get("whole_article", {}).get("SBERT Similarity"),
            "Sentence Embedding Precision": score_by_section.get("whole_article", {}).get("Sentence Embedding Precision"),
            "Sentence Embedding Recall": score_by_section.get("whole_article", {}).get("Sentence Embedding Recall"),
            "Sentence Embedding F1": score_by_section.get("whole_article", {}).get("Sentence Embedding F1"),
            "BERTScore F1": score_by_section.get("whole_article", {}).get("BERTScore F1"),
            "Citation DOI Jaccard": score_by_section.get("whole_article", {}).get("Citation DOI Jaccard"),
            "Citation Title-Year Jaccard": score_by_section.get("whole_article", {}).get("Citation Title-Year Jaccard"),
            "Keyphrase Coverage": score_by_section.get("whole_article", {}).get("Keyphrase Coverage"),
        },
        "outputs": {
            "by_section_csv": str(out_dir / "evaluation_metrics_by_section.csv"),
            "table_csv": str(out_dir / "evaluation_metrics_table.csv"),
            "section_summary_csv": str(out_dir / "section_comparison_summary.csv"),
            "composite_provenance_csv": str(out_dir / "composite_provenance_by_section.csv"),
            "summary_json": str(out_dir / "evaluation_summary.json"),
            "printable_table": str(out_dir / "evaluation_printable_table.txt"),
            "section_printable_table": str(out_dir / "evaluation_section_summary_table.txt"),
            "overall_printable_table": str(out_dir / "evaluation_overall_metrics_table.txt"),
            "composite_provenance_table": str(out_dir / "composite_provenance_table.txt"),
            "metric_unavailable_reasons": str(out_dir / "metric_unavailable_reasons.json"),
            "section_source_report": str(out_dir / "section_source_report.json"),
            "citation_overlap_report": str(out_dir / "citation_overlap_report.json"),
        },
    }
    write_json(out_dir / "metric_unavailable_reasons.json", metric_unavailable_reasons)
    write_json(out_dir / "evaluation_summary.json", summary)
    return summary


def summarize_project_result(project_dir: Path, actual_pdf: Path, generated_pdf: Path, summary: Dict[str, Any], ref_info: Dict[str, Any]) -> Dict[str, Any]:
    key = summary.get("key_metrics_whole_article", {})
    section_scores = summary.get("section_composite_scores", {})
    composite_rows = summary.get("composite_provenance_by_section", [])
    whole_light = next((r for r in composite_rows if r.get("section") == "whole_article" and r.get("composite") == "Light Composite Review Score"), {})
    whole_heavy = next((r for r in composite_rows if r.get("section") == "whole_article" and r.get("composite") == "Heavy Composite Review Score"), {})
    return {
        "project": project_dir.name,
        "project_title": load_project_title(project_dir),
        "actual_pdf": str(actual_pdf),
        "generated_pdf": str(generated_pdf),
        "reference_match_method": ref_info.get("method", ""),
        "reference_match_score": round_or_blank(ref_info.get("score"), 4) if ref_info.get("score") is not None else "",
        "whole_composite": round_or_blank(summary.get("whole_article_composite_score"), 4),
        "whole_light_composite": round_or_blank(summary.get("whole_article_light_composite_score"), 4),
        "whole_heavy_composite": round_or_blank(summary.get("whole_article_heavy_composite_score"), 4),
        "whole_primary_composite": "Light Composite Review Score",
        "whole_light_composite_metrics_used": whole_light.get("used_metrics", ""),
        "whole_light_composite_missing_metrics": whole_light.get("missing_metrics", ""),
        "whole_heavy_composite_metrics_used": whole_heavy.get("used_metrics", ""),
        "whole_heavy_composite_missing_metrics": whole_heavy.get("missing_metrics", ""),
        "whole_tfidf": round_or_blank(key.get("TF-IDF Cosine Similarity"), 4),
        "whole_rouge1": round_or_blank(key.get("ROUGE-1"), 4),
        "whole_rouge2": round_or_blank(key.get("ROUGE-2"), 4),
        "whole_rougeL": round_or_blank(key.get("ROUGE-L"), 4),
        "whole_bleu4": round_or_blank(key.get("BLEU-4"), 4),
        "whole_jaccard": round_or_blank(key.get("Jaccard Similarity"), 4),
        "whole_sbert": round_or_blank(key.get("SBERT Similarity"), 4),
        "whole_sentence_precision": round_or_blank(key.get("Sentence Embedding Precision"), 4),
        "whole_sentence_recall": round_or_blank(key.get("Sentence Embedding Recall"), 4),
        "whole_sentence_f1": round_or_blank(key.get("Sentence Embedding F1"), 4),
        "whole_bertscore_f1": round_or_blank(key.get("BERTScore F1"), 4),
        "whole_citation_doi_jaccard": round_or_blank(key.get("Citation DOI Jaccard"), 4),
        "whole_citation_title_year_jaccard": round_or_blank(key.get("Citation Title-Year Jaccard"), 4),
        "whole_keyphrase_coverage": round_or_blank(key.get("Keyphrase Coverage"), 4),
        "abstract_composite": round_or_blank(section_scores.get("abstract"), 4),
        "introduction_composite": round_or_blank(section_scores.get("introduction"), 4),
        "methods_composite": round_or_blank(section_scores.get("methods"), 4),
        "results_composite": round_or_blank(section_scores.get("results"), 4),
        "main_findings_composite": round_or_blank(section_scores.get("main_findings"), 4),
        "discussion_composite": round_or_blank(section_scores.get("discussion"), 4),
        "conclusion_composite": round_or_blank(section_scores.get("conclusion"), 4),
        "warnings": len(summary.get("warnings", [])),
        "eval_dir": str(Path(summary["outputs"]["summary_json"]).parent),
    }


# ============================================================
# Main runners
# ============================================================

def run_explicit_pair(args: argparse.Namespace) -> None:
    actual_pdf = Path(args.actual_pdf).expanduser().resolve()
    generated_pdf = Path(args.generated_pdf).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else generated_pdf.parent / EVALUATION_DIR

    if not actual_pdf.exists():
        raise SystemExit(f"ERROR: actual PDF not found: {actual_pdf}")
    if not generated_pdf.exists():
        raise SystemExit(f"ERROR: generated PDF not found: {generated_pdf}")

    print("=" * 120)
    print("PDF EVALUATION EXPLICIT PAIR")
    print("=" * 120)
    print("Actual PDF    :", actual_pdf)
    print("Generated PDF :", generated_pdf)
    print("Output dir    :", out_dir)
    print("=" * 120)

    summary = evaluate_pdf_pair(actual_pdf, generated_pdf, out_dir, args, project_name=generated_pdf.parent.parent.name)
    print()
    print("Section sources:")
    print("  Actual   :", summary.get("actual_section_source", {}).get("source"))
    print("  Generated:", summary.get("generated_section_source", {}).get("source"))
    print()
    print("SECTION-WISE COMPARISON SUMMARY")
    print("-" * 120)
    print(read_text(out_dir / "evaluation_section_summary_table.txt"))
    print()
    print("SECTION-WISE METRIC TABLE")
    print("-" * 120)
    print(read_text(out_dir / "evaluation_printable_table.txt"))
    print()
    print("OVERALL METRICS")
    print("-" * 120)
    print(read_text(out_dir / "evaluation_overall_metrics_table.txt"))
    print()
    print("COMPOSITE PROVENANCE")
    print("-" * 120)
    print(read_text(out_dir / "composite_provenance_table.txt"))
    if summary.get("metric_unavailable_reasons"):
        print()
        print("Metric availability notes:")
        for metric, reason in sorted(summary.get("metric_unavailable_reasons", {}).items()):
            print(f" - {metric}: {reason}")
    if summary.get("warnings"):
        print()
        print("Warnings:")
        for warning in summary.get("warnings", []):
            print(" -", warning)
    print("Saved:", out_dir)


def run_projects(args: argparse.Namespace) -> None:
    projects = select_projects(args)
    if not projects:
        raise SystemExit("ERROR: no projects selected.")

    benchmark_root = Path(args.benchmark_root)
    benchmark_index = scan_benchmark_pdfs(benchmark_root)

    global_dir = Path(args.global_eval_dir)
    global_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("STEP20 PDF EVALUATION ALL PROJECTS")
    print("=" * 120)
    print("Projects dir       :", Path(args.projects_dir).resolve())
    print("Projects selected  :", len(projects))
    print("FinalArticle dir   :", args.final_article_dir)
    print("Benchmark root     :", benchmark_root.resolve())
    print("Benchmark PDFs     :", len(benchmark_index))
    print("Model metrics      :", not args.skip_model_metrics)
    print("Match threshold    :", args.match_threshold)
    print("=" * 120)

    summaries: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for i, project_dir in enumerate(projects, start=1):
        print()
        print("-" * 120)
        print(f"[{i}/{len(projects)}] {project_dir.name}")

        generated_pdf = generated_pdf_for_project(project_dir, args.final_article_dir)
        if not generated_pdf.exists():
            print("SKIP: generated PDF not found:", generated_pdf)
            skipped.append({
                "project": project_dir.name,
                "reason": "generated_pdf_not_found",
                "generated_pdf": str(generated_pdf),
            })
            continue

        actual_pdf, ref_info = find_reference_pdf_for_project(project_dir, args, benchmark_index)
        if not actual_pdf or not actual_pdf.exists():
            print("SKIP: reference PDF not found")
            skipped.append({
                "project": project_dir.name,
                "reason": "reference_pdf_not_found",
                "generated_pdf": str(generated_pdf),
                "reference_match_info": json.dumps(ref_info, ensure_ascii=False),
            })
            continue

        print("Actual PDF    :", actual_pdf)
        print("Generated PDF :", generated_pdf)
        print("Reference via :", ref_info.get("method"), "score=", ref_info.get("score"))

        out_dir = project_dir / args.final_article_dir / EVALUATION_DIR
        try:
            summary = evaluate_pdf_pair(actual_pdf, generated_pdf, out_dir, args, project_name=project_dir.name)
            row = summarize_project_result(project_dir, actual_pdf, generated_pdf, summary, ref_info)
            summaries.append(row)
            print(
                "Whole light composite:",
                row["whole_light_composite"],
                "| Heavy composite:",
                row["whole_heavy_composite"],
                "| TF-IDF:",
                row["whole_tfidf"],
                "| ROUGE-L:",
                row["whole_rougeL"],
            )
            print("Section sources:")
            print("  Actual   :", summary.get("actual_section_source", {}).get("source"))
            print("  Generated:", summary.get("generated_section_source", {}).get("source"))
            if not args.no_print_section_tables:
                print()
                print("SECTION-WISE COMPARISON SUMMARY")
                print("-" * 120)
                print(read_text(out_dir / "evaluation_section_summary_table.txt"))
                print()
                print("SECTION-WISE METRIC TABLE")
                print("-" * 120)
                print(read_text(out_dir / "evaluation_printable_table.txt"))
                print()
                print("OVERALL METRICS")
                print("-" * 120)
                print(read_text(out_dir / "evaluation_overall_metrics_table.txt"))
                print()
                print("COMPOSITE PROVENANCE")
                print("-" * 120)
                print(read_text(out_dir / "composite_provenance_table.txt"))
            if summary.get("metric_unavailable_reasons"):
                print()
                print("Metric availability notes:")
                for metric, reason in sorted(summary.get("metric_unavailable_reasons", {}).items()):
                    print(f" - {metric}: {reason}")
            if summary.get("warnings"):
                print()
                print("Warnings:")
                for warning in summary.get("warnings", []):
                    print(" -", warning)
        except Exception as e:
            print("ERROR:", repr(e))
            skipped.append({
                "project": project_dir.name,
                "reason": "evaluation_failed",
                "error": repr(e),
                "actual_pdf": str(actual_pdf),
                "generated_pdf": str(generated_pdf),
            })

    fields = [
        "project", "project_title", "actual_pdf", "generated_pdf",
        "reference_match_method", "reference_match_score",
        "whole_composite", "whole_light_composite", "whole_heavy_composite",
        "whole_primary_composite",
        "whole_light_composite_metrics_used", "whole_light_composite_missing_metrics",
        "whole_heavy_composite_metrics_used", "whole_heavy_composite_missing_metrics",
        "whole_tfidf", "whole_rouge1", "whole_rouge2",
        "whole_rougeL", "whole_bleu4", "whole_jaccard", "whole_sbert",
        "whole_sentence_precision", "whole_sentence_recall", "whole_sentence_f1",
        "whole_bertscore_f1", "whole_citation_doi_jaccard",
        "whole_citation_title_year_jaccard", "whole_keyphrase_coverage",
        "abstract_composite", "introduction_composite", "methods_composite",
        "results_composite", "main_findings_composite", "discussion_composite", "conclusion_composite",
        "warnings", "eval_dir",
    ]
    write_csv(global_dir / "all_projects_pdf_evaluation_summary.csv", summaries, fields=fields)
    write_csv(global_dir / "skipped_projects.csv", skipped)
    write_json(global_dir / "all_projects_pdf_evaluation_summary.json", {
        "projects_selected": len(projects),
        "projects_evaluated": len(summaries),
        "projects_skipped": len(skipped),
        "benchmark_root": str(benchmark_root),
        "benchmark_pdfs_indexed": len(benchmark_index),
        "model_backed_metrics": not args.skip_model_metrics,
        "summary_csv": str(global_dir / "all_projects_pdf_evaluation_summary.csv"),
        "skipped_csv": str(global_dir / "skipped_projects.csv"),
        "rows": summaries,
        "skipped": skipped,
    })

    print()
    print("=" * 120)
    print("ALL PROJECTS PDF EVALUATION DONE")
    print("=" * 120)
    print("Evaluated:", len(summaries))
    print("Skipped  :", len(skipped))
    print("Saved:")
    print(" ", global_dir / "all_projects_pdf_evaluation_summary.csv")
    print(" ", global_dir / "all_projects_pdf_evaluation_summary.json")
    print(" ", global_dir / "skipped_projects.csv")
    print("=" * 120)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("positional_project", nargs="?", default="", help="Optional project index, e.g. 1")
    parser.add_argument("--projects-dir", default=str(PROJECTS_DIR))
    parser.add_argument("--project-index", type=int, default=0)
    parser.add_argument("--project-name", default="")
    parser.add_argument("--all-projects", action="store_true", help="Evaluate all projects. If no project is supplied, all projects are used by default.")
    parser.add_argument("--final-article-dir", default=FINAL_ARTICLE_DIR)

    parser.add_argument("--actual-pdf", default="", help="Benchmark/reference PDF path. Use mainly for one project or explicit pair.")
    parser.add_argument("--generated-pdf", default="", help="Generated PDF path. If supplied with --actual-pdf, evaluates explicit pair.")
    parser.add_argument("--out-dir", default="", help="Output directory for explicit pair mode.")

    parser.add_argument("--benchmark-root", default=str(BENCHMARK_ROOT), help="Folder containing benchmark review PDFs.")
    parser.add_argument("--global-eval-dir", default=str(GLOBAL_EVAL_DIR), help="Global output directory for all-project evaluation.")
    parser.add_argument("--match-threshold", type=float, default=0.18, help="Minimum fuzzy score for matching project to benchmark PDF.")

    parser.add_argument("--digits", type=int, default=4)

    parser.add_argument("--no-section-files", action="store_true", help="Do not use Projects/<project>/section_files for the reference article sections.")
    parser.add_argument("--no-generated-tex-sections", action="store_true", help="Do not use FinalArticle/*.tex files for generated article sections.")
    parser.add_argument("--no-print-section-tables", action="store_true", help="Do not print the detailed section tables during project runs.")

    parser.add_argument("--skip-model-metrics", action="store_true", help="Skip model-backed metrics for a quick debug run. By default, Step20 computes all available metrics.")
    parser.add_argument("--heavy", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--sbert-model", default=DEFAULT_SBERT_MODEL)
    parser.add_argument("--bertscore-model", default=DEFAULT_BERTSCORE_MODEL)
    parser.add_argument("--no-bertscore-rescale-with-baseline", dest="bertscore_rescale_with_baseline", action="store_false", help="Use raw BERTScore instead of baseline-rescaled BERTScore.")
    parser.set_defaults(bertscore_rescale_with_baseline=True)
    parser.add_argument("--codebert-model", default="microsoft/codebert-base", help="Model used for CodeBERT Similarity.")
    parser.add_argument("--spacy-model", default="en_core_web_md", help="spaCy model used for spaCy/WMD-style similarity.")
    parser.add_argument("--entailment-model", default="roberta-large-mnli", help="NLI model used for Entailment Score.")
    parser.add_argument("--model-metric-chars", type=int, default=200000, help="Maximum characters passed to model-backed metrics before model-specific token limits are applied.")
    parser.add_argument("--model-metric-tokens", type=int, default=512, help="Maximum tokens passed to transformer-backed optional metrics.")
    parser.add_argument("--wmd-max-tokens", type=int, default=250, help="Maximum vectorized tokens used for WMD-style similarity.")
    parser.add_argument("--sentence-metric-max-sentences", type=int, default=1000, help="Maximum sentences per document/section for sentence embedding coverage.")
    parser.add_argument("--sentence-metric-batch-size", type=int, default=64, help="Batch size for sentence embedding coverage.")
    parser.add_argument("--keyphrase-top-k", type=int, default=50, help="Number of reference keyphrases used for keyphrase coverage.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.actual_pdf and args.generated_pdf:
        run_explicit_pair(args)
    else:
        run_projects(args)


if __name__ == "__main__":
    main()
 
