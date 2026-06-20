# Step4A-BuildStructuredEvidencePack-GROBID.py
from __future__ import annotations

import json
import os
import re
import shutil
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests
from grobid_client.grobid_client import GrobidClient


# ============================================================
# CONFIG
# ============================================================

PROJECTS_DIR = Path("Projects")

GROBID_CONFIG_PATH = Path("config.json")
GROBID_INPUT_DIR = Path("_grobid_input_pdfs")
GROBID_OUTPUT_DIR = Path("_grobid_output_tei")

MAX_WORKERS = max(4, (os.cpu_count() or 4))
GROBID_N = 10

REQUEST_TIMEOUT = 25
OVERWRITE = True

RESOLVE_OPENALEX_REFERENCE_DOIS = True
OPENALEX_SLEEP_SECONDS = 0.05

MIN_SECTION_WORDS_FOR_SCORING = 50

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}

OPENALEX_WORK_CACHE: Dict[str, Dict[str, Any]] = {}
OPENALEX_WORK_CACHE_LOCK = Lock()

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


# ============================================================
# SECTION PATTERNS
# ============================================================

SECTION_PATTERNS = {
    "abstract": [
        r"^abstract$",
        r"^summary$",
    ],
    "introduction": [
        r"^introduction$",
        r"^background$",
    ],
    "methods": [
        r"^methods$",
        r"^method$",
        r"^materials and methods$",
        r"^materials & methods$",
        r"^methodology$",
        r"^review methods$",
        r"^study design$",
        r"^search strategy$",
        r"^data sources$",
        r"^data extraction$",
        r"^eligibility criteria$",
        r"^inclusion and exclusion criteria$",
        r"^study selection$",
        r"^selection criteria$",
        r"^information sources$",
        r"^quality assessment$",
        r"^risk of bias assessment$",
    ],
    "results": [
        r"^results$",
        r"^findings$",
        r"^review findings$",
        r"^results and discussion$",
    ],
    "discussion": [
        r"^discussion$",
        r"^general discussion$",
    ],
    "conclusion": [
        r"^conclusion$",
        r"^conclusions$",
        r"^discussion and conclusion$",
        r"^discussion & conclusion$",
        r"^concluding remarks$",
        r"^summary and conclusion$",
        r"^summary and conclusions$",
        r"^conclusions and future directions$",
    ],
    "limitations": [
        r"^limitations$",
        r"^study limitations$",
        r"^limitations and future work$",
        r"^strengths and limitations$",
    ],
    "funding": [
        r"^funding$",
        r"^financial support$",
        r"^funding statement$",
    ],
    "data_availability": [
        r"^data availability$",
        r"^availability of data and materials$",
        r"^data access$",
        r"^data statement$",
    ],
    "acknowledgements": [
        r"^acknowledg(e)?ments$",
        r"^acknowledgements$",
        r"^acknowledgment$",
    ],
    "references_section": [
        r"^references$",
        r"^bibliography$",
        r"^works cited$",
        r"^literature cited$",
    ],
}

CORE_EVALUATION_SECTIONS = [
    "abstract",
    "introduction",
    "methods",
    "results",
    "discussion",
    "conclusion",
    "limitations",
]


# ============================================================
# BASIC HELPERS
# ============================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def clean_text(text: Optional[Any]) -> str:
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def slugify(text: str, max_len: int = 90) -> str:
    text = clean_text(text)
    text = re.sub(r"\s+", "_", text.lower())
    text = re.sub(r"[^a-z0-9_]+", "", text)
    return text[:max_len].strip("_") or "untitled"


def strip_xml_namespace(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def xml_text(elem: Optional[ET.Element]) -> str:
    if elem is None:
        return ""
    return clean_text(" ".join(t for t in elem.itertext() if clean_text(t)))


def normalize_heading(line: str) -> str:
    s = clean_text(line).lower()
    s = re.sub(r"^\d+(\.\d+)*\.?\s*", "", s)
    s = re.sub(r"[^a-z0-9& ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def canonical_for_heading(raw_heading: str) -> Optional[str]:
    norm = normalize_heading(raw_heading)

    if not norm:
        return None

    for section_name, patterns in SECTION_PATTERNS.items():
        for pat in patterns:
            if re.fullmatch(pat, norm, flags=re.I):
                return section_name

    for section_name, patterns in SECTION_PATTERNS.items():
        for pat in patterns:
            base = pat.strip("^$")
            base = base.replace(r"\s+", " ")
            base = base.replace("\\", "")
            base = re.sub(r"[^a-z0-9& ]+", " ", base)
            base = re.sub(r"\s+", " ", base).strip()

            if base and norm.startswith(base + " "):
                return section_name

    return None


def is_blank_config_value(value: Any) -> bool:
    """
    Safe blank check for config values.

    This avoids:
        TypeError: unhashable type: 'list'

    because config values like coordinates are lists.
    """
    if value is None:
        return True

    if isinstance(value, str):
        return value.strip() == ""

    if isinstance(value, (list, tuple, dict, set)):
        return len(value) == 0

    return False


# ============================================================
# PROJECT DISCOVERY AND GROBID INPUT PREP
# ============================================================

def get_project_dirs() -> List[Path]:
    if not PROJECTS_DIR.exists():
        raise FileNotFoundError(f"Projects directory not found: {PROJECTS_DIR}")

    return [p for p in sorted(PROJECTS_DIR.iterdir()) if p.is_dir()]


def prepare_grobid_input(project_dirs: List[Path]) -> None:
    """
    Copy Projects/<project>/article.pdf to _grobid_input_pdfs/<project>.pdf.
    """
    ensure_dir(GROBID_INPUT_DIR)
    ensure_dir(GROBID_OUTPUT_DIR)

    for project_dir in project_dirs:
        pdf_path = project_dir / "article.pdf"

        if not pdf_path.exists():
            continue

        safe_name = slugify(project_dir.name, max_len=120)
        target_pdf = GROBID_INPUT_DIR / f"{safe_name}.pdf"

        if OVERWRITE or not target_pdf.exists():
            shutil.copy2(pdf_path, target_pdf)


def ensure_grobid_config() -> None:
    """
    Create/update config.json.

    Important:
    Your installed grobid-client-python does not accept output_path=...
    so we store the output directory in config.json instead.
    """
    default_config = {
        "grobid_server": "http://localhost:8070",
        "batch_size": 1000,
        "sleep_time": 5,
        "timeout": 180,
        "output": str(GROBID_OUTPUT_DIR),
        "coordinates": [
            "persName",
            "figure",
            "ref",
            "biblStruct",
            "formula",
            "s",
        ],
    }

    if not GROBID_CONFIG_PATH.exists():
        write_json(GROBID_CONFIG_PATH, default_config)
        print(f"Created config: {GROBID_CONFIG_PATH}")
        return

    try:
        config = read_json(GROBID_CONFIG_PATH)
    except Exception:
        write_json(GROBID_CONFIG_PATH, default_config)
        print(f"Recreated invalid config: {GROBID_CONFIG_PATH}")
        return

    if not isinstance(config, dict):
        write_json(GROBID_CONFIG_PATH, default_config)
        print(f"Recreated non-dict config: {GROBID_CONFIG_PATH}")
        return

    changed = False

    for key, value in default_config.items():
        if key not in config or is_blank_config_value(config.get(key)):
            config[key] = value
            changed = True

    # Force output path to our expected TEI folder.
    if clean_text(config.get("output", "")) != str(GROBID_OUTPUT_DIR):
        config["output"] = str(GROBID_OUTPUT_DIR)
        changed = True

    if changed:
        write_json(GROBID_CONFIG_PATH, config)
        print(f"Updated config: {GROBID_CONFIG_PATH}")


def run_grobid_fulltext(project_dirs: List[Path]) -> None:
    """
    Run GROBID processFulltextDocument on all project article.pdf files.

    Compatible with your installed API:
        client.process("processFulltextDocument", "/path/to/pdfs", n=10)
    """
    prepare_grobid_input(project_dirs)
    ensure_grobid_config()

    if not any(GROBID_INPUT_DIR.glob("*.pdf")):
        print("No PDFs found for GROBID input.")
        return

    ensure_dir(GROBID_OUTPUT_DIR)

    print("Running GROBID processFulltextDocument...")
    print(f"Input : {GROBID_INPUT_DIR.resolve()}")
    print(f"Output: {GROBID_OUTPUT_DIR.resolve()}")

    client = GrobidClient(config_path=str(GROBID_CONFIG_PATH))

    # Compatible call for your installed grobid-client-python.
    client.process(
        "processFulltextDocument",
        str(GROBID_INPUT_DIR),
        n=GROBID_N,
    )

    print("GROBID processing finished.")


def find_grobid_tei_for_project(project_name: str) -> Optional[Path]:
    """
    GROBID client output names can vary by version.

    Check:
      _grobid_output_tei/<safe>.grobid.tei.xml
      _grobid_output_tei/<safe>.tei.xml
      _grobid_output_tei/<safe>.xml
      _grobid_input_pdfs/<safe>.grobid.tei.xml
    """
    safe_name = slugify(project_name, max_len=120)

    candidates = [
        GROBID_OUTPUT_DIR / f"{safe_name}.grobid.tei.xml",
        GROBID_OUTPUT_DIR / f"{safe_name}.tei.xml",
        GROBID_OUTPUT_DIR / f"{safe_name}.xml",
        GROBID_INPUT_DIR / f"{safe_name}.grobid.tei.xml",
        GROBID_INPUT_DIR / f"{safe_name}.tei.xml",
        GROBID_INPUT_DIR / f"{safe_name}.xml",
    ]

    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return path

    search_roots = [GROBID_OUTPUT_DIR, GROBID_INPUT_DIR]

    for root in search_roots:
        if not root.exists():
            continue

        for path in root.rglob("*.xml"):
            if safe_name in path.name and path.stat().st_size > 0:
                return path

    return None


# ============================================================
# TEI SECTION EXTRACTION
# ============================================================

def immediate_child_text(elem: ET.Element) -> str:
    """
    Text from this element, excluding nested <div> content.
    This avoids duplicating subsection text.
    """
    parts: List[str] = []

    for child in list(elem):
        tag = strip_xml_namespace(child.tag)

        if tag == "div":
            continue

        if tag == "head":
            continue

        text = xml_text(child)

        if text:
            parts.append(text)

    return clean_text("\n\n".join(parts))


def flatten_tei_divs(div: ET.Element, level: int = 1) -> List[Dict[str, Any]]:
    """
    Flatten GROBID TEI <div> sections into ordered heading blocks.
    """
    blocks: List[Dict[str, Any]] = []

    head = div.find("./tei:head", TEI_NS)
    heading = xml_text(head)

    body_text = immediate_child_text(div)

    if heading or body_text:
        blocks.append({
            "level": level,
            "heading": heading if heading else f"__UNHEADED_DIV_LEVEL_{level}__",
            "heading_normalized": normalize_heading(heading),
            "canonical": canonical_for_heading(heading),
            "text": body_text,
            "word_count": len(body_text.split()),
        })

    for child_div in div.findall("./tei:div", TEI_NS):
        blocks.extend(flatten_tei_divs(child_div, level=level + 1))

    return blocks


def extract_tei_abstract(root: ET.Element) -> str:
    abstracts = root.findall(".//tei:profileDesc/tei:abstract", TEI_NS)

    if not abstracts:
        abstracts = root.findall(".//tei:abstract", TEI_NS)

    parts = [xml_text(a) for a in abstracts if xml_text(a)]
    return clean_text("\n\n".join(parts))


def extract_tei_body_blocks(root: ET.Element) -> List[Dict[str, Any]]:
    body = root.find(".//tei:text/tei:body", TEI_NS)

    if body is None:
        return []

    blocks: List[Dict[str, Any]] = []

    for div in body.findall("./tei:div", TEI_NS):
        blocks.extend(flatten_tei_divs(div, level=1))

    return blocks


def build_sections_from_grobid_tei(tei_path: Path) -> Dict[str, Any]:
    """
    Build canonical sections from GROBID TEI.

    Logic:
    - GROBID gives structured <div><head>...</head><p>...</p></div>.
    - Heading text is mapped to canonical sections.
    - Non-canonical subsections are folded into the current canonical section.
    """
    tree = ET.parse(str(tei_path))
    root = tree.getroot()

    abstract_text = extract_tei_abstract(root)
    blocks = extract_tei_body_blocks(root)

    sections: Dict[str, Dict[str, Any]] = {}

    for section_name in SECTION_PATTERNS:
        sections[section_name] = {
            "found": False,
            "heading": None,
            "text": "",
            "word_count": 0,
            "blocks": [],
        }

    if abstract_text:
        sections["abstract"]["found"] = True
        sections["abstract"]["heading"] = "Abstract"
        sections["abstract"]["text"] = abstract_text
        sections["abstract"]["word_count"] = len(abstract_text.split())
        sections["abstract"]["blocks"].append({
            "level": 0,
            "heading": "Abstract",
            "text": abstract_text,
        })

    heading_map: List[Dict[str, Any]] = []
    current_section: Optional[str] = None

    for block in blocks:
        heading = clean_text(block.get("heading", ""))
        canonical = block.get("canonical")
        text = clean_text(block.get("text", ""))
        level = block.get("level")

        heading_map.append({
            "level": level,
            "heading": heading,
            "heading_normalized": block.get("heading_normalized", ""),
            "canonical": canonical,
            "body_word_count": len(text.split()),
        })

        if canonical is not None:
            current_section = canonical

            piece = clean_text(f"{heading}\n\n{text}" if text else heading)

            if sections[canonical]["text"]:
                sections[canonical]["text"] = clean_text(sections[canonical]["text"] + "\n\n" + piece)
            else:
                sections[canonical]["text"] = piece
                sections[canonical]["heading"] = heading

            sections[canonical]["found"] = True
            sections[canonical]["blocks"].append({
                "level": level,
                "heading": heading,
                "text": text,
            })

        else:
            if current_section is None:
                continue

            piece = clean_text(f"{heading}\n\n{text}" if text else heading)

            if piece:
                sections[current_section]["text"] = clean_text(
                    sections[current_section]["text"] + "\n\n" + piece
                )
                sections[current_section]["blocks"].append({
                    "level": level,
                    "heading": heading,
                    "text": text,
                })

    for section_name, sec in sections.items():
        sec["text"] = clean_text(sec["text"])
        sec["word_count"] = len(sec["text"].split())

        if sec["word_count"] == 0:
            sec["found"] = False

    return {
        "sectioning_method": "grobid_tei_div_head_sections",
        "tei_path": str(tei_path),
        "sections": sections,
        "heading_map": heading_map,
        "n_headings": len(heading_map),
    }


# ============================================================
# FIGURES AND TABLES FROM TEI
# ============================================================

def extract_tei_figures_and_tables(tei_path: Path) -> Dict[str, Any]:
    tree = ET.parse(str(tei_path))
    root = tree.getroot()

    figures = []
    tables = []

    for fig in root.findall(".//tei:figure", TEI_NS):
        fig_type = clean_text(fig.attrib.get("type", ""))
        label = xml_text(fig.find("./tei:label", TEI_NS))
        head = xml_text(fig.find("./tei:head", TEI_NS))
        desc = xml_text(fig.find("./tei:figDesc", TEI_NS))
        full_text = xml_text(fig)

        payload = {
            "type": fig_type or "figure",
            "label": label,
            "head": head,
            "description": desc,
            "text": full_text,
        }

        if fig_type.lower() == "table":
            tables.append(payload)
        else:
            figures.append(payload)

    return {
        "figures": figures,
        "tables": tables,
    }


# ============================================================
# OPENALEX DOI RESOLUTION
# ============================================================

def normalize_openalex_work_id(work_id: Any) -> str:
    work_id = clean_text(work_id)

    if not work_id:
        return ""

    if work_id.startswith("https://openalex.org/"):
        return work_id.rstrip("/").rsplit("/", 1)[-1]

    if work_id.startswith("http://openalex.org/"):
        return work_id.rstrip("/").rsplit("/", 1)[-1]

    if work_id.startswith("https://api.openalex.org/works/"):
        return work_id.rstrip("/").rsplit("/", 1)[-1]

    return work_id


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


def fetch_openalex_work(work_id: Any) -> Dict[str, Any]:
    norm_id = normalize_openalex_work_id(work_id)

    if not norm_id:
        return {}

    with OPENALEX_WORK_CACHE_LOCK:
        if norm_id in OPENALEX_WORK_CACHE:
            return OPENALEX_WORK_CACHE[norm_id]

    url = f"https://api.openalex.org/works/{quote(norm_id, safe='')}"
    params = {
        "select": "id,doi,title,publication_year,primary_location,type",
    }

    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json() or {}

        primary_location = data.get("primary_location") or {}
        source = primary_location.get("source") or {}

        payload = {
            "openalex_id": data.get("id", ""),
            "work_id": norm_id,
            "doi": normalize_doi(data.get("doi", "")),
            "title": clean_text(data.get("title", "")),
            "year": data.get("publication_year", ""),
            "type": clean_text(data.get("type", "")),
            "source": clean_text(source.get("display_name", "")),
        }

    except Exception as e:
        payload = {
            "openalex_id": f"https://openalex.org/{norm_id}",
            "work_id": norm_id,
            "doi": "",
            "title": "",
            "year": "",
            "type": "",
            "source": "",
            "error": f"{type(e).__name__}: {e}",
        }

    with OPENALEX_WORK_CACHE_LOCK:
        OPENALEX_WORK_CACHE[norm_id] = payload

    time.sleep(OPENALEX_SLEEP_SECONDS)

    return payload


def resolve_metadata_referenced_works(metadata: Dict[str, Any]) -> Dict[str, Any]:
    referenced_works = metadata.get("referenced_works") or []

    if not isinstance(referenced_works, list):
        referenced_works = []

    resolved: List[Dict[str, Any]] = []

    for work_id in referenced_works:
        resolved.append(fetch_openalex_work(work_id))

    dois = sorted({
        normalize_doi(x.get("doi", ""))
        for x in resolved
        if normalize_doi(x.get("doi", ""))
    })

    return {
        "source": "metadata.referenced_works_openalex",
        "n_referenced_works": len(referenced_works),
        "n_resolved": len(resolved),
        "n_with_doi": len(dois),
        "dois": dois,
        "works": resolved,
    }


def write_reference_outputs(project_dir: Path, referenced_works_payload: Dict[str, Any]) -> Dict[str, str]:
    dois = referenced_works_payload.get("dois", []) or []

    references_txt_path = project_dir / "references.txt"
    references_json_path = project_dir / "references.json"

    write_text(references_txt_path, "\n".join(dois))
    write_json(references_json_path, referenced_works_payload)

    return {
        "references_txt": str(references_txt_path),
        "references_json": str(references_json_path),
    }


# ============================================================
# SECTION FILE WRITING
# ============================================================

def write_section_files(project_dir: Path, sections: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    section_dir = project_dir / "section_files"
    ensure_dir(section_dir)

    manifest: Dict[str, Any] = {
        "section_dir": str(section_dir),
        "files": {},
        "section_quality": {},
    }

    combined_blocks: List[str] = []

    for section_name in CORE_EVALUATION_SECTIONS:
        info = sections.get(section_name, {})
        found = bool(info.get("found", False))
        text = clean_text(info.get("text", ""))

        if not found or not text:
            manifest["section_quality"][section_name] = {
                "found": False,
                "word_count": 0,
                "file": "",
                "usable_for_scoring": False,
                "reason": "section_not_detected",
            }
            continue

        out_path = section_dir / f"{section_name}.txt"

        if OVERWRITE or not out_path.exists():
            write_text(out_path, text)

        word_count = len(text.split())
        usable = word_count >= MIN_SECTION_WORDS_FOR_SCORING

        manifest["files"][section_name] = str(out_path)
        manifest["section_quality"][section_name] = {
            "found": True,
            "word_count": word_count,
            "file": str(out_path),
            "usable_for_scoring": usable,
            "reason": "ok" if usable else "too_short_check_manually",
        }

        combined_blocks.append(
            f"============================================================\n"
            f"{section_name.upper()}\n"
            f"============================================================\n\n"
            f"{text}"
        )

    combined_path = section_dir / "all_sections_combined.txt"

    if OVERWRITE or not combined_path.exists():
        write_text(combined_path, clean_text("\n\n\n".join(combined_blocks)))

    manifest["files"]["all_sections_combined"] = str(combined_path)

    manifest_path = section_dir / "section_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)

    return manifest


# ============================================================
# CONFIG EVIDENCE
# ============================================================

def build_config_evidence(
    metadata: Dict[str, Any],
    section_parse: Dict[str, Any],
    figures_tables: Dict[str, Any],
    section_file_manifest: Dict[str, Any],
    referenced_works_payload: Dict[str, Any],
    reference_files: Dict[str, str],
) -> Dict[str, Any]:

    sections = section_parse["sections"]

    return {
        "article_identity": {
            "title": clean_text(metadata.get("title", "")),
            "authors": clean_text(metadata.get("authors", "")),
            "year": clean_text(metadata.get("year", "")),
            "journal": clean_text(metadata.get("journal", "")),
            "doi": clean_text(metadata.get("doi", "")),
            "openalex_id": clean_text(metadata.get("openalex_id", "")),
            "source": clean_text(metadata.get("source", "")),
            "landing_page_url": clean_text(metadata.get("landing_page_url", "")),
        },
        "sectioning_method": "grobid_tei_div_head_sections",
        "core_sections": {
            "abstract": sections.get("abstract", {}).get("text", ""),
            "introduction": sections.get("introduction", {}).get("text", ""),
            "methods": sections.get("methods", {}).get("text", ""),
            "results": sections.get("results", {}).get("text", ""),
            "discussion": sections.get("discussion", {}).get("text", ""),
            "conclusion": sections.get("conclusion", {}).get("text", ""),
            "limitations": sections.get("limitations", {}).get("text", ""),
        },
        "section_files": section_file_manifest,
        "reference_validation_targets": {
            "source": "metadata.referenced_works_openalex",
            "references_txt": reference_files.get("references_txt", ""),
            "references_json": reference_files.get("references_json", ""),
            "n_referenced_works": referenced_works_payload.get("n_referenced_works", 0),
            "n_with_doi": referenced_works_payload.get("n_with_doi", 0),
            "dois": referenced_works_payload.get("dois", []),
        },
        "supporting_evidence": {
            "figures": figures_tables.get("figures", []),
            "tables": figures_tables.get("tables", []),
        },
        "section_presence": {
            sec: {
                "found": info.get("found", False),
                "heading": info.get("heading", None),
                "word_count": info.get("word_count", 0),
                "usable_for_scoring": (
                    sec in CORE_EVALUATION_SECTIONS
                    and info.get("found", False)
                    and int(info.get("word_count", 0) or 0) >= MIN_SECTION_WORDS_FOR_SCORING
                ),
            }
            for sec, info in sections.items()
        },
        "heading_map": section_parse.get("heading_map", []),
    }


# ============================================================
# PROJECT PROCESSING
# ============================================================

@dataclass
class Step4AResult:
    project_name: str
    success: bool
    n_sections_found: int
    n_section_files: int
    n_referenced_works: int = 0
    n_reference_dois: int = 0
    error: str = ""


def process_project(project_dir: Path) -> Step4AResult:
    try:
        metadata_path = project_dir / "article_metadata.json"

        if not metadata_path.exists():
            return Step4AResult(
                project_name=project_dir.name,
                success=False,
                n_sections_found=0,
                n_section_files=0,
                error="Missing article_metadata.json",
            )

        metadata = read_json(metadata_path)

        tei_path = find_grobid_tei_for_project(project_dir.name)

        if tei_path is None:
            return Step4AResult(
                project_name=project_dir.name,
                success=False,
                n_sections_found=0,
                n_section_files=0,
                error="Missing GROBID TEI XML output",
            )

        project_tei_path = project_dir / "grobid_article.tei.xml"

        if OVERWRITE or not project_tei_path.exists():
            shutil.copy2(tei_path, project_tei_path)

        # 1) Extract canonical sections from GROBID TEI.
        section_parse = build_sections_from_grobid_tei(project_tei_path)
        sections = section_parse["sections"]

        # 2) Write one text file per section.
        section_file_manifest = write_section_files(project_dir, sections)

        # 3) Extract figure/table text from GROBID TEI.
        figures_tables = extract_tei_figures_and_tables(project_tei_path)

        # 4) Resolve DOI validation targets from metadata["referenced_works"].
        if RESOLVE_OPENALEX_REFERENCE_DOIS:
            referenced_works_payload = resolve_metadata_referenced_works(metadata)
        else:
            referenced_works = metadata.get("referenced_works") or []
            referenced_works_payload = {
                "source": "metadata.referenced_works_openalex_unresolved",
                "n_referenced_works": len(referenced_works) if isinstance(referenced_works, list) else 0,
                "n_resolved": 0,
                "n_with_doi": 0,
                "dois": [],
                "works": referenced_works if isinstance(referenced_works, list) else [],
            }

        reference_files = write_reference_outputs(project_dir, referenced_works_payload)

        # 5) Save sectioned article JSON.
        sectioned_article = {
            "project_name": project_dir.name,
            "metadata": metadata,
            "sectioning_method": "grobid_tei_div_head_sections",
            "note": (
                "Sections are split from GROBID TEI XML <div>/<head> structure. "
                "DOI validation targets are generated from metadata['referenced_works']; "
                "the PDF reference list is not used as the gold reference target."
            ),
            "tei_file": str(project_tei_path),
            "sections": sections,
            "heading_map": section_parse.get("heading_map", []),
            "section_files": section_file_manifest,
            "figures": figures_tables.get("figures", []),
            "tables": figures_tables.get("tables", []),
            "reference_validation_targets": {
                "references_txt": reference_files.get("references_txt", ""),
                "references_json": reference_files.get("references_json", ""),
                "n_referenced_works": referenced_works_payload.get("n_referenced_works", 0),
                "n_with_doi": referenced_works_payload.get("n_with_doi", 0),
                "dois": referenced_works_payload.get("dois", []),
            },
        }
        write_json(project_dir / "sectioned_article.json", sectioned_article)

        # 6) Save config evidence.
        config_evidence = build_config_evidence(
            metadata=metadata,
            section_parse=section_parse,
            figures_tables=figures_tables,
            section_file_manifest=section_file_manifest,
            referenced_works_payload=referenced_works_payload,
            reference_files=reference_files,
        )
        write_json(project_dir / "config_evidence.json", config_evidence)

        n_sections_found = sum(
            1 for sec in CORE_EVALUATION_SECTIONS
            if sections.get(sec, {}).get("found")
        )

        n_section_files = len([
            k for k, v in section_file_manifest.get("files", {}).items()
            if k != "all_sections_combined" and str(v)
        ])

        log = {
            "step": "4A",
            "status": "success",
            "project_name": project_dir.name,
            "sectioning_method": "grobid_tei_div_head_sections",
            "files_created": [
                str(project_dir / "grobid_article.tei.xml"),
                str(project_dir / "sectioned_article.json"),
                str(project_dir / "config_evidence.json"),
                str(project_dir / "references.txt"),
                str(project_dir / "references.json"),
                str(project_dir / "section_files"),
            ],
            "counts": {
                "n_sections_found": n_sections_found,
                "n_section_files": n_section_files,
                "n_referenced_works": referenced_works_payload.get("n_referenced_works", 0),
                "n_reference_dois": referenced_works_payload.get("n_with_doi", 0),
                "n_figures": len(figures_tables.get("figures", [])),
                "n_tables": len(figures_tables.get("tables", [])),
            },
            "section_quality": section_file_manifest.get("section_quality", {}),
        }
        write_json(project_dir / "step4a_log.json", log)

        return Step4AResult(
            project_name=project_dir.name,
            success=True,
            n_sections_found=n_sections_found,
            n_section_files=n_section_files,
            n_referenced_works=int(referenced_works_payload.get("n_referenced_works", 0) or 0),
            n_reference_dois=int(referenced_works_payload.get("n_with_doi", 0) or 0),
            error="",
        )

    except Exception as e:
        err = f"{type(e).__name__}: {e}"

        try:
            write_json(project_dir / "step4a_log.json", {
                "step": "4A",
                "status": "failed",
                "project_name": project_dir.name,
                "error": err,
            })
        except Exception:
            pass

        return Step4AResult(
            project_name=project_dir.name,
            success=False,
            n_sections_found=0,
            n_section_files=0,
            error=err,
        )


# ============================================================
# MAIN
# ============================================================

def main():
    project_dirs = get_project_dirs()

    if not project_dirs:
        print("No project folders found.")
        return

    print(f"Found {len(project_dirs)} project folders")
    print("Step 4A mode: GROBID TEI section extraction + DOI targets")
    print("Reference target source: metadata['referenced_works']")
    print(f"GROBID concurrency n={GROBID_N}")

    # First run GROBID over all article.pdf files.
    run_grobid_fulltext(project_dirs)

    # Then parse the TEI XML and write project outputs.
    results: List[Step4AResult] = []

    print(f"Parsing GROBID TEI outputs with {MAX_WORKERS} workers")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_project, p): p for p in project_dirs}

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            if result.success:
                print(
                    f"OK: {result.project_name} | "
                    f"sections_found={result.n_sections_found} | "
                    f"section_files={result.n_section_files} | "
                    f"referenced_works={result.n_referenced_works} | "
                    f"reference_dois={result.n_reference_dois}"
                )
            else:
                print(f"FAILED: {result.project_name} | {result.error}")

    successful = [r for r in results if r.success]

    summary = {
        "step": "4A",
        "mode": "grobid_tei_section_file_generation_and_reference_doi_targets",
        "n_total_projects": len(results),
        "n_success": sum(1 for r in results if r.success),
        "n_failed": sum(1 for r in results if not r.success),
        "n_section_files_total": sum(r.n_section_files for r in successful),
        "n_referenced_works_total": sum(r.n_referenced_works for r in successful),
        "n_reference_dois_total": sum(r.n_reference_dois for r in successful),
        "results": [asdict(r) for r in sorted(results, key=lambda x: x.project_name)],
    }

    write_json(PROJECTS_DIR / "step4a_summary.json", summary)

    print("\nDone.")
    print(f"Summary saved to: {PROJECTS_DIR / 'step4a_summary.json'}")


if __name__ == "__main__":
    main()