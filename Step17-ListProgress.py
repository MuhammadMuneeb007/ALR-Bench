#!/usr/bin/env python3
from pathlib import Path
import json
import csv

PROJECTS_DIR = Path("Projects")
MODEL_NAME = "Qwen__Qwen2.5-3B-Instruct"
TOP_N = 200

MODEL_SAFE = MODEL_NAME.replace("/", "__")

def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None

def count_ranking_rows(project_dir):
    ranking = project_dir / "Rankings" / "full_rank_strategy_10.csv"
    if not ranking.exists():
        return 0

    try:
        with ranking.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            return min(TOP_N, sum(1 for _ in csv.DictReader(f)))
    except Exception:
        return 0

def main():
    projects = sorted([
        p for p in PROJECTS_DIR.iterdir()
        if p.is_dir() and (p / "config.json").exists()
    ])

    total_expected = 0
    total_done = 0
    total_include = 0
    total_exclude = 0
    total_uncertain = 0
    total_failed = 0

    print("=" * 120)
    print("STEP16 PROGRESS BY PROJECT")
    print("=" * 120)
    print(f"{'Project':60s} {'Done':>8s} {'Expected':>8s} {'%':>7s} {'Inc':>6s} {'Exc':>6s} {'Unc':>6s} {'Fail':>6s}")
    print("-" * 120)

    for project_dir in projects:
        expected = count_ranking_rows(project_dir)

        article_dir = (
            project_dir
            / "Extractions"
            / f"Step16_{MODEL_SAFE}"
            / "article_json"
        )

        files = sorted(article_dir.glob("global_*_article_*.json")) if article_dir.exists() else []

        done = 0
        include = 0
        exclude = 0
        uncertain = 0
        failed = 0

        for f in files:
            rec = read_json(f)

            if not rec:
                failed += 1
                continue

            status = str(rec.get("_status", "")).lower()
            decision = str(rec.get("final_decision", "")).lower()

            if status == "done" or decision:
                done += 1
            else:
                failed += 1

            if decision == "include":
                include += 1
            elif decision == "exclude":
                exclude += 1
            elif decision == "uncertain":
                uncertain += 1

        pct = (done / expected * 100) if expected else 0

        total_expected += expected
        total_done += done
        total_include += include
        total_exclude += exclude
        total_uncertain += uncertain
        total_failed += failed

        print(
            f"{project_dir.name[:60]:60s} "
            f"{done:8d} {expected:8d} {pct:6.1f}% "
            f"{include:6d} {exclude:6d} {uncertain:6d} {failed:6d}"
        )

    print("-" * 120)

    total_pct = (total_done / total_expected * 100) if total_expected else 0

    print(
        f"{'TOTAL':60s} "
        f"{total_done:8d} {total_expected:8d} {total_pct:6.1f}% "
        f"{total_include:6d} {total_exclude:6d} {total_uncertain:6d} {total_failed:6d}"
    )

    print("=" * 120)

if __name__ == "__main__":
    main()