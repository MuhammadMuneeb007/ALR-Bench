# Project34-ALR FinalCode Pipeline

This repository implements an automated literature-review pipeline. The pipeline begins with known review articles, uses those reviews to define review-specific search and screening configurations, searches for candidate primary articles, ranks candidates, downloads and converts full text, extracts structured evidence cards with an LLM, synthesizes theme-level evidence, generates a review article, and evaluates the generated output.

In simple terms, what we did is:

1. Downloaded review articles for the evaluation dataset.
2. Converted those review articles into Markdown and structured project folders.
3. Extracted metadata, references, and dataset statistics from the review articles.
4. Generated one review configuration per project.
5. Searched literature databases using each configuration.
6. Cleaned and deduplicated retrieved candidate articles.
7. Ranked candidate articles using several abstract/title strategies.
8. Downloaded full text for top candidates.
9. Converted PDFs/full text into searchable text.
10. Re-ranked candidates using full-text-aware strategies.
11. Used an LLM to create structured evidence cards for top-ranked articles.
12. Built canonical counts, themes, article-theme assignments, evidence packs, and synthesis briefs.
13. Generated article sections, tables, figures, references, validation reports, and PDF output.
14. Evaluated generated outputs against reference material.

Run commands from this folder:

```bash
cd FinalCode
```

## Current Stats

These values were read from the current local workspace. Where information is not available yet, the README uses `TODO`.

| Metric | Current value | Source / note |
|---|---:|---|
| Review projects found | 14 | `Projects/*`, excluding shared `Projects/Search` and `Projects/Rankings` |
| Step16 article evidence cards found | 1,918 | `Projects/*/Extractions/Step16_*/article_json/*.json` |
| Included cards | 1,631 | `final_decision == include` |
| Excluded cards | 210 | `final_decision == exclude` |
| Uncertain cards | 77 | `final_decision == uncertain` |
| Cards with full text | 1,884 | `has_fulltext == 1` |
| Projects with final canonical truth file | 1 | `FinalArticle/project_truth.json`; TODO: run Step18.1 for remaining projects |
| Empty placeholder scripts | 4 | TODO: implement/remove `Step11.1`, `Step15`, `Step18.1-CombineInformation`, `Step9-FinalRankingChecking` |

## Completed Project Example

The project below currently has `FinalArticle/project_truth.json`:

`a_survey_on_large_language_model_based_autonomous_agents`

| Canonical count | Value | Meaning |
|---|---:|---|
| Records downloaded | 5,271 | Database retrieval output |
| Records after DOI/metadata cleaning | 4,962 | Cleaned candidate records |
| Records ranked by abstract/title | 4,962 | Abstract/title ranking pool |
| Top-ranked screening pool | 1,000 | Full-text-aware ranking pool; do **not** call these retrieved records |
| LLM screening target | 200 | Top records sent to Step16 evidence-card generation |
| Article cards generated | 200 | Step16 card count |
| Cards used for synthesis | 187 | Included cards used in article package |
| Included | 187 | Final decision include |
| Excluded | 11 | Final decision exclude |
| Uncertain | 2 | Final decision uncertain |
| Cards with full text | 200 | Full text available in Step16 cards |
| Cards without full text | 0 | Abstract/metadata-only cards |

For all other projects, canonical counts are currently `TODO`. Run Step18.1 for each project to generate `FinalArticle/project_truth.json`.

## Step-by-Step Pipeline

### Step 1: Download Review Articles

We first build the evaluation dataset by downloading known review articles. These review articles are the starting point for each project because each one defines a review topic and provides references that can be used as gold-standard evidence.

| Script | What it does | Input | Output | Instruction |
|---|---|---|---|---|
| `Step1-DownloadResearchArticleForSystemEvaluation.py` | Downloads/collects review articles for system evaluation. | Review search settings/OpenAlex metadata | Downloaded review article records and files | `python Step1-DownloadResearchArticleForSystemEvaluation.py` |
| `Step2-DownloadResearchArticle-Information.py` | Saves review-article information and dataset report metadata. | Downloaded review metadata | Dataset CSV/report files | `python Step2-DownloadResearchArticle-Information.py` |

Important note: review articles should carry their OpenAlex `referenced_works` when available so the pipeline can later resolve cited works into DOI-based gold references.

### Step 2: Convert Review Articles To Markdown

After downloading review PDFs/articles, we convert them into Markdown so the pipeline can parse and inspect them consistently.

| Script | What it does | Input | Output | Instruction |
|---|---|---|---|---|
| `Step3-ConvertArticlesToMarkdown.py` | Converts downloaded review articles into Markdown/project folders. | Downloaded PDFs/articles | `Projects/<project>/article.md` and metadata files | `python Step3-ConvertArticlesToMarkdown.py` |

### Step 3: Structure Review Articles And References

Next, we structure the review article content. This includes extracting metadata, reference sections, structured evidence, and gold-standard references when possible.

| Script | What it does | Input | Output | Instruction |
|---|---|---|---|---|
| `Step4-StructureArticles.py` | Builds structured evidence/reference packs from review articles. | Project Markdown and metadata | Structured evidence files and reference files | `python Step4-StructureArticles.py` |

Expected important outputs:

| Output | Purpose |
|---|---|
| `article_metadata.json` | Carries title, DOI, year, and source metadata |
| `references_gold.json` | Gold DOI list for cited works when available |
| Structured article text files | Used by later configuration/statistics steps |

### Step 4: Print Dataset Statistics

Before generating configurations, we inspect the dataset to understand what we have.

| Script | What it does | Input | Output | Instruction |
|---|---|---|---|---|
| `Step5-DatasetStatistics.py` | Prints dataset statistics and article BibTeX/metadata information. | `Projects/*` | Console statistics | `python Step5-DatasetStatistics.py` |

This helps confirm how many review projects exist, how much metadata exists, and whether references/DOIs are available.

### Step 5: Generate Review Configuration

For each review article, the pipeline creates a flat configuration describing the review objective, research questions, keywords, inclusion criteria, exclusion criteria, databases, and quality/extraction questions.

| Script | What it does | Input | Output | Instruction |
|---|---|---|---|---|
| `Step6.1-GenerateConfiguration.py` | Generates one flat `config.json` per review project. | Review article Markdown/metadata | `Projects/<project>/config.json` | `python Step6.1-GenerateConfiguration.py` |
| `Step6.2-ValidateConfiguration.py` | Checks whether config terms appear in the review article text. | `config.json`, article text | Validation report/statistics | `python Step6.2-ValidateConfiguration.py` |

The configuration is important because later search and ranking strategies use it as the review-specific intent.

### Step 6: Search Literature Databases

Using each `config.json`, we search literature databases for candidate articles related to the review topic.

| Script | What it does | Input | Output | Instruction |
|---|---|---|---|---|
| `Step7-SearchLiterature.py` | Searches configured databases and saves candidate article metadata. | `Projects/<project>/config.json` | `Projects/<project>/Search/articles.csv` | `python Step7-SearchLiterature.py` |

Search output can include records from sources such as OpenAlex, Crossref, Semantic Scholar, Europe PMC, and PubMed depending on the project configuration and available APIs.

### Step 7: Clean Candidate Articles

Raw search results contain duplicates, missing DOI records, and noisy metadata. We clean and standardize them before ranking.

| Script | What it does | Input | Output | Instruction |
|---|---|---|---|---|
| `Step8-Cleaning.py` | Cleans retrieved metadata, removes duplicates, filters DOI metadata, and prints common references. | `Search/articles.csv` | `Search/articles_clean_doi_only.csv` | `python Step8-Cleaning.py` |
| `Step8.1-CalculateCeiling.py` | Computes the retrievable ceiling against gold DOI references. | Cleaned candidates and gold references | Ceiling/diagnostic reports | `python Step8.1-CalculateCeiling.py` |

The ceiling step tells us the maximum possible recall if the candidate pool does not contain all gold references.

### Step 8: Rank Candidates Using Abstract/Title Metadata

We then rank candidate articles using multiple strategies. These are metadata-based rankings that rely mostly on title, abstract, keywords, source metadata, and the review configuration.

| Strategy | Script | What it does | Instruction |
|---:|---|---|---|
| 1 | `Step9-Rank1-ConfigBM25.py` | Config-weighted BM25 / lexical baseline. | `python Step9-Rank1-ConfigBM25.py` |
| 2 | `Step9-Rank2-ReviewIntentSBERT.py` | Review-intent semantic/cross-encoder ranking. | `python Step9-Rank2-ReviewIntentSBERT.py --force` |
| 3 | `Step9-Rank3-TitleObjectiveSentenceTransformer.py` | Title/objective semantic ranking. | `python Step9-Rank3-TitleObjectiveSentenceTransformer.py --force` |
| 4 | `Step9-Rank4-ConceptKeywordCoverage.py` | Concept and keyword phrase coverage. | `python Step9-Rank4-ConceptKeywordCoverage.py` |
| 5 | `Step9-Rank5-GeneratedQueryMatch.py` | Generated/reported query matching. | `python Step9-Rank5-GeneratedQueryMatch.py` |
| 6 | `Step9-Rank6-BooleanSoftMatch.py` | Boolean-style soft matching. | `python Step9-Rank6-BooleanSoftMatch.py` |
| 7 | `Step9-Rank7-FieldWeightedLexical.py` | Field-weighted lexical ranking. | `python Step9-Rank7-FieldWeightedLexical.py` |
| 8 | `Step9-Rank8-SciBERTSemantic.py` | Scientific/biomedical semantic ranking. | `python Step9-Rank8-SciBERTSemantic.py` |
| 9 | `Step9-Rank9-SPECTER2Semantic.py` | SPECTER/SPECTER2 paper-level semantic ranking. | `python Step9-Rank9-SPECTER2Semantic.py` |
| 10 | `Step9-Rank10-HybridFusion.py` | Hybrid fusion of ranking strategies. | `python Step9-Rank10-HybridFusion.py` |
| Final | `Step9-FinalRanking.py` | Final metadata ranking and DOI-match reporting. | `python Step9-FinalRanking.py` |

Helpful reporting scripts:

| Script | What it does | Instruction |
|---|---|---|
| `Step9-PrintRanking.py` | Prints ranking tables for Step9 strategies. | `python Step9-PrintRanking.py` |
| `Step9-Diagnosis.py` | Prints deeper ranking diagnostics such as C@10 through C@3000. | `python Step9-Diagnosis.py` |

### Step 9: Download Full Text

After metadata ranking, the pipeline downloads full text/PDFs for top candidate articles.

| Script | What it does | Input | Output | Instruction |
|---|---|---|---|---|
| `Step11-DownloadFullText.py` | Downloads full text and PDFs for candidate articles. | Ranked candidate files | Full-text/PDF files under project/database folders | `python Step11-DownloadFullText.py` |
| `Step12-PrintStatistics.py` | Prints full-text download statistics. | Downloaded files | Console statistics | `python Step12-PrintStatistics.py` |

### Step 10: Convert Downloaded Full Text

Downloaded PDFs/full-text files are converted into text so ranking and LLM extraction can use them.

| Script | What it does | Input | Output | Instruction |
|---|---|---|---|---|
| `Step13-ConvertToFullText.py` | Converts PDFs/full text into text/Markdown. | Downloaded PDFs/full text | Converted text files such as `COMBINED.md` | `python Step13-ConvertToFullText.py` |
| `Step13.1-PrintConvertTextStatistics.py` | Prints conversion statistics. | Converted text files | Console statistics | `python Step13.1-PrintConvertTextStatistics.py` |

### Step 11: Full-Text-Aware Ranking

The pipeline then re-ranks candidates using full text where available. If full text is missing, metadata/title/abstract fallback can be used.

| Strategy | Script | What it does | Instruction |
|---:|---|---|---|
| 1 | `Step14-FullRank1-ConfigBM25.py` | Full-text-aware config BM25 ranking. | `python Step14-FullRank1-ConfigBM25.py` |
| 2 | `Step14-FullRank2-ReviewIntentSBERT.py` | Full-text-aware review-intent semantic ranking. | `python Step14-FullRank2-ReviewIntentSBERT.py` |
| 3 | `Step14-FullRank3-TitleObjectiveSentenceTransformer.py` | Full-text-aware title/objective semantic ranking. | `python Step14-FullRank3-TitleObjectiveSentenceTransformer.py` |
| 4 | `Step14-FullRank4-ConceptKeywordCoverage.py` | Full-text-aware concept/keyword coverage ranking. | `python Step14-FullRank4-ConceptKeywordCoverage.py` |
| 5 | `Step14-FullRank5-GeneratedQueryMatch.py` | Full-text-aware generated query matching. | `python Step14-FullRank5-GeneratedQueryMatch.py` |
| 6 | `Step14-FullRank6-BooleanSoftMatch.py` | Full-text-aware Boolean soft-match ranking. | `python Step14-FullRank6-BooleanSoftMatch.py` |
| 7 | `Step14-FullRank7-FieldWeightedLexical.py` | Full-text-aware field-weighted lexical ranking. | `python Step14-FullRank7-FieldWeightedLexical.py` |
| 8 | `Step14-FullRank8-SciBERTSemantic.py` | Full-text-aware SciBERT/BGE-style semantic ranking. | `python Step14-FullRank8-SciBERTSemantic.py` |
| 9 | `Step14-FullRank9-SPECTER2Semantic.py` | Full-text-aware SPECTER/SPECTER2 semantic ranking. | `python Step14-FullRank9-SPECTER2Semantic.py` |
| 10 | `Step14-FullRank10-HybridFusion.py` | Hybrid fusion of full-text rankings. | `python Step14-FullRank10-HybridFusion.py` |

Helpful reporting scripts:

| Script | What it does | Instruction |
|---|---|---|
| `Step14-PrintRanking.py` | Prints full ranking performance at C@10 to C@1000. | `python Step14-PrintRanking.py` |
| `Step14-PrintEachFullRankingTable.py` | Prints one table per full-text ranking strategy. | `python Step14-PrintEachFullRankingTable.py` |

### Step 12: Generate LLM Evidence Cards

The top-ranked articles are passed to an LLM to create structured article evidence cards. These cards are the core synthesis unit for the generated review article.

| Script | What it does | Input | Output | Instruction |
|---|---|---|---|---|
| `Step16-Synthesis.py` | Generates structured evidence cards from top-ranked full-text articles. | Full-text ranking CSVs and full text | `Projects/<project>/Extractions/Step16_*/article_json/*.json` | `python Step16-Synthesis.py` |
| `TestingLLM.py` | Experimental/config-driven LLM eligibility and extraction script. | Full-rank top candidates | LLM quality/ranking outputs | `python TestingLLM.py --help` |

Current evidence-card totals:

| Decision | Count |
|---|---:|
| Include | 1,631 |
| Exclude | 210 |
| Uncertain | 77 |
| Total | 1,918 |

### Step 13: Track Progress

Use this to quickly see what projects are complete or missing outputs.

| Script | What it does | Instruction |
|---|---|---|
| `Step17-ListProgress.py` | Lists progress across extraction and article-generation stages. | `python Step17-ListProgress.py` |

### Step 14: Generate Final Review Article

`Step18.1-GenerateArticle.py` builds the final article package. It creates canonical counts, a canonical theme schema, article-theme assignments, evidence packs, synthesis briefs, section prompts, article sections, references, validation reports, and optionally compiles the PDF.

Fast rebuild without LLM:

```bash
python Step18.1-GenerateArticle.py 1 --max-cards 200 --max-table-rows 200 --included-only --generate 0 --compile 0
```

Full generation with DeepSeek/LLM:

```bash
python Step18.1-GenerateArticle.py 1 --max-cards 200 --max-table-rows 200 --included-only --generate 1 --max-themes 8 --force --compile 1 --progress-interval 10
```

Key outputs:

| Output | Purpose |
|---|---|
| `FinalArticle/Main.tex` | Main article source |
| `FinalArticle/Main.pdf` | Compiled article PDF when `--compile 1` |
| `FinalArticle/reference.bib` | DOI-enriched references |
| `FinalArticle/bibtex_doi_cache.json` | DOI-to-BibTeX cache to avoid repeated API calls |
| `FinalArticle/project_truth.json` | Locked canonical counts |
| `FinalArticle/review_theme_schema.json` | Canonical theme schema and number of themes |
| `FinalArticle/article_theme_assignments.json` | Multi-theme article assignments |
| `FinalArticle/theme_evidence_packs.json` | Evidence grouped by canonical themes |
| `FinalArticle/theme_synthesis_briefs.json` | Per-theme synthesis briefs |
| `FinalArticle/GeneratedTables.tex` | Deterministic flow diagram and descriptive tables |
| `FinalArticle/Reports/FINAL_citation_report.json` | Citation validation |
| `FinalArticle/Reports/FINAL_numeric_provenance_report.json` | Numeric provenance validation |
| `FinalArticle/Reports/FINAL_theme_consistency_report.json` | Theme consistency validation |
| `FinalArticle/Reports/FINAL_count_wording_report.json` | Count/wording validation |
| `FinalArticle/Reports/FINAL_compile_report.json` | PDF compile log summary |

### Step 15: Generate Diagrams

This script generates local diagrams, tables, and summaries for projects.

| Script | What it does | Instruction |
|---|---|---|
| `Step19-GenerateDiagrams.py` | Generates local diagrams/tables/JSON summaries. | `python Step19-GenerateDiagrams.py 1` |

### Step 16: Evaluate Generated Output

Finally, generated PDFs/articles can be compared against reference articles using multiple text-similarity metrics.

| Script | What it does | Instruction |
|---|---|---|
| `Step20-Evaluate.py` | Evaluates generated PDFs/articles against benchmark/reference PDFs. | `python Step20-Evaluate.py` |

Example metric families include exact match, Levenshtein similarity, Jaccard similarity, TF-IDF cosine similarity, ROUGE, BLEU, METEOR, BERTScore, SBERT similarity, WMD similarity, and entailment-style scores.

## Script Inventory

| File | Role | Status |
|---|---|---|
| `Step1-DownloadResearchArticleForSystemEvaluation.py` | Download review articles for evaluation. | Active |
| `Step2-DownloadResearchArticle-Information.py` | Save review metadata/report information. | Active |
| `Step3-ConvertArticlesToMarkdown.py` | Convert review articles to Markdown/project files. | Active |
| `Step4-StructureArticles.py` | Structure review article content and references. | Active |
| `Step5-DatasetStatistics.py` | Print dataset statistics and BibTeX/metadata info. | Active |
| `Step6.1-GenerateConfiguration.py` | Generate review configuration. | Active |
| `Step6.2-ValidateConfiguration.py` | Validate generated configuration. | Active |
| `Step7-SearchLiterature.py` | Search candidate literature. | Active |
| `Step8-Cleaning.py` | Clean/deduplicate candidate metadata. | Active |
| `Step8.1-CalculateCeiling.py` | Compute retrievable ceiling. | Active |
| `Step9-Rank*.py` | Abstract/title ranking strategies. | Active |
| `Step9-FinalRanking.py` | Final metadata ranking. | Active |
| `Step9-PrintRanking.py` | Print Step9 ranking tables. | Active |
| `Step11-DownloadFullText.py` | Download full text/PDFs. | Active |
| `Step12-PrintStatistics.py` | Print full-text statistics. | Active |
| `Step13-ConvertToFullText.py` | Convert full text/PDFs to text. | Active |
| `Step13.1-PrintConvertTextStatistics.py` | Print conversion statistics. | Active |
| `Step14-FullRank*.py` | Full-text-aware ranking strategies. | Active |
| `Step14-PrintRanking.py` | Print full-text ranking summary. | Active |
| `Step16-Synthesis.py` | Generate LLM evidence cards. | Active |
| `Step17-ListProgress.py` | List pipeline progress. | Active |
| `Step18.1-GenerateArticle.py` | Generate final review article package. | Active |
| `Step19-GenerateDiagrams.py` | Generate diagrams/tables. | Active |
| `Step20-Evaluate.py` | Evaluate generated outputs. | Active |
| `TestingLLM.py` | Experimental LLM extraction/eligibility script. | Active/experimental |
| `Step11.1-DownloadTest.py` | Empty. | TODO |
| `Step15-FullTextSummary.py` | Empty. | TODO |
| `Step18.1-CombineInformation.py` | Empty. | TODO |
| `Step9-FinalRankingChecking.py` | Empty. | TODO |
| `Step18.1-GenerateArticle copy.py` | Duplicate/backup. | TODO: archive or remove |

## Important Wording Rules

| Use this wording | Avoid this wording |
|---|---|
| `5,271 records were downloaded/retrieved from configured databases` | `1,000 records were retrieved` |
| `1,000 records formed the top-ranked screening pool` | Treating top-1000 as original retrieval count |
| `evidence-card completeness and extraction audit` | `quality assessment`, unless formal quality scores exist |
| `studies may belong to multiple themes` | Treating theme counts as mutually exclusive when multi-assignment is enabled |

## LLM Configuration

`Step18.1-GenerateArticle.py` reads `.env` automatically.

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` | Provider label, e.g. `deepseek` |
| `LLM_BASE_URL` | OpenAI-compatible base URL |
| `LLM_MODEL` | Model name |
| `LLM_API_KEY` | API key; do not commit this |
| `LLM_TEMPERATURE` | Temperature |
| `LLM_TOP_P` | Top-p sampling |
| `LLM_MAX_TOKENS` | Max output tokens |
| `LLM_REASONING_EFFORT` | Optional provider-specific setting |
| `PROGRESS_INTERVAL` | Heartbeat interval for DOI/LLM waits |

## Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| References do not appear in PDF | BibTeX was not run after first LaTeX pass | Run `pdflatex Main.tex`, `bibtex Main`, `pdflatex Main.tex`, `pdflatex Main.tex` |
| Citations show as `?` | Invalid cite keys or stale `.bbl` | Check `Reports/FINAL_citation_report.json` and recompile |
| Step18.1 seems stuck | DOI lookup or LLM request is running | Use `--progress-interval 10`; DOI cache is saved in `bibtex_doi_cache.json` |
| Numeric report has many unknown numbers | Generated text includes numbers not found in truth/cards/theme packs | Check `Reports/FINAL_numeric_provenance_report.json` |
| Theme counts exceed included count | Multi-theme assignment is enabled | This is expected; include the multi-assignment disclaimer |
| LLM key missing | `.env` not loaded or key not set | Add `LLM_API_KEY` to `.env` or pass `--llm-api-key` |

## Remaining TODOs

| TODO | Why it matters |
|---|---|
| Run Step18.1 for all 14 projects | Creates canonical truth, themes, reports, and PDFs for every project |
| Review empty placeholder scripts | Avoid confusion and dead files |
| Archive/remove `Step18.1-GenerateArticle copy.py` | Prevent editing the wrong article generator |
| Run Step20 after final PDFs are built | Produces final evaluation metrics |

