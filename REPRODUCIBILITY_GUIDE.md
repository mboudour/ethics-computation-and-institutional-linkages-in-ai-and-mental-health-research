# Reproducibility Guide

## Scope

This repository supports inspection and reconstruction of the computational workflow for *Ethics, Computation, and Institutional Linkages in AI and Mental-Health Research*. It is a **data-free reproducibility release**: code, query specifications, manuscript sources, figures, and aggregate outputs are public; licensed record-level Dimensions content is not.

## What is public

| Material | Location | Purpose |
| --- | --- | --- |
| Exact candidate-publication query | [`queries/publication_anchor_query.dsl`](queries/publication_anchor_query.dsl) | Documents the initial Dimensions retrieval. |
| Retrieval and local screening code | [`code/retrieve_candidate_publications.py`](code/retrieve_candidate_publications.py), [`code/retrieval_and_screening.py`](code/retrieval_and_screening.py) | Reconstructs the candidate retrieval and deterministic literal screen for authorized users. |
| High-specificity corpus and thematic workflow | [`code/thematic/`](code/thematic/) | Constructs the analytic subset, audits the hard partition, estimates the four-theme text model, and produces final analyses. |
| Relation construction and analysis code | [`code/`](code/) | Supports attachment of recorded grants, policy documents, affiliations, funders, issuers, robustness checks, and figures. |
| Aggregate results | [`derived_outputs/`](derived_outputs/) | Provides screen totals, thematic counts, regression coefficients, diagnostics, landscape summaries, and permutation summaries. |
| Accepted-manuscript sources | [`manuscript/`](manuscript/) | Provides the LaTeX source, bibliography, and supplementary-material source. |
| Data-free dashboard | [`streamlit_app.py`](streamlit_app.py) | Presents selected aggregate results interactively. |

## What is not public

The repository does **not** contain or distribute:

- raw Dimensions publication, grant, or policy-document records;
- record-level titles, abstracts, DOIs, Dimensions IDs, linkouts, or issuer IDs;
- publication--grant or publication--policy-document edge lists;
- `supporting_grant_ids`, `publication_ids`, credentials, cached API responses, or raw exports;
- serialized files, spreadsheets, archives, or record-level model matrices.

Researchers must obtain their own authorized Dimensions access and comply with the Dimensions data-use agreement. Raw reconstructed records must remain local and are excluded by [`.gitignore`](.gitignore).

## Final workflow

The final design consists of five stages. The complete process is documented in the source scripts; this sequence indicates their role.

1. **Retrieve candidate publications.** Run the exact query in [`queries/`](queries/) through an authorized Dimensions Analytics API session. The query targets article and review records from 2000--2025 with explicit AI/ML and core mental-health phrase blocks.
2. **Apply the deterministic local screen.** [`code/retrieval_and_screening.py`](code/retrieval_and_screening.py) retains records only when an approved AI/ML phrase and an approved core mental-health phrase occur literally in the title or available abstract. It is case-insensitive, uses no stemming or wildcard expansion, makes no manual inclusion decisions, and uses no outcome information.
3. **Construct the high-specificity analytic corpus and four-theme text partition.** [`code/thematic/build_high_specificity_ai_subset.py`](code/thematic/build_high_specificity_ai_subset.py) imposes the outcome-blind abstract and phrase restriction. The final text workflow evaluates deterministic 2/3/4-component NMF fits, retains the four-component solution, audits hard assignments, and saves model evidence and assignment-margin diagnostics. Theme labels are operational descriptions, not claims of naturally bounded research domains.
4. **Attach recorded relations after thematic construction.** The relation workflow uses Dimensions-recorded links and attributes for publication--grant, publication--policy-document, publication--author-affiliation, grant--funder, and policy-document--issuer relations. It does not use grant, policy, affiliation, country, citation, or linkage information to construct themes.
5. **Estimate aggregate analyses and figures.** The final scripts estimate HC3 logistic-regression models, policy-window and assignment-margin sensitivity checks, institution/country summaries, and the 10,000-permutation publication-year-stratified local-alignment test. The six public figures can be regenerated using the figure scripts once authorized reconstruction inputs have been created locally.

## Environment

The original workflow uses Python 3.11+ with at least:

```bash
conda install -y numpy pandas scipy statsmodels patsy openpyxl
python -m pip install dimcli
```

The dashboard is independent of the Dimensions API and can be run from repository aggregates alone:

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

Several scripts contain paths suited to the original controlled workflow. In a new reconstruction, configure local input/output paths outside the repository or in an ignored directory. Never commit a Dimensions API key; `key.txt`, environment files, raw data directories, serialized files, and spreadsheets are ignored by default.

## Interpretation boundary

All findings concern **Dimensions-recorded relations** and publication-text structure. The aggregate results are descriptive or associational. They do not establish causal effects, funding decisions, institutional intentions, policy endorsement, policy use, or policy impact.
