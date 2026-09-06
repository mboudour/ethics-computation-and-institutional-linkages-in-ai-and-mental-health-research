# Ethics, Computation, and Institutional Linkages in AI and Mental-Health Research

## Reproducibility materials

This repository provides the **code, exact retrieval specification, deterministic screening rules, derived aggregate outputs, and manuscript sources** for the study. It contains **no Dimensions records** and no restricted record-level content.

> **Important:** No raw Dimensions publication, grant, or policy-document records are deposited here. In particular, this repository contains no record-level titles, abstracts, DOIs, Dimensions IDs, linkouts, `supporting_grant_ids`, `publication_ids`, issuer IDs, full edge lists, or record-level model data. Reproduction of the retrieval requires authorized access to Dimensions and must comply with the applicable Dimensions data-use agreement.

## Study design

The study uses a publication-centred relational design. It first retrieves candidate AI-and-mental-health publications and applies a deterministic local title/abstract screen. A rule-defined, outcome-blind high-specificity subset then requires an available abstract with an explicit core mental-health phrase and an explicit AI/ML phrase other than \emph{neural network} alone. Grants are retained only through identifiers recorded on publications in that subset; policy documents are retained only where their recorded publication identifiers contain a retained publication identifier.

Publication titles and abstracts are represented with deterministic TF--IDF preprocessing and a four-component nonnegative matrix factorization (NMF). Component labels are operational descriptions based on ranked terms and fixed evidence packets; the full component weights and assignment margins are retained. The study estimates theme associations with Dimensions-recorded funding-support and policy-document coverage, describes author-affiliation, grant-funder, and policy-issuer institutions and countries, and tests local relation alignment with a 10,000-permutation publication-year-stratified null.

## Repository contents

| Directory | Content | Record-level Dimensions data? |
| --- | --- | --- |
| `queries/` | Exact candidate-publication query and retrieval specification. | No |
| `code/` | Retrieval, screening, relation-construction, analysis, robustness, and figure scripts. | No |
| `derived_outputs/primary/` | Aggregate principal-model and network summaries. | No |
| `derived_outputs/robustness/` | Aggregate sensitivity, diagnostic, and permutation summaries. | No |
| `derived_outputs/screening/` | Aggregate screening-audit totals and rule-validation results. | No |
| `derived_outputs/thematic/` | Aggregate four-theme composition, policy robustness, institutional-country, and alignment summaries. | No |
| `manuscript/` | Manuscript source, bibliography, and supplementary-material source. | No |

## Exact candidate-publication query

The executable Dimensions DSL query is in [`queries/publication_anchor_query.dsl`](queries/publication_anchor_query.dsl). It retrieves article and review records from 2000--2025 matching one explicit AI/ML expression and one explicit core mental-health expression in the Dimensions `title_abstract_only` search field.

The candidate query is only the first stage. The retained publication anchor requires an additional local, case-insensitive literal screen: at least one approved AI/ML phrase **and** at least one approved core mental-health phrase must appear in the record title or available abstract. The approved phrase lists and matching implementation are in `code/retrieval_and_screening.py`.

## Reproduction requirements

Use Python 3.11+ and an authorized Dimensions Analytics API account. The analysis scripts use `pandas`, `numpy`, `scipy`, `statsmodels`, `patsy`, `openpyxl`, and, for retrieval, `dimcli`. A typical conda installation is:

```bash
conda install -y numpy pandas scipy statsmodels patsy openpyxl
pip install dimcli
```

Place a valid Dimensions API key in a local `key.txt` file that is **not** committed to version control. Recreate the candidate retrieval with the query in `queries/`, apply the local screen, retrieve the linked records, and then run the scripts in the numbered workflow indicated by their filenames and headers. Paths in the scripts reflect the original local workflow and may need to be set to the directories created in a new authorized reconstruction.

## Scope and interpretive boundary

All reported relations are **Dimensions-recorded linkages**. The results are descriptive or associational. They do not establish causal effects, funding decisions, institutional intentions, policy impact, endorsement, policy use, or topical affinity between linked policy documents and publications.

## Availability statement for the manuscript

The associated manuscript should cite this repository as follows after it has been published:

> Reproducibility materials are available at **https://github.com/mboudour/ai-mental-health-institutional-linkages-reproducibility**. The repository contains the exact retrieval specification and run metadata, local screening rules and audit summaries, high-specificity subset rule, NMF and relation-construction code, derived aggregate outputs, and reconstruction instructions for authorized Dimensions users. Raw Dimensions records are not redistributed where restricted by the Dimensions data-use agreement.

## License

Unless stated otherwise, code is released under the MIT License. The repository contains no Dimensions data. Use of Dimensions, any reconstructed data, and any derived files must comply with the applicable Dimensions terms and data-use agreement.
