# Ethics, Computation, and Institutional Linkages in AI and Mental-Health Research

> **Accepted manuscript reproducibility repository.** This repository contains the public, data-free computational materials for the study.

The study constructs a publication-centred relational system from Dimensions.ai and analyzes a high-specificity corpus of AI-and-mental-health publications. It identifies four stable, operational publication-text themes and examines their association with Dimensions-recorded funding-support and policy-document coverage relations, along with descriptive institutional and country representations.

## Explore the published aggregates

**Open the live companion dashboard:** [ai-and-mental-health-research.streamlit.app](https://ai-and-mental-health-research.streamlit.app/)

A data-free interactive companion is included at [`streamlit_app.py`](streamlit_app.py). It reads only the repository's aggregate CSV outputs and presents the four-theme partition, recorded-relation rates and logistic-regression estimates, local alignment permutation results, institution/country landscape summaries, and a full gallery of the six final manuscript figures.

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

The application deliberately contains **no record-level Dimensions content** and does not call the Dimensions API.

## Study at a glance

| Item | Public aggregate result |
| --- | ---: |
| Candidate publications returned by the initial query | 13,574 |
| Broad screened publication anchor ($P_0$) | 11,102 |
| High-specificity analytic corpus ($P$) | 7,146 |
| Recorded linked grants | 3,022 |
| Recorded linked policy documents | 214 |
| Publication-text themes | 4 |
| Local-alignment permutations | 10,000 |

### Four publication-text themes

| Theme | Publications | Share of analytic corpus |
| --- | ---: | ---: |
| Mental-health risk prediction | 1,603 | 22.4% |
| Digital mental-health care and ethics | 2,621 | 36.7% |
| Social and affective detection | 1,445 | 20.2% |
| Neuropsychiatric assessment | 1,477 | 20.7% |

The thematic partition is based on deterministic TF--IDF preprocessing and a four-component non-negative matrix factorization (NMF) model fitted to publication titles and available abstracts. The labels are operational descriptions based on model evidence; they are not claims about naturally bounded or established research domains.

## Acceptance-release materials

| Location | Contents | Record-level Dimensions data? |
| --- | --- | --- |
| [`streamlit_app.py`](streamlit_app.py) | Interactive companion for aggregate results. | No |
| [`queries/`](queries/) | Exact candidate-publication query. | No |
| [`code/`](code/) | Retrieval, screening, thematic, relation-construction, analysis, robustness, and figure-generation scripts. | No |
| [`derived_outputs/`](derived_outputs/) | Aggregate screening, thematic, regression, diagnostics, institutional-country, and permutation summaries. | No |
| [`figures/`](figures/) | Six manuscript figures in PDF format. | No |
| [`manuscript/`](manuscript/) | Manuscript source, bibliography, and supplementary-material source. | No |
| [`REPRODUCIBILITY_GUIDE.md`](REPRODUCIBILITY_GUIDE.md) | Reconstruction sequence, environment, and data-use boundary. | No |

### Manuscript figures

1. [Publication selection and four-theme partition](figures/figure_1_study_design.pdf)
2. [Four-theme composition](figures/figure_2_four_theme_composition.pdf)
3. [Theme-based logistic-regression estimates](figures/figure_3_four_theme_associations.pdf)
4. [Country representation across themes](figures/figure_4_four_theme_country_landscapes.pdf)
5. [Institutional representation across themes](figures/figure_5_institution_landscapes.pdf)
6. [Local alignment of funding-support and policy-document coverage relations](figures/figure_6_four_theme_local_alignment.pdf)

## Data and interpretation boundary

> **No raw Dimensions records are distributed.** The repository contains no record-level titles, abstracts, DOIs, Dimensions identifiers, linkouts, publication--grant or publication--policy-document edge lists, `supporting_grant_ids`, `publication_ids`, issuer identifiers, credentials, serialized data, spreadsheets, or archived raw exports.

Reconstruction requires authorized access to Dimensions and must comply with the applicable Dimensions data-use agreement. All public results concern **Dimensions-recorded relations**. They are descriptive or associational and do not establish causal effects, funding decisions, institutional intentions, policy endorsement, policy use, or policy impact.

## Reproducibility

See the [reproducibility guide](REPRODUCIBILITY_GUIDE.md) for the required environment, final workflow, and data-handling safeguards. The code is provided to make the computational design inspectable and reproducible by researchers with authorized access; it is not a redistribution of the licensed source records.

## Citation

The manuscript has been accepted for publication. Full bibliographic metadata and a persistent DOI will be added when they are available.

## License

Unless stated otherwise, code is released under the [MIT License](LICENSE). Use of Dimensions, any reconstructed data, and any locally generated derivatives must comply with the applicable Dimensions terms and data-use agreement.
