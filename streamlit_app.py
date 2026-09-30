"""Data-free companion dashboard for the accepted manuscript.

This application reads only public aggregate outputs committed in this repository.
It does not retrieve, display, or transmit raw Dimensions records.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


TITLE = "Ethics, Computation, and Institutional Linkages in AI and Mental-Health Research"
ROOT = Path(__file__).resolve().parent
THEMATIC = ROOT / "derived_outputs" / "thematic"
FIGURES = ROOT / "figures"
REPOSITORY_URL = "https://github.com/mboudour/ethics-computation-and-institutional-linkages-in-ai-and-mental-health-research"

THEME_ORDER = [
    "Mental-health risk prediction",
    "Digital mental-health care and ethics",
    "Social and affective detection",
    "Neuropsychiatric assessment",
]
THEME_COLORS = {
    "Mental-health risk prediction": "#2962a3",
    "Digital mental-health care and ethics": "#2e8b72",
    "Social and affective detection": "#d17a22",
    "Neuropsychiatric assessment": "#8f4b9e",
}
MANUSCRIPT_FIGURES = (
    {
        "number": 1,
        "title": "Publication selection and four-theme text partition",
        "pdf": "figure_1_study_design.pdf",
        "caption": "The diagram summarizes the two-stage construction of the publication corpus, the high-specificity analytic filter, and the outcome-blind four-component NMF partition.",
    },
    {
        "number": 2,
        "title": "Four-theme composition",
        "pdf": "figure_2_four_theme_composition.pdf",
        "caption": "Hard assignment to the four publication-text themes in the high-specificity analytic corpus (n = 7,146). The companion panel reports assignment-margin categories.",
    },
    {
        "number": 3,
        "title": "Theme-based logistic-regression estimates",
        "pdf": "figure_3_four_theme_associations.pdf",
        "caption": "Adjusted associations of three publication themes with recorded funding-support and policy-document coverage relations, relative to digital mental-health care and ethics. Points show adjusted odds ratios and bars show 95% confidence intervals.",
    },
    {
        "number": 4,
        "title": "Country representation across themes",
        "pdf": "figure_4_four_theme_country_landscapes.pdf",
        "caption": "Country representation across the four publication themes for publication author affiliations, grant funders, and policy issuers. Representations are nonexclusive descriptive counts normalized within each relation-theme panel.",
    },
    {
        "number": 5,
        "title": "Institutional representation across themes",
        "pdf": "figure_5_institution_landscapes.pdf",
        "caption": "Institutional representation across the four publication themes for author affiliations, grant funders, and policy issuers. Within each panel, the eight institutions with the largest representation in the full thematic corpus are shown.",
    },
    {
        "number": 6,
        "title": "Local alignment of recorded relations",
        "pdf": "figure_6_four_theme_local_alignment.pdf",
        "caption": "Local alignment of recorded funding-support and policy-document coverage relations. The null panels compare observed extensive and intensive alignment statistics with 10,000 publication-year-stratified permutations.",
    },
)


@st.cache_data(show_spinner=False)
def read_csv(filename: str) -> pd.DataFrame:
    """Read a known aggregate-output file from the repository."""
    return pd.read_csv(THEMATIC / filename)


def primary_rates() -> pd.DataFrame:
    rates = read_csv("four_theme_linkage_rates.csv")
    return rates.loc[
        rates["model"].eq("all_themed_publications__funding_support"),
        [
            "theme",
            "n_publications",
            "n_funding_supported_P",
            "funding_support_pct",
            "n_policy_covered_P",
            "policy_coverage_pct",
        ],
    ].copy()


def primary_coefficients(outcome: str) -> pd.DataFrame:
    coefficients = read_csv("four_theme_linkage_model_coefficients.csv")
    model = {
        "Funding support": "all_themed_publications__funding_support",
        "Policy-document coverage (publications through 2021)": "all_themed_publications__policy_coverage_through_2021",
    }[outcome]
    terms = coefficients.loc[
        coefficients["model"].eq(model)
        & coefficients["term"].str.startswith("theme_"),
        ["term", "odds_ratio", "ci_low", "ci_high", "p_value", "n_publications", "n_outcome_events"],
    ].copy()
    terms["theme"] = terms["term"].str.removeprefix("theme_")
    return terms


def format_percent(value: float) -> str:
    return f"{value:.1f}%"


def setup_page() -> None:
    st.set_page_config(page_title=TITLE, page_icon="", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {max-width: 1320px; padding-top: 2.4rem; padding-bottom: 3.2rem;}
        h1 {letter-spacing: -0.03em;}
        .quiet {color: #52616b; font-size: 0.95rem;}
        div[data-testid="stMetric"] {background: #f7fafc; border-radius: 10px; padding: 0.7rem 0.9rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def overview() -> None:
    st.title(TITLE)
    st.markdown(
        "**Accepted-manuscript companion dashboard.** This application uses only the public aggregate outputs in the repository; it does not contain, request, or display raw Dimensions records."
    )
    st.divider()

    stats = read_csv("high_specificity_ai_subset_summary.csv")
    high_specificity = stats.loc[
        stats["candidate_subset"].eq(
            "strict_available_abstract_specific_ai_and_core_mh_in_abstract"
        )
    ].iloc[0]
    partition = read_csv("four_theme_partition_summary.csv")
    alignment = read_csv("alignment_null_summary.csv")

    metrics = st.columns(5)
    metrics[0].metric("Candidate query records", "13,574")
    metrics[1].metric("Screened anchor ($P_0$)", "11,102")
    metrics[2].metric("Analytic corpus ($P$)", f"{int(high_specificity['n_publications']):,}")
    metrics[3].metric("Recorded linked grants", "3,022")
    metrics[4].metric("Recorded linked policy documents", "214")

    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        chart = px.bar(
            partition.sort_values("n_publications", ascending=True),
            x="n_publications",
            y="candidate_theme",
            orientation="h",
            color="candidate_theme",
            color_discrete_map=THEME_COLORS,
            text="n_publications",
            labels={"candidate_theme": "", "n_publications": "Publications"},
            title="Four-theme operational partition",
        )
        chart.update_layout(showlegend=False, margin=dict(l=0, r=10, t=55, b=10), height=360)
        chart.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
        st.plotly_chart(chart, width="stretch")
    with right:
        st.subheader("What the dashboard shows")
        st.markdown(
            "- **Thematic partition:** component sizes and assignment diagnostics.\n"
            "- **Recorded relations:** theme-specific funding-support and policy-document coverage rates, plus adjusted logistic-regression estimates.\n"
            "- **Institution-country landscape:** aggregate descriptive coverage for author affiliations, funders, and policy issuers.\n"
            "- **Local alignment:** year-stratified permutation results for the co-occurrence of the two recorded relations."
        )
        st.caption(
            "Theme labels are operational descriptions based on a deterministic TF--IDF/NMF model. They are not claims about naturally bounded or established research domains."
        )

    st.subheader("Local relation alignment")
    metric_columns = st.columns(2)
    for column, (_, row) in zip(metric_columns, alignment.iterrows(), strict=True):
        column.metric(row["statistic"], f"{int(row['observed']):,}")
        column.caption(
            f"Null mean {row['null_mean']:.2f}; enrichment {row['enrichment']:.2f}; two-sided p = {row['two_sided_p']:.4f}."
        )


def thematic_partition() -> None:
    st.header("Thematic partition")
    st.write(
        "Publication titles and available abstracts were represented with deterministic TF--IDF preprocessing and assigned to the largest normalized weight of a four-component NMF model. Grant, policy-document, institution, country, citation, and relation information were not used to create the partition."
    )

    partition = read_csv("four_theme_partition_summary.csv")
    partition = partition.set_index("candidate_theme").loc[THEME_ORDER].reset_index()
    display = partition.rename(
        columns={
            "candidate_theme": "Theme",
            "n_publications": "Publications",
            "pct_of_subset": "Corpus share (%)",
            "mean_dominant_share": "Mean dominant weight",
            "mean_assignment_margin": "Mean assignment margin",
            "pct_margin_under_0_10": "Margin below .10 (%)",
        }
    )[
        [
            "Theme",
            "Publications",
            "Corpus share (%)",
            "Mean dominant weight",
            "Mean assignment margin",
            "Margin below .10 (%)",
        ]
    ]
    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        column_config={
            "Corpus share (%)": st.column_config.NumberColumn(format="%.1f"),
            "Mean dominant weight": st.column_config.NumberColumn(format="%.3f"),
            "Mean assignment margin": st.column_config.NumberColumn(format="%.3f"),
            "Margin below .10 (%)": st.column_config.NumberColumn(format="%.1f"),
        },
    )

    terms = read_csv("four_component_terms.csv")
    st.subheader("Ranked component terms")
    selected = st.selectbox("Select a theme", THEME_ORDER, key="term_theme")
    component = int(
        partition.loc[partition["candidate_theme"].eq(selected), "dominant_component"].iloc[0]
    )
    subset = terms.loc[terms["component"].eq(component)].copy()
    st.dataframe(subset, hide_index=True, width="stretch")
    st.caption(
        "Terms support an operational component characterization. The full model evidence and assignment-margin checks are retained for sensitivity analysis."
    )


def recorded_relations() -> None:
    st.header("Recorded funding-support and policy-document coverage relations")
    st.write(
        "The measures below are Dimensions-recorded relations. They are descriptive or associational, not evidence of causal funding decisions, policy use, endorsement, or impact."
    )

    rates = primary_rates().set_index("theme").loc[THEME_ORDER].reset_index()
    rate_chart = rates.melt(
        id_vars="theme",
        value_vars=["funding_support_pct", "policy_coverage_pct"],
        var_name="Recorded relation",
        value_name="Percentage",
    )
    rate_chart["Recorded relation"] = rate_chart["Recorded relation"].map(
        {
            "funding_support_pct": "Funding support",
            "policy_coverage_pct": "Policy-document coverage",
        }
    )
    chart = px.bar(
        rate_chart,
        x="theme",
        y="Percentage",
        color="Recorded relation",
        barmode="group",
        color_discrete_map={"Funding support": "#2962a3", "Policy-document coverage": "#d17a22"},
        labels={"theme": "", "Percentage": "Publications with recorded relation (%)"},
        title="Unadjusted recorded-relation rates across themes",
    )
    chart.update_layout(legend_title_text="", margin=dict(l=0, r=10, t=55, b=10), height=440)
    st.plotly_chart(chart, width="stretch")

    outcome = st.radio(
        "Adjusted logistic-regression result",
        ["Funding support", "Policy-document coverage (publications through 2021)"],
        horizontal=True,
    )
    coefficients = primary_coefficients(outcome).set_index("theme").loc[
        [theme for theme in THEME_ORDER if theme != "Digital mental-health care and ethics"]
    ].reset_index()

    forest = go.Figure()
    forest.add_trace(
        go.Scatter(
            x=coefficients["odds_ratio"],
            y=coefficients["theme"],
            mode="markers",
            marker=dict(size=11, color="#2962a3"),
            error_x=dict(
                type="data",
                symmetric=False,
                array=coefficients["ci_high"] - coefficients["odds_ratio"],
                arrayminus=coefficients["odds_ratio"] - coefficients["ci_low"],
                thickness=1.5,
            ),
            hovertemplate="%{y}<br>Odds ratio: %{x:.2f}<extra></extra>",
        )
    )
    forest.add_vline(x=1, line_dash="dash", line_color="#68737d")
    forest.update_layout(
        title=f"Adjusted odds ratios relative to digital mental-health care and ethics ({outcome.lower()})",
        xaxis_title="Odds ratio (95% confidence interval)",
        yaxis_title="",
        margin=dict(l=0, r=20, t=55, b=10),
        height=330,
    )
    st.plotly_chart(forest, width="stretch")

    result_table = coefficients.rename(
        columns={
            "theme": "Theme",
            "odds_ratio": "Odds ratio",
            "ci_low": "95% CI low",
            "ci_high": "95% CI high",
            "p_value": "p value",
        }
    )[["Theme", "Odds ratio", "95% CI low", "95% CI high", "p value"]]
    st.dataframe(
        result_table,
        hide_index=True,
        width="stretch",
        column_config={
            "Odds ratio": st.column_config.NumberColumn(format="%.2f"),
            "95% CI low": st.column_config.NumberColumn(format="%.2f"),
            "95% CI high": st.column_config.NumberColumn(format="%.2f"),
            "p value": st.column_config.NumberColumn(format="%.4f"),
        },
    )
    st.caption(
        "All displayed models use HC3-robust standard errors and adjust for centered publication year and log(1 + times cited). Digital mental-health care and ethics is the reference theme."
    )


def institutional_landscape() -> None:
    st.header("Institution and country landscape")
    st.write(
        "These are nonexclusive descriptive aggregations of recorded author-affiliation, funder, and policy-issuer attributes. They are not population-adjusted and may reflect uneven database indexing and entity resolution."
    )

    landscape = read_csv("theme_institution_country_landscape_summary.csv")
    selected = st.selectbox("Select corpus or theme", landscape["theme"].drop_duplicates().tolist())
    subset = landscape.loc[landscape["theme"].eq(selected)].copy()
    subset = subset.loc[~subset["relation"].eq("P publications")]
    display = subset.rename(
        columns={
            "relation": "Recorded relation or attribute",
            "n_source_records": "Source records",
            "n_target_entities": "Distinct institutions/countries/grants/documents",
            "n_unique_ties": "Recorded ties",
            "source_coverage_pct": "Source coverage (%)",
        }
    )[
        [
            "Recorded relation or attribute",
            "Source records",
            "Distinct institutions/countries/grants/documents",
            "Recorded ties",
            "Source coverage (%)",
        ]
    ]
    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        column_config={"Source coverage (%)": st.column_config.NumberColumn(format="%.1f")},
    )
    st.caption(
        "A source record may have more than one institution or country; counts are therefore nonexclusive. Aggregate tables contain no underlying record identifiers."
    )


def manuscript_figures() -> None:
    st.header("Manuscript figures")
    st.write(
        "This gallery displays the six final figures cited in the accepted manuscript. The browser previews are rendered from the repository's authoritative PDFs; the PDF source for each figure is available below it."
    )

    for figure in MANUSCRIPT_FIGURES:
        pdf_path = FIGURES / figure["pdf"]
        preview_path = FIGURES / "previews" / f"{pdf_path.stem}.png"
        st.subheader(f"Figure {figure['number']}. {figure['title']}")

        if not preview_path.exists() or not pdf_path.exists():
            st.error(f"The display preview for {figure['pdf']} is not available.")
            continue

        st.image(preview_path, caption=figure["caption"], width="stretch")
        download, source = st.columns(2)
        with download:
            st.download_button(
                "Download final PDF",
                data=pdf_path.read_bytes(),
                file_name=figure["pdf"],
                mime="application/pdf",
                key=f"download_figure_{figure['number']}",
                width="stretch",
            )
        with source:
            st.link_button(
                "Open final PDF on GitHub",
                f"{REPOSITORY_URL}/blob/master/figures/{figure['pdf']}",
                width="stretch",
            )
        st.divider()


def methods_and_access() -> None:
    st.header("Methods, access, and safeguards")
    st.markdown(
        """
        ### Final analytical design
        1. **Candidate retrieval and local screen.** The Dimensions query returned 13,574 candidates. A deterministic, literal title/abstract rule retained 11,102 screened publications.
        2. **High-specificity corpus.** A pre-specified, outcome-blind subset required an available abstract with a core mental-health phrase and an explicit AI/ML phrase other than *neural network* alone, yielding 7,146 publications.
        3. **Publication-text partition.** Deterministic TF--IDF and four-component NMF produced the operational four-theme partition. The text model used publication information only.
        4. **Recorded-relation analysis.** Grants, policy documents, author affiliations, funders, and policy issuers were attached only after the corpus and themes were fixed.
        5. **Local alignment.** A 10,000-permutation, publication-year-stratified null tested whether funding-support and policy-document coverage relations co-occurred locally more than expected.

        ### Public-data boundary
        This repository and dashboard distribute only code, query specifications, manuscript sources, figures, and aggregate outputs. They do **not** distribute raw Dimensions records, record-level titles or abstracts, IDs, DOIs, linkouts, edge lists, credentials, spreadsheets, serialized files, or archives.
        """
    )
    st.info(
        "Researchers who wish to reconstruct the workflow need their own authorized Dimensions access and must follow the applicable Dimensions data-use agreement."
    )


def main() -> None:
    setup_page()
    page = st.sidebar.radio(
        "Navigate",
        [
            "Overview",
            "Thematic partition",
            "Recorded relations",
            "Institution and country",
            "Manuscript figures",
            "Methods and data access",
        ],
    )
    st.sidebar.divider()
    st.sidebar.caption("Public aggregate companion\nNo raw Dimensions records")

    {
        "Overview": overview,
        "Thematic partition": thematic_partition,
        "Recorded relations": recorded_relations,
        "Institution and country": institutional_landscape,
        "Manuscript figures": manuscript_figures,
        "Methods and data access": methods_and_access,
    }[page]()


if __name__ == "__main__":
    main()
