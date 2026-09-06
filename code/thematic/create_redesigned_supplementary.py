#!/usr/bin/env python3
"""Build compilation-ready supplementary materials for the four-theme redesign."""
from pathlib import Path
import re
import pandas as pd

BASE=Path('/home/ubuntu')
OUT=BASE/'redesigned_manuscript'/'supplementary_materials.tex'
THEMES=['Mental-health risk prediction','Digital mental-health care and ethics','Social and affective detection','Neuropsychiatric assessment']
SHORT={'Mental-health risk prediction':'Risk prediction','Digital mental-health care and ethics':'Digital care and ethics','Social and affective detection':'Social/affective detection','Neuropsychiatric assessment':'Neuropsychiatric assessment'}

def tex(x):
    if pd.isna(x): return ''
    x=str(x)
    return x.replace('\\','\\textbackslash{}').replace('&','\\&').replace('%','\\%').replace('_','\\_').replace('#','\\#').replace('$','\\$')

def fnum(x,d=2):
    try: return f'{float(x):.{d}f}'
    except: return tex(x)

def model_label(value):
    s=str(value)
    if 'funding_support' in s:
        base='Funding support'
    elif 'policy_coverage_through_2021' in s:
        base='Policy coverage, P through 2021'
    elif 'policy_coverage_through_2020' in s:
        base='Policy coverage, P through 2020'
    else:
        base=s.replace('_',' ')
    if 'margin_at_least_0_10' in s:
        base += '; margin at least .10'
    elif 'margin_at_least_0_20' in s:
        base += '; margin at least .20'
    return base

def longtable(headers, rows, widths, label, caption):
    spec='@{}' + '@{\\hspace{0.4em}}'.join(f'p{{{w}\\linewidth}}' for w in widths) + '@{}'
    h=' & '.join(headers)+r' \\'
    out=[r'\begin{longtable}{'+spec+r'}',r'\caption{'+caption+r'}\label{'+label+r'}\\',r'\toprule',h,r'\midrule',r'\endfirsthead',r'\multicolumn{'+str(len(headers))+r'}{l}{\small\itshape Continued from previous page}\\',r'\toprule',h,r'\midrule',r'\endhead',r'\bottomrule',r'\endfoot']
    for row in rows: out.append(' & '.join(row)+r' \\')
    out.append(r'\end{longtable}')
    return '\n'.join(out)

def main():
    partition=pd.read_csv(BASE/'four_theme_hard_partition_audit'/'four_theme_partition_summary.csv')
    clarity=pd.read_csv(BASE/'four_theme_hard_partition_audit'/'four_theme_assignment_clarity.csv')
    model_summary=pd.read_csv(BASE/'high_specificity_ai_topic_test'/'topic_model_outputs'/'topic_model_summary.csv')
    stability=pd.read_csv(BASE/'high_specificity_ai_topic_test'/'topic_model_outputs'/'topic_stability_summary.csv')
    terms=pd.read_csv(BASE/'four_theme_evidence_packets'/'four_component_terms.csv')
    coeff=pd.read_csv(BASE/'four_theme_linkage_models'/'four_theme_linkage_model_coefficients.csv')
    rates=pd.read_csv(BASE/'four_theme_linkage_models'/'four_theme_linkage_rates.csv')
    policy_robust=pd.read_csv(BASE/'four_theme_policy_robustness'/'four_theme_policy_robustness_coefficients.csv')
    policy_diag=pd.read_csv(BASE/'four_theme_policy_robustness'/'four_theme_policy_robustness_diagnostics.csv')
    landscape=pd.read_csv(BASE/'four_theme_institution_country_landscapes'/'theme_institution_country_landscape_summary.csv')
    conc=pd.read_csv(BASE/'four_theme_institution_country_landscapes'/'theme_institution_country_concentration.csv')
    top=pd.read_csv(BASE/'four_theme_institution_country_landscapes'/'theme_top_institutions_and_countries.csv')
    config=pd.read_csv(BASE/'four_theme_local_alignment'/'configuration_by_theme.csv')
    null=pd.read_csv(BASE/'four_theme_local_alignment'/'alignment_null_summary.csv')
    subset=pd.read_csv(BASE/'high_specificity_ai_topic_test'/'high_specificity_ai_subset_summary.csv')

    lines=[r'\documentclass[11pt]{article}',r'\usepackage[utf8]{inputenc}',r'\usepackage[T1]{fontenc}',r'\usepackage[margin=0.8in]{geometry}',r'\usepackage{booktabs,longtable,array}',r'\usepackage{microtype}',r'\microtypesetup{expansion=false}',r'\setlength{\LTpre}{0.4em}',r'\setlength{\LTpost}{0.4em}',r'\begin{document}',r'\begin{center}',r'{\Large Supplementary Materials}\\[0.35em]',r'{\large Ethics, Computation, and Institutional Linkages in AI and Mental-Health Research}',r'\end{center}',r'\section*{S1. Analytic corpus and thematic construction}',
    'The broad screened publication anchor contained 11,102 records. The high-specificity analytic corpus retained publications with an available abstract containing a core mental-health phrase and an explicit AI/ML phrase other than \emph{neural network} alone. The final thematic corpus contained 7,146 publications. Text modelling used deterministic TF--IDF preprocessing of titles and abstracts. Two-, three-, and four-component NMF solutions were compared before the four-component solution was selected. Grant, policy-document, institution, country, citation, and linkage information were not used to construct the text model or choose the number of components.']

    sdict={str(r.iloc[0]): r.iloc[1] for _,r in subset.iterrows()}
    s1_rows=[['Screened publication anchor $P_0$','11,102 publications'],['Records with available abstracts',f"{int(sdict.get('n_with_abstract',10676)):,} publications"],['High-specificity analytic corpus $P$','7,146 publications (64.4\\% of $P_0$)']]
    lines.append(longtable(['Item','Value'],s1_rows,[.50,.40],'tab:S1','High-specificity analytic-corpus construction.'))
    ms=[]
    for _,r in model_summary.iterrows():
        k=r.get('n_topics',r.get('k',''))
        st=stability.loc[stability.iloc[:,0]==k]
        if len(st):
            sr=st.iloc[0]
            cosine=sr.get('mean_component_cosine_similarity',sr.iloc[2] if len(sr)>2 else '')
            agreement=sr.get('mean_dominant_assignment_agreement',sr.iloc[-2] if len(sr)>2 else '')
            s=f'8 resamples; mean cosine {fnum(cosine,3)}; mean assignment agreement {fnum(agreement,3)}'
        else:
            s='Not available'
        ms.append([tex(k),fnum(r.get('mean_dominant_share',r.get('average_dominant_weight','')),3),fnum(r.get('share_dominant_ge_060',r.get('share_assignment_at_least_060','')),3),tex(s)])
    lines.append(longtable(['Components','Mean dominant weight','Share $\geq .60$','Stability summary'],ms,[.12,.18,.18,.42],'tab:S2','Small-NMF solution diagnostics.'))

    lines.append(r'\section*{S2. Component evidence and hard assignment}')
    comp_col='component' if 'component' in terms.columns else terms.columns[0]
    term_col='term' if 'term' in terms.columns else terms.columns[1]
    term_rows=[]
    for c in sorted(terms[comp_col].unique()):
        x=terms.loc[terms[comp_col]==c].head(12)
        term_rows.append([f'Component {int(c)+1 if str(c).isdigit() else c}',tex(', '.join(x[term_col].astype(str).tolist()))])
    lines.append(longtable(['Component','Twelve highest-weighted terms'],term_rows,[.18,.72],'tab:S3','Ranked component terms used with fixed publication evidence packets to formulate the operational theme descriptions.'))
    part_rows=[]
    for _,r in partition.iterrows(): part_rows.append([tex(SHORT.get(r['candidate_theme'],r['candidate_theme'])),f'{int(r.n_publications):,}',f'{float(r.pct_of_subset):.1f}\%'])
    lines.append(longtable(['Hard-assignment theme','Publications','Share'],part_rows,[.48,.20,.18],'tab:S4','Four-theme hard partition of the analytic corpus.'))
    cl_rows=[]
    clarity_labels={'clear_margin_at_least_0_20':'Clear margin (at least .20)','moderately_ambiguous_margin_0_10_to_under_0_20':'Moderate ambiguity (.10 to <.20)','highly_ambiguous_margin_under_0_10':'High ambiguity (<.10)'}
    for _,r in clarity.iterrows(): cl_rows.append([clarity_labels.get(r.assignment_clarity,tex(r.assignment_clarity)),f'{int(r.n_publications):,}',f'{float(r.pct_of_subset):.1f}\\%'])
    lines.append(longtable(['Largest-minus-second-largest margin','Publications','Share'],cl_rows,[.48,.20,.18],'tab:S5','Hard-assignment clarity.'))

    lines.append(r'\section*{S3. Theme-based association models}')
    lines.append('Digital mental-health care and ethics is the reference theme. All models use HC3-robust standard errors and adjust for centred publication year and $\log(1+\mathrm{times\ cited})$. Policy-document coverage models are restricted to the stated publication-year eligibility sample.')
    crows=[]
    for _,r in coeff.iterrows():
        term=str(r['term']).replace('theme_','')
        crows.append([tex(model_label(r['model'])),tex(SHORT.get(term,term)),fnum(r['odds_ratio']),f'[{fnum(r["ci_low"])}, {fnum(r["ci_high"])}]',fnum(r['p_value'],3)])
    lines.append(longtable(['Model','Term','OR','95\% CI','$p$'],crows,[.30,.25,.10,.20,.08],'tab:S6','Theme-contrast coefficients from the adjusted funding-support and policy-document coverage models.'))
    rate_rows=[]
    primary_rates=rates.loc[rates['model'].isin(['all_themed_publications__funding_support','all_themed_publications__policy_coverage_through_2021','all_themed_publications__policy_coverage_through_2020'])]
    for _,r in primary_rates.iterrows():
        rate_rows.append([tex(model_label(r['model'])),tex(SHORT.get(r['theme'],r['theme'])),f'{int(r["n_publications"]):,}',f'{float(r["funding_support_pct"]):.1f}\\%',f'{float(r["policy_coverage_pct"]):.1f}\\%'])
    lines.append(longtable(['Analysis','Theme','Publications','Funding support','Policy coverage'],rate_rows,[.28,.25,.13,.13,.13],'tab:S7','Unadjusted recorded funding-support and policy-document coverage shares by hard-assignment theme.'))
    lines.append(r'\subsection*{Policy-document coverage robustness}')
    robust_labels={
        'primary_hc3_linear_year_citation_adjusted':'HC3 logistic: linear year, citation adjusted',
        'hc3_linear_year_citation_omitted':'HC3 logistic: linear year, citation omitted',
        'hc3_spline_year_citation_adjusted':'HC3 logistic: spline year, citation adjusted',
        'firth_linear_year_citation_adjusted':'Firth logistic: linear year, citation adjusted',
    }
    rrows=[]
    focal=policy_robust.loc[policy_robust['term'].str.startswith('theme_')].copy()
    for _,r in focal.iterrows():
        theme=SHORT.get(str(r['term']).replace('theme_',''),str(r['term']).replace('theme_',''))
        rrows.append([tex(robust_labels.get(r['model'],r['model'])),tex(theme),fnum(r['odds_ratio']),f'[{fnum(r["ci_low"])}, {fnum(r["ci_high"])}]',fnum(r['p_value'],3)])
    lines.append(longtable(['Specification','Theme contrast','OR','95\\% CI','$p$'],rrows,[.34,.24,.09,.19,.08],'tab:S7a','Focal theme contrasts in rare-outcome and specification robustness checks for policy-document coverage through 2021. Reference theme: digital care and ethics.'))
    drows=[]
    for _,r in policy_diag.iterrows():
        drows.append([tex(robust_labels.get(r['model'],r['model'])),f'{int(r["n_publications"]):,}',f'{int(r["n_events"]):,}',f'{float(r["events_per_parameter"]):.1f}',tex(r['converged']),tex(r['all_themes_have_event'])])
    lines.append(longtable(['Specification','$N$','Events','EPV','Converged','All themes >0 events'],drows,[.31,.07,.08,.08,.10,.16],'tab:S7b','Diagnostics for policy-document coverage robustness models. EPV denotes events per model parameter.'))

    lines.append(r'\section*{S4. Institutional and country representations}')
    land_rows=[]
    for _,r in landscape.iterrows():
        nsource='' if pd.isna(r['n_source_records']) else f'{int(r["n_source_records"]):,}'
        ntarget='' if pd.isna(r['n_target_entities']) else f'{int(r["n_target_entities"]):,}'
        nties='' if pd.isna(r['n_unique_ties']) else f'{int(r["n_unique_ties"]):,}'
        land_rows.append([tex(r['relation']),tex(SHORT.get(r['theme'],r['theme'])),nsource,ntarget,nties])
    lines.append(longtable(['Relation','Theme','Source records','Target entities','Unique ties'],land_rows,[.22,.30,.15,.15,.12],'tab:S8','Theme-specific institution and country representations. Relation labels indicate whether target entities are institutions or countries; counts are nonexclusive.'))
    conc_rows=[]
    for _,r in conc.iterrows():
        conc_rows.append([tex(r['relation']),tex(SHORT.get(r['theme'],r['theme'])),f'{int(r["n_target_entities"]):,}',f'{float(r["top_1_share_pct"]):.1f}\\%',f'{float(r["top_5_share_pct"]):.1f}\\%',fnum(r['hhi'],3)])
    lines.append(longtable(['Relation','Theme','Entities','Top-one share','Top-five share','HHI'],conc_rows,[.20,.24,.10,.13,.13,.10],'tab:S9','Descriptive concentration of theme-specific institution and country representations.'))
    top_rows=[]
    balanced_top=(top.sort_values(['relation','theme','n_source_records','target_name'],ascending=[True,True,False,True])
                    .groupby(['relation','theme'],sort=False,as_index=False).head(5))
    for _,r in balanced_top.iterrows():
        top_rows.append([tex(r['relation']),tex(SHORT.get(r['theme'],r['theme'])),tex(r['target_name']),f'{int(r["n_source_records"]):,}'])
    lines.append(longtable(['Relation','Theme','Institution/country','Represented source records'],top_rows,[.20,.28,.30,.15],'tab:S10','Five leading institution and country representations for every relation-theme combination. Counts are nonexclusive; complete rankings are included in the reproducibility repository.'))

    lines.append(r'\section*{S5. Local relation alignment}')
    cfg_rows=[]
    for _,r in config.iterrows(): cfg_rows.append([tex(SHORT.get(r['candidate_theme'],r['candidate_theme'])),tex(r['configuration']),f'{int(r["n_publications"]):,}'])
    lines.append(longtable(['Theme','Local configuration','Publications'],cfg_rows,[.45,.30,.15],'tab:S11','Funding-support and policy-document coverage configurations by hard-assignment theme.'))
    null_rows=[]
    for _,r in null.iterrows(): null_rows.append([tex(r['statistic']),f'{float(r["observed"]):.0f}',fnum(r['null_mean']),f'[{fnum(r["null_2_5"])}, {fnum(r["null_97_5"])}]',fnum(r['enrichment']),fnum(r['two_sided_p'],3)])
    lines.append(longtable(['Statistic','Obs.','Null mean','95\\% interval','Enrich.','$p$'],null_rows,[.25,.07,.09,.16,.08,.08],'tab:S12','Publication-year-stratified local-alignment permutation results (10,000 permutations).'))
    lines += [r'\section*{S6. Interpretive boundary}', 'The hard partition is an operational summary of a continuous four-component NMF representation. The component weights and assignment-margin diagnostics are retained in the reproducibility materials. Institution and country counts are descriptive and nonexclusive. Grant and policy-document relations are database-recorded relations; they do not establish funding decisions, institutional intent, policy endorsement, policy influence, or policy impact.',r'\end{document}']
    OUT.write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    print('Wrote',OUT)
if __name__=='__main__': main()
