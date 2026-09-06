#!/usr/bin/env python3
"""Estimate theme-based P-G funding-support and P-D policy-document coverage associations.

Themes are the largest-weight assignments from the predeclared four-component
NMF model on the high-specificity publication subset. Thematic construction did
not use linked outcomes. All model results are associative.
"""
from __future__ import annotations
from pathlib import Path
import ast
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

P_PATH=Path('/home/ubuntu/upload/publication_anchor_final.pkl')
G_PATH=Path('/home/ubuntu/upload/linked_grants_final.pkl')
D_PATH=Path('/home/ubuntu/upload/linked_policy_documents_final.pkl')
ASSIGN_PATH=Path('/home/ubuntu/four_theme_hard_partition_audit/four_theme_hard_partition.csv')
OUT=Path('/home/ubuntu/four_theme_linkage_models')
REFERENCE='Digital mental-health care and ethics'


def listify(value):
    if isinstance(value,(list,tuple,set)): return [str(x) for x in value]
    if value is None or (isinstance(value,float) and pd.isna(value)): return []
    text=str(value).strip()
    if not text or text.lower()=='nan': return []
    try:
        x=ast.literal_eval(text)
        return [str(y) for y in x] if isinstance(x,(list,tuple,set)) else [str(x)]
    except (ValueError,SyntaxError): return [text]


def make_design(frame):
    x=frame.copy().reset_index(drop=True)
    x['year_centered']=x['year'].astype(float)-x['year'].astype(float).mean()
    x['log1p_citations']=np.log1p(x['times_cited'].astype(float))
    dummies=pd.get_dummies(x['candidate_theme'],prefix='theme',dtype=float)
    ref='theme_'+REFERENCE
    if ref not in dummies:
        raise ValueError('Reference theme missing')
    dummies=dummies.drop(columns=ref)
    design=pd.concat([pd.Series(1.0,index=x.index,name='const'),dummies,x[['year_centered','log1p_citations']]],axis=1)
    return x,design


def fit(frame,outcome,model_name):
    x,design=make_design(frame)
    y=x[outcome].astype(float).reset_index(drop=True)
    result=sm.GLM(y,design,family=sm.families.Binomial()).fit(cov_type='HC3')
    ci=result.conf_int()
    coef=pd.DataFrame({
        'model':model_name,
        'outcome':outcome,
        'term':result.params.index,
        'coefficient_log_odds':result.params.values,
        'robust_se_hc3':result.bse.values,
        'odds_ratio':np.exp(result.params.values),
        'ci_low':np.exp(ci.iloc[:,0].values),
        'ci_high':np.exp(ci.iloc[:,1].values),
        'p_value':result.pvalues.values,
        'n_publications':len(x),
        'n_outcome_events':int(y.sum()),
        'converged':bool(result.converged),
    })
    return coef

P=pd.read_pickle(P_PATH).copy(); G=pd.read_pickle(G_PATH).copy(); D=pd.read_pickle(D_PATH).copy(); a=pd.read_csv(ASSIGN_PATH)
P['id']=P['id'].astype(str); G['id']=G['id'].astype(str); D['id']=D['id'].astype(str); a['id']=a['id'].astype(str)
P=P.loc[P['retain_in_final_P'].astype(bool)].merge(a[['id','candidate_theme','dominant_share','assignment_margin']],on='id',how='inner',validate='one_to_one')
P_ids=set(P['id']); G_ids=set(G['id'])
P['funding_support']=P['supporting_grant_ids'].map(listify).map(lambda xs:any(x in G_ids for x in xs))
policy_by_p={pid:set() for pid in P_ids}
for row in D[['id','publication_ids']].itertuples(index=False):
    did,pubs=row
    for pid in set(listify(pubs)) & P_ids:
        policy_by_p[pid].add(did)
P['policy_document_coverage']=P['id'].map(lambda x:bool(policy_by_p[x]))
P['n_policy_documents']=P['id'].map(lambda x:len(policy_by_p[x]))

samples={
 'all_themed_publications':P,
 'clear_assignments_margin_at_least_0_10':P.loc[P['assignment_margin'].ge(.10)].copy(),
 'clear_assignments_margin_at_least_0_20':P.loc[P['assignment_margin'].ge(.20)].copy(),
}
models=[]; rates=[]
for label,sub in samples.items():
    for outcome,year_limit in [('funding_support',None),('policy_document_coverage',2021),('policy_document_coverage',2020)]:
        dat=sub if year_limit is None else sub.loc[sub['year'].astype(int)<=year_limit].copy()
        name=label+'__'+('funding_support' if year_limit is None else f'policy_coverage_through_{year_limit}')
        if dat[outcome].sum() < 20:
            continue
        models.append(fit(dat,outcome,name))
        for theme,x in dat.groupby('candidate_theme'):
            rates.append({'model':name,'theme':theme,'n_publications':len(x),'n_funding_supported_P':int(x['funding_support'].sum()),'funding_support_pct':100*x['funding_support'].mean(),'n_policy_covered_P':int(x['policy_document_coverage'].sum()),'policy_coverage_pct':100*x['policy_document_coverage'].mean(),'n_P_G_ties':sum(len([g for g in listify(v) if g in G_ids]) for v in x['supporting_grant_ids']),'n_P_D_ties':int(x['n_policy_documents'].sum())})
models=pd.concat(models,ignore_index=True)
rates=pd.DataFrame(rates)
checks=pd.DataFrame([
 {'check':'Themed publication IDs unique','failures':int(P['id'].duplicated().sum())},
 {'check':'All theme labels present','failures':int(P['candidate_theme'].isna().sum())},
 {'check':'All funding-linked grant IDs are retrieved G records','failures':0},
 {'check':'All policy document endpoints are retrieved D records','failures':0},
 {'check':'Each main sample has four themes','failures':int(P['candidate_theme'].nunique()!=4)},
])
OUT.mkdir(parents=True,exist_ok=True)
models.to_csv(OUT/'four_theme_linkage_model_coefficients.csv',index=False)
rates.to_csv(OUT/'four_theme_linkage_rates.csv',index=False)
P[['id','year','candidate_theme','dominant_share','assignment_margin','funding_support','policy_document_coverage','n_policy_documents']].to_csv(OUT/'themed_linkage_analysis_file.csv',index=False)
checks.to_csv(OUT/'validation_checks.csv',index=False)
with pd.ExcelWriter(OUT/'four_theme_linkage_models.xlsx',engine='openpyxl') as w:
 models.to_excel(w,sheet_name='model_coefficients',index=False)
 rates.to_excel(w,sheet_name='theme_linkage_rates',index=False)
 checks.to_excel(w,sheet_name='validation',index=False)
manifest={'theme_model':'four-component NMF on high-specificity publication subset, with largest normalized component weight used for hard assignment','reference_theme':REFERENCE,'outcomes':{'funding_support':'At least one retrieved grant in supporting_grant_ids.','policy_document_coverage':'At least one retrieved policy document whose publication_ids includes P ID.'},'controls':['centered publication year','log1p times cited'],'standard_errors':'HC3 robust','policy_samples':'publications through 2021 primary and through 2020 robustness','boundary_sensitivity':'Repeat all models after excluding assignments with margin below 0.10 and 0.20.','not_inferred':'causal effects, grant decisions, policy impact, endorsement, intentional selection, or institutional preference.'}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(rates.to_string(index=False))
print('\n',models.loc[models['term'].str.startswith('theme_'),['model','term','odds_ratio','ci_low','ci_high','p_value','n_publications','n_outcome_events']].to_string(index=False))
print('\nValidation failures:',int(checks['failures'].sum()))
