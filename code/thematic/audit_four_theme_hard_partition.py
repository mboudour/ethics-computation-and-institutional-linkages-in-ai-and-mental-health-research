#!/usr/bin/env python3
"""Audit the hard four-theme NMF partition and boundary ambiguity.

Uses the outcome-blind four-component NMF assignment file. It does not use
funding, policy, institution, country, citation, or linkage outcomes.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

INPUT=Path('/home/ubuntu/four_theme_evidence_packets/four_component_hard_assignments.csv')
OUT=Path('/home/ubuntu/four_theme_hard_partition_audit')
LABELS={
  1:'Mental-health risk prediction',
  2:'Digital mental-health care and ethics',
  3:'Social and affective detection',
  4:'Neuropsychiatric assessment',
}

a=pd.read_csv(INPUT)
required=['id','dominant_component','dominant_share','assignment_margin']+[f'component_{i}_share' for i in range(1,5)]
missing=[c for c in required if c not in a]
if missing: raise SystemExit('Missing columns: '+', '.join(missing))
a['candidate_theme']=a['dominant_component'].map(LABELS)
a['boundary_flag_under_0_10']=a['assignment_margin'].lt(.10)
a['boundary_flag_under_0_20']=a['assignment_margin'].lt(.20)
a['assignment_clarity']=np.select(
 [a['assignment_margin'].lt(.10),a['assignment_margin'].lt(.20)],
 ['highly_ambiguous_margin_under_0_10','moderately_ambiguous_margin_0_10_to_under_0_20'],
 default='clear_margin_at_least_0_20')
shares=a[[f'component_{i}_share' for i in range(1,5)]].to_numpy()
with np.errstate(divide='ignore', invalid='ignore'):
    entropy_terms = np.where(shares > 0, shares * np.log(shares), 0.0)
a['normalized_entropy']=-(entropy_terms.sum(axis=1)/np.log(4))

theme_summary=(a.groupby(['dominant_component','candidate_theme'],as_index=False).agg(
 n_publications=('id','size'),
 pct_of_subset=('id',lambda s:100*len(s)/len(a)),
 mean_dominant_share=('dominant_share','mean'),
 median_dominant_share=('dominant_share','median'),
 mean_assignment_margin=('assignment_margin','mean'),
 median_assignment_margin=('assignment_margin','median'),
 pct_margin_under_0_10=('boundary_flag_under_0_10','mean'),
 pct_margin_under_0_20=('boundary_flag_under_0_20','mean'),
 mean_normalized_entropy=('normalized_entropy','mean'),
).sort_values('dominant_component'))
for col in ['pct_margin_under_0_10','pct_margin_under_0_20']:
    theme_summary[col]*=100
clarity=(a['assignment_clarity'].value_counts(dropna=False).rename_axis('assignment_clarity').reset_index(name='n_publications'))
clarity['pct_of_subset']=100*clarity['n_publications']/len(a)

# Fixed reproducible review samples: the highest membership, closest-to-median
# membership, and most ambiguous rows per component.
review=[]
for component,label in LABELS.items():
    x=a.loc[a['dominant_component'].eq(component)].copy()
    high=x.nlargest(20,'dominant_share').copy(); high.insert(0,'sample_type','highest_membership')
    med=x['dominant_share'].median(); typical=x.assign(_d=(x['dominant_share']-med).abs()).nsmallest(20,'_d').drop(columns='_d').copy(); typical.insert(0,'sample_type','typical_membership')
    boundary=x.nsmallest(20,'assignment_margin').copy(); boundary.insert(0,'sample_type','smallest_margin')
    for z in [high,typical,boundary]:
        review.append(z)
review=pd.concat(review,ignore_index=True)

checks=pd.DataFrame([
 {'check':'One hard assignment per publication','failures':int(a['dominant_component'].isna().sum())},
 {'check':'Hard assignment is in 1 through 4','failures':int((~a['dominant_component'].isin(LABELS)).sum())},
 {'check':'Theme labels are complete','failures':int(a['candidate_theme'].isna().sum())},
 {'check':'All component shares are nonnegative','failures':int((shares<-1e-12).any(axis=1).sum())},
 {'check':'All assignment margins are nonnegative','failures':int(a['assignment_margin'].lt(-1e-12).sum())},
])
OUT.mkdir(parents=True,exist_ok=True)
theme_summary.to_csv(OUT/'four_theme_partition_summary.csv',index=False)
clarity.to_csv(OUT/'four_theme_assignment_clarity.csv',index=False)
a.to_csv(OUT/'four_theme_hard_partition.csv',index=False)
review.to_csv(OUT/'four_theme_partition_review_samples.csv',index=False)
checks.to_csv(OUT/'validation_checks.csv',index=False)
with pd.ExcelWriter(OUT/'four_theme_hard_partition_audit.xlsx',engine='openpyxl') as w:
 theme_summary.to_excel(w,sheet_name='theme_summary',index=False)
 clarity.to_excel(w,sheet_name='assignment_clarity',index=False)
 a.to_excel(w,sheet_name='hard_partition',index=False)
 review.to_excel(w,sheet_name='review_samples',index=False)
 checks.to_excel(w,sheet_name='validation',index=False)
manifest={'purpose':'Outcome-blind hard-partition and boundary-ambiguity audit for four-component NMF solution.','hard_assignment':'Largest normalized NMF component weight.','candidate_labels':LABELS,'ambiguity_flags':{'highly_ambiguous':'assignment margin < 0.10','moderately_ambiguous':'assignment margin from 0.10 to <0.20'},'not_used':'grants, policy documents, institutions, countries, citations, or linkage outcomes.'}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(theme_summary.to_string(index=False))
print('\n'+clarity.to_string(index=False))
print('\nValidation failures:',int(checks['failures'].sum()))
