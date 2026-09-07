#!/usr/bin/env python3
"""Audit stricter literal evidence subsets within the final publication anchor.

The purpose is feasibility testing only. It uses the saved literal phrase-match
fields created during publication construction; it does not inspect grants,
policy documents, institutions, countries, citations, or outcomes.
"""
from __future__ import annotations
from pathlib import Path
import ast
import json
import pandas as pd

INPUT = Path('/home/ubuntu/upload/publication_anchor_final.pkl')
OUT = Path('/home/ubuntu/literal_evidence_subset_audit')


def as_list(value):
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() == 'nan':
        return []
    try:
        parsed = ast.literal_eval(text)
        return list(parsed) if isinstance(parsed, (list, tuple, set)) else [str(parsed)]
    except (SyntaxError, ValueError):
        return [text]

frame = pd.read_pickle(INPUT).copy()
frame = frame.loc[frame['retain_in_final_P'].astype(bool)].copy()
required = ['ai_ml_terms_title','ai_ml_terms_abstract','core_mental_health_terms_title','core_mental_health_terms_abstract']
missing = [c for c in required if c not in frame.columns]
if missing:
    raise SystemExit('Missing saved screening fields: ' + ', '.join(missing))
for col in required:
    frame[col + '_n'] = frame[col].map(as_list).map(len)
frame['ai_title'] = frame['ai_ml_terms_title_n'].gt(0)
frame['ai_abstract'] = frame['ai_ml_terms_abstract_n'].gt(0)
frame['mh_title'] = frame['core_mental_health_terms_title_n'].gt(0)
frame['mh_abstract'] = frame['core_mental_health_terms_abstract_n'].gt(0)
frame['has_abstract'] = frame['abstract'].fillna('').astype(str).str.strip().ne('')

rules = {
    'current_screen': pd.Series(True, index=frame.index),
    'both_explicit_terms_in_available_abstract': frame['has_abstract'] & frame['ai_abstract'] & frame['mh_abstract'],
    'both_explicit_terms_in_title_or_available_abstract': (frame['ai_title'] | frame['ai_abstract']) & (frame['mh_title'] | frame['mh_abstract']),
    'both_explicit_terms_in_title_plus_abstract_with_abstract_available': frame['has_abstract'] & (frame['ai_title'] | frame['ai_abstract']) & (frame['mh_title'] | frame['mh_abstract']),
    'both_explicit_terms_in_title': frame['ai_title'] & frame['mh_title'],
}
rows=[]
for name, mask in rules.items():
    rows.append({
        'candidate_subset':name,
        'n_publications':int(mask.sum()),
        'pct_of_current_P':100*float(mask.mean()),
        'with_abstract':int((mask & frame['has_abstract']).sum()),
        'pct_with_abstract':100*float(frame.loc[mask,'has_abstract'].mean()) if mask.any() else 0.0,
        'median_abstract_length':float(frame.loc[mask,'abstract'].fillna('').astype(str).str.len().median()) if mask.any() else 0.0,
    })
summary=pd.DataFrame(rows)

# A reproducible sample of records excluded only by the strict abstract-evidence screen.
strict = rules['both_explicit_terms_in_available_abstract']
excluded = frame.loc[~strict, ['id','year','title','abstract','abstract_available','ai_ml_terms_title','ai_ml_terms_abstract','core_mental_health_terms_title','core_mental_health_terms_abstract']].copy()
excluded['abstract'] = excluded['abstract'].fillna('').astype(str).str.slice(0, 1000)
excluded['exclusion_pattern'] = (
    (~excluded['abstract_available'].astype(bool)).map({True:'no_abstract',False:''}) +
    excluded['ai_ml_terms_abstract'].map(as_list).map(lambda x: '' if x else '|no_ai_term_in_abstract') +
    excluded['core_mental_health_terms_abstract'].map(as_list).map(lambda x: '' if x else '|no_mh_term_in_abstract')
).str.strip('|')
review_parts = []
for pattern, group in excluded.groupby('exclusion_pattern', sort=True):
    sample = group.sample(min(25, len(group)), random_state=20260902).copy()
    review_parts.append(sample)
review = pd.concat(review_parts, ignore_index=True).sort_values(['exclusion_pattern', 'id']) if review_parts else excluded.head(0).copy()

OUT.mkdir(parents=True,exist_ok=True)
summary.to_csv(OUT/'literal_evidence_subset_summary.csv',index=False)
review.to_csv(OUT/'strict_abstract_screen_exclusion_samples.csv',index=False)
with pd.ExcelWriter(OUT/'literal_evidence_subset_audit.xlsx',engine='openpyxl') as writer:
    summary.to_excel(writer,sheet_name='subset_summary',index=False)
    review.to_excel(writer,sheet_name='excluded_samples',index=False)
manifest={
 'purpose':'Outcome-blind feasibility audit of stricter literal title/abstract evidence rules within current P.',
 'input':str(INPUT),
 'rules':{
  'both_explicit_terms_in_available_abstract':'Available abstract contains at least one saved explicit AI/ML phrase and at least one saved explicit core mental-health phrase.',
  'both_explicit_terms_in_title_plus_abstract_with_abstract_available':'Abstract is available and each phrase block appears in title or abstract.'
 },
 'not_used':'grants, policies, institutions, countries, citations, linkage outcomes, NMF results, or topical labels'
}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(summary.to_string(index=False))
print(f'Output directory: {OUT}')
