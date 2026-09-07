#!/usr/bin/env python3
"""Outcome-blind audit of objective publication-record cleaning candidates.

This script does not create a final filtered corpus. It identifies only records
that can be flagged by transparent non-substantive record markers or by missing
substantive abstract text. It never reads grants, policies, institutions,
countries, citations, or linkage outcomes.
"""
from __future__ import annotations

from pathlib import Path
import json
import re
import pandas as pd

INPUT = Path('/home/ubuntu/upload/publication_anchor_final.pkl')
OUT = Path('/home/ubuntu/publication_cleaning_candidate_audit')

ARTIFACT_RULES = {
    'retraction_or_removal_notice': r'\b(retracted|retraction|removed|withdrawn)\b',
    'correction_or_erratum': r'\b(correction|corrigendum|erratum|errata)\b',
    'issue_or_cover_matter': r'\b(issue\s+cover|cover\s+image|cover\s+page)\b',
    'meeting_or_programme_matter': r'\b(poster\s+session|conference\s+programme|conference\s+program|investigators[’\']?\s+workshop)\b',
}

frame = pd.read_pickle(INPUT).copy()
for col in ('id', 'title', 'abstract', 'retain_in_final_P'):
    if col not in frame.columns:
        raise SystemExit(f'Missing required column: {col}')
frame = frame.loc[frame['retain_in_final_P'].astype(bool)].copy()
frame['title_text'] = frame['title'].fillna('').astype(str).str.strip()
frame['abstract_text'] = frame['abstract'].fillna('').astype(str).str.strip()
frame['has_abstract'] = frame['abstract_text'].ne('')
frame['abstract_length'] = frame['abstract_text'].str.len()
for label, pattern in ARTIFACT_RULES.items():
    frame[label] = frame['title_text'].str.contains(pattern, flags=re.I, regex=True, na=False)
artifact_columns = list(ARTIFACT_RULES)
frame['any_objective_artifact_marker'] = frame[artifact_columns].any(axis=1)
frame['candidate_objective_clean_subset'] = frame['has_abstract'] & ~frame['any_objective_artifact_marker']

rows = []
def add_rule(name: str, mask: pd.Series, description: str) -> None:
    rows.append({'candidate_rule': name, 'n_flagged_or_retained': int(mask.sum()), 'pct_of_P': 100*float(mask.mean()), 'description': description})

add_rule('all_publications', pd.Series(True, index=frame.index), 'Current retained publication anchor.')
add_rule('abstract_available', frame['has_abstract'], 'Title plus available abstract can be inspected.')
add_rule('abstract_at_least_150_characters', frame['abstract_length'].ge(150), 'Excludes title-only and extremely short abstract records.')
add_rule('any_objective_artifact_marker', frame['any_objective_artifact_marker'], 'Flag only; not automatic exclusion from the publication anchor.')
add_rule('objective_clean_subset', frame['candidate_objective_clean_subset'], 'Abstract available and no objective title artifact marker.')
summary = pd.DataFrame(rows)

flagged = frame.loc[frame['any_objective_artifact_marker'] | ~frame['has_abstract'], ['id','year','title','has_abstract','abstract_length'] + artifact_columns].copy()
flagged.insert(0, 'candidate_status', 'flagged_for_transparent_review')
flagged = flagged.sort_values(['has_abstract','year','id'], ascending=[True, False, True])

samples = []
for label in artifact_columns:
    subset = frame.loc[frame[label], ['id','year','title','abstract_length','has_abstract']].sort_values(['year','id'], ascending=[False, True]).head(100).copy()
    subset.insert(0, 'flag_reason', label)
    samples.append(subset)
no_abs = frame.loc[~frame['has_abstract'], ['id','year','title','abstract_length']].sort_values(['year','id'], ascending=[False, True]).head(100).copy()
no_abs.insert(0, 'flag_reason', 'no_abstract')
samples.append(no_abs)
samples = pd.concat(samples, ignore_index=True)

OUT.mkdir(parents=True, exist_ok=True)
summary.to_csv(OUT/'cleaning_candidate_summary.csv', index=False)
flagged.to_csv(OUT/'objective_record_flags.csv', index=False)
samples.to_csv(OUT/'flag_review_samples.csv', index=False)
with pd.ExcelWriter(OUT/'publication_cleaning_candidate_audit.xlsx', engine='openpyxl') as writer:
    summary.to_excel(writer, sheet_name='summary', index=False)
    flagged.to_excel(writer, sheet_name='objective_flags', index=False)
    samples.to_excel(writer, sheet_name='review_samples', index=False)
manifest = {
    'purpose': 'Outcome-blind audit of objective record-quality cleaning candidates.',
    'input': str(INPUT),
    'n_publications': int(len(frame)),
    'objective_artifact_rules': ARTIFACT_RULES,
    'not_used': 'grants, policy documents, institutions, countries, citations, linkage outcomes, NMF results, and thematic labels',
    'boundary': 'The audit identifies transparent flags. It does not infer that every flagged record is out of scope or create a final subset.'
}
(OUT/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(summary.to_string(index=False))
print(f'Output directory: {OUT}')
