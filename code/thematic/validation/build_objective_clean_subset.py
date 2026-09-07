#!/usr/bin/env python3
"""Build the objective-cleaning subset used for topic-model feasibility testing."""
from __future__ import annotations
from pathlib import Path
import json
import re
import pandas as pd

INPUT = Path('/home/ubuntu/upload/publication_anchor_final.pkl')
OUT_DIR = Path('/home/ubuntu/objective_clean_topic_test')
OUT = OUT_DIR / 'publication_anchor_objective_clean.pkl'

RULES = {
    'retraction_or_removal_notice': r'\b(?:retracted|retraction|removed|withdrawn)\b',
    'correction_or_erratum': r'\b(?:correction|corrigendum|erratum|errata)\b',
    'issue_or_cover_matter': r'\b(?:issue\s+cover|cover\s+image|cover\s+page)\b',
    'meeting_or_programme_matter': r'\b(?:poster\s+session|conference\s+programme|conference\s+program|investigators[’\']?\s+workshop)\b',
}

frame = pd.read_pickle(INPUT).copy()
frame = frame.loc[frame['retain_in_final_P'].astype(bool)].copy()
title = frame['title'].fillna('').astype(str).str.strip()
abstract = frame['abstract'].fillna('').astype(str).str.strip()
artifact = pd.Series(False, index=frame.index)
for pattern in RULES.values():
    artifact |= title.str.contains(pattern, flags=re.I, regex=True, na=False)
keep = abstract.ne('') & ~artifact
clean = frame.loc[keep].copy()
if clean.empty:
    raise SystemExit('Objective-clean subset is empty.')
OUT_DIR.mkdir(parents=True, exist_ok=True)
clean.to_pickle(OUT)
manifest = {
    'input_records': int(len(frame)),
    'retained_records': int(len(clean)),
    'retained_pct': 100 * float(len(clean)) / len(frame),
    'rule': 'available abstract and no transparent title marker for retraction/removal, correction/erratum, issue/cover matter, or meeting/programme matter',
    'rules': RULES,
    'not_used': 'grants, policy documents, institutions, countries, citations, linkage outcomes, topic-model results, or thematic labels',
}
(OUT_DIR / 'objective_clean_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(json.dumps(manifest, indent=2))
