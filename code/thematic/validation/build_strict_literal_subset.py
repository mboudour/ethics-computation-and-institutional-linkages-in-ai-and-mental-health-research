#!/usr/bin/env python3
"""Build a strict abstract-evidence subset from saved final-screen match fields."""
from __future__ import annotations
from pathlib import Path
import ast
import json
import pandas as pd

INPUT = Path('/home/ubuntu/upload/publication_anchor_final.pkl')
OUT_DIR = Path('/home/ubuntu/strict_literal_topic_test')
OUT = OUT_DIR/'publication_anchor_strict_abstract.pkl'

def to_list(value):
    if isinstance(value,(list,tuple,set)):
        return list(value)
    if value is None or (isinstance(value,float) and pd.isna(value)):
        return []
    text=str(value).strip()
    if not text or text.lower()=='nan':
        return []
    try:
        parsed=ast.literal_eval(text)
        return list(parsed) if isinstance(parsed,(list,tuple,set)) else [str(parsed)]
    except (ValueError,SyntaxError):
        return [text]

frame=pd.read_pickle(INPUT).copy()
frame=frame.loc[frame['retain_in_final_P'].astype(bool)].copy()
abstract_available=frame['abstract'].fillna('').astype(str).str.strip().ne('')
ai_abstract=frame['ai_ml_terms_abstract'].map(to_list).map(len).gt(0)
mh_abstract=frame['core_mental_health_terms_abstract'].map(to_list).map(len).gt(0)
keep=abstract_available & ai_abstract & mh_abstract
subset=frame.loc[keep].copy()
OUT_DIR.mkdir(parents=True,exist_ok=True)
subset.to_pickle(OUT)
manifest={
 'input_records':int(len(frame)),
 'retained_records':int(len(subset)),
 'retained_pct':100*float(len(subset))/len(frame),
 'rule':'available abstract containing at least one saved explicit AI/ML phrase and at least one saved explicit core mental-health phrase',
 'not_used':'grants, policy documents, institutions, countries, citations, linkage outcomes, NMF results, or thematic labels',
}
(OUT_DIR/'strict_literal_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(manifest,indent=2))
