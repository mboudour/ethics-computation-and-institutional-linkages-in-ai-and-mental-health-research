#!/usr/bin/env python3
"""Audit ambiguous versus specific saved AI/ML phrase evidence in the publication anchor."""
from __future__ import annotations
from pathlib import Path
import ast
import pandas as pd

INPUT=Path('/home/ubuntu/upload/publication_anchor_final.pkl')
OUT=Path('/home/ubuntu/ai_phrase_specificity_audit')

def values(value):
    if isinstance(value,(list,tuple,set)): return [str(x) for x in value]
    if value is None or (isinstance(value,float) and pd.isna(value)): return []
    try:
        x=ast.literal_eval(str(value))
        return [str(y) for y in x] if isinstance(x,(list,tuple,set)) else [str(x)]
    except (ValueError,SyntaxError): return [str(value)]

frame=pd.read_pickle(INPUT)
frame=frame.loc[frame['retain_in_final_P'].astype(bool)].copy()
frame['ai_terms']=frame['ai_ml_terms_anywhere'].map(values)
exploded=frame[['id','year','title','abstract_available','ai_terms']].explode('ai_terms')
exploded['ai_terms']=exploded['ai_terms'].fillna('').astype(str).str.strip().str.lower()
counts=exploded.loc[exploded['ai_terms'].ne(''),'ai_terms'].value_counts().rename_axis('saved_ai_phrase').reset_index(name='n_publications_with_phrase')
print(counts.to_string(index=False))
OUT.mkdir(parents=True,exist_ok=True)
counts.to_csv(OUT/'saved_ai_phrase_prevalence.csv',index=False)
