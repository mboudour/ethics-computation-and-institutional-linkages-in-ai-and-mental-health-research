#!/usr/bin/env python3
"""Build an outcome-blind high-specificity AI-evidence subset for feasibility testing."""
from __future__ import annotations
from pathlib import Path
import ast
import json
import pandas as pd

INPUT=Path('/home/ubuntu/upload/publication_anchor_final.pkl')
OUT_DIR=Path('/home/ubuntu/high_specificity_ai_topic_test')
OUT=OUT_DIR/'publication_anchor_high_specificity_ai.pkl'
SPECIFIC={'artificial intelligence','machine learning','deep learning','large language model','llm'}

def to_list(value):
    if isinstance(value,(list,tuple,set)): return [str(x).strip().lower() for x in value]
    if value is None or (isinstance(value,float) and pd.isna(value)): return []
    text=str(value).strip()
    if not text or text.lower()=='nan': return []
    try:
        parsed=ast.literal_eval(text)
        return [str(x).strip().lower() for x in parsed] if isinstance(parsed,(list,tuple,set)) else [str(parsed).strip().lower()]
    except (SyntaxError,ValueError):
        return [text.lower()]

frame=pd.read_pickle(INPUT).copy()
frame=frame.loc[frame['retain_in_final_P'].astype(bool)].copy()
frame['ai_terms_anywhere']=frame['ai_ml_terms_anywhere'].map(to_list)
frame['has_specific_ai_phrase']=frame['ai_terms_anywhere'].map(lambda terms: any(term in SPECIFIC for term in terms))
frame['abstract_available']=frame['abstract'].fillna('').astype(str).str.strip().ne('')
# Candidate rule: an available abstract, core mental-health phrase in that
# abstract, and an AI/ML phrase more specific than neural-network alone.
frame['ai_terms_abstract']=frame['ai_ml_terms_abstract'].map(to_list)
frame['mh_terms_abstract']=frame['core_mental_health_terms_abstract'].map(to_list)
frame['specific_ai_in_abstract']=frame['ai_terms_abstract'].map(lambda terms: any(term in SPECIFIC for term in terms))
keep=frame['abstract_available'] & frame['mh_terms_abstract'].map(len).gt(0) & frame['specific_ai_in_abstract']
subset=frame.loc[keep].copy()
OUT_DIR.mkdir(parents=True,exist_ok=True)
subset.to_pickle(OUT)
summary=pd.DataFrame([
 {'candidate_subset':'current_publication_anchor','n_publications':len(frame),'pct_of_current_P':100.0},
 {'candidate_subset':'contains_at_least_one_specific_ai_phrase_anywhere','n_publications':int(frame['has_specific_ai_phrase'].sum()),'pct_of_current_P':100*float(frame['has_specific_ai_phrase'].mean())},
 {'candidate_subset':'strict_available_abstract_specific_ai_and_core_mh_in_abstract','n_publications':len(subset),'pct_of_current_P':100*float(len(subset))/len(frame)},
])
summary.to_csv(OUT_DIR/'high_specificity_ai_subset_summary.csv',index=False)
manifest={
 'input_records':int(len(frame)),
 'retained_records':int(len(subset)),
 'retained_pct':100*float(len(subset))/len(frame),
 'rule':'Available abstract contains at least one core mental-health phrase and at least one high-specificity AI phrase: artificial intelligence, machine learning, deep learning, large language model, or LLM. Neural network alone is not sufficient.',
 'not_used':'grants, policy documents, institutions, countries, citations, linkage outcomes, topic-model results, or thematic labels'
}
(OUT_DIR/'high_specificity_ai_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(summary.to_string(index=False))
print(json.dumps(manifest,indent=2))
