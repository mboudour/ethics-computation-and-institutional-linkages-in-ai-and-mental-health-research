#!/usr/bin/env python3
"""Count publication-to-grant support volume under outcome-blind candidate subsets."""
from __future__ import annotations
from pathlib import Path
import ast
import pandas as pd

SOURCE=Path('/home/ubuntu/upload/publication_anchor_final.pkl')
HIGH=Path('/home/ubuntu/high_specificity_ai_topic_test/publication_anchor_high_specificity_ai.pkl')
STRICT=Path('/home/ubuntu/strict_literal_topic_test/publication_anchor_strict_abstract.pkl')
OBJECTIVE=Path('/home/ubuntu/objective_clean_topic_test/publication_anchor_objective_clean.pkl')

def ids(value):
    if isinstance(value,(list,tuple,set)): return [str(x) for x in value if str(x).strip()]
    if value is None or (isinstance(value,float) and pd.isna(value)): return []
    text=str(value).strip()
    if not text or text.lower()=='nan': return []
    try:
        data=ast.literal_eval(text)
        return [str(x) for x in data] if isinstance(data,(list,tuple,set)) else [str(data)]
    except (ValueError,SyntaxError): return [text]

rows=[]
for name,path in [('current_P',SOURCE),('objective_clean',OBJECTIVE),('strict_abstract',STRICT),('high_specificity_ai',HIGH)]:
    f=pd.read_pickle(path)
    f=f.loc[f['retain_in_final_P'].astype(bool)].copy()
    grants=f['supporting_grant_ids'].map(ids)
    rows.append({'candidate_subset':name,'n_publications':len(f),'grant_linked_publications':int(grants.map(bool).sum()),'pct_grant_linked':100*float(grants.map(bool).mean()),'P_G_ties':int(grants.map(len).sum()),'distinct_grant_ids':int(len({grant for value in grants for grant in value}))})
out=pd.DataFrame(rows)
out.to_csv('/home/ubuntu/candidate_subset_grant_volume.csv',index=False)
print(out.to_string(index=False))
