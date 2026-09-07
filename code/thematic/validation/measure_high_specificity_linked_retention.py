#!/usr/bin/env python3
"""Measure retained final P-G and P-D links for outcome-blind candidate publication subsets."""
from __future__ import annotations

from pathlib import Path
import ast
import json
import pandas as pd

P_PATH = Path('/home/ubuntu/upload/publication_anchor_final.pkl')
G_PATH = Path('/home/ubuntu/upload/linked_grants_final.pkl')
D_PATH = Path('/home/ubuntu/upload/linked_policy_documents_final.pkl')
OUT = Path('/home/ubuntu/high_specificity_linked_retention')
SPECIFIC = {'artificial intelligence','machine learning','deep learning','large language model','llm'}


def listify(value):
    if isinstance(value, (list, tuple, set)):
        return [str(x) for x in value if str(x).strip()]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() == 'nan':
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            return [str(x) for x in parsed if str(x).strip()]
        return [str(parsed)]
    except (SyntaxError, ValueError):
        return [text]

P = pd.read_pickle(P_PATH).copy()
G = pd.read_pickle(G_PATH).copy()
D = pd.read_pickle(D_PATH).copy()
P = P.loc[P['retain_in_final_P'].astype(bool)].copy()
for frame, name in [(P,'P'),(G,'G'),(D,'D')]:
    if 'id' not in frame.columns:
        raise SystemExit(f'{name} has no id column')
    frame['id'] = frame['id'].astype(str)
P_ids = set(P['id'])
G_ids = set(G['id'])
D_ids = set(D['id'])

for col in ['supporting_grant_ids','ai_ml_terms_abstract','core_mental_health_terms_abstract']:
    if col not in P.columns:
        raise SystemExit(f'P missing {col}')
for col in ['publication_ids']:
    if col not in D.columns:
        raise SystemExit(f'D missing {col}')

P['abstract_present'] = P['abstract'].fillna('').astype(str).str.strip().ne('')
P['specific_ai_abstract'] = P['ai_ml_terms_abstract'].map(listify).map(lambda terms: any(term.strip().lower() in SPECIFIC for term in terms))
P['core_mh_abstract'] = P['core_mental_health_terms_abstract'].map(listify).map(len).gt(0)
P['candidate_high_specificity'] = P['abstract_present'] & P['specific_ai_abstract'] & P['core_mh_abstract']

P['grant_ids_final'] = P['supporting_grant_ids'].map(listify).map(lambda xs: sorted(set(x for x in xs if x in G_ids)))
D['publication_ids_final'] = D['publication_ids'].map(listify).map(lambda xs: sorted(set(x for x in xs if x in P_ids)))

rows=[]
for subset_name, mask in [('current_P', pd.Series(True, index=P.index)), ('high_specificity_AI_and_core_MH_abstract', P['candidate_high_specificity'])]:
    subset=P.loc[mask].copy()
    ids=set(subset['id'])
    pg=[(pid,gid) for pid, grants in zip(subset['id'],subset['grant_ids_final']) for gid in grants]
    pd_edges=[(pid,did) for did, pubs in zip(D['id'],D['publication_ids_final']) for pid in pubs if pid in ids]
    pg_grants={g for _,g in pg}
    pd_docs={d for _,d in pd_edges}
    pd_pubs={p for p,_ in pd_edges}
    through_2021=set(subset.loc[subset['year'].astype(int)<=2021,'id'])
    pd_edges_2021=[(p,d) for p,d in pd_edges if p in through_2021]
    rows.append({
        'subset':subset_name,
        'P_records':len(subset),
        'P_pct_of_current':100*len(subset)/len(P),
        'grant_linked_P':sum(bool(x) for x in subset['grant_ids_final']),
        'P_G_ties':len(pg),
        'linked_G_records':len(pg_grants),
        'policy_linked_P':len(pd_pubs),
        'P_D_ties':len(pd_edges),
        'linked_D_records':len(pd_docs),
        'P_through_2021':len(through_2021),
        'policy_linked_P_through_2021':len({p for p,_ in pd_edges_2021}),
        'P_D_ties_through_2021':len(pd_edges_2021),
        'linked_D_records_through_2021':len({d for _,d in pd_edges_2021}),
    })

out=pd.DataFrame(rows)
OUT.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT/'high_specificity_linked_retention_summary.csv',index=False)
manifest={
    'purpose':'Measure P-G and P-D records retained by a predeclared outcome-blind high-specificity publication subset.',
    'candidate_rule':'available abstract with a saved explicit core mental-health phrase and a saved explicit AI phrase other than neural network alone',
    'specific_ai_phrases':sorted(SPECIFIC),
    'raw_input_counts':{'P':len(P),'G':len(G),'D':len(D)},
    'note':'Grant and policy links are measured only after the candidate subset is defined. They are not used to create it.'
}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(out.to_string(index=False))
print(f'Wrote {OUT}')
