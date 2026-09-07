#!/usr/bin/env python3
"""Inventory organization and country fields in final linked grants and policy documents."""
from pathlib import Path
import json
import pandas as pd

paths={
 'publications':Path('/home/ubuntu/upload/publication_anchor_final.pkl'),
 'grants':Path('/home/ubuntu/upload/linked_grants_final.pkl'),
 'policy_documents':Path('/home/ubuntu/upload/linked_policy_documents_final.pkl'),
}
out={}
for name,path in paths.items():
 f=pd.read_pickle(path)
 cols=[c for c in f.columns if any(k in c.lower() for k in ['org','institution','funder','publisher','country','author','publication_id'])]
 examples={}
 for c in cols:
  s=f[c].dropna()
  examples[c]=str(s.iloc[0])[:1000] if len(s) else None
 out[name]={'n_records':len(f),'relevant_columns':cols,'examples':examples}
Path('/home/ubuntu/final_organization_field_inventory.json').write_text(json.dumps(out,indent=2,default=str),encoding='utf-8')
print(json.dumps(out,indent=2,default=str))
