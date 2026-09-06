#!/usr/bin/env python3
"""Record-year ordering and two-mode structure for the high-specificity thematic corpus.

Descriptive only. Record years are database metadata, not causal lags. The two-mode
structure is summarized independently for P-G and P-D; no global community or latent
network geometry is estimated.
"""
from __future__ import annotations
import ast, json
from pathlib import Path
import numpy as np
import pandas as pd

P_PATH=Path('/home/ubuntu/upload/publication_anchor_final.pkl')
G_PATH=Path('/home/ubuntu/upload/linked_grants_final.pkl')
D_PATH=Path('/home/ubuntu/upload/linked_policy_documents_final.pkl')
A_PATH=Path('/home/ubuntu/four_theme_hard_partition_audit/four_theme_hard_partition.csv')
OUT=Path('/home/ubuntu/themed_timing_and_structure')

def listify(v):
    if isinstance(v,(list,tuple,set)): return [str(x) for x in v]
    if v is None or (isinstance(v,float) and pd.isna(v)): return []
    s=str(v).strip()
    if not s or s.lower()=='nan': return []
    try:
        x=ast.literal_eval(s)
        return [str(y) for y in x] if isinstance(x,(list,tuple,set)) else [str(x)]
    except (ValueError,SyntaxError): return [s]

def components(edges,left,right):
    """Union-find component summary for a two-mode edge set."""
    parent={x:x for x in set(left)|set(right)}
    size={x:1 for x in parent}
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a==b:return
        if size[a]<size[b]: a,b=b,a
        parent[b]=a;size[a]+=size[b]
    for a,b in edges: union(a,b)
    comp={}
    for x in parent: comp.setdefault(find(x),[]).append(x)
    sizes=sorted((len(x) for x in comp.values()),reverse=True)
    return len(comp),sizes[0] if sizes else 0,len(parent)

def summarize_diffs(df,label):
    x=df['difference'].dropna().astype(float)
    return {'relation':label,'n_ties_with_valid_years':len(x),'negative_n':int((x<0).sum()),'zero_n':int((x==0).sum()),'positive_n':int((x>0).sum()),'median':float(x.median()),'q1':float(x.quantile(.25)),'q3':float(x.quantile(.75)),'minimum':float(x.min()),'maximum':float(x.max())}

def main():
    P=pd.read_pickle(P_PATH).copy();G=pd.read_pickle(G_PATH).copy();D=pd.read_pickle(D_PATH).copy();A=pd.read_csv(A_PATH)
    for x in [P,G,D,A]:x['id']=x['id'].astype(str)
    P=P.loc[P['retain_in_final_P'].astype(bool)].merge(A[['id','candidate_theme']],on='id',how='inner',validate='one_to_one')
    P['year']=pd.to_numeric(P['year'],errors='coerce');G['start_year']=pd.to_numeric(G['start_year'],errors='coerce');D['year']=pd.to_numeric(D['year'],errors='coerce')
    pids=set(P.id);gids=set(G.id);pyear=P.set_index('id').year.to_dict();gstart=G.set_index('id').start_year.to_dict();dyear=D.set_index('id').year.to_dict();theme=P.set_index('id').candidate_theme.to_dict()
    pg=[]
    for r in P[['id','supporting_grant_ids']].itertuples(index=False):
        for gid in set(listify(r.supporting_grant_ids))&gids: pg.append((r.id,gid))
    PG=pd.DataFrame(pg,columns=['publication_id','grant_id']).drop_duplicates()
    PG['publication_year']=PG.publication_id.map(pyear);PG['grant_start_year']=PG.grant_id.map(gstart);PG['difference']=PG.grant_start_year-PG.publication_year;PG['candidate_theme']=PG.publication_id.map(theme)
    pdrows=[]
    for r in D[['id','publication_ids']].itertuples(index=False):
        for pid in set(listify(r.publication_ids))&pids:pdrows.append((pid,r.id))
    PD=pd.DataFrame(pdrows,columns=['publication_id','policy_document_id']).drop_duplicates()
    PD['publication_year']=PD.publication_id.map(pyear);PD['policy_year']=PD.policy_document_id.map(dyear);PD['difference']=PD.policy_year-PD.publication_year;PD['candidate_theme']=PD.publication_id.map(theme)
    timing=[summarize_diffs(PG,'P-G: grant start year minus publication year'),summarize_diffs(PD,'P-D: policy-document year minus publication year')]
    timing_theme=[]
    for rel,frame in [('P-G',PG),('P-D',PD)]:
        for t,sub in frame.groupby('candidate_theme'):
            y=summarize_diffs(sub,rel); y['candidate_theme']=t; timing_theme.append(y)
    structs=[]
    for label,frame,left,right in [('P-G funding-support two-mode network',PG,'publication_id','grant_id'),('P-D policy-document coverage two-mode network',PD,'publication_id','policy_document_id')]:
        edges=list(frame[[left,right]].itertuples(index=False,name=None));l=frame[left].unique();r=frame[right].unique();nc,largest,nodes=components(edges,l,r)
        ldeg=frame.groupby(left).size();rdeg=frame.groupby(right).size()
        structs.append({'network':label,'unique_ties':len(frame),'left_mode_connected_nodes':len(l),'right_mode_connected_nodes':len(r),'connected_nodes_total':nodes,'components':nc,'largest_component_nodes':largest,'largest_component_pct_connected_nodes':100*largest/nodes,'left_mode_mean_degree':float(ldeg.mean()),'right_mode_mean_degree':float(rdeg.mean()),'left_mode_max_degree':int(ldeg.max()),'right_mode_max_degree':int(rdeg.max())})
    checks=pd.DataFrame([{'check':'Themed P IDs unique','failures':int(P.id.duplicated().sum())},{'check':'P-G endpoints valid','failures':int((~PG.publication_id.isin(pids)).sum()+ (~PG.grant_id.isin(gids)).sum())},{'check':'P-D endpoints valid','failures':int((~PD.publication_id.isin(pids)).sum()+ (~PD.policy_document_id.isin(set(D.id))).sum())},{'check':'P-G valid year differences available','failures':int(PG.difference.isna().sum())},{'check':'P-D valid year differences available','failures':int(PD.difference.isna().sum())}])
    OUT.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(timing).to_csv(OUT/'themed_record_year_difference_summary.csv',index=False);pd.DataFrame(timing_theme).to_csv(OUT/'themed_record_year_difference_by_theme.csv',index=False);pd.DataFrame(structs).to_csv(OUT/'themed_two_mode_structure.csv',index=False);checks.to_csv(OUT/'validation_checks.csv',index=False)
    with pd.ExcelWriter(OUT/'themed_timing_and_two_mode_structure.xlsx',engine='openpyxl') as w:
        pd.DataFrame(timing).to_excel(w,sheet_name='year_differences',index=False);pd.DataFrame(timing_theme).to_excel(w,sheet_name='by_theme',index=False);pd.DataFrame(structs).to_excel(w,sheet_name='two_mode_structure',index=False);checks.to_excel(w,sheet_name='validation',index=False)
    (OUT/'manifest.json').write_text(json.dumps({'subset':'High-specificity thematic publication corpus','timing':'Grant start year minus publication year and policy-document record year minus publication year; descriptive ordering only, not causal lags.','network_method':'Components and degree statistics separately for P-G and P-D two-mode networks.','not_inferred':'causal time order, knowledge translation, network communities, or institutional intent.'},indent=2))
    print(pd.DataFrame(timing).to_string(index=False));print(pd.DataFrame(structs).to_string(index=False));print('validation failures',int(checks.failures.sum()))
if __name__=='__main__':main()
