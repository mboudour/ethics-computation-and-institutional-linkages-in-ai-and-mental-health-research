#!/usr/bin/env python3
"""Re-estimate publication-level funding-support/policy-coverage alignment in the high-specificity four-theme subset.

The null keeps every P-G funding-support tie fixed and reassigns each complete
P-D policy-document incidence profile within the publication-year stratum. It
therefore preserves P-D degree profiles, D endpoints, and D-DI issuer links.
Themes are not used in the permutations; they describe observed configurations.
"""
from __future__ import annotations
from pathlib import Path
import ast
import json
import numpy as np
import pandas as pd

P_PATH=Path('/home/ubuntu/upload/publication_anchor_final.pkl')
G_PATH=Path('/home/ubuntu/upload/linked_grants_final.pkl')
D_PATH=Path('/home/ubuntu/upload/linked_policy_documents_final.pkl')
ASSIGN_PATH=Path('/home/ubuntu/four_theme_hard_partition_audit/four_theme_hard_partition.csv')
OUT=Path('/home/ubuntu/four_theme_local_alignment')
SEED=20260902
N_PERM=10000


def listify(x):
 if isinstance(x,(list,tuple,set)): return [str(v) for v in x]
 if x is None or (isinstance(x,float) and pd.isna(x)): return []
 s=str(x).strip()
 if not s or s.lower()=='nan': return []
 try:
  y=ast.literal_eval(s); return [str(v) for v in y] if isinstance(y,(list,tuple,set)) else [str(y)]
 except (SyntaxError,ValueError): return [s]

def two_sided(obs,values):
 upper=(1+int((values>=obs).sum()))/(1+len(values)); lower=(1+int((values<=obs).sum()))/(1+len(values))
 return min(1.0,2*min(upper,lower))
def upper(obs,values): return (1+int((values>=obs).sum()))/(1+len(values))
def config(pg,pd):
 out=np.full(len(pg),'neither',dtype=object)
 out[(pg>0)&(pd==0)]='funding_only'; out[(pg==0)&(pd>0)]='policy_only'; out[(pg>0)&(pd>0)]='both'
 return out

P=pd.read_pickle(P_PATH); G=pd.read_pickle(G_PATH); D=pd.read_pickle(D_PATH); a=pd.read_csv(ASSIGN_PATH)
for f in [P,G,D,a]: f['id']=f['id'].astype(str)
P=P.loc[P['retain_in_final_P'].astype(bool)].merge(a[['id','candidate_theme','dominant_share','assignment_margin']],on='id',how='inner',validate='one_to_one').reset_index(drop=True)
p_ids=P['id'].to_numpy(); p_index={x:i for i,x in enumerate(p_ids)}; g_ids=set(G['id'])
pg=[]
for row in P[['id','supporting_grant_ids']].itertuples(index=False):
 for gid in set(listify(row.supporting_grant_ids)) & g_ids: pg.append((row.id,gid))
pg=pd.DataFrame(pg,columns=['publication_id','grant_id']).drop_duplicates()
pd_rows=[]
for row in D[['id','publication_ids']].itertuples(index=False):
 for pid in set(listify(row.publication_ids)) & set(p_ids): pd_rows.append((pid,row.id))
pd_edges=pd.DataFrame(pd_rows,columns=['publication_id','policy_document_id']).drop_duplicates()
pg_ix=pg['publication_id'].map(p_index).to_numpy(dtype=int); pd_ix=pd_edges['publication_id'].map(p_index).to_numpy(dtype=int)
pg_deg=np.bincount(pg_ix,minlength=len(P)); pd_deg=np.bincount(pd_ix,minlength=len(P)); obs_config=config(pg_deg,pd_deg)
P['funding_support_degree']=pg_deg; P['policy_document_coverage_degree']=pd_deg; P['configuration']=obs_config
summary=(P.groupby(['candidate_theme','configuration'],as_index=False).agg(n_publications=('id','size'),mean_funding_support_degree=('funding_support_degree','mean'),mean_policy_coverage_degree=('policy_document_coverage_degree','mean')))
totals=(P.groupby('configuration',as_index=False).size().rename(columns={'size':'n_publications'})); totals['pct_of_themed_subset']=100*totals['n_publications']/len(P)
obs_both=int((obs_config=='both').sum()); obs_intensive=int((pg_deg*pd_deg).sum())
# Profile reassignment by publication year preserves the P-D degree profile multiset and document endpoints.
years=P['year'].astype(int).to_numpy(); rng=np.random.default_rng(SEED); n_both=[]; intensive=[]
for replicate in range(N_PERM):
 targets=np.arange(len(P))
 for year in np.unique(years):
  ix=np.where(years==year)[0]
  if len(ix)>1: targets[ix]=rng.permutation(ix)
 perm_pd=np.bincount(targets[pd_ix],minlength=len(P))
 labels=config(pg_deg,perm_pd)
 n_both.append(int((labels=='both').sum())); intensive.append(int((pg_deg*perm_pd).sum()))
n_both=np.asarray(n_both); intensive=np.asarray(intensive)
null=pd.DataFrame({'replicate':np.arange(1,N_PERM+1),'N_both':n_both,'intensive_degree_product_sum':intensive})
results=pd.DataFrame([
 {'statistic':'N publications with both funding-support and policy-document coverage','observed':obs_both,'null_mean':float(n_both.mean()),'null_2_5':float(np.quantile(n_both,.025)),'null_97_5':float(np.quantile(n_both,.975)),'enrichment':float(obs_both/n_both.mean()),'two_sided_p':two_sided(obs_both,n_both),'upper_tail_p':upper(obs_both,n_both)},
 {'statistic':'Sum of publication funding-support degree × policy-coverage degree','observed':obs_intensive,'null_mean':float(intensive.mean()),'null_2_5':float(np.quantile(intensive,.025)),'null_97_5':float(np.quantile(intensive,.975)),'enrichment':float(obs_intensive/intensive.mean()),'two_sided_p':two_sided(obs_intensive,intensive),'upper_tail_p':upper(obs_intensive,intensive)},
])
checks=pd.DataFrame([
 {'check':'Themed publication IDs unique','failures':int(P['id'].duplicated().sum())},
 {'check':'All P-G ties have themed P endpoint','failures':int((~pg['publication_id'].isin(p_index)).sum())},
 {'check':'All P-D ties have themed P endpoint','failures':int((~pd_edges['publication_id'].isin(p_index)).sum())},
 {'check':'All four themes represented','failures':int(P['candidate_theme'].nunique()!=4)},
 {'check':'All publication years available for stratification','failures':int(P['year'].isna().sum())},
])
OUT.mkdir(parents=True,exist_ok=True)
P[['id','year','candidate_theme','dominant_share','assignment_margin','funding_support_degree','policy_document_coverage_degree','configuration']].to_csv(OUT/'themed_publication_configurations.csv',index=False)
summary.to_csv(OUT/'configuration_by_theme.csv',index=False); totals.to_csv(OUT/'configuration_totals.csv',index=False); null.to_csv(OUT/'alignment_null_replicates.csv',index=False); results.to_csv(OUT/'alignment_null_summary.csv',index=False); checks.to_csv(OUT/'validation_checks.csv',index=False)
with pd.ExcelWriter(OUT/'four_theme_local_alignment.xlsx',engine='openpyxl') as w:
 results.to_excel(w,sheet_name='alignment_results',index=False); summary.to_excel(w,sheet_name='by_theme',index=False); totals.to_excel(w,sheet_name='overall_configurations',index=False); checks.to_excel(w,sheet_name='validation',index=False)
manifest={'subset':'High-specificity AI-and-core-mental-health publication subset with four-theme hard NMF assignment.','primary_null':'Complete P-D incidence profiles reassigned within publication-year strata; P-G ties fixed; policy-document endpoints and issuer attributes retained.','permutations':N_PERM,'seed':SEED,'themes':'Descriptive only; no theme result enters the null.','not_inferred':'causation, funding decisions, policy impact, endorsement, intentional selection, or institutional preference.'}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(results.to_string(index=False)); print('\nValidation failures:',int(checks['failures'].sum()))
