#!/usr/bin/env python3
"""Create outcome-blind evidence packets for interpreting a four-component NMF solution.

This script reproduces the high-specificity subset and the previously used TF-IDF
and NMF settings. It does not use grants, policy documents, institutions,
countries, citations, or linkage outcomes.
"""
from __future__ import annotations
from pathlib import Path
import ast
import json
import re
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

INPUT=Path('/home/ubuntu/upload/publication_anchor_final.pkl')
OUT=Path('/home/ubuntu/four_theme_evidence_packets')
SEED=20260902
SPECIFIC={'artificial intelligence','machine learning','deep learning','large language model','llm'}
ANCHOR=(r'\bartificial\s+intelligence\b',r'\bmachine\s+learning\b',r'\bmental\s+health\b',r'\bmental\s+illness\b')
CUSTOM={
'abstract','aim','analysis','approach','article','articles','author','background','case','cases','conclusion','conclusions','data','dataset','datasets','design','discussion','effect','effects','finding','findings','health','intelligence','learning','machine','method','methods','model','models','objective','objectives','paper','patient','patients','research','result','results','review','study','studies','system','systems','use','using','used','mental','artificial','machinelearning','artificialintelligence','mentalhealth'}
STOP=set(ENGLISH_STOP_WORDS).union(CUSTOM)

def listify(value):
    if isinstance(value,(list,tuple,set)): return [str(x).lower().strip() for x in value]
    if value is None or (isinstance(value,float) and pd.isna(value)): return []
    text=str(value).strip()
    if not text or text.lower()=='nan': return []
    try:
        parsed=ast.literal_eval(text)
        return [str(x).lower().strip() for x in parsed] if isinstance(parsed,(list,tuple,set)) else [str(parsed).lower().strip()]
    except (ValueError,SyntaxError): return [text.lower()]

def doc_text(title,abstract):
    title='' if pd.isna(title) else str(title).strip()
    abstract='' if pd.isna(abstract) else str(abstract).strip()
    text=' '.join(x for x in [title,title,abstract] if x)
    for phrase in ANCHOR:
        text=re.sub(phrase,' ',text,flags=re.I)
    return text

P=pd.read_pickle(INPUT).copy()
P=P.loc[P['retain_in_final_P'].astype(bool)].copy()
P['abstract_text']=P['abstract'].fillna('').astype(str).str.strip()
P['ai_terms_abstract']=P['ai_ml_terms_abstract'].map(listify)
P['mh_terms_abstract']=P['core_mental_health_terms_abstract'].map(listify)
keep=P['abstract_text'].ne('') & P['mh_terms_abstract'].map(len).gt(0) & P['ai_terms_abstract'].map(lambda terms:any(t in SPECIFIC for t in terms))
P=P.loc[keep].copy().reset_index(drop=True)
P['model_text']=[doc_text(t,a) for t,a in zip(P['title'],P['abstract_text'])]
vec=TfidfVectorizer(lowercase=True,strip_accents='unicode',stop_words=sorted(STOP),ngram_range=(1,2),min_df=10,max_features=15000,sublinear_tf=True)
X=vec.fit_transform(P['model_text'])
model=NMF(n_components=4,init='nndsvda',solver='cd',beta_loss='frobenius',max_iter=300,random_state=SEED,l1_ratio=0.0,alpha_W=0.0,alpha_H=0.0)
W=model.fit_transform(X)
shares=np.divide(W,W.sum(axis=1,keepdims=True),out=np.zeros_like(W),where=W.sum(axis=1,keepdims=True)>0)
P['dominant_component']=np.argmax(W,axis=1)+1
P['dominant_share']=shares.max(axis=1)
ordered=np.sort(shares,axis=1)
P['assignment_margin']=ordered[:,-1]-ordered[:,-2]
for k in range(4): P[f'component_{k+1}_share']=shares[:,k]
terms=np.array(vec.get_feature_names_out())
term_rows=[]
packet_rows=[]
for k in range(4):
    weights=model.components_[k]
    top=np.argsort(weights)[::-1][:40]
    for rank,idx in enumerate(top,1): term_rows.append({'component':k+1,'rank':rank,'term':terms[idx],'weight':float(weights[idx])})
    cols=['id','year','title','abstract_text','dominant_component','dominant_share','assignment_margin']+[f'component_{i}_share' for i in range(1,5)]
    comp=P.loc[P['dominant_component'].eq(k+1),cols].copy()
    high=comp.sort_values([f'component_{k+1}_share','id'],ascending=[False,True]).head(35).copy(); high.insert(0,'sample_type','high_membership')
    # Typical documents: centered around median component share, but retain a range of document types.
    med=comp[f'component_{k+1}_share'].median()
    typical=comp.assign(distance=(comp[f'component_{k+1}_share']-med).abs()).sort_values(['distance','id']).head(35).drop(columns='distance').copy(); typical.insert(0,'sample_type','typical_membership')
    boundary=comp.sort_values(['assignment_margin','id']).head(35).copy(); boundary.insert(0,'sample_type','smallest_assignment_margin')
    for sample in [high,typical,boundary]:
        sample['abstract_excerpt']=sample.pop('abstract_text').str.slice(0,1200)
        sample.insert(1,'component',k+1)
        packet_rows.append(sample)
packet=pd.concat(packet_rows,ignore_index=True)
term_table=pd.DataFrame(term_rows)
assignments=P[['id','year','title','dominant_component','dominant_share','assignment_margin']+[f'component_{i}_share' for i in range(1,5)]].copy()
summary=(assignments.groupby('dominant_component',as_index=False).agg(n_publications=('id','size'),pct_of_subset=('id',lambda x:100*len(x)/len(assignments)),mean_dominant_share=('dominant_share','mean'),median_dominant_share=('dominant_share','median'),mean_assignment_margin=('assignment_margin','mean'),share_margin_under_0_10=('assignment_margin',lambda x:float((x<0.10).mean())),share_margin_under_0_20=('assignment_margin',lambda x:float((x<0.20).mean()))))
OUT.mkdir(parents=True,exist_ok=True)
term_table.to_csv(OUT/'four_component_terms.csv',index=False)
packet.to_csv(OUT/'four_component_evidence_packets.csv',index=False)
assignments.to_csv(OUT/'four_component_hard_assignments.csv',index=False)
summary.to_csv(OUT/'four_component_assignment_summary.csv',index=False)
with pd.ExcelWriter(OUT/'four_component_evidence_packets.xlsx',engine='openpyxl') as writer:
    summary.to_excel(writer,sheet_name='assignment_summary',index=False)
    term_table.to_excel(writer,sheet_name='component_terms',index=False)
    packet.to_excel(writer,sheet_name='evidence_packets',index=False)
    assignments.to_excel(writer,sheet_name='hard_assignments',index=False)
manifest={'input_records':11102,'subset_records':int(len(P)),'subset_rule':'available abstract with explicit core mental-health phrase and explicit high-specificity AI phrase (artificial intelligence, machine learning, deep learning, large language model, or LLM) in abstract; neural network alone is not sufficient','n_components':4,'vectorizer':{'ngram_range':[1,2],'min_df':10,'max_features':15000,'sublinear_tf':True,'anchor_phrases_removed':ANCHOR},'nmf':{'seed':SEED,'max_iter':300,'solver':'coordinate descent'},'not_used':'grants, policies, institutions, countries, citations, linkage outcomes, or thematic labels'}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(summary.to_string(index=False))
print('Output:',OUT)
