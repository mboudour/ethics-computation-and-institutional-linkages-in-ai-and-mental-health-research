#!/usr/bin/env python3
"""Construct theme-specific institutional and country landscapes for final linked records.

Raw relations
-------------
P--PI : publication to author-affiliation organization
G--GI : grant to funding organization
D--DI : policy document to issuing organization

Thematic views are constructed only after publications receive the audited hard
four-theme NMF assignment. A grant or policy document can therefore contribute
to more than one theme when its linked publications belong to more than one
theme. No result below is a rate, causal effect, funding decision, policy
impact, endorsement, or intentional-selection measure.
"""
from __future__ import annotations

from pathlib import Path
import ast
import json
import re
from collections import defaultdict
from typing import Any

import pandas as pd

P_PATH = Path('/home/ubuntu/upload/publication_anchor_final.pkl')
G_PATH = Path('/home/ubuntu/upload/linked_grants_final.pkl')
D_PATH = Path('/home/ubuntu/upload/linked_policy_documents_final.pkl')
ASSIGN_PATH = Path('/home/ubuntu/four_theme_hard_partition_audit/four_theme_hard_partition.csv')
OUT = Path('/home/ubuntu/four_theme_institution_country_landscapes')

THEMES = {
    1: 'Mental-health risk prediction',
    2: 'Digital mental-health care and ethics',
    3: 'Social and affective detection',
    4: 'Neuropsychiatric assessment',
}


def listify(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() == 'nan':
        return []
    try:
        parsed = ast.literal_eval(text)
        return list(parsed) if isinstance(parsed, (list, tuple, set)) else [parsed]
    except (ValueError, SyntaxError):
        return [text]


def norm_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ''
    return re.sub(r'\s+', ' ', str(value)).strip()


def org_key(item: Any, prefix: str) -> tuple[str, str, str]:
    """Return stable key, display name, and country for a structured organization item."""
    if isinstance(item, dict):
        identifier = norm_text(item.get('id'))
        name = norm_text(item.get('name') or item.get('name_original'))
        country = norm_text(item.get('country_name') or item.get('country') or item.get('country_code'))
        key = f'{prefix}:id:{identifier}' if identifier else (f'{prefix}:name:{name.lower()}' if name else '')
        return key, name or identifier, country
    name = norm_text(item)
    return (f'{prefix}:name:{name.lower()}' if name else ''), name, ''


def country_key(item: Any) -> tuple[str, str]:
    if isinstance(item, dict):
        identifier = norm_text(item.get('id') or item.get('country_code'))
        name = norm_text(item.get('name') or item.get('country_name'))
        return identifier or name, name or identifier
    name = norm_text(item)
    return name, name


def add_relation(rows: list[dict[str, Any]], theme: str, relation: str, source_id: str, target_id: str, target_name: str, country: str = '') -> None:
    if target_id:
        rows.append({'theme': theme, 'relation': relation, 'source_id': source_id, 'target_id': target_id, 'target_name': target_name, 'country': country})


P = pd.read_pickle(P_PATH).copy()
G = pd.read_pickle(G_PATH).copy()
D = pd.read_pickle(D_PATH).copy()
a = pd.read_csv(ASSIGN_PATH)
for frame, label in [(P, 'P'), (G, 'G'), (D, 'D'), (a, 'assignments')]:
    if 'id' not in frame.columns:
        raise SystemExit(f'{label} has no id field')
    frame['id'] = frame['id'].astype(str)
P = P.loc[P['retain_in_final_P'].astype(bool)].copy()

# The assignment file is already derived from a transparent high-specificity subset.
a = a[['id', 'dominant_component', 'candidate_theme', 'dominant_share', 'assignment_margin']].copy()
if not a['id'].is_unique:
    raise SystemExit('Theme assignment file has duplicate publication IDs.')
if set(a['candidate_theme']) != set(THEMES.values()):
    raise SystemExit('Theme labels in assignment file do not match the approved four-theme audit.')
P = P.merge(a, on='id', how='inner', validate='one_to_one')
if len(P) != len(a):
    raise SystemExit('Some assigned publications are missing from the final publication anchor.')
P_ids = set(P['id'])
G_ids = set(G['id'])
D_ids = set(D['id'])

# P--PI and P--country affiliation relations
p_i_rows: list[dict[str, Any]] = []
p_c_rows: list[dict[str, Any]] = []
for row in P[['id','candidate_theme','research_orgs','research_org_countries']].itertuples(index=False):
    pid, theme, orgs, countries = row
    seen_orgs: set[str] = set()
    for item in listify(orgs):
        key, name, country = org_key(item, 'PI')
        if key and key not in seen_orgs:
            add_relation(p_i_rows, theme, 'P-PI author-affiliation relation', pid, key, name, country)
            seen_orgs.add(key)
    seen_countries: set[str] = set()
    for item in listify(countries):
        key, name = country_key(item)
        if key and key not in seen_countries:
            add_relation(p_c_rows, theme, 'P-country author-affiliation relation', pid, key, name)
            seen_countries.add(key)

# P--G funding-support links.
def ids(value: Any) -> list[str]:
    return [str(x) for x in listify(value) if norm_text(x)]

pg_rows: list[dict[str, Any]] = []
for row in P[['id','candidate_theme','supporting_grant_ids']].itertuples(index=False):
    pid, theme, grants = row
    for gid in sorted(set(ids(grants))):
        if gid in G_ids:
            pg_rows.append({'theme': theme, 'publication_id': pid, 'grant_id': gid})
pg = pd.DataFrame(pg_rows, columns=['theme','publication_id','grant_id']).drop_duplicates()

# G--GI funding-organization and country relations, represented by P--G--GI paths.
g_map = G.set_index('id')
g_i_rows: list[dict[str, Any]] = []
g_c_rows: list[dict[str, Any]] = []
for item in pg.itertuples(index=False):
    grant = g_map.loc[item.grant_id]
    seen_orgs: set[str] = set()
    for org in listify(grant.get('funder_orgs')):
        key, name, country = org_key(org, 'GI')
        if key and key not in seen_orgs:
            add_relation(g_i_rows, item.theme, 'G-GI funder association via P-G', item.grant_id, key, name, country)
            seen_orgs.add(key)
    seen_countries: set[str] = set()
    # Countries are taken first from structured funder organizations. The grant-level
    # research-org country field is not substituted because it describes recipients,
    # not funders.
    for org in listify(grant.get('funder_orgs')):
        _, _, country = org_key(org, 'GI')
        if country:
            key = country
            if key not in seen_countries:
                add_relation(g_c_rows, item.theme, 'G-country funder association via P-G', item.grant_id, key, country)
                seen_countries.add(key)

g_i = pd.DataFrame(g_i_rows, columns=['theme','relation','source_id','target_id','target_name','country']).drop_duplicates()
g_c = pd.DataFrame(g_c_rows, columns=['theme','relation','source_id','target_id','target_name','country']).drop_duplicates()

# P--D policy-document coverage links.
pd_rows: list[dict[str, Any]] = []
for drow in D[['id','publication_ids']].itertuples(index=False):
    did, publications = drow
    for pid in sorted(set(ids(publications)) & P_ids):
        theme = P.loc[P['id'].eq(pid), 'candidate_theme'].iloc[0]
        pd_rows.append({'theme': theme, 'publication_id': pid, 'policy_document_id': did})
pd_edges = pd.DataFrame(pd_rows, columns=['theme','publication_id','policy_document_id']).drop_duplicates()

# D--DI issuer and issuer-country relations, represented by P--D--DI paths.
d_map = D.set_index('id')
d_i_rows: list[dict[str, Any]] = []
d_c_rows: list[dict[str, Any]] = []
for item in pd_edges.itertuples(index=False):
    doc = d_map.loc[item.policy_document_id]
    issuer_id = norm_text(doc.get('publisher_org.id'))
    issuer_name = norm_text(doc.get('publisher_org.name'))
    issuer_country = norm_text(doc.get('publisher_org.country_name') or doc.get('publisher_org_country.name') or doc.get('publisher_org.country_code') or doc.get('publisher_org_country.id'))
    key = f'DI:id:{issuer_id}' if issuer_id else (f'DI:name:{issuer_name.lower()}' if issuer_name else '')
    add_relation(d_i_rows, item.theme, 'D-DI issuer association via P-D', item.policy_document_id, key, issuer_name or issuer_id, issuer_country)
    if issuer_country:
        add_relation(d_c_rows, item.theme, 'D-country issuer association via P-D', item.policy_document_id, issuer_country, issuer_country)

d_i = pd.DataFrame(d_i_rows, columns=['theme','relation','source_id','target_id','target_name','country']).drop_duplicates()
d_c = pd.DataFrame(d_c_rows, columns=['theme','relation','source_id','target_id','target_name','country']).drop_duplicates()
p_i = pd.DataFrame(p_i_rows, columns=['theme','relation','source_id','target_id','target_name','country']).drop_duplicates()
p_c = pd.DataFrame(p_c_rows, columns=['theme','relation','source_id','target_id','target_name','country']).drop_duplicates()

# Add a de-duplicated all-theme baseline. Grants and policy documents may appear
# in several thematic paths, but appear only once per source-target pair here.
ALL_THEME = 'All high-specificity publications'
def with_all_theme(table):
    all_rows = table.drop(columns='theme').drop_duplicates().copy()
    all_rows.insert(0, 'theme', ALL_THEME)
    return pd.concat([table, all_rows], ignore_index=True)

p_i = with_all_theme(p_i)
p_c = with_all_theme(p_c)
g_i = with_all_theme(g_i)
g_c = with_all_theme(g_c)
d_i = with_all_theme(d_i)
d_c = with_all_theme(d_c)
pg_all = pg.drop(columns='theme').drop_duplicates().copy(); pg_all.insert(0, 'theme', ALL_THEME)
pd_all = pd_edges.drop(columns='theme').drop_duplicates().copy(); pd_all.insert(0, 'theme', ALL_THEME)
pg = pd.concat([pg, pg_all], ignore_index=True)
pd_edges = pd.concat([pd_edges, pd_all], ignore_index=True)

# Summaries.
relations = {
    'P-PI author-affiliation institutions': p_i,
    'P-country author-affiliation countries': p_c,
    'G-GI funder institutions via P-G': g_i,
    'G-country funder countries via P-G': g_c,
    'D-DI policy-issuer institutions via P-D': d_i,
    'D-country policy-issuer countries via P-D': d_c,
}
summary_rows=[]
for theme in [ALL_THEME] + list(THEMES.values()):
    ptheme = P if theme == ALL_THEME else P.loc[P['candidate_theme'].eq(theme)]
    summary_rows.append({'theme':theme,'relation':'P publications','n_source_records':len(ptheme),'n_target_entities':'','n_unique_ties':'','source_coverage_pct':100.0})
    # Recorded P--G and P--D linkage scale determines the denominator for
    # downstream grant and policy-document organization relations.
    xpg=pg.loc[pg['theme'].eq(theme)]
    xpd=pd_edges.loc[pd_edges['theme'].eq(theme)]
    relation_denominators={
        'P-PI author-affiliation institutions': len(ptheme),
        'P-country author-affiliation countries': len(ptheme),
        'G-GI funder institutions via P-G': xpg['grant_id'].nunique(),
        'G-country funder countries via P-G': xpg['grant_id'].nunique(),
        'D-DI policy-issuer institutions via P-D': xpd['policy_document_id'].nunique(),
        'D-country policy-issuer countries via P-D': xpd['policy_document_id'].nunique(),
    }
    for relation, table in relations.items():
        if isinstance(table,list):
            table=pd.DataFrame(table)
        x=table.loc[table['theme'].eq(theme)] if not table.empty else table
        n_source=x['source_id'].nunique() if not x.empty else 0
        n_target=x['target_id'].nunique() if not x.empty else 0
        n_ties=len(x.drop_duplicates(['source_id','target_id'])) if not x.empty else 0
        denominator=relation_denominators[relation]
        coverage=100*n_source/denominator if denominator else 0.0
        summary_rows.append({'theme':theme,'relation':relation,'n_source_records':n_source,'n_target_entities':n_target,'n_unique_ties':n_ties,'source_coverage_pct':coverage})
    summary_rows.append({'theme':theme,'relation':'P-G funding-support linkage','n_source_records':xpg['publication_id'].nunique(),'n_target_entities':xpg['grant_id'].nunique(),'n_unique_ties':len(xpg),'source_coverage_pct':100*xpg['publication_id'].nunique()/len(ptheme)})
    summary_rows.append({'theme':theme,'relation':'P-D policy-document coverage linkage','n_source_records':xpd['publication_id'].nunique(),'n_target_entities':xpd['policy_document_id'].nunique(),'n_unique_ties':len(xpd),'source_coverage_pct':100*xpd['publication_id'].nunique()/len(ptheme)})
summary=pd.DataFrame(summary_rows)

# Top entity representation per relation/theme. Each entity count is the number of
# distinct source records, plus total source-target ties; calculated nonexclusively by theme.
top_rows=[]
for relation, table in relations.items():
    if isinstance(table,list): table=pd.DataFrame(table)
    if table.empty: continue
    for theme, x in table.groupby('theme'):
        stat=(x.groupby(['target_id','target_name'],as_index=False).agg(n_source_records=('source_id','nunique'),n_relation_ties=('source_id','size')).sort_values(['n_source_records','n_relation_ties','target_name'],ascending=[False,False,True]).head(30))
        total=stat['n_source_records'].sum() # only display denominator is replaced below
        overall=x['source_id'].nunique()
        stat.insert(0,'theme',theme)
        stat.insert(1,'relation',relation)
        stat['pct_of_records_with_relation']=100*stat['n_source_records']/overall if overall else 0.0
        top_rows.append(stat)
top=pd.concat(top_rows,ignore_index=True) if top_rows else pd.DataFrame()

# Cross-theme entity recurrence indicates whether the same organizations/countries appear in multiple thematic landscapes.
recurrence_rows=[]
for relation, table in relations.items():
    if isinstance(table,list): table=pd.DataFrame(table)
    if table.empty: continue
    table = table.loc[table['theme'].ne(ALL_THEME)].copy()
    z=(table.groupby(['target_id','target_name'],as_index=False).agg(n_themes=('theme','nunique'),themes=('theme',lambda s:'; '.join(sorted(set(s)))),n_source_records=('source_id','nunique')))
    z.insert(0,'relation',relation)
    recurrence_rows.append(z)
recurrence=pd.concat(recurrence_rows,ignore_index=True) if recurrence_rows else pd.DataFrame()

# Target-entity concentration is descriptive: it summarizes the distribution of
# recorded source-target relations within a theme and relation type.
concentration_rows=[]
for relation, table in relations.items():
    if isinstance(table, list): table=pd.DataFrame(table)
    if table.empty: continue
    for theme, x in table.groupby('theme'):
        counts=x.groupby('target_id')['source_id'].nunique().sort_values(ascending=False)
        total=int(counts.sum())
        shares=counts/total if total else counts.astype(float)
        concentration_rows.append({
            'theme':theme,
            'relation':relation,
            'n_target_entities':int(len(counts)),
            'n_source_entity_incidence':total,
            'top_1_share_pct':float(100*shares.iloc[0]) if len(shares) else 0.0,
            'top_5_share_pct':float(100*shares.head(5).sum()) if len(shares) else 0.0,
            'hhi':float((shares**2).sum()) if len(shares) else 0.0,
        })
concentration=pd.DataFrame(concentration_rows)

# Paths are outputs for audit and potential later network work, not independent raw layers.

OUT.mkdir(parents=True,exist_ok=True)
summary.to_csv(OUT/'theme_institution_country_landscape_summary.csv',index=False)
top.to_csv(OUT/'theme_top_institutions_and_countries.csv',index=False)
recurrence.to_csv(OUT/'institution_country_cross_theme_recurrence.csv',index=False)
concentration.to_csv(OUT/'theme_institution_country_concentration.csv',index=False)
P[['id','year','candidate_theme','dominant_share','assignment_margin']].to_csv(OUT/'themed_publications.csv',index=False)
pg.to_csv(OUT/'themed_P_G_funding_support_links.csv',index=False)
pd_edges.to_csv(OUT/'themed_P_D_policy_coverage_links.csv',index=False)
p_i.to_csv(OUT/'themed_P_PI_author_affiliations.csv',index=False)
p_c.to_csv(OUT/'themed_P_country_author_affiliations.csv',index=False)
g_i.to_csv(OUT/'themed_G_GI_funder_associations.csv',index=False)
g_c.to_csv(OUT/'themed_G_country_funder_associations.csv',index=False)
d_i.to_csv(OUT/'themed_D_DI_issuer_associations.csv',index=False)
d_c.to_csv(OUT/'themed_D_country_issuer_associations.csv',index=False)
with pd.ExcelWriter(OUT/'four_theme_institution_country_landscapes.xlsx',engine='openpyxl') as writer:
    summary.to_excel(writer,sheet_name='landscape_summary',index=False)
    top.to_excel(writer,sheet_name='top_entities',index=False)
    recurrence.to_excel(writer,sheet_name='cross_theme_recurrence',index=False)
    concentration.to_excel(writer,sheet_name='concentration',index=False)
    P[['id','year','candidate_theme','dominant_share','assignment_margin']].to_excel(writer,sheet_name='themed_P',index=False)
    pg.to_excel(writer,sheet_name='P_G_links',index=False)
    pd_edges.to_excel(writer,sheet_name='P_D_links',index=False)

checks=pd.DataFrame([
 {'check':'Theme assignment IDs are unique','failures':int(a['id'].duplicated().sum())},
 {'check':'Themed P records equal hard-assignment records','failures':abs(len(P)-len(a))},
 {'check':'All P-G endpoints exist in G','failures':int((~pg['grant_id'].isin(G_ids)).sum()) if len(pg) else 0},
 {'check':'All P-D endpoints exist in D','failures':int((~pd_edges['policy_document_id'].isin(D_ids)).sum()) if len(pd_edges) else 0},
 {'check':'All D-DI rows have issuer target','failures':int(d_i['target_id'].eq('').sum()) if len(d_i) else 0},
])
checks.to_csv(OUT/'validation_checks.csv',index=False)
manifest={
 'raw_relations':{
  'P-PI':'publication-to-recorded author-affiliation organization',
  'G-GI':'grant-to-recorded funding organization',
  'D-DI':'policy-document-to-recorded issuing organization',
  'P-G':'publication-to-recorded supporting grant',
  'P-D':'publication-to-recorded policy document linkage'
 },
 'theme_assignment':'largest normalized four-component NMF weight in the predeclared high-specificity publication subset',
 'important_boundary':'Theme-specific G-GI and D-DI tables are relation paths through publications in a theme; grants and policy documents can therefore appear in multiple theme views.',
 'country_boundary':'Country counts are nonexclusive representations, not rates or population-adjusted shares.',
 'not_inferred':'causation, funding decisions, policy impact, endorsement, intentional selection, or institutional preference.'
}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(summary.to_string(index=False))
print('\nValidation failures:', int(checks['failures'].sum()))
print('Output:',OUT)
