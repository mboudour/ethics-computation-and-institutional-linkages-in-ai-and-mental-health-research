#!/usr/bin/env python3
"""Rare-outcome and specification robustness checks for four-theme policy coverage.

All classification inputs are the fixed high-specificity corpus and its previously
created hard theme assignments. This script does not alter the thematic model.
It estimates only publication-year-eligible policy-document coverage models.

Specifications:
  1. HC3 logistic, linear year + log1p citations (primary replication)
  2. HC3 logistic, linear year without citation adjustment
  3. HC3 logistic, cubic B-spline year + log1p citations
  4. Firth penalized logistic, linear year + log1p citations

The Firth implementation uses Jeffreys-prior bias reduction and reports
Wald-style intervals based on the final expected-information inverse.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit
from patsy import dmatrix

P_PATH = Path('/home/ubuntu/upload/publication_anchor_final.pkl')
D_PATH = Path('/home/ubuntu/upload/linked_policy_documents_final.pkl')
ASSIGN_PATH = Path('/home/ubuntu/four_theme_hard_partition_audit/four_theme_hard_partition.csv')
OUT = Path('/home/ubuntu/four_theme_policy_robustness')
REFERENCE = 'Digital mental-health care and ethics'
ALPHA = 0.05
MAX_ITER = 1000
TOL = 1e-9


def listify(value):
    if isinstance(value, (list, tuple, set)):
        return [str(x) for x in value]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() == 'nan':
        return []
    try:
        parsed = ast.literal_eval(text)
        return [str(x) for x in parsed] if isinstance(parsed, (list, tuple, set)) else [str(parsed)]
    except (ValueError, SyntaxError):
        return [text]


def load_analysis_frame():
    p = pd.read_pickle(P_PATH).copy()
    d = pd.read_pickle(D_PATH).copy()
    a = pd.read_csv(ASSIGN_PATH).copy()
    for frame in (p, d, a):
        frame['id'] = frame['id'].astype(str)
    p = p.loc[p['retain_in_final_P'].astype(bool)].merge(
        a[['id', 'candidate_theme', 'dominant_share', 'assignment_margin']],
        on='id', how='inner', validate='one_to_one'
    )
    pids = set(p['id'])
    covered = set()
    for row in d[['id', 'publication_ids']].itertuples(index=False):
        did, publication_ids = row
        covered.update(set(listify(publication_ids)) & pids)
    p['policy_document_coverage'] = p['id'].isin(covered).astype(int)
    p['year'] = pd.to_numeric(p['year'], errors='coerce')
    p['times_cited'] = pd.to_numeric(p['times_cited'], errors='coerce').fillna(0).clip(lower=0)
    p = p.dropna(subset=['year', 'candidate_theme']).copy()
    p['year'] = p['year'].astype(int)
    return p


def theme_dummies(frame):
    dummies = pd.get_dummies(frame['candidate_theme'], prefix='theme', dtype=float)
    ref = 'theme_' + REFERENCE
    if ref not in dummies.columns:
        raise ValueError(f'Missing reference theme: {REFERENCE}')
    return dummies.drop(columns=ref)


def build_design(frame, specification):
    x = frame.reset_index(drop=True).copy()
    pieces = [pd.Series(1.0, index=x.index, name='const'), theme_dummies(x)]
    if specification['year'] == 'linear':
        pieces.append(pd.Series(x['year'].astype(float) - x['year'].astype(float).mean(), index=x.index, name='year_centered'))
    elif specification['year'] == 'spline':
        spline = dmatrix('0 + bs(year, df=4, degree=3, include_intercept=False)', {'year': x['year'].astype(float)}, return_type='dataframe')
        spline.index = x.index
        spline.columns = [f'year_spline_{i+1}' for i in range(spline.shape[1])]
        pieces.append(spline)
    else:
        raise ValueError('Unknown year specification')
    if specification['citation']:
        pieces.append(pd.Series(np.log1p(x['times_cited'].astype(float)), index=x.index, name='log1p_citations'))
    design = pd.concat(pieces, axis=1).astype(float)
    y = x['policy_document_coverage'].astype(float).reset_index(drop=True)
    return x, y, design


def loglik_firth(beta, X, y):
    eta = np.clip(X @ beta, -35, 35)
    mu = expit(eta)
    w = np.maximum(mu * (1 - mu), 1e-12)
    info = X.T @ (w[:, None] * X)
    sign, logdet = np.linalg.slogdet(info)
    if sign <= 0:
        return -np.inf
    ll = np.sum(y * np.log(np.maximum(mu, 1e-15)) + (1 - y) * np.log(np.maximum(1 - mu, 1e-15)))
    return float(ll + 0.5 * logdet)


def fit_firth(y, design):
    X = design.to_numpy(dtype=float)
    yy = np.asarray(y, dtype=float)
    beta = np.zeros(X.shape[1], dtype=float)
    current = loglik_firth(beta, X, yy)
    converged = False
    iterations = 0
    for iterations in range(1, MAX_ITER + 1):
        eta = np.clip(X @ beta, -35, 35)
        mu = expit(eta)
        w = np.maximum(mu * (1 - mu), 1e-12)
        info = X.T @ (w[:, None] * X)
        try:
            info_inv = np.linalg.inv(info)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError('Firth expected-information matrix is singular') from exc
        h = w * np.einsum('ij,jk,ik->i', X, info_inv, X)
        modified_score = X.T @ (yy - mu + h * (0.5 - mu))
        step = info_inv @ modified_score
        step_scale = 1.0
        accepted = False
        for _ in range(40):
            candidate = beta + step_scale * step
            candidate_ll = loglik_firth(candidate, X, yy)
            if np.isfinite(candidate_ll) and candidate_ll >= current - 1e-12:
                beta = candidate
                current = candidate_ll
                accepted = True
                break
            step_scale *= 0.5
        if not accepted:
            break
        if np.max(np.abs(step_scale * step)) < TOL:
            converged = True
            break
    eta = np.clip(X @ beta, -35, 35)
    mu = expit(eta)
    w = np.maximum(mu * (1 - mu), 1e-12)
    info = X.T @ (w[:, None] * X)
    cov = np.linalg.inv(info)
    se = np.sqrt(np.diag(cov))
    return beta, se, converged, iterations, current


def fit_hc3(y, design):
    result = sm.GLM(y, design, family=sm.families.Binomial()).fit(cov_type='HC3')
    params = result.params.to_numpy()
    se = result.bse.to_numpy()
    return params, se, bool(result.converged), int(getattr(result, 'fit_history', {}).get('iteration', 0) or 0), float(result.llf)


def coefficient_table(params, se, design, specification, n, events, converged, iterations, loglik):
    z = 1.959963984540054
    low = params - z * se
    high = params + z * se
    p = 2 * (1 - 0.5 * (1 + np.vectorize(__import__('math').erf)(np.abs(params / se) / np.sqrt(2))))
    return pd.DataFrame({
        'model': specification['name'],
        'estimator': specification['estimator'],
        'year_specification': specification['year'],
        'citation_adjusted': specification['citation'],
        'term': design.columns,
        'coefficient_log_odds': params,
        'standard_error': se,
        'odds_ratio': np.exp(params),
        'ci_low': np.exp(low),
        'ci_high': np.exp(high),
        'p_value': p,
        'n_publications': n,
        'n_outcome_events': events,
        'converged': converged,
        'iterations': iterations,
        'penalized_or_model_loglikelihood': loglik,
    })


def diagnostics(frame, design, specification, converged, iterations):
    theme_event = frame.groupby('candidate_theme')['policy_document_coverage'].agg(['size', 'sum']).reset_index()
    theme_event['model'] = specification['name']
    theme_event['estimator'] = specification['estimator']
    theme_event = theme_event.rename(columns={'size': 'n_publications', 'sum': 'n_events'})
    corr = design.drop(columns='const', errors='ignore').corr().stack().reset_index()
    corr.columns = ['predictor_1', 'predictor_2', 'correlation']
    corr = corr.loc[corr['predictor_1'] < corr['predictor_2']].copy()
    corr['model'] = specification['name']
    # VIF via auxiliary least-squares regressions; intercept omitted from reported predictors.
    vif_rows = []
    cols = [c for c in design.columns if c != 'const']
    for col in cols:
        others = [c for c in cols if c != col]
        if not others:
            vif = np.nan
        else:
            res = sm.OLS(design[col], sm.add_constant(design[others], has_constant='add')).fit()
            vif = np.inf if res.rsquared >= 1 else 1 / (1 - res.rsquared)
        vif_rows.append({'model': specification['name'], 'predictor': col, 'vif': vif})
    summary = pd.DataFrame([{
        'model': specification['name'],
        'estimator': specification['estimator'],
        'year_specification': specification['year'],
        'citation_adjusted': specification['citation'],
        'n_publications': len(frame),
        'n_events': int(frame['policy_document_coverage'].sum()),
        'events_per_parameter': float(frame['policy_document_coverage'].sum()) / design.shape[1],
        'n_parameters': design.shape[1],
        'converged': converged,
        'iterations': iterations,
        'all_themes_have_event': bool((theme_event['n_events'] > 0).all()),
        'all_themes_have_non_event': bool((theme_event['n_events'] < theme_event['n_publications']).all()),
    }])
    return summary, theme_event, pd.DataFrame(vif_rows), corr


def main():
    out = OUT
    out.mkdir(parents=True, exist_ok=True)
    p = load_analysis_frame()
    p = p.loc[p['year'] <= 2021].copy()  # primary age-eligible policy sample
    specifications = [
        {'name': 'primary_hc3_linear_year_citation_adjusted', 'estimator': 'HC3 logistic', 'year': 'linear', 'citation': True},
        {'name': 'hc3_linear_year_citation_omitted', 'estimator': 'HC3 logistic', 'year': 'linear', 'citation': False},
        {'name': 'hc3_spline_year_citation_adjusted', 'estimator': 'HC3 logistic', 'year': 'spline', 'citation': True},
        {'name': 'firth_linear_year_citation_adjusted', 'estimator': 'Firth penalized logistic', 'year': 'linear', 'citation': True},
    ]
    coef_all=[]; diag_all=[]; event_all=[]; vif_all=[]; corr_all=[]
    for spec in specifications:
        frame, y, design = build_design(p, spec)
        if spec['estimator'].startswith('Firth'):
            params, se, conv, iters, ll = fit_firth(y, design)
        else:
            params, se, conv, iters, ll = fit_hc3(y, design)
        coef_all.append(coefficient_table(params, se, design, spec, len(frame), int(y.sum()), conv, iters, ll))
        diag, ev, vif, corr = diagnostics(frame, design, spec, conv, iters)
        diag_all.append(diag); event_all.append(ev); vif_all.append(vif); corr_all.append(corr)
    coeff = pd.concat(coef_all, ignore_index=True)
    diag = pd.concat(diag_all, ignore_index=True)
    events = pd.concat(event_all, ignore_index=True)
    vif = pd.concat(vif_all, ignore_index=True)
    corr = pd.concat(corr_all, ignore_index=True)
    checks = pd.DataFrame([
        {'check': 'Primary policy sample has 111 events', 'failures': int((p['policy_document_coverage'].sum()) != 111)},
        {'check': 'Four themes represented in primary sample', 'failures': int(p['candidate_theme'].nunique() != 4)},
        {'check': 'All specifications converged', 'failures': int((~diag['converged']).sum())},
        {'check': 'No theme has zero policy events', 'failures': int((~diag['all_themes_have_event']).sum())},
        {'check': 'No theme has only policy events', 'failures': int((~diag['all_themes_have_non_event']).sum())},
    ])
    coeff.to_csv(out/'four_theme_policy_robustness_coefficients.csv', index=False)
    diag.to_csv(out/'four_theme_policy_robustness_diagnostics.csv', index=False)
    events.to_csv(out/'four_theme_policy_robustness_theme_events.csv', index=False)
    vif.to_csv(out/'four_theme_policy_robustness_vif.csv', index=False)
    corr.to_csv(out/'four_theme_policy_robustness_correlations.csv', index=False)
    checks.to_csv(out/'validation_checks.csv', index=False)
    with pd.ExcelWriter(out/'four_theme_policy_robustness.xlsx', engine='openpyxl') as writer:
        coeff.to_excel(writer, sheet_name='coefficients', index=False)
        diag.to_excel(writer, sheet_name='diagnostics', index=False)
        events.to_excel(writer, sheet_name='theme_events', index=False)
        vif.to_excel(writer, sheet_name='vif', index=False)
        corr.to_excel(writer, sheet_name='correlations', index=False)
        checks.to_excel(writer, sheet_name='validation', index=False)
    manifest = {
        'sample': 'High-specificity thematic corpus; publications through 2021 only.',
        'outcome': 'At least one Dimensions-recorded policy-document coverage linkage.',
        'reference_theme': REFERENCE,
        'specifications': specifications,
        'firth_intervals': 'Approximate Wald 95% intervals based on inverse final expected information.',
        'theme_construction_not_used': 'Grant, policy, institution, country, citation, and linkage outcomes were not used to construct themes.',
    }
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Validation failures:', int(checks['failures'].sum()))
    print(coeff.loc[coeff['term'].str.startswith('theme_'), ['model','term','odds_ratio','ci_low','ci_high','p_value']].to_string(index=False))

if __name__ == '__main__':
    main()
