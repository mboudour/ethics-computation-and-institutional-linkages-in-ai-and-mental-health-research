#!/usr/bin/env python3
from pathlib import Path
import re
import subprocess

root = Path('/home/ubuntu/redesigned_manuscript')
tex_path = root / 'manuscript.tex'
bib_path = root / 'references.bib'
repo_tex = Path('/home/ubuntu/ai-mental-health-institutional-linkages-reproducibility/manuscript/manuscript.tex')
repo_supp = Path('/home/ubuntu/ai-mental-health-institutional-linkages-reproducibility/manuscript/supplementary_materials.tex')
tex = tex_path.read_text(encoding='utf-8')
bib = bib_path.read_text(encoding='utf-8')
used = set()
for group in re.findall(r'\\cite\w*\{([^}]+)\}', tex):
    used.update(x.strip() for x in group.split(','))
available = set(re.findall(r'@\w+\{([^,]+),', bib))
figures = re.findall(r'\\maybefigure\{[^}]*\}\{([^}]+)\}', tex)
checks = {
    'citation_keys_missing': sorted(used - available),
    'all_figure_paths_present': all((root / p).exists() for p in figures),
    'figure_paths_missing': [p for p in figures if not (root / p).exists()],
    'predeclared_high_specificity_absent': 'predeclared high-specificity' not in tex,
    'ten_thousand_permutations_reported': '10,000 permutations' in tex,
    'firth_robustness_reported': 'Firth penalized logistic model' in tex,
    'repo_manuscript_matches': repo_tex.exists() and repo_tex.read_bytes() == tex_path.read_bytes(),
    'repo_supplement_matches': repo_supp.exists() and repo_supp.read_bytes() == (root / 'supplementary_materials.tex').read_bytes(),
}
for key, value in checks.items():
    print(f'{key}: {value}')
print(f'citation_keys_used: {len(used)}')
print(f'figures_referenced: {len(figures)}')
