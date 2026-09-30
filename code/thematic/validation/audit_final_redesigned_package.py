#!/usr/bin/env python3
from pathlib import Path
import re

repository_root = Path(__file__).resolve().parents[3]
manuscript_root = repository_root / 'manuscript'
tex_path = manuscript_root / 'manuscript.tex'
bib_path = manuscript_root / 'references.bib'
tex = tex_path.read_text(encoding='utf-8')
bib = bib_path.read_text(encoding='utf-8')
used = set()
for group in re.findall(r'\\cite\w*\{([^}]+)\}', tex):
    used.update(x.strip() for x in group.split(','))
available = set(re.findall(r'@\w+\{([^,]+),', bib))
figures = re.findall(r'\\maybefigure\{[^}]*\}\{([^}]+)\}', tex)
checks = {
    'citation_keys_missing': sorted(used - available),
    'all_figure_paths_present': all((repository_root / p).exists() for p in figures),
    'figure_paths_missing': [p for p in figures if not (repository_root / p).exists()],
    'predeclared_high_specificity_absent': 'predeclared high-specificity' not in tex,
    'ten_thousand_permutations_reported': '10,000 permutations' in tex,
    'firth_sensitivity_reported': 'Firth' in tex,
    'supplementary_source_present': (manuscript_root / 'supplementary_materials.tex').exists(),
}
for key, value in checks.items():
    print(f'{key}: {value}')
print(f'citation_keys_used: {len(used)}')
print(f'figures_referenced: {len(figures)}')
