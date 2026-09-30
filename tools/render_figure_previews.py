#!/usr/bin/env python3
"""Render final manuscript figure PDFs as browser-friendly PNG previews.

The original PDF figures remain the authoritative source. PNG previews exist solely
for the Streamlit gallery and preserve the full figure page without cropping.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


FIGURE_FILENAMES = (
    "figure_1_study_design.pdf",
    "figure_2_four_theme_composition.pdf",
    "figure_3_four_theme_associations.pdf",
    "figure_4_four_theme_country_landscapes.pdf",
    "figure_5_institution_landscapes.pdf",
    "figure_6_four_theme_local_alignment.pdf",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dpi", type=int, default=180, help="Rasterization DPI (default: 180).")
    args = parser.parse_args()

    if args.dpi <= 0:
        parser.error("--dpi must be a positive integer")
    renderer = shutil.which("pdftocairo")
    if renderer is None:
        raise SystemExit("pdftocairo is required to render figure previews.")

    repository_root = Path(__file__).resolve().parents[1]
    figures_dir = repository_root / "figures"
    preview_dir = figures_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    for filename in FIGURE_FILENAMES:
        source = figures_dir / filename
        if not source.exists():
            raise FileNotFoundError(f"Missing final manuscript figure: {source}")
        target_base = preview_dir / source.stem
        target_png = target_base.with_suffix(".png")
        target_png.unlink(missing_ok=True)
        subprocess.run(
            [renderer, "-png", "-singlefile", "-r", str(args.dpi), str(source), str(target_base)],
            check=True,
        )
        if not target_png.exists() or target_png.stat().st_size == 0:
            raise RuntimeError(f"Preview rendering failed for {source.name}")
        print(f"Rendered {target_png.relative_to(repository_root)}")


if __name__ == "__main__":
    main()
