#!/usr/bin/env python3
"""Render the data-construction and thematic-partition flow diagram.

This figure documents only publication retrieval, screening, high-specificity
filtering, and the publication-text NMF partition. It deliberately excludes
subsequent grant, policy-document, institution, and country analyses.
"""
from pathlib import Path
import math
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path('/home/ubuntu/redesigned_manuscript/figures')
INK = '#24465E'
TEXT = '#1D3548'


def draw_box(ax, x, y, w, h, text, fill, fontsize=9.3, edge=INK):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.010,rounding_size=0.008',
        facecolor=fill, edgecolor=edge, linewidth=1.25, zorder=2
    ))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fontsize, color=TEXT, linespacing=1.18, zorder=3)


def arrow(ax, start_edge, end_edge, gap=.012):
    """Draw an arrow that begins and ends in the whitespace outside boxes."""
    dx = end_edge[0] - start_edge[0]
    dy = end_edge[1] - start_edge[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return
    ux, uy = dx / length, dy / length
    start = (start_edge[0] + gap * ux, start_edge[1] + gap * uy)
    end = (end_edge[0] - gap * ux, end_edge[1] - gap * uy)
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle='-|>', mutation_scale=13, linewidth=1.25,
        color=INK, shrinkA=0, shrinkB=0, zorder=1,
        connectionstyle='arc3,rad=0'
    ))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13.2, 7.4))
    fig.subplots_adjust(left=.025, right=.985, top=.93, bottom=.055)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    ax.text(.02, .978, 'Publication selection and four-theme text partition',
            ha='left', va='top', fontsize=15, fontweight='bold', color='#142C3E')
    ax.text(.02, .943,
            'Outcome-blind data-construction workflow. Counts refer to publication records.',
            ha='left', va='top', fontsize=9.3, color='#59636D')

    # Main horizontal selection flow.
    draw_box(ax, .035, .715, .205, .120,
             'Candidate publication retrieval\nfrom Dimensions.ai\n13,574 records', '#DCEAF2', 9.7)
    draw_box(ax, .290, .715, .220, .120,
             'Local literal screen\nAI/ML + core mental-health\nevidence\n11,102 retained', '#F9E6BF', 9.2)
    draw_box(ax, .560, .715, .185, .120,
             'Publication anchor $P$\n11,102 records', '#D9F0ED', 10.0)
    draw_box(ax, .795, .715, .175, .120,
             'High-specificity\nanalytic corpus\n7,146 records', '#D8E9DD', 9.5)

    arrow(ax, (.240, .775), (.290, .775))
    arrow(ax, (.510, .775), (.560, .775))
    arrow(ax, (.745, .775), (.795, .775))

    # Explicit exclusions.
    draw_box(ax, .290, .545, .220, .075,
             'Excluded: 2,472\nfailed literal screen', '#F6E3E3', 8.7, edge='#A76B6B')
    arrow(ax, (.400, .715), (.400, .620))
    draw_box(ax, .795, .545, .175, .075,
             'Excluded: 3,956\nno high-specificity evidence', '#F6E3E3', 8.2, edge='#A76B6B')
    arrow(ax, (.8825, .715), (.8825, .620))

    # NMF stage.
    draw_box(ax, .315, .360, .370, .120,
             'Four-component NMF text model\nDeterministic TF--IDF of titles + available abstracts\nNo grant, policy-document, institution, country, citation, or linkage data used',
             '#E7E0F2', 8.9)
    arrow(ax, (.8825, .715), (.500, .480))

    # Four hard-assigned themes.
    themes = [
        (.035, 'T1  Risk prediction\n1,603 records (22.4%)'),
        (.280, 'T2  Digital care and ethics\n2,621 records (36.7%)'),
        (.545, 'T3  Social/affective detection\n1,445 records (20.2%)'),
        (.790, 'T4  Neuropsychiatric assessment\n1,477 records (20.7%)'),
    ]
    for x, label in themes:
        draw_box(ax, x, .165, .180, .095, label, '#E7E0F2', 8.1)

    # Orthogonal branch: no diagonal connector crosses the outer theme boxes.
    branch_y = .300
    arrow(ax, (.500, .360), (.500, branch_y))
    ax.plot([.125, .880], [branch_y, branch_y], color=INK, linewidth=1.25, zorder=1)
    for x, _ in themes:
        arrow(ax, (x + .090, branch_y), (x + .090, .260), gap=.008)

    fig.savefig(OUT / 'figure_1_study_design.pdf', bbox_inches='tight', pad_inches=.10)
    fig.savefig(OUT / 'figure_1_study_design.png', dpi=300, bbox_inches='tight', pad_inches=.10)
    plt.close(fig)
    print('Wrote', OUT / 'figure_1_study_design.pdf')


if __name__ == '__main__':
    main()
