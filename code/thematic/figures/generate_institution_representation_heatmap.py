#!/usr/bin/env python3
"""Top-institution representation heatmaps parallel to the country figure.

For each relation, select the eight institutions with the largest all-corpus
representation, then show the percentage of source records with the relation
in each hard-assignment theme. Percentages are nonexclusive and are normalized
within relation-theme columns.
"""
from pathlib import Path
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INFILE=Path('/home/ubuntu/four_theme_institution_country_landscapes/theme_top_institutions_and_countries.csv')
OUTDIR=Path('/home/ubuntu/redesigned_manuscript/figures')
ALL='All high-specificity publications'
THEMES=['Mental-health risk prediction','Digital mental-health care and ethics','Social and affective detection','Neuropsychiatric assessment']
CODES=['T1','T2','T3','T4']
PANELS=[
    ('P-PI author-affiliation institutions','A. Author affiliations','Blues'),
    ('G-GI funder institutions via P-G','B. Grant funders','Greens'),
    ('D-DI policy-issuer institutions via P-D','C. Policy issuers','Purples'),
]

def wrap(text,width=26):
    return '\n'.join(textwrap.wrap(str(text),width=width,break_long_words=False,break_on_hyphens=False))

def main():
    df=pd.read_csv(INFILE)
    OUTDIR.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(18.0,8.4),constrained_layout=False)
    fig.subplots_adjust(left=.075,right=.985,top=.805,bottom=.11,wspace=1.22)
    fig.text(.075,.975,'Institutional representation across publication themes',ha='left',va='top',fontsize=15,fontweight='bold',color='#202124')
    fig.text(.075,.942,'Top eight institutions within each relation; cells are percentages of relation-specific source records in each theme.',ha='left',va='top',fontsize=9.5,color='#555555')
    for ax,(relation,title,cmap) in zip(axes,PANELS):
        overall=(df[(df.theme.eq(ALL)) & (df.relation.eq(relation))]
                 .sort_values(['n_source_records','target_name'],ascending=[False,True]).head(8))
        names=overall.target_name.tolist()
        data=[]
        for name in names:
            vals=[]
            for theme in THEMES:
                q=df[(df.theme.eq(theme))&(df.relation.eq(relation))&(df.target_name.eq(name))]
                vals.append(float(q.pct_of_records_with_relation.iloc[0]) if len(q) else 0.0)
            data.append(vals)
        arr=np.array(data)
        im=ax.imshow(arr,aspect='auto',cmap=cmap,vmin=0,vmax=max(arr.max(),1))
        ax.set_title(title,loc='left',fontsize=10.7,fontweight='bold',pad=10,color='#202124')
        ax.set_xticks(range(4),CODES,fontsize=9,fontweight='bold')
        ax.set_yticks(range(len(names)),[wrap(x,22) for x in names],fontsize=7.3)
        ax.tick_params(axis='both',length=0)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                value=arr[i,j]
                ax.text(j,i,f'{value:.1f}',ha='center',va='center',fontsize=7.4,color='white' if value>.58*arr.max() else '#344054')
        for spine in ax.spines.values():spine.set_color('#B7C0CC')
    fig.text(.075,.060,'T1 Risk prediction    T2 Digital care and ethics    T3 Social/affective detection    T4 Neuropsychiatric assessment',ha='left',va='center',fontsize=8.2,color='#444444')
    fig.savefig(OUTDIR/'figure_5_institution_landscapes.pdf',bbox_inches='tight',pad_inches=.12)
    fig.savefig(OUTDIR/'figure_5_institution_landscapes.png',dpi=300,bbox_inches='tight',pad_inches=.12)
    plt.close(fig)
    print('Wrote',OUTDIR/'figure_5_institution_landscapes.pdf')
if __name__=='__main__':main()
