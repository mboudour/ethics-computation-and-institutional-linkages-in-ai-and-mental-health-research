#!/usr/bin/env python3
"""Generate the five figures for the redesigned four-theme manuscript.
All figures use only aggregate outputs produced by the themed analysis scripts.
"""
from pathlib import Path
import json
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.ticker import FuncFormatter

BASE=Path('/home/ubuntu')
OUT=BASE/'redesigned_manuscript'/'figures'
THEME_ORDER=['Mental-health risk prediction','Digital mental-health care and ethics','Social and affective detection','Neuropsychiatric assessment']
SHORT={'Mental-health risk prediction':'Risk prediction','Digital mental-health care and ethics':'Digital care and ethics','Social and affective detection':'Social/affective detection','Neuropsychiatric assessment':'Neuropsychiatric assessment'}
COLORS={'Mental-health risk prediction':'#1F4E79','Digital mental-health care and ethics':'#188977','Social and affective detection':'#D99B3B','Neuropsychiatric assessment':'#7864A8'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titleweight':'bold','axes.edgecolor':'#9aa3aa','axes.labelcolor':'#313b44','xtick.color':'#313b44','ytick.color':'#313b44','figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})

def save(fig,name):
    fig.savefig(OUT/f'{name}.png',dpi=260,bbox_inches='tight',pad_inches=.08)
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight',pad_inches=.08)
    plt.close(fig)

def panel_title(ax,label,title):
    ax.text(0,1.04,f'{label}. {title}',transform=ax.transAxes,ha='left',va='bottom',fontweight='bold',fontsize=10,color='#1e2e3b')

def fig1():
    fig,ax=plt.subplots(figsize=(11.5,5.1)); ax.set_axis_off(); ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.text(.02,.96,'Two-stage construction, four-theme partition, and recorded relations',fontsize=14,fontweight='bold',ha='left',va='top',color='#142c3e')
    boxes=[(.03,.66,.19,.16,'Candidate retrieval\n13,574 records','#DCEAF2'),(.275,.66,.19,.16,'Literal screen\n11,102 publications','#F9E6BF'),(.52,.66,.21,.16,'High-specificity corpus $P$\n7,146 publications','#D9F0ED'),(.785,.66,.18,.16,'Four-theme partition\nHard assignment + margin','#E5DEF3'),(.09,.22,.22,.16,'Author affiliations $PI$\nInstitutions and countries','#E6EEF5'),(.39,.22,.22,.16,'Grants $G$ and funders $GI$\nFunding-support relations','#E6EEF5'),(.69,.22,.22,.16,'Policy documents $D$ and issuers $DI$\nCoverage relations','#E6EEF5')]
    for x,y,w,h,label,color in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012,rounding_size=0.008',fc=color,ec='#24465E',lw=1.25))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=9.5,color='#1D3548')
    for a,b in [((.22,.74),(.275,.74)),((.465,.74),(.52,.74)),((.73,.74),(.785,.74))]: ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,color='#24465E',lw=1.2))
    for a,b in [((.64,.66),(.20,.38)),((.64,.66),(.50,.38)),((.64,.66),(.80,.38))]: ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,color='#24465E',lw=1.15))
    save(fig,'figure_1_study_design')

def fig2():
    s=pd.read_csv(BASE/'four_theme_hard_partition_audit'/'four_theme_partition_summary.csv')
    c=pd.read_csv(BASE/'four_theme_hard_partition_audit'/'four_theme_assignment_clarity.csv')
    s=s.set_index('candidate_theme').loc[THEME_ORDER].reset_index()
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(11.3,4.8),gridspec_kw={'width_ratios':[1.5,1]})
    panel_title(ax1,'A','Four-theme hard partition')
    y=np.arange(len(s)); vals=s['n_publications'].to_numpy(); colors=[COLORS[x] for x in s['candidate_theme']]
    ax1.barh(y,vals,color=colors,height=.62); ax1.set_yticks(y,[SHORT[x] for x in s['candidate_theme']]); ax1.invert_yaxis(); ax1.set_xlabel('Publications'); ax1.grid(axis='x',alpha=.25); ax1.spines[['top','right','left']].set_visible(False)
    for yi,v,p in zip(y,vals,s['pct_of_subset']): ax1.text(v+55,yi,f'{v:,} ({p:.1f}%)',va='center',fontsize=9)
    ax1.set_xlim(0,vals.max()*1.28)
    panel_title(ax2,'B','Hard-assignment margin')
    order=['clear_margin_at_least_0_20','moderately_ambiguous_margin_0_10_to_under_0_20','highly_ambiguous_margin_under_0_10']
    labels=['Clear\nmargin ≥ .20','Moderate ambiguity\n.10–<.20','High ambiguity\n< .10']
    cc=c.set_index('assignment_clarity').loc[order].reset_index(); bars=ax2.bar(np.arange(3),cc['pct_of_subset'],color=['#5F8A6D','#D9A441','#BF5B5B'],width=.65)
    ax2.set_ylim(0,85); ax2.set_xticks(range(3),labels); ax2.set_ylabel('Share of publications (%)'); ax2.grid(axis='y',alpha=.25); ax2.spines[['top','right']].set_visible(False)
    for b,n,p in zip(bars,cc['n_publications'],cc['pct_of_subset']): ax2.text(b.get_x()+b.get_width()/2,p+2,f'{n:,}\n({p:.1f}%)',ha='center',va='bottom',fontsize=9)
    fig.suptitle('Four-theme publication-text partition',x=.02,ha='left',fontsize=14,fontweight='bold',color='#142c3e')
    fig.tight_layout(rect=(0,0,.99,.91)); save(fig,'figure_2_four_theme_composition')

def fig3():
    m=pd.read_csv(BASE/'four_theme_linkage_models'/'four_theme_linkage_model_coefficients.csv')
    groups=[('all_themed_publications__funding_support','Funding-support linkage: all thematic publications'),('all_themed_publications__policy_coverage_through_2021','Policy-document coverage: publications through 2021'),('all_themed_publications__policy_coverage_through_2020','Policy-document coverage: publications through 2020')]
    rows=[]
    for gid,glabel in groups:
        x=m.loc[(m.model==gid)&m.term.str.startswith('theme_')].copy()
        for theme in THEME_ORDER:
            term='theme_'+theme
            if term in set(x.term):
                r=x.loc[x.term==term].iloc[0]; rows.append((glabel,theme,r.odds_ratio,r.ci_low,r.ci_high,r.p_value))
    # Reference is displayed as a labelled diamond at OR 1 for all models, then contrasts below.
    fig,ax=plt.subplots(figsize=(11.2,7.2)); ypos=[]; labels=[]; y=0; boundaries=[]
    for group,theme,od,lo,hi,p in rows:
        if not ypos or group!=rows[len(ypos)-1][0]:
            if ypos: boundaries.append(y-.4)
        ypos.append(y); labels.append(SHORT[theme]); y+=1
        if len(rows)>len(ypos) and rows[len(ypos)][0]!=group: y+=.65
    group_y={}
    for yy,(group,*_) in zip(ypos,rows): group_y.setdefault(group,[]).append(yy)
    for yy,(group,theme,od,lo,hi,p) in zip(ypos,rows):
        ax.errorbar(od,yy,xerr=[[od-lo],[hi-od]],fmt='o',color=COLORS[theme],ecolor=COLORS[theme],capsize=3,markersize=6,zorder=3)
        ax.text(5.55,yy,f'{od:.2f} [{lo:.2f}, {hi:.2f}]',va='center',fontsize=9,color='#24313b')
    for b in boundaries: ax.axhline(b,color='#D9DEE2',lw=.8)
    ax.axvline(1,color='#6C737A',ls='--',lw=1); ax.set_xscale('log'); ax.set_xlim(.18,7.1); ax.set_xticks([.25,.5,1,2,4]); ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:g}')); ax.minorticks_off()
    ax.set_yticks(ypos,labels); ax.invert_yaxis(); ax.set_xlabel('Adjusted odds ratio (log scale)'); ax.grid(axis='x',alpha=.22); ax.spines[['top','right']].set_visible(False)
    for group,ys in group_y.items(): ax.text(.19,np.mean(ys)-.45,group,ha='left',va='bottom',fontsize=9,fontweight='bold',color='#1E2E3B',bbox=dict(boxstyle='round,pad=.18',fc='white',ec='none',alpha=.96))
    ax.text(5.55,-.75,'OR [95% CI]',fontweight='bold',fontsize=9)
    fig.suptitle('Adjusted associations of publication themes with recorded relations',x=.02,ha='left',fontsize=14,fontweight='bold',color='#142c3e')
    fig.tight_layout(rect=(0.02,.08,.99,.94)); save(fig,'figure_3_four_theme_associations')

def country_matrix(relation):
    f=BASE/'four_theme_institution_country_landscapes'/'themed_P_country_author_affiliations.csv'
    if relation=='P': x=pd.read_csv(f); target='target_name'
    elif relation=='G': x=pd.read_csv(BASE/'four_theme_institution_country_landscapes'/'themed_G_country_funder_associations.csv'); target='target_name'
    else: x=pd.read_csv(BASE/'four_theme_institution_country_landscapes'/'themed_D_country_issuer_associations.csv'); target='target_name'
    x=x.loc[x.theme!='All high-specificity publications'].copy()
    top=x.groupby(target).source_id.nunique().nlargest(8).index.tolist()
    x=x.loc[x[target].isin(top)]
    mat=x.groupby(['target_name','theme']).source_id.nunique().unstack(fill_value=0).reindex(index=top,columns=THEME_ORDER,fill_value=0)
    return mat.div(mat.sum(axis=0),axis=1)*100

def fig4():
    fig,axes=plt.subplots(1,3,figsize=(12.2,5.8),sharey=False)
    specs=[('P','A','Author affiliations'),('G','B','Grant funders'),('D','C','Policy issuers')]
    all_mats=[country_matrix(key) for key,_,_ in specs]
    vmax=max(20.0,max(float(m.to_numpy().max()) for m in all_mats))
    for ax,(key,letter,title),mat in zip(axes,specs,all_mats):
        ax.imshow(mat.to_numpy(),cmap='Blues',vmin=0,vmax=vmax,aspect='auto')
        ax.set_xticks(range(4),['T1','T2','T3','T4'],fontsize=9,fontweight='bold')
        ax.set_yticks(range(len(mat)),mat.index,fontsize=8.5)
        panel_title(ax,letter,title)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v=mat.iloc[i,j]; ax.text(j,i,f'{v:.1f}',ha='center',va='center',fontsize=7.2,color='white' if v>vmax*.48 else '#1E2E3B')
    fig.suptitle('Country representation across publication themes',x=.02,ha='left',fontsize=14,fontweight='bold',color='#142c3e')
    fig.text(.02,.035,'T1 Risk prediction     T2 Digital care and ethics     T3 Social/affective detection     T4 Neuropsychiatric assessment',fontsize=8.7,color='#364651')
    fig.tight_layout(rect=(.01,.085,.99,.91)); save(fig,'figure_4_four_theme_country_landscapes')

def fig5():
    cfg=pd.read_csv(BASE/'four_theme_local_alignment'/'configuration_by_theme.csv'); null=pd.read_csv(BASE/'four_theme_local_alignment'/'alignment_null_replicates.csv'); res=pd.read_csv(BASE/'four_theme_local_alignment'/'alignment_null_summary.csv')
    fig,axes=plt.subplots(1,3,figsize=(12,4.8),gridspec_kw={'width_ratios':[1.15,1,1]})
    ax=axes[0]; panel_title(ax,'A','Observed local configurations')
    order=['funding_only','policy_only','both','neither']; display=['Funding\nonly','Policy\nonly','Both','Neither']; bottom=np.zeros(4)
    for theme in THEME_ORDER:
        x=cfg.set_index(['candidate_theme','configuration']).reindex([(theme,k) for k in order],fill_value=0)['n_publications'].to_numpy()
        ax.bar(range(4),x,bottom=bottom,color=COLORS[theme],label=SHORT[theme],width=.65); bottom+=x
    ax.set_yscale('log'); ax.set_xticks(range(4),display); ax.set_ylabel('Publications (log scale)'); ax.grid(axis='y',alpha=.22); ax.spines[['top','right']].set_visible(False)
    ax.legend(fontsize=7.5,frameon=False,loc='upper center',bbox_to_anchor=(.5,-.24),ncol=2)
    for i,v in enumerate(bottom): ax.text(i,v*1.18,f'{int(v):,}',ha='center',va='bottom',fontsize=8.5)
    for ax,col,title,obs in [(axes[1],'N_both','Extensive-margin alignment','N publications with both funding-support and policy-document coverage'),(axes[2],'intensive_degree_product_sum','Intensive-margin alignment','Sum of publication funding-support degree × policy-coverage degree')]:
        panel_title(ax,'B' if col=='N_both' else 'C',title); values=null[col].to_numpy(); r=res.loc[res.statistic==obs].iloc[0]
        ax.hist(values,bins=20,color='#DCEAF2',edgecolor='#7895AA',lw=.6); ax.axvline(r.observed,color='#B54B4B',lw=2)
        xmax=max(values.max()*1.05, r.observed*1.05)
        if r.observed>values.max()*1.08:
            ax.set_xlim(values.min()*.95,values.max()*1.08); ax.text(.97,.93,f'Observed = {int(r.observed)}\nNull maximum = {int(values.max())}\nEnrichment = {r.enrichment:.2f}\nTwo-sided p = {r.two_sided_p:.3f}',transform=ax.transAxes,ha='right',va='top',fontsize=7.8,bbox=dict(boxstyle='round,pad=.28',fc='white',ec='#B54B4B',lw=.8))
        else: ax.set_xlim(values.min()*.95,xmax); ax.text(.97,.93,f'Observed = {int(r.observed)}\nEnrichment = {r.enrichment:.2f}\nTwo-sided p = {r.two_sided_p:.3f}',transform=ax.transAxes,ha='right',va='top',fontsize=7.8,bbox=dict(boxstyle='round,pad=.28',fc='white',ec='#B54B4B',lw=.8))
        ax.set_xlabel('Null statistic'); ax.set_ylabel('Permutations'); ax.grid(axis='y',alpha=.18); ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Local alignment of recorded funding-support and policy-document coverage relations',x=.02,ha='left',fontsize=14,fontweight='bold',color='#142c3e')
    fig.text(.02,.015,'Primary null: complete policy-document coverage profiles are reassigned within publication-year strata while funding-support ties remain fixed.',fontsize=8.3,color='#59636d')
    fig.tight_layout(rect=(.01,.10,.99,.91)); save(fig,'figure_5_four_theme_local_alignment')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    fig1(); fig2(); fig3(); fig4(); fig5()
    (OUT/'figure_data_sources.json').write_text(json.dumps({'figure_1':'study design and counts from manuscript construction','figure_2':'four_theme_hard_partition_audit','figure_3':'four_theme_linkage_models','figure_4':'four_theme_institution_country_landscapes','figure_5':'four_theme_local_alignment'},indent=2),encoding='utf-8')
    print('Generated five redesigned manuscript figures in',OUT)
if __name__=='__main__': main()
