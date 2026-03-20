"""
Figure generation from BC prediction result Excel files.
Reads data dynamically (no hardcoded values). 600 DPI TIFF, Arial font.

Usage:
  python bc_prediction_figures.py --results-dir ../results_YYYYMMDD/
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir', type=str, required=True)
    p.add_argument('--output-dir', type=str, default=None)
    p.add_argument('--model', type=str, default='XGB', choices=['LR', 'RF', 'XGB'])
    p.add_argument('--dpi', type=int, default=600)
    return p.parse_args()


def load_pipeline(results_dir, pipeline):
    fname = os.path.join(results_dir, f'bc_prediction_results_{pipeline}.xlsx')
    if not os.path.exists(fname):
        sys.exit(f"File not found: {fname}")
    a = pd.read_excel(fname, sheet_name='PartA_Results')
    b = pd.read_excel(fname, sheet_name='PartB_Results')
    b = b.drop_duplicates(subset=['Variable_Set', 'Model', 'Training', 'Testing']).reset_index(drop=True)
    h = pd.read_excel(fname, sheet_name='Hospital_Summary')
    return a, b, h


def load_both(results_dir):
    ra, rb, h = load_pipeline(results_dir, 'raw')
    na, nb, _ = load_pipeline(results_dir, 'normalized')
    return ra, rb, na, nb, h


def pivot_part_a(part_a):
    sets = part_a['Variable_Set'].unique().tolist()
    models = part_a['Model'].unique().tolist()
    auroc, auprc = {'Variable_Set': sets}, {'Variable_Set': sets}
    nf = {}
    for m in models:
        mdf = part_a[part_a['Model'] == m].set_index('Variable_Set')
        auroc[m] = [mdf.loc[s, 'Test_AUROC'] for s in sets]
        auprc[m] = [mdf.loc[s, 'Test_AUPRC'] for s in sets]
    for s in sets:
        nf[s] = int(part_a[part_a['Variable_Set'] == s].iloc[0]['N_Features'])
    return auroc, auprc, nf, models


def cross_val_matrix(part_b, vset, model):
    sub = part_b[(part_b['Variable_Set'] == vset) & (part_b['Model'] == model)
                 & (part_b['Testing'] != 'Pooled')]
    result = {}
    for src in sub['Training'].unique():
        result[src] = {}
        for _, r in sub[sub['Training'] == src].iterrows():
            result[src][r['Testing']] = r['AUROC']
    return result


# --- Style ---
SET_SHORT = {'Set1_Inflammation': 'Inflam', 'Set2_Hematology': 'Hemat',
             'Set3_Other': 'Other', 'Set4_Inflam+Hemat': 'I+H',
             'Set5_Inflam+Other': 'I+O', 'Set6_Hemat+Other': 'H+O',
             'Set7_All': 'All'}
SET_FULL = {'Set1_Inflammation': 'Inflam', 'Set2_Hematology': 'Hemat',
            'Set3_Other': 'Other', 'Set4_Inflam+Hemat': 'Inflam+\nHemat',
            'Set5_Inflam+Other': 'Inflam+\nOther',
            'Set6_Hemat+Other': 'Hemat+\nOther', 'Set7_All': 'All'}
COLORS = {'LR': '#4ECDC4', 'RF': '#FF6B6B', 'XGB': '#45B7D1'}
MDISP = {'LR': 'Logistic Regression', 'RF': 'Random Forest', 'XGB': 'XGBoost*'}
MNOTE = '*XGBoost: native missing value handling; LR/RF: median imputation'


def save_fig(fig, name, outdir, dpi):
    fig.savefig(os.path.join(outdir, f"{name}.tiff"), dpi=dpi,
                bbox_inches='tight', facecolor='white', format='tiff',
                pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)


def fig1(auroc, auprc, nf, models, outdir, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sets = auroc['Variable_Set']
    labels = [f"{SET_FULL.get(s, s)}\n(n={nf[s]})" for s in sets]
    x = np.arange(len(sets))
    w = 0.25
    for ai, (data, metric, title) in enumerate([
        (auroc, 'AUROC', 'A) AUROC by Variable Set'),
        (auprc, 'AUPRC', 'B) AUPRC by Variable Set'),
    ]):
        ax = axes[ai]
        off = np.linspace(-w, w, len(models))
        for mi, m in enumerate(models):
            vals = data[m]
            ax.bar(x + off[mi], vals, w, label=MDISP.get(m, m),
                   color=COLORS.get(m, f'C{mi}'), edgecolor='white', linewidth=0.5)
            for i, v in enumerate(vals):
                ax.text(x[i] + off[mi], v + 0.008, f'{v:.3f}',
                        ha='center', va='bottom', fontsize=7, fontweight='bold')
        ax.set_ylabel(metric, fontweight='bold')
        ax.set_xlabel('Variable Set', fontweight='bold')
        ax.set_title(title, fontweight='bold', fontsize=13)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        if metric == 'AUROC':
            ax.set_ylim(0.55, 0.88)
            ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5)
        else:
            av = [v for m in models for v in data[m]]
            ax.set_ylim(min(av) - 0.05, max(av) + 0.08)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=len(models),
               bbox_to_anchor=(0.5, -0.02), framealpha=0.9)
    fig.text(0.5, -0.08, MNOTE, ha='center', fontsize=9, fontstyle='italic')
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    save_fig(fig, 'figure1_variable_set_comparison', outdir, dpi)


def fig2(raw_a, norm_a, outdir, dpi):
    models = raw_a['Model'].unique().tolist()
    sets = raw_a['Variable_Set'].unique().tolist()
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    w, x = 0.35, np.arange(len(sets))
    sl = [SET_SHORT.get(s, s) for s in sets]
    pc = {'Raw': '#2C3E50', 'Normalized': '#E67E22'}
    for mi, m in enumerate(models):
        rm = raw_a[raw_a['Model'] == m].set_index('Variable_Set')
        nm = norm_a[norm_a['Model'] == m].set_index('Variable_Set')
        for ri, met in enumerate(['Test_AUROC', 'Test_AUPRC']):
            ax = axes[ri, mi]
            rv = [rm.loc[s, met] for s in sets]
            nv = [nm.loc[s, met] for s in sets]
            ax.bar(x - w/2, rv, w, color=pc['Raw'], edgecolor='white', linewidth=0.5)
            ax.bar(x + w/2, nv, w, color=pc['Normalized'], edgecolor='white', linewidth=0.5)
            for i, (r, n) in enumerate(zip(rv, nv)):
                ax.text(i - w/2, r + 0.005, f'{r:.3f}', ha='center', fontsize=7, fontweight='bold')
                ax.text(i + w/2, n + 0.005, f'{n:.3f}', ha='center', fontsize=7, fontweight='bold')
            ax.set_xticks(x); ax.set_xticklabels(sl, fontsize=9)
            if ri == 0: ax.set_title(MDISP.get(m, m), fontweight='bold', fontsize=12)
            if ri == 1: ax.set_xlabel('Variable Set', fontweight='bold')
            if mi == 0: ax.set_ylabel(met.replace('Test_', ''), fontweight='bold')
            if 'AUROC' in met:
                ax.set_ylim(0.55, 0.88)
                ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5)
            else:
                av = rv + nv
                ax.set_ylim(min(av) - 0.05, max(av) + 0.08)
    h = [Patch(facecolor=pc['Raw'], label='Raw'),
         Patch(facecolor=pc['Normalized'], label='Normalized')]
    fig.legend(handles=h, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.01))
    fig.text(0.5, -0.05, MNOTE, ha='center', fontsize=9, fontstyle='italic')
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    save_fig(fig, 'figure2_pipeline_comparison', outdir, dpi)


def fig3(part_b, model, outdir, dpi):
    sel = [('Set2_Hematology', 'Set2: Hematology'),
           ('Set6_Hemat+Other', 'Set6: Hematology + Other'),
           ('Set7_All', 'Set7: All Features')]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    train_ord = ['Pooled', 'A', 'B', 'C']
    test_ord = ['A', 'B', 'C']
    for idx, (sn, title) in enumerate(sel):
        ax = axes[idx]
        cv = cross_val_matrix(part_b, sn, model)
        mat = np.array([[cv.get(tr, {}).get(te, np.nan) for te in test_ord] for tr in train_ord])
        im = ax.imshow(mat, cmap='RdYlGn', aspect='auto', vmin=0.55, vmax=0.85)
        ax.set_xticks(range(len(test_ord))); ax.set_xticklabels(test_ord)
        ax.set_yticks(range(len(train_ord))); ax.set_yticklabels(train_ord)
        ax.set_xlabel('Test Hospital', fontweight='bold')
        if idx == 0: ax.set_ylabel('Training Data', fontweight='bold')
        ax.set_title(title, fontweight='bold', fontsize=11)
        for i in range(len(train_ord)):
            for j in range(len(test_ord)):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                            fontsize=10, fontweight='bold')
                    if i > 0 and train_ord[i] == test_ord[j]:
                        ax.add_patch(plt.Rectangle((j-.5, i-.5), 1, 1,
                                                   fill=False, edgecolor='black', lw=2))
    fig.colorbar(im, cax=fig.add_axes([0.93, 0.15, 0.02, 0.7])).set_label('AUROC')
    plt.tight_layout(rect=[0, 0.05, 0.92, 1])
    save_fig(fig, 'figure3_multicenter_heatmap', outdir, dpi)


def fig4(part_b, model, outdir, dpi):
    sel = [('Set2_Hematology', 'Set2: Hematology'),
           ('Set6_Hemat+Other', 'Set6: Hematology + Other'),
           ('Set7_All', 'Set7: All Features')]
    hosps = ['A', 'B', 'C']
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for idx, (sn, title) in enumerate(sel):
        ax = axes[idx]
        cv = cross_val_matrix(part_b, sn, model)
        pp = [cv.get('Pooled', {}).get(h, np.nan) for h in hosps]
        sp = [cv.get(h, {}).get(h, np.nan) for h in hosps]
        x, w = np.arange(len(hosps)), 0.35
        ax.bar(x - w/2, pp, w, label='Pooled', color='#3498DB', edgecolor='white')
        ax.bar(x + w/2, sp, w, label='Hospital-Specific', color='#E74C3C', edgecolor='white')
        for i, (p, s) in enumerate(zip(pp, sp)):
            ax.text(i - w/2, p + 0.005, f'{p:.3f}', ha='center', fontsize=9, fontweight='bold')
            ax.text(i + w/2, s + 0.005, f'{s:.3f}', ha='center', fontsize=9, fontweight='bold')
            d = p - s
            ax.annotate(f'\u0394={d:+.3f}', xy=(i, max(p, s) + 0.02),
                        ha='center', fontsize=8, color='green' if d > 0 else 'red',
                        fontweight='bold')
        ax.set_ylabel('AUROC', fontweight='bold')
        ax.set_xlabel('Hospital', fontweight='bold')
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_xticks(x); ax.set_xticklabels(hosps)
        av = pp + sp
        ax.set_ylim(min(av) - 0.05, max(av) + 0.08)
        ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    save_fig(fig, 'figure4_pooled_vs_specific', outdir, dpi)


def main():
    args = parse_args()
    outdir = args.output_dir or args.results_dir
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 10,
                         'axes.labelsize': 11, 'axes.titlesize': 12})
    raw_a, raw_b, norm_a, norm_b, _ = load_both(args.results_dir)
    auroc, auprc, nf, models = pivot_part_a(raw_a)
    fig1(auroc, auprc, nf, models, outdir, args.dpi)
    fig2(raw_a, norm_a, outdir, args.dpi)
    fig3(raw_b, args.model, outdir, args.dpi)
    fig4(raw_b, args.model, outdir, args.dpi)


if __name__ == "__main__":
    main()
