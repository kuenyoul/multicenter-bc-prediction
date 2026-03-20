"""
SHAP Analysis + Decision Curve Analysis (DCA) for Blood Culture Prediction

Loads trained models from pickle artifacts produced by bc_prediction_analysis.py.
Generates SHAP summary plot (XGBoost, Set7) and DCA comparing Set1 vs Set2 vs Set7.
"""

import argparse
import datetime
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dcurves import dca
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import shap
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')
warnings.filterwarnings('ignore', category=FutureWarning)

# --- CLI ---
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--artifacts', type=str, default=None,
                    help='Path to model_artifacts.pkl from main analysis')
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
parser.add_argument('--outdir', type=str,
                    default=os.path.abspath(os.path.join(
                        os.path.dirname(__file__), '..', f'results_shap_dca_{ts}')))
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)

# --- Variable definitions ---
common_vars = ['sex', 'age']
set1 = ['crp', 'procal']
set2 = ['wbc', 'hb', 'hct', 'rbc', 'plt', 'mcv', 'mch', 'mchc',
        'rdw', 'pdw', 'mpv', 'neut', 'lympho', 'mono', 'eosino',
        'baso', 'anc', 'esr']
set3 = ['bun', 'crea', 'glucose', 'totalbil', 'ast', 'alt', 'alk',
        'albumin', 'ld_ratio', 'ferr_ratio_log10', 'lactic',
        'sodium', 'pota', 'chlo', 'calc', 'mg', 'phos',
        'pt', 'inr', 'aptt', 'ph', 'pco2', 'po2', 'hco3']

variable_sets = {
    'Set1_Inflammation': common_vars + set1,
    'Set2_Hematology': common_vars + set2,
    'Set7_All': common_vars + set1 + set2 + set3,
}

DISPLAY = {
    'sex': 'Sex', 'age': 'Age', 'crp': 'CRP', 'procal': 'Procalcitonin',
    'wbc': 'WBC', 'hb': 'Hemoglobin', 'hct': 'Hematocrit', 'rbc': 'RBC',
    'plt': 'Platelet', 'mcv': 'MCV', 'mch': 'MCH', 'mchc': 'MCHC',
    'rdw': 'RDW', 'pdw': 'PDW', 'mpv': 'MPV',
    'neut': 'Neutrophil %', 'lympho': 'Lymphocyte %', 'mono': 'Monocyte %',
    'eosino': 'Eosinophil %', 'baso': 'Basophil %', 'anc': 'ANC', 'esr': 'ESR',
    'bun': 'BUN', 'crea': 'Creatinine', 'glucose': 'Glucose',
    'totalbil': 'Total bilirubin', 'ast': 'AST', 'alt': 'ALT', 'alk': 'ALP',
    'albumin': 'Albumin', 'ld_ratio': 'LDH/ULN ratio',
    'ferr_ratio_log10': 'log10(Ferritin/ULN)', 'lactic': 'Lactic acid',
    'sodium': 'Sodium', 'pota': 'Potassium', 'chlo': 'Chloride',
    'calc': 'Calcium', 'mg': 'Magnesium', 'phos': 'Phosphorus',
    'pt': 'PT', 'inr': 'INR', 'aptt': 'aPTT',
    'ph': 'pH', 'pco2': 'pCO2', 'po2': 'pO2', 'hco3': 'HCO3',
}

# --- Data ---
df = pd.read_excel(args.input)
df.columns = df.columns.str.lower()
df['outcome'] = df['bc_pos_ncont'].fillna(0).astype(int)
df_raw = df.drop(columns=['ldh', 'ferr'], errors='ignore')
df_raw['strat_var'] = df_raw['hospital'].astype(str) + '_' + df_raw['outcome'].astype(str)
train_df, test_df = train_test_split(df_raw, test_size=0.2, random_state=42,
                                     stratify=df_raw['strat_var'])
y_test = test_df['outcome'].values

# --- Load or train models ---
artifacts = {}
if args.artifacts and os.path.exists(args.artifacts):
    with open(args.artifacts, 'rb') as f:
        artifacts = pickle.load(f)


def lasso_select(dataframe, features):
    avail = [f for f in features if f in dataframe.columns]
    X = dataframe[avail].copy()
    cols = X.columns[X.notna().any()].tolist()
    X = X[cols]
    y = dataframe['outcome'].values
    imp = SimpleImputer(strategy='median', keep_empty_features=True)
    sc = StandardScaler()
    X_sc = sc.fit_transform(imp.fit_transform(X))
    lasso = LassoCV(cv=5, random_state=42, n_jobs=-1, max_iter=2000)
    lasso.fit(X_sc, y)
    sel = [cols[i] for i, c in enumerate(lasso.coef_) if c != 0]
    return sel if sel else cols


def get_model(set_name):
    key = ('Raw', set_name, 'XGB')
    if key in artifacts:
        a = artifacts[key]
        return a['model'], a['selected_features']
    feats = lasso_select(train_df, variable_sets[set_name])
    base = XGBClassifier(random_state=42, use_label_encoder=False,
                         eval_metric='logloss', verbosity=0)
    gs = GridSearchCV(base, {'n_estimators': [100, 200], 'learning_rate': [0.1, 0.05]},
                      cv=3, scoring='roc_auc', n_jobs=-1)
    gs.fit(train_df[feats].copy(), train_df['outcome'].values)
    return gs.best_estimator_, feats


# --- SHAP ---
xgb7, sel7 = get_model('Set7_All')
X_test7 = test_df[sel7].copy()
explainer = shap.TreeExplainer(xgb7)
shap_vals = explainer.shap_values(X_test7)

X_disp = X_test7.copy()
X_disp.columns = [DISPLAY.get(c, c) for c in sel7]

plt.rcParams.update({'font.family': 'Arial', 'font.size': 10})

fig, _ = plt.subplots(figsize=(10, 12))
shap.summary_plot(shap_vals, X_disp, plot_type='dot', max_display=20,
                  show=False, plot_size=None)
plt.tight_layout()
fig.savefig(os.path.join(args.outdir, 'figure_shap_summary.tiff'),
            dpi=600, bbox_inches='tight', facecolor='white',
            format='tiff', pil_kwargs={'compression': 'tiff_lzw'})
plt.close(fig)

fig, _ = plt.subplots(figsize=(10, 10))
shap.summary_plot(shap_vals, X_disp, plot_type='bar', max_display=20,
                  show=False, plot_size=None)
plt.tight_layout()
fig.savefig(os.path.join(args.outdir, 'figure_shap_bar.tiff'),
            dpi=600, bbox_inches='tight', facecolor='white',
            format='tiff', pil_kwargs={'compression': 'tiff_lzw'})
plt.close(fig)

pd.DataFrame({
    'Feature': sel7,
    'Display_Name': [DISPLAY.get(c, c) for c in sel7],
    'Mean_abs_SHAP': np.abs(shap_vals).mean(axis=0),
}).sort_values('Mean_abs_SHAP', ascending=False).to_csv(
    os.path.join(args.outdir, 'shap_feature_importance.csv'), index=False)

# --- DCA ---
preds = {}
for sn in ['Set1_Inflammation', 'Set2_Hematology', 'Set7_All']:
    m, f = get_model(sn)
    preds[sn] = m.predict_proba(test_df[f].copy())[:, 1]

dca_df = pd.DataFrame({
    'outcome': y_test,
    'Inflammation': preds['Set1_Inflammation'],
    'Hematology': preds['Set2_Hematology'],
    'All_variables': preds['Set7_All'],
})
dca_result = dca(data=dca_df, outcome='outcome',
                 modelnames=['Inflammation', 'Hematology', 'All_variables'],
                 thresholds=np.arange(0.0, 0.51, 0.01))

fig, ax = plt.subplots(figsize=(10, 7))
cfg = {
    'all':  ('#333', 'Treat all', 1, '--'),
    'none': ('#999', 'Treat none', 1, '-'),
    'Inflammation':  ('#FF6B6B', 'Set 1: Inflammation (CRP + PCT)', 2, '-'),
    'Hematology':    ('#4ECDC4', 'Set 2: Hematology (CBC)', 2, '-'),
    'All_variables': ('#45B7D1', 'Set 7: All variables', 2.5, '-'),
}
for mn in ['all', 'none', 'Inflammation', 'Hematology', 'All_variables']:
    sub = dca_result[dca_result['model'] == mn]
    if sub.empty:
        continue
    c, lbl, lw, ls = cfg[mn]
    ax.plot(sub['threshold'], sub['net_benefit'], color=c, label=lbl,
            linewidth=lw, linestyle=ls)
ax.set_xlabel('Threshold Probability', fontweight='bold', fontsize=12)
ax.set_ylabel('Net Benefit', fontweight='bold', fontsize=12)
ax.set_xlim(0, 0.50)
ax.set_ylim(-0.05, max(dca_result['net_benefit'].max() * 1.1, 0.15))
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='black', linewidth=0.5)
plt.tight_layout()
fig.savefig(os.path.join(args.outdir, 'figure_dca.tiff'),
            dpi=600, bbox_inches='tight', facecolor='white',
            format='tiff', pil_kwargs={'compression': 'tiff_lzw'})
plt.close(fig)

dca_result.to_csv(os.path.join(args.outdir, 'dca_results.csv'), index=False)
