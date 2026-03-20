"""
Blood Culture Positivity Prediction Model Analysis

Part A: 7 variable sets x 3 models (LR, RF, XGB) with 10-fold CV
Part B: Multi-center generalization analysis (pooled vs hospital-specific)
Part C: Bootstrap 95% CI, Brier score, Calibration plot

XGBoost uses native missing value handling (no imputation).
LR and RF use median imputation with per-fold fitting.
"""

import argparse
import datetime
import os
import pickle
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss)
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     GridSearchCV)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')
warnings.filterwarnings('ignore', category=FutureWarning)

# --- CLI ---
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='data_analysis.xlsx')
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
parser.add_argument('--outdir', type=str,
                    default=os.path.abspath(os.path.join(
                        os.path.dirname(__file__), '..', f'results_{ts}')))
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)

N_BOOTSTRAP = 1000

# --- Variable definitions ---
common_vars = ['sex', 'age']
set1_inflammation = ['crp', 'procal']
set2_hematology = ['wbc', 'hb', 'hct', 'rbc', 'plt', 'mcv', 'mch', 'mchc',
                   'rdw', 'pdw', 'mpv', 'neut', 'lympho', 'mono', 'eosino',
                   'baso', 'anc', 'esr']
set3_other = ['bun', 'crea', 'glucose', 'totalbil', 'ast', 'alt', 'alk',
              'albumin', 'ldh', 'ferr', 'ld_ratio', 'ferr_ratio_log10',
              'lactic', 'sodium', 'pota', 'chlo', 'calc', 'mg', 'phos',
              'pt', 'inr', 'aptt', 'ph', 'pco2', 'po2', 'hco3']

variable_sets = {
    'Set1_Inflammation': common_vars + set1_inflammation,
    'Set2_Hematology': common_vars + set2_hematology,
    'Set3_Other': common_vars + set3_other,
    'Set4_Inflam+Hemat': common_vars + set1_inflammation + set2_hematology,
    'Set5_Inflam+Other': common_vars + set1_inflammation + set3_other,
    'Set6_Hemat+Other': common_vars + set2_hematology + set3_other,
    'Set7_All': common_vars + set1_inflammation + set2_hematology + set3_other,
}

hospital_names = {1: 'Hospital A', 2: 'Hospital B', 3: 'Hospital C'}

# --- Data loading ---
df = pd.read_excel(args.input)
df.columns = df.columns.str.lower()
df['outcome'] = df['bc_pos_ncont'].fillna(0).astype(int)

# --- Missing value analysis ---
all_vars = common_vars + set1_inflammation + set2_hematology + set3_other
missing_report = []
for var in all_vars:
    if var in df.columns:
        n_miss = df[var].isna().sum()
        pct = (n_miss / len(df)) * 100
        missing_report.append({
            'Variable': var, 'N_Total': len(df),
            'N_Present': len(df) - n_miss, 'N_Missing': n_miss,
            'Pct_Present': round(100 - pct, 2), 'Pct_Missing': round(pct, 2),
        })
missing_df = pd.DataFrame(missing_report)

# --- Exclude variables with >95% missing ---
vars_to_exclude = set(
    missing_df.loc[missing_df['Pct_Missing'] > 95, 'Variable'].tolist()
)
variable_sets = {
    name: [v for v in feats if v not in vars_to_exclude]
    for name, feats in variable_sets.items()
}

# --- Baseline characteristics ---
def calc_baseline(df, var, group='outcome'):
    pos = df[df[group] == 1][var].dropna()
    neg = df[df[group] == 0][var].dropna()
    if df[var].dtype not in ['int64', 'float64']:
        return None
    _, p = stats.ttest_ind(pos, neg, equal_var=False) if len(pos) > 1 and len(neg) > 1 else (0, np.nan)
    return {
        'Variable': var,
        'BC_Positive': f"{pos.mean():.1f} ({pos.std():.1f})",
        'BC_Positive_N': len(pos),
        'BC_Negative': f"{neg.mean():.1f} ({neg.std():.1f})",
        'BC_Negative_N': len(neg),
        'P_value': f"{p:.4f}" if not np.isnan(p) else "N/A",
    }

lab_categories = {
    'Hematology': set2_hematology,
    'Inflammation': ['crp', 'procal', 'lactic', 'ld_ratio', 'ferr_ratio_log10'],
    'Chemistry': ['bun', 'crea', 'glucose', 'totalbil', 'ast', 'alt', 'alk', 'albumin'],
    'Electrolytes': ['sodium', 'pota', 'chlo', 'calc', 'mg', 'phos'],
    'Coagulation': ['pt', 'inr', 'aptt'],
    'Blood Gas': ['ph', 'pco2', 'po2', 'hco3'],
}
baseline_results = []
for cat, vlist in lab_categories.items():
    for var in vlist:
        if var in df.columns:
            r = calc_baseline(df, var)
            if r:
                baseline_results.append({**r, 'Category': cat})
baseline_df = pd.DataFrame(baseline_results)

# Hospital summary
hospital_stats = df.groupby('hospital').agg(
    N_samples=('outcome', 'count'),
    N_patients=('id_patient', 'nunique'),
    N_positive=('outcome', 'sum'),
    Positive_rate=('outcome', 'mean'),
).round(4)
hospital_stats['Positive_rate_pct'] = (hospital_stats['Positive_rate'] * 100).round(2)
hospital_stats.index = hospital_stats.index.map(lambda x: hospital_names.get(x, f'Hospital {x}'))

# --- Train/Test split ---
df['strat_var'] = df['hospital'].astype(str) + '_' + df['outcome'].astype(str)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42,
                                     stratify=df['strat_var'])

# --- Pipeline setup (Raw vs Normalized) ---
continuous_vars = [v for v in (set1_inflammation + set2_hematology + set3_other + ['age'])
                   if v in df.columns]

train_norm, test_norm = train_df.copy(), test_df.copy()
for var in continuous_vars:
    if abs(train_norm[var].skew()) > 2.0:
        mn = train_norm[var].min()
        offset = abs(mn) + 1 if mn <= 0 else 0
        train_norm[var] = np.log1p(np.maximum(train_norm[var] + offset, 0))
        test_norm[var] = np.log1p(np.maximum(test_norm[var] + offset, 0))

scaler_global = StandardScaler()
train_norm[continuous_vars] = scaler_global.fit_transform(train_norm[continuous_vars])
test_norm[continuous_vars] = scaler_global.transform(test_norm[continuous_vars])

train_raw = train_df.drop(columns=['ldh', 'ferr'], errors='ignore')
test_raw = test_df.drop(columns=['ldh', 'ferr'], errors='ignore')
train_norm = train_norm.drop(columns=['ld_ratio', 'ferr_ratio_log10'], errors='ignore')
test_norm = test_norm.drop(columns=['ld_ratio', 'ferr_ratio_log10'], errors='ignore')

pipelines = {
    'Raw': {'train': train_raw, 'test': test_raw},
    'Normalized': {'train': train_norm, 'test': test_norm},
}


# --- Helper functions ---
def get_valid_features(dataframe, feature_list):
    return [f for f in feature_list if f in dataframe.columns]


def lasso_select(dataframe, features, target='outcome'):
    available = get_valid_features(dataframe, features)
    X = dataframe[available].copy()
    valid_cols = X.columns[X.notna().any()].tolist()
    X = X[valid_cols]
    y = dataframe[target].values
    imp = SimpleImputer(strategy='median', keep_empty_features=True)
    sc = StandardScaler()
    X_sc = sc.fit_transform(imp.fit_transform(X))
    lasso = LassoCV(cv=5, random_state=42, n_jobs=-1, max_iter=2000)
    lasso.fit(X_sc, y)
    selected = [valid_cols[i] for i, c in enumerate(lasso.coef_) if c != 0]
    return selected if selected else valid_cols


def evaluate_cv(model, X_raw, y, cv=10, model_name='XGB'):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    aurocs, auprcs = [], []
    for tr_idx, vl_idx in skf.split(X_raw, y):
        X_tr, X_vl = X_raw.iloc[tr_idx], X_raw.iloc[vl_idx]
        y_tr, y_vl = y[tr_idx], y[vl_idx]
        if model_name in ('LR', 'RF'):
            imp = SimpleImputer(strategy='median', keep_empty_features=True)
            X_tr = pd.DataFrame(imp.fit_transform(X_tr), columns=X_raw.columns)
            X_vl = pd.DataFrame(imp.transform(X_vl), columns=X_raw.columns)
            if model_name == 'LR':
                sc = StandardScaler()
                X_tr = pd.DataFrame(sc.fit_transform(X_tr), columns=X_raw.columns)
                X_vl = pd.DataFrame(sc.transform(X_vl), columns=X_raw.columns)
        m = clone(model)
        m.fit(X_tr, y_tr)
        yp = m.predict_proba(X_vl)[:, 1]
        aurocs.append(roc_auc_score(y_vl, yp))
        auprcs.append(average_precision_score(y_vl, yp))
    return {'AUROC_mean': np.mean(aurocs), 'AUROC_std': np.std(aurocs),
            'AUPRC_mean': np.mean(auprcs), 'AUPRC_std': np.std(auprcs)}


def bootstrap_ci(y_true, y_prob, n_boot=N_BOOTSTRAP):
    rng = np.random.RandomState(42)
    n = len(y_true)
    aurocs, auprcs, briers = [], [], []
    for _ in range(n_boot * 10):
        if len(aurocs) >= n_boot:
            break
        idx = rng.randint(0, n, size=n)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        aurocs.append(roc_auc_score(yt, yp))
        auprcs.append(average_precision_score(yt, yp))
        briers.append(brier_score_loss(yt, yp))
    return {
        'AUROC_CI_lower': round(np.percentile(aurocs, 2.5), 4),
        'AUROC_CI_upper': round(np.percentile(aurocs, 97.5), 4),
        'AUPRC_CI_lower': round(np.percentile(auprcs, 2.5), 4),
        'AUPRC_CI_upper': round(np.percentile(auprcs, 97.5), 4),
        'Brier': round(brier_score_loss(y_true, y_prob), 4),
        'Brier_CI_lower': round(np.percentile(briers, 2.5), 4),
        'Brier_CI_upper': round(np.percentile(briers, 97.5), 4),
    }


# --- Part A + Test evaluation ---
pipeline_results_a = {}
pipeline_results_test_a = {}
pipeline_results_b = {}
pipeline_predictions = {}
model_artifacts = {}

for pipe_name, data in pipelines.items():
    cur_train, cur_test = data['train'], data['test']

    models = {
        'LR': LogisticRegression(max_iter=1000, random_state=42),
        'RF': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'XGB': XGBClassifier(n_estimators=100, random_state=42,
                             use_label_encoder=False, eval_metric='logloss',
                             verbosity=0),
    }
    results_a, test_results_a = [], []

    for set_name, features in variable_sets.items():
        sel = lasso_select(cur_train, features)

        imputer = SimpleImputer(strategy='median', keep_empty_features=True)
        X_train_imp = pd.DataFrame(imputer.fit_transform(cur_train[sel]),
                                   columns=sel, index=cur_train.index)
        X_train_native = cur_train[sel].copy()
        y_train = cur_train['outcome'].values

        # 10-fold CV
        for mn, model in models.items():
            cv = evaluate_cv(model, X_train_native, y_train, cv=10, model_name=mn)
            results_a.append({
                'Variable_Set': set_name, 'Model': mn,
                'N_Features': len(sel),
                'AUROC_mean': round(cv['AUROC_mean'], 4),
                'AUROC_std': round(cv['AUROC_std'], 4),
                'AUPRC_mean': round(cv['AUPRC_mean'], 4),
                'AUPRC_std': round(cv['AUPRC_std'], 4),
            })

        # Test set evaluation
        X_test_imp = pd.DataFrame(imputer.transform(cur_test[sel]),
                                  columns=sel, index=cur_test.index)
        X_test_native = cur_test[sel].copy()
        y_test = cur_test['outcome'].values

        for mn, model in models.items():
            mc = clone(model)
            if mn == 'RF':
                gs = GridSearchCV(mc, {'n_estimators': [100, 200], 'max_depth': [None, 10]},
                                  cv=3, scoring='roc_auc', n_jobs=-1)
                gs.fit(X_train_imp, y_train)
                mc = gs.best_estimator_
                yp = mc.predict_proba(X_test_imp)[:, 1]
            elif mn == 'XGB':
                gs = GridSearchCV(mc, {'n_estimators': [100, 200], 'learning_rate': [0.1, 0.05]},
                                  cv=3, scoring='roc_auc', n_jobs=-1)
                gs.fit(X_train_native, y_train)
                mc = gs.best_estimator_
                yp = mc.predict_proba(X_test_native)[:, 1]
            else:  # LR
                sc = StandardScaler()
                X_tr_sc = pd.DataFrame(sc.fit_transform(X_train_imp), columns=sel)
                X_ts_sc = pd.DataFrame(sc.transform(X_test_imp), columns=sel)
                mc.fit(X_tr_sc, y_train)
                yp = mc.predict_proba(X_ts_sc)[:, 1]

            ci = bootstrap_ci(y_test, yp)
            pipeline_predictions[(pipe_name, set_name, mn)] = (y_test, yp)
            model_artifacts[(pipe_name, set_name, mn)] = {
                'model': mc, 'selected_features': sel,
            }
            test_results_a.append({
                'Variable_Set': set_name, 'Model': mn,
                'Test_AUROC': round(roc_auc_score(y_test, yp), 4),
                'Test_AUPRC': round(average_precision_score(y_test, yp), 4),
                **ci,
            })

    pipeline_results_a[pipe_name] = pd.DataFrame(results_a)
    pipeline_results_test_a[pipe_name] = pd.DataFrame(test_results_a)

    # --- Part B: Multi-center generalization ---
    selected_sets = ['Set2_Hematology', 'Set6_Hemat+Other', 'Set7_All']
    results_b = []
    hospitals = [1, 2, 3]
    hnames_b = {1: 'A', 2: 'B', 3: 'C'}

    for set_name in selected_sets:
        sel = lasso_select(cur_train, variable_sets[set_name])

        # Pooled model
        imp_p = SimpleImputer(strategy='median', keep_empty_features=True)
        X_tr_imp = pd.DataFrame(imp_p.fit_transform(cur_train[sel]), columns=sel)
        X_tr_nat = cur_train[sel].copy()
        y_tr = cur_train['outcome'].values
        sc_p = StandardScaler()
        X_tr_sc = pd.DataFrame(sc_p.fit_transform(X_tr_imp), columns=sel)

        pooled = {
            'LR': LogisticRegression(max_iter=1000, random_state=42).fit(X_tr_sc, y_tr),
            'RF': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(X_tr_imp, y_tr),
            'XGB': XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False,
                                 eval_metric='logloss', verbosity=0).fit(X_tr_nat, y_tr),
        }

        for th in ['Pooled'] + hospitals:
            t_df = cur_test if th == 'Pooled' else cur_test[cur_test['hospital'] == th]
            if len(t_df) == 0:
                continue
            Xt_imp = pd.DataFrame(imp_p.transform(t_df[sel]), columns=sel)
            Xt_nat = t_df[sel].copy()
            Xt_sc = pd.DataFrame(sc_p.transform(Xt_imp), columns=sel)
            yt = t_df['outcome'].values
            for mn in ['LR', 'RF', 'XGB']:
                Xu = Xt_sc if mn == 'LR' else (Xt_imp if mn == 'RF' else Xt_nat)
                auc = roc_auc_score(yt, pooled[mn].predict_proba(Xu)[:, 1])
                lbl = 'Pooled' if th == 'Pooled' else hnames_b[th]
                results_b.append({'Variable_Set': set_name, 'Model': mn,
                                  'Training': 'Pooled', 'Testing': lbl,
                                  'AUROC': round(auc, 4)})

        # Hospital-specific
        for trh in hospitals:
            tr_h = cur_train[cur_train['hospital'] == trh]
            if len(tr_h) == 0:
                continue
            imp_h = SimpleImputer(strategy='median', keep_empty_features=True)
            X_h_imp = pd.DataFrame(imp_h.fit_transform(tr_h[sel]), columns=sel)
            X_h_nat = tr_h[sel].copy()
            y_h = tr_h['outcome'].values
            sc_h = StandardScaler()
            X_h_sc = pd.DataFrame(sc_h.fit_transform(X_h_imp), columns=sel)
            hmodels = {
                'LR': LogisticRegression(max_iter=1000, random_state=42).fit(X_h_sc, y_h),
                'RF': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(X_h_imp, y_h),
                'XGB': XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False,
                                     eval_metric='logloss', verbosity=0).fit(X_h_nat, y_h),
            }
            for tsh in hospitals:
                t_h = cur_test[cur_test['hospital'] == tsh]
                if len(t_h) == 0:
                    continue
                Xt_imp = pd.DataFrame(imp_h.transform(t_h[sel]), columns=sel)
                Xt_nat = t_h[sel].copy()
                Xt_sc = pd.DataFrame(sc_h.transform(Xt_imp), columns=sel)
                yt = t_h['outcome'].values
                for mn in ['LR', 'RF', 'XGB']:
                    Xu = Xt_sc if mn == 'LR' else (Xt_imp if mn == 'RF' else Xt_nat)
                    auc = roc_auc_score(yt, hmodels[mn].predict_proba(Xu)[:, 1])
                    results_b.append({'Variable_Set': set_name, 'Model': mn,
                                      'Training': hnames_b[trh], 'Testing': hnames_b[tsh],
                                      'AUROC': round(auc, 4)})

    pipeline_results_b[pipe_name] = pd.DataFrame(results_b)

# --- Save results ---
for pn in pipelines:
    merged = pipeline_results_a[pn].merge(pipeline_results_test_a[pn],
                                          on=['Variable_Set', 'Model'])
    fname = os.path.join(args.outdir, f"bc_prediction_results_{pn.lower()}.xlsx")
    with pd.ExcelWriter(fname, engine='openpyxl') as w:
        merged.to_excel(w, sheet_name='PartA_Results', index=False)
        pipeline_results_b[pn].to_excel(w, sheet_name='PartB_Results', index=False)
        missing_df.to_excel(w, sheet_name='Missing_Analysis', index=False)
        baseline_df.to_excel(w, sheet_name='Baseline_LabData', index=False)
        hospital_stats.to_excel(w, sheet_name='Hospital_Summary')

# --- Calibration plot ---
plt.rcParams.update({'font.family': 'Arial', 'font.size': 10})
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
colors = {'LR': '#4ECDC4', 'RF': '#FF6B6B', 'XGB': '#45B7D1'}
display = {'LR': 'Logistic Regression', 'RF': 'Random Forest', 'XGB': 'XGBoost'}

for idx, mn in enumerate(['LR', 'RF', 'XGB']):
    ax = axes[idx]
    key = ('Raw', 'Set7_All', mn)
    if key not in pipeline_predictions:
        continue
    yt, yp = pipeline_predictions[key]
    brier = brier_score_loss(yt, yp)
    prob_true, prob_pred = calibration_curve(yt, yp, n_bins=10, strategy='uniform')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    ax.plot(prob_pred, prob_true, 's-', color=colors[mn], label=display[mn],
            linewidth=2, markersize=6)
    ax.set_xlabel('Mean predicted probability', fontweight='bold')
    if idx == 0:
        ax.set_ylabel('Observed proportion', fontweight='bold')
    ax.set_title(f'{display[mn]}\n(Brier = {brier:.4f})', fontweight='bold', fontsize=11)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal'); ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    inset = ax.inset_axes([0.05, 0.65, 0.35, 0.25])
    inset.hist(yp[yt == 0], bins=30, alpha=0.5, color='gray', density=True)
    inset.hist(yp[yt == 1], bins=30, alpha=0.5, color=colors[mn], density=True)
    inset.set_xlim(0, 1); inset.set_yticks([])
    inset.tick_params(labelsize=7)

plt.tight_layout()
fig.savefig(os.path.join(args.outdir, 'figure_calibration.tiff'),
            dpi=600, bbox_inches='tight', facecolor='white',
            format='tiff', pil_kwargs={'compression': 'tiff_lzw'})
plt.close(fig)

# --- Save model artifacts ---
with open(os.path.join(args.outdir, 'model_artifacts.pkl'), 'wb') as f:
    pickle.dump(model_artifacts, f)
