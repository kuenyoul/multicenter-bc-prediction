# Multicenter Blood Culture Prediction

Analysis code for: **"A pooled multicenter model outperforms hospital-specific models for laboratory-based bacteremia prediction"**

## Overview

Machine learning models predicting blood culture positivity using routine laboratory data from three tertiary hospitals. Compares pooled multicenter versus hospital-specific model strategies.

## Scripts

| File | Description |
|------|-------------|
| `analysis/bc_prediction_analysis.py` | Main analysis: 7 variable sets x 3 models, multicenter generalization, bootstrap CI |
| `analysis/bc_prediction_shap_dca.py` | SHAP feature importance + Decision Curve Analysis |
| `analysis/bc_prediction_figures.py` | Figure generation from result Excel files |

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run main analysis
python analysis/bc_prediction_analysis.py --input <data.xlsx> --outdir results/

# Run SHAP + DCA (requires model artifacts from main analysis)
python analysis/bc_prediction_shap_dca.py --input <data.xlsx> --artifacts results/model_artifacts.pkl --outdir results_shap_dca/

# Generate figures
python analysis/bc_prediction_figures.py --results-dir results/
```

## Data

De-identified data are available from the corresponding author upon reasonable request, subject to institutional data sharing agreements.

## License

This project is licensed under the MIT License.
