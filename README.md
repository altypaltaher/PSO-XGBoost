# PSO-XGBoost with calibrated triage for Android malware network traffic

Code and results for the paper *"PSO-Tuned XGBoost with Calibrated Probabilities and a Reject Option for Reliable Triage of Android Malware Network Traffic"* (Taha et al., submitted to *Future Internet*).

XGBoost tuned by particle swarm optimisation (PSO-XGBoost) is combined with three reliability components:

- post-hoc probability calibration (Platt scaling; isotonic regression compared),
- Chow's two-threshold reject option, which sends a chosen share of flows to an analyst,
- Mondrian (class-conditional) conformal prediction.

These are evaluated on CICAndMal2017 and CIC-AAGM2017 against XGBoost, LightGBM, AdaBoost and GBM.

## Protocol

Each dataset uses stratified 5-fold cross-validation (seed 42). Within each fold, the training part is split into:

| Split | Share of flows | Used for |
|---|---|---|
| FIT | 60% | PSO search and model fitting |
| CAL | 20% | calibration, reject thresholds and conformal scores |
| TEST | 20% | reporting only |

PSO settings:
- 8 particles × 10 iterations, inertia w = 0.7, c1 = c2 = 1.5.
- Fitness = 1 − mean 3-fold F1 on FIT.
- Search space: learning rate [0.01, 0.3], maximum depth {3–10}, number of trees {50–300}, subsample [0.6, 1.0].

Statistical tests use the corrected resampled t-test (Nadeau & Bengio, 2003) with Holm adjustment.

## Repository layout

| Path | Content |
|---|---|
| `src/common.py` | paths, seed, data loader, baseline settings |
| `src/prepare_data.py` | builds the two datasets from the original CIC files (see `data/README.md`) |
| `src/train.py` | Stage 1: PSO search, model fitting, saves CAL/TEST probabilities per fold |
| `src/analysis.py` | Stage 2: discrimination, calibration, reject option, conformal prediction, statistics, Figures 2–3 |
| `src/shap_analysis.py` | refits PSO-XGBoost per fold (verifies identical predictions), exact TreeSHAP, Figures 4–5 |
| `src/fig1_framework.py` | Figure 1 |
| `results/fold_predictions/` | saved per-fold CAL/TEST probabilities of all models (output of `train.py`) |
| `results/*.csv` | per-fold metrics and statistical tests behind Tables 3–9 |
| `results/analysis_report.txt` | printed summary of the analysis |
| `figures/` | Figures 1–5 of the paper |

`train.py` also trains PSO-LightGBM with the same budget. It is the preliminary tuned-LightGBM comparison mentioned in the limitations and is not included in the paper's tables.

## Reproducing the results

```bash
pip install -r requirements.txt
cd src
python prepare_data.py <CICAndMal2017.csv> <TotalFeatures-ISCXFlowMeter.csv>
python train.py CICAndMal2017     # about 1 h on 2 CPU cores; resumable per fold
python train.py CIC-AAGM2017
python analysis.py
python shap_analysis.py
python fig1_framework.py
```

The per-fold predictions used in the paper are included. To rerun the analysis and SHAP stages without retraining, prepare the data and skip `train.py`. `analysis.py` alone needs only the saved predictions plus the prepared data, which it uses for the fold sizes.

Environment used for the paper: Python 3.13.16, XGBoost 3.4.1, LightGBM 4.7.0, scikit-learn 1.9.1, SHAP 0.54.1, on a two-core CPU.

## Mapping to the paper

| Paper | File |
|---|---|
| Table 3 | `results/pso_params_perfold.csv` |
| Table 4 | `results/discrimination_perfold.csv` |
| Table 5 | `results/calibration_perfold.csv` |
| Tables 6–7 | `results/triage_perfold.csv` |
| Table 8 | `results/conformal_perfold.csv` |
| Table 9 | `results/stats.csv` |
| Section 4.7 | `results/shap_importance.csv`, `results/shap_deferral_summary.csv` |
