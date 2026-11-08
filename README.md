# Calibrated, Cost-Aware Triage of Android Malware Network Traffic on Unseen Applications

Code, data samples, per-fold predictions and results for the revised manuscript
"Calibrated, Cost-Aware Triage of Android Malware Network Traffic on Unseen Applications"
(A. A. Taha et al., King Abdulaziz University). The code for the first submission is kept in `legacy/`.

## What the study does
- **CICAndMal2017**, rebuilt from the 2126 public per-application flow files: de-duplicated, at most 20 flows per
  application (41,900 flows, 2112 apps, 412 malicious). Evaluated by 3 × 5-fold cross-validation **grouped by
  application**; tuning, fitting and calibration use disjoint application sets, so every test application is unseen.
- **CIC-AAGM2017**: uniform sample of 20,000 de-duplicated flows at the natural malware share (34.8%); stratified
  5-fold CV by flow (the dataset has no application identifiers).
- **Equal-budget tuning** (48 configurations each): PSO, random search and TPE (Optuna) for XGBoost; PSO for
  LightGBM, random forest and ExtraTrees; default XGBoost, LightGBM, AdaBoost and GBM as references.
- **Triage layer**: application score = mean flow probability; Platt calibration on held-out applications;
  workload-constrained deferral (cost-weighted window vs symmetric Chow-type band vs unweighted window);
  Mondrian conformal prediction; prior-shift analysis at 10/5/1(/0.1)% prevalence; TreeSHAP.

## Layout
| Path | Content |
| --- | --- |
| `src/prepare_cicandmal_app.py` | builds `data/cicandmal_app_K20.parquet` from the per-app CSVs (`Data/<Category>/<Family>/*.pcap_ISCX.csv`) |
| `src/prepare_aagm.py` | builds `data/aagm_natural20k.parquet` from `TotalFeatures-ISCXFlowMeter.csv` |
| `src/train_v2.py` | tuning + fitting; writes per-fold probabilities to `predictions/*.npz` (args: dataset protocol features repeat modelset) |
| `src/queue.sh` | the exact list of training runs used in the paper |
| `src/analysis_app.py` | application-level analyses (CICAndMal2017) → `results/{A,K,AC,AS,AP,AM,AF,AV}.csv` |
| `src/analysis_v2.py` | flow-level analyses (both datasets) → `results/{D,C,S,P,M,V}.csv` |
| `src/prevalence_v2.py` | prevalence analysis → `results/PREV.csv` |
| `src/stats_v2.py` | corrected repeated k-fold t-tests with Holm adjustment → `results/T.csv` |
| `src/shap_v2.py` | TreeSHAP + error-prediction test → `results/shap_*.csv`, `figures/shap_*.png` |
| `src/figs_v2.py` | manuscript figures |
| `src/extra_checks.py` | capture-day check, alternative aggregations → `results/CAPDAY.csv`, `results/AGG.csv` |
| `src/manuscript_numbers.py` | computes every number quoted in the manuscript from `results/` and `predictions/` |
| `results/train_log.txt` | training log with the selected hyperparameters per fold |

## Reproduce
```bash
pip install -r requirements.txt
python src/prepare_cicandmal_app.py <path-to>/Data 20 data/cicandmal_app_K20.parquet
python src/prepare_aagm.py <path-to>/TotalFeatures-ISCXFlowMeter.csv data/aagm_natural20k.parquet
bash src/queue.sh            # resumable; about 30-50 min per fold for the full model set on 2 CPU cores
python src/analysis_app.py && python src/analysis_v2.py && python src/prevalence_v2.py && python src/stats_v2.py
python src/shap_v2.py andmal app main full && python src/shap_v2.py andmal app ports small && python src/shap_v2.py aagm flow main full
python src/figs_v2.py
```
Set `TRIAGE_ROOT` to run from another directory. The prepared data samples and all per-fold predictions are
included, so the analyses can be rerun without retraining.

## Data sources
CICAndMal2017 and CIC-AAGM2017 are available from the Canadian Institute for Cybersecurity
(https://www.unb.ca/cic/datasets/andmal2017.html, https://www.unb.ca/cic/datasets/android-adware.html).
