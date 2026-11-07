# Data

The datasets are not redistributed here. Download them from the Canadian Institute for Cybersecurity:

- CICAndMal2017: https://www.unb.ca/cic/datasets/andmal2017.html (class-balanced flow-level CSV with the label column `Label1`)
- CIC-AAGM2017: https://www.unb.ca/cic/datasets/android-adware.html (`TotalFeatures-ISCXFlowMeter.csv`, label column `calss`)

Then build the analysis datasets:

```bash
cd src
python prepare_data.py <path/to/CICAndMal2017.csv> <path/to/TotalFeatures-ISCXFlowMeter.csv>
```

This writes `data/CICAndMal2017.parquet` (14,997 flows, 68 features) and `data/CIC-AAGM2017.parquet` (20,000 flows, 74 features).
