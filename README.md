# clinical-multimodal-learning

## Prerequisites
Have MIMIC III GCP accessibility through physionet.

## Step 1. Environment Configuration

Create environment
```
conda env create -f environment.yml
```

Activate environment
```
conda activate clinical-multimodal
```

## Step 2. MIMIC-III Data

- Download [MIMIC-Extract data](https://console.cloud.google.com/storage/browser/mimic_extract) in GCP from the pipeline which is reprocessed by default parameters. Save the data `all_hourly_data.h5` under `data` folder.
- Save MIMIC-III csv data files `ADMISSIONS.csv`, `ICUSTAYS.csv`, `NOTEEVENTS.csv` in 
`data` folder.
