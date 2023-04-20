# clinical-multimodal-learning

## Prerequisites
Have MIMIC III GCP accessibility through physionet.

## Working directory
Our working directory is 
```
clincial-multimodal-learning/
```
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

## Step 3. Data Preparation
In this step, we extract time series features, select and preprocess eligiable clincal notes.
```
python src/data_prep.py
```

## Step 4. 
Download pretrained `Word2Vec` embeddings into `embeddings` folder. [Link](https://drive.google.com/file/d/14EOqvvjJ8qUxihQ_SFnuRsjK9pOTrP-6/view). 
```
python src/embedding.py
```

## Step 5.
Prepare data sets: traininng data, validation data, and testing data.
```
python src/train_val_test_prep.py
```

## Step 6.
Train time series baseline model, multimodal baseline model, and proposed CNN model. Each model has been modulized with hyperparameters as inputs. Users can build and train their own model by changing the hyperparamters, such as `batch_size`, `filter_num` (number of filters in CNN architecture) , `unit_sizes` (unit size of hidden layer), and so on.
```
python src/train_model.py
```

