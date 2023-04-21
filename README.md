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

## Step 4. Embedding\
Download pretrained `Word2Vec` embeddings into `embeddings` folder. [Link](https://drive.google.com/file/d/14EOqvvjJ8qUxihQ_SFnuRsjK9pOTrP-6/view). 
```
python src/embedding.py
```

## Step 5. Data Split
Prepare data sets: traininng data, validation data, and testing data.
```
python src/train_val_test_prep.py
```

## Step 6. Model Train
Train time series baseline model, multimodal baseline model, and proposed CNN model. Each model has been modulized with hyperparameters as inputs. Users can build and train their own model by changing the hyperparamters, such as `batch_size`, `filter_num` (number of filters in CNN architecture) , `unit_sizes` (unit size of hidden layer), and so on.
```
python src/train_model.py
```

## Step 7. Evaluation
After finding the best model, the metrics is in `clinical-multimodal-learning/src/evaluation.ipynb`

# Relavent Data
Note that the following link can only be accessed via @gatech.edu email.
- Input data for the models is [here](https://gtvault.sharepoint.com/:f:/s/CSE6252-BD/Eiz1Ez6UcxVIsL1mUNdBkagBgnBcSmlAW5B_sKsb0nqIZA?e=1JgMVo)
- Models are [here](https://gtvault.sharepoint.com/:f:/s/CSE6252-BD/EtnuwXOZk7JHlhkte_0jE0wBZpJU27FZYfNxFEAb9OjEQA?e=FJzdjN)
- Results are [here](https://gtvault.sharepoint.com/:f:/s/CSE6252-BD/Eug1uIPwB25AjQ4J9nkyHhoB4vMwz2VztTQGbsmPQSeyUg?e=ItiNYD)
