import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

lvl2_train =  pd.read_pickle("data/lvl2_imputer_train.pkl")
lvl2_dev =  pd.read_pickle("data/lvl2_imputer_dev.pkl")
lvl2_test =  pd.read_pickle("data/lvl2_imputer_test.pkl")

Ys =  pd.read_pickle("data/Ys.pkl")
Ys_train =  pd.read_pickle("data/Ys_train.pkl")
Ys_dev =  pd.read_pickle("data/Ys_dev.pkl")
Ys_test =  pd.read_pickle("data/Ys_test.pkl")

all_train_ids = set()
for i in Ys_train.itertuples():
    all_train_ids.add(i.Index[0])

all_dev_ids = set()
for i in Ys_dev.itertuples():
    all_dev_ids.add(i.Index[0])

all_test_ids = set()
for i in Ys_test.itertuples():
    all_test_ids.add(i.Index[0])

new_word2vec_dict = pd.read_pickle("data/new_ner_word2vec_dict.pkl")
new_keys = set(new_word2vec_dict.keys())
new_train_ids = sorted(all_train_ids.intersection(new_keys))
new_dev_ids = sorted(all_dev_ids.intersection(new_keys))
new_test_ids = sorted(all_test_ids.intersection(new_keys))

pd.to_pickle(new_train_ids, "data/new_train_ids.pkl")
pd.to_pickle(new_test_ids, "data/new_test_ids.pkl")
pd.to_pickle(new_dev_ids, "data/new_dev_ids.pkl")

data_ids = [(new_train_ids, new_dev_ids, new_test_ids)]
data_names = ["new"]

for i, (tr, de, te) in zip(data_names, data_ids):
    y_train = Ys_train.loc[tr]
    y_dev = Ys_dev.loc[de]
    y_test = Ys_test.loc[te]

    sub_train = lvl2_train.loc[tr]
    sub_train = sub_train.loc[:, pd.IndexSlice[:, 'mean']]

    sub_dev = lvl2_dev.loc[de]
    sub_dev = sub_dev.loc[:, pd.IndexSlice[:, 'mean']]

    sub_test = lvl2_test.loc[te]
    sub_test = sub_test.loc[:, pd.IndexSlice[:, 'mean']]

    sub_train = sub_train.values
    sub_dev = sub_dev.values
    sub_test = sub_test.values

    # reshape the data for timeseries prediction
    x_train_lstm = sub_train.reshape(int(sub_train.shape[0] / 24), 24, 104)
    x_dev_lstm = sub_dev.reshape(int(sub_dev.shape[0] / 24), 24, 104)
    x_test_lstm = sub_test.reshape(int(sub_test.shape[0] / 24), 24, 104)

    pd.to_pickle(x_train_lstm, "data/" + i + "_x_train.pkl")
    pd.to_pickle(x_dev_lstm, "data/" + i + "_x_dev.pkl")
    pd.to_pickle(x_test_lstm, "data/" + i + "_x_test.pkl")

    pd.to_pickle(y_train, "data/" + i + "_y_train.pkl")
    pd.to_pickle(y_dev, "data/" + i + "_y_dev.pkl")
    pd.to_pickle(y_test, "data/" + i + "_y_test.pkl")
