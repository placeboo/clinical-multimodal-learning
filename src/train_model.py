import pandas as pd
import numpy as np
from models import train_baseline, create_dataset, train_multimodal_baseline, train_proposedmodel, get_subvector_data
import collections

import warnings

warnings.filterwarnings('ignore')

type_of_ner = "new"

x_train_lstm = pd.read_pickle("data/" + type_of_ner + "_x_train.pkl")
x_dev_lstm = pd.read_pickle("data/" + type_of_ner + "_x_dev.pkl")
x_test_lstm = pd.read_pickle("data/" + type_of_ner + "_x_test.pkl")

y_train = pd.read_pickle("data/" + type_of_ner + "_y_train.pkl")
y_dev = pd.read_pickle("data/" + type_of_ner + "_y_dev.pkl")
y_test = pd.read_pickle("data/" + type_of_ner + "_y_test.pkl")
ner_word2vec = pd.read_pickle("data/" + type_of_ner + "_ner_word2vec_dict.pkl")

train_ids = pd.read_pickle("data/" + type_of_ner + "_train_ids.pkl")
dev_ids = pd.read_pickle("data/" + type_of_ner + "_dev_ids.pkl")
test_ids = pd.read_pickle("data/" + type_of_ner + "_test_ids.pkl")

temp_train_ner = dict((k, ner_word2vec[k]) for k in train_ids)
temp_dev_ner = dict((k, ner_word2vec[k]) for k in dev_ids)
temp_test_ner = dict((k, ner_word2vec[k]) for k in test_ids)

# baseline model training
train_baseline(x_train=x_train_lstm, y_train=y_train,
               x_dev=x_dev_lstm, y_dev=y_dev,
               x_test=x_test_lstm, y_test=y_test,
               epoch_num=100,
               model_patience=3,
               batch_size=128,
               unit_sizes=[128, 256],
               iter_num=11,
               layers=["LSTM", "GRU"])

# multimodel_baseline model
x_train_ner = create_dataset(temp_train_ner)
x_dev_ner = create_dataset(temp_dev_ner)
x_test_ner = create_dataset(temp_test_ner)

train_multimodal_baseline(x_train=x_train_lstm, x_train_ner=x_train_ner, y_train=y_train,
                          x_dev=x_dev_lstm, x_dev_ner=x_dev_ner, y_dev=y_dev,
                          x_test=x_test_lstm, x_test_ner=x_test_ner, y_test=y_test,
                          epoch_num=100,
                          model_patience=5,
                          batch_size=64,
                          iter_num=11,
                          unit_sizes=[128, 256],
                          layers=['GRU'])

# train proposed model
ner_representation_limit = 64
x_train_dict = get_subvector_data(ner_representation_limit, temp_train_ner)
x_dev_dict = get_subvector_data(ner_representation_limit, temp_dev_ner)
x_test_dict = get_subvector_data(ner_representation_limit, temp_test_ner)
x_train_dict_sorted = collections.OrderedDict(sorted(x_train_dict.items()))
x_dev_dict_sorted = collections.OrderedDict(sorted(x_dev_dict.items()))
x_test_dict_sorted = collections.OrderedDict(sorted(x_test_dict.items()))

x_train_ner = np.asarray(list(x_train_dict_sorted.values()))
x_dev_ner = np.asarray(list(x_dev_dict_sorted.values()))
x_test_ner = np.asarray(list(x_test_dict_sorted.values()))

train_proposedmodel(x_train=x_train_lstm, x_train_ner=x_train_ner, y_train=y_train,
                    x_dev=x_dev_lstm, x_dev_ner=x_dev_ner, y_dev=y_dev,
                    x_test=x_test_lstm, x_test_ner=x_test_ner, y_test=y_test,
                    epoch_num=100,
                    model_patience=5,
                    batch_size=64,
                    iter_num=11,
                    unit_sizes=[256],
                    filter_num=32,
                    ner_representation_limit=ner_representation_limit,
                    layers=['GRU'])
