import pandas as pd
from models import train_baseline
import os
import numpy as np
from gensim.models import Word2Vec, FastText
# import glove
# from glove import Corpus

import collections
import gc

import keras
from keras import backend as K
from keras import regularizers
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Input, concatenate, Activation, Concatenate, LSTM, GRU
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv1D, BatchNormalization, GRU, Convolution1D, LSTM
from keras.layers import UpSampling1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D, MaxPool1D

from keras.optimizers import Adam

from keras.callbacks import EarlyStopping, ModelCheckpoint, History, ReduceLROnPlateau
from keras.utils import np_utils
from keras.backend import set_session, clear_session, get_session
import tensorflow as tf

from sklearn.utils import class_weight
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score

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
