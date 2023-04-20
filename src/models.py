import pandas as pd
import os
import numpy as np
import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout, concatenate, Input, Conv1D, GRU, LSTM, GlobalMaxPooling1D
from keras.optimizers import Adam
from keras.backend import clear_session, get_session
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score
import gc
import warnings

warnings.filterwarnings('ignore')


def reset_keras(model):
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()

    try:
        del model  # this is from global space - change this as you need
    except:
        pass


def make_prediction(model, test_data):
    probs = model.predict(test_data)
    y_pred = [1 if i >= 0.5 else 0 for i in probs]
    return probs, y_pred


def save_scores(framework, predictions, probs, ground_truth, model_name,
                problem_type, iteration, hidden_unit_size, type_of_ner, sequence_name='GRU'):
    file_name = ''
    auc = roc_auc_score(ground_truth, probs)
    auprc = average_precision_score(ground_truth, probs)
    acc = accuracy_score(ground_truth, predictions)
    F1 = f1_score(ground_truth, predictions)
    result_dict = {}
    result_dict['auc'] = auc
    result_dict['auprc'] = auprc
    result_dict['acc'] = acc
    result_dict['F1'] = F1

    result_path = 'results/' + framework
    if framework == 'baseline':
        file_name = str(hidden_unit_size) + "-" + model_name + "-" + problem_type + "-" + str(
            iteration) + "-" + type_of_ner + ".p"

    if framework == 'multimodal_baseline':
        file_name = str(sequence_name) + "-" + str(hidden_unit_size) + "-" + model_name
        file_name = file_name + "-" + problem_type + "-" + str(iteration) + "-" + type_of_ner + "-avg-.p"

    if framework == 'cnn':
        file_name = str(sequence_name) + "-" + str(hidden_unit_size) + "-" + model_name
        file_name = file_name + "-" + problem_type + "-" + str(iteration) + "-" + type_of_ner + "-cnn-.p"

    pd.to_pickle(result_dict, os.path.join(result_path, file_name))
    print(auc, auprc, acc, F1)


def timeseries_model(layer_name, number_of_unit):
    K.clear_session()

    sequence_input = Input(shape=(24, 104), name="timeseries_input")

    if layer_name == "LSTM":
        x = LSTM(number_of_unit)(sequence_input)
    else:
        x = GRU(number_of_unit)(sequence_input)

    logits_regularizer = tf.keras.regularizers.L2(l2=0.01)
    # logits_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
    sigmoid_pred = Dense(1, activation='sigmoid', use_bias=False,
                         kernel_initializer=tf.keras.initializers.glorot_normal(),
                         kernel_regularizer=logits_regularizer)(x)

    model = Model(inputs=sequence_input, outputs=sigmoid_pred)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return model


def create_dataset(dict_of_ner):
    temp_data = []
    for k, v in sorted(dict_of_ner.items()):
        temp = []
        for embed in v:
            temp.append(embed)
        temp_data.append(np.mean(temp, axis=0))
    return np.asarray(temp_data)


def avg_ner_model(layer_name, number_of_unit, embedding_name):
    if embedding_name == "concat":
        input_dimension = 200
    else:
        input_dimension = 100

    sequence_input = Input(shape=(24, 104))

    input_avg = Input(shape=(input_dimension,), name="avg")
    #     x_1 = Dense(256, activation='relu')(input_avg)
    #     x_1 = Dropout(0.3)(x_1)

    if layer_name == "GRU":
        x = GRU(number_of_unit)(sequence_input)
    elif layer_name == "LSTM":
        x = LSTM(number_of_unit)(sequence_input)

    x = keras.layers.Concatenate()([x, input_avg])

    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)

    logits_regularizer = tf.keras.regularizers.L2(l2=0.01)

    preds = Dense(1, activation='sigmoid', use_bias=False,
                  kernel_initializer=tf.keras.initializers.glorot_normal(),
                  kernel_regularizer=logits_regularizer)(x)

    opt = Adam(lr=0.001, decay=0.01)
    model = Model(inputs=[sequence_input, input_avg], outputs=preds)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['acc'])
    return model


def get_subvector_data(size, embed_name, data):
    if embed_name == "concat":
        vector_size = 200
    else:
        vector_size = 100

    x_data = {}
    for k, v in data.items():
        number_of_additional_vector = len(v) - size
        vector = []
        for i in v:
            vector.append(i)
        if number_of_additional_vector < 0:
            number_of_additional_vector = np.abs(number_of_additional_vector)

            temp = vector[:size]
            for i in range(0, number_of_additional_vector):
                temp.append(np.zeros(vector_size))
            x_data[k] = np.asarray(temp)
        else:
            x_data[k] = np.asarray(vector[:size])

    return x_data


def proposedmodel(layer_name, number_of_unit, embedding_name, ner_limit, num_filter):
    if embedding_name == "concat":
        input_dimension = 200
    else:
        input_dimension = 100

    sequence_input = Input(shape=(24, 104))

    input_img = Input(shape=(ner_limit, input_dimension), name="cnn_input")

    convs = []
    filter_sizes = [2, 3, 4]

    text_conv1d = Conv1D(filters=num_filter, kernel_size=3,
                         padding='valid', strides=1, dilation_rate=1, activation='relu',
                         kernel_initializer=tf.keras.initializers.glorot_normal())(input_img)

    text_conv1d = Conv1D(filters=num_filter * 2, kernel_size=3,
                         padding='valid', strides=1, dilation_rate=1, activation='relu',
                         kernel_initializer=tf.keras.initializers.glorot_normal())(text_conv1d)

    text_conv1d = Conv1D(filters=num_filter * 3, kernel_size=3,
                         padding='valid', strides=1, dilation_rate=1, activation='relu',
                         kernel_initializer=tf.keras.initializers.glorot_normal())(text_conv1d)

    text_embeddings = GlobalMaxPooling1D()(text_conv1d)

    if layer_name == "GRU":
        x = GRU(number_of_unit)(sequence_input)
    elif layer_name == "LSTM":
        x = LSTM(number_of_unit)(sequence_input)

    concatenated = concatenate([x, text_embeddings], axis=1)

    concatenated = Dense(512, activation='relu')(concatenated)
    concatenated = Dropout(0.2)(concatenated)
    logits_regularizer = tf.keras.regularizers.L2(l2=0.01)
    preds = Dense(1, activation='sigmoid', use_bias=False,
                  kernel_initializer=tf.keras.initializers.glorot_normal(),
                  kernel_regularizer=logits_regularizer)(concatenated)

    opt = Adam(lr=1e-3, decay=0.01)

    # opt = Adam(lr=0.001)

    model = Model(inputs=[sequence_input, input_img], outputs=preds)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['acc'])

    return model


def train_baseline(x_train, y_train,
                   x_dev, y_dev,
                   x_test, y_test,
                   epoch_num=100,
                   model_patience=3,
                   monitor_criteria='val_loss',
                   batch_size=128,
                   unit_sizes=None,
                   iter_num=11,
                   target_problems=None,
                   layers=None,
                   type_of_ner=None):
    if layers is None:
        layers = ["LSTM", "GRU"]
    if unit_sizes is None:
        unit_sizes = [128, 256]
    if target_problems is None:
        target_problems = ['mort_hosp', 'mort_icu', 'los_3', 'los_7']
    if type_of_ner is None:
        type_of_ner = 'new'

    for each_layer in layers:
        print("Layer: ", each_layer)
        for each_unit_size in unit_sizes:
            print("Hidden unit: ", each_unit_size)
            for iteration in range(1, iter_num):
                print("Iteration number: ", iteration)
                print("=============================")

                for each_problem in target_problems:
                    print("Problem type: ", each_problem)
                    print("__________________")

                    early_stopping_monitor = EarlyStopping(monitor=monitor_criteria, patience=model_patience)
                    best_model_name = str(each_layer) + "-" + str(each_unit_size) + "-" + str(
                        each_problem) + "-" + "best_model.hdf5"
                    checkpoint = ModelCheckpoint(best_model_name, monitor='val_loss', verbose=1,
                                                 save_best_only=True, mode='min', period=1)

                    callbacks = [early_stopping_monitor, checkpoint]

                    model = timeseries_model(each_layer, each_unit_size)
                    model.fit(x_train, y_train[each_problem], epochs=epoch_num, verbose=1,
                              validation_data=(x_dev, y_dev[each_problem]), callbacks=callbacks,
                              batch_size=batch_size)

                    model.load_weights(best_model_name)

                    probs, predictions = make_prediction(model, x_test)
                    save_scores(framework='baseline',
                                predictions=predictions,
                                probs=probs,
                                ground_truth=y_test[each_problem].values,
                                model_name=str(each_layer),
                                problem_type=each_problem,
                                iteration=iteration, hidden_unit_size=each_unit_size,
                                type_of_ner=type_of_ner)
                    reset_keras(model)
                    # del model
                    clear_session()
                    gc.collect()
