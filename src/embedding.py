import pandas as pd
import numpy as np
from gensim.models import Word2Vec

import warnings
warnings.filterwarnings('ignore')

def mean(a):
    return sum(a) / len(a)

new_notes = pd.read_pickle("data/ner_df.p") # med7
w2vec = Word2Vec.load("embeddings/word2vec.model")
null_index_list = []
for i in new_notes.itertuples():

    if len(i.ner) == 0:
        null_index_list.append(i.Index)
new_notes.drop(null_index_list, inplace=True)

med7_ner_data = {}

for ii in new_notes.itertuples():

    p_id = ii.SUBJECT_ID
    ind = ii.Index

    try:
        new_ner = new_notes.loc[ind].ner
    except:
        new_ner = []

    unique = set()
    new_temp = []

    for j in new_ner:
        for k in j:
            unique.add(k[0])
            new_temp.append(k)

    if p_id in med7_ner_data:
        for i in new_temp:
            med7_ner_data[p_id].append(i)
    else:
        med7_ner_data[p_id] = new_temp

pd.to_pickle(med7_ner_data, "data/new_ner_word_dict.pkl")

data_types = [med7_ner_data]
data_names = ["new_ner"]

for data, names in zip(data_types, data_names):
    new_word2vec = {}
    print("w2vec starting..")
    for k,v in data.items():

        patient_temp = []
        for i in v:
            try:
                patient_temp.append(w2vec.wv[i[0]])
            except:
                avg = []
                num = 0
                temp = []

                if len(i[0].split(" ")) > 1:
                    for each_word in i[0].split(" "):
                        try:
                            temp = w2vec.wv[each_word]
                            avg.append(temp)
                            num += 1
                        except:
                            pass
                    if num == 0: continue
                    avg = np.asarray(avg)
                    t = np.asarray(list(map(mean, zip(*avg))))
                    patient_temp.append(t)
        if len(patient_temp) == 0: continue
        new_word2vec[k] = patient_temp

    print("word2vec length is {}", len(new_word2vec))
    pd.to_pickle(new_word2vec, "data/" + names + "_word2vec_dict.pkl")