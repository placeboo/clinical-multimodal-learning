import pandas as pd
import os
import numpy as np
from data_prep_helper import simple_imputer, getSentences
import nltk
import re
import warnings
import spacy

warnings.filterwarnings('ignore')

# Step 1. Extract time series features
GAP_TIME = 6  # In hours
WINDOW_SIZE = 24  # In hours
SEED = 10
GPU = '2'

os.environ['CUDA_VISIBLE_DEVICES'] = GPU
np.random.seed(SEED)

MIMIC_EXTRACT_DATA = "data/all_hourly_data.h5"

data_full_lvl2 = pd.read_hdf(MIMIC_EXTRACT_DATA, "vitals_labs")
data_full_raw = pd.read_hdf(MIMIC_EXTRACT_DATA, "vitals_labs")
statics = pd.read_hdf(MIMIC_EXTRACT_DATA, 'patients')

Ys = statics[statics.max_hours > WINDOW_SIZE + GAP_TIME][['mort_hosp', 'mort_icu', 'los_icu']]
Ys['los_3'] = Ys['los_icu'] > 3
Ys['los_7'] = Ys['los_icu'] > 7
Ys.drop(columns=['los_icu'], inplace=True)
Ys.astype(float)

lvl2, raw = [df[
                 (df.index.get_level_values('icustay_id').isin(set(Ys.index.get_level_values('icustay_id')))) &
                 (df.index.get_level_values('hours_in') < WINDOW_SIZE)
                 ] for df in (data_full_lvl2, data_full_raw)]

raw.columns = raw.columns.droplevel(level=['LEVEL2'])

train_frac, dev_frac, test_frac = 0.7, 0.1, 0.2
lvl2_subj_idx, raw_subj_idx, Ys_subj_idx = [df.index.get_level_values('subject_id') for df in (lvl2, raw, Ys)]
lvl2_subjects = set(lvl2_subj_idx)
assert lvl2_subjects == set(Ys_subj_idx), "Subject ID pools differ!"
assert lvl2_subjects == set(raw_subj_idx), "Subject ID pools differ!"

np.random.seed(SEED)
subjects, N = np.random.permutation(list(lvl2_subjects)), len(lvl2_subjects)
N_train, N_dev, N_test = int(train_frac * N), int(dev_frac * N), int(test_frac * N)
train_subj = subjects[:N_train]
dev_subj = subjects[N_train:N_train + N_dev]
test_subj = subjects[N_train + N_dev:]

[(lvl2_train, lvl2_dev, lvl2_test), (raw_train, raw_dev, raw_test), (Ys_train, Ys_dev, Ys_test)] = [
    [df[df.index.get_level_values('subject_id').isin(s)] for s in (train_subj, dev_subj, test_subj)] \
    for df in (lvl2, raw, Ys)
]

idx = pd.IndexSlice
lvl2_means, lvl2_stds = lvl2_train.loc[:, idx[:, 'mean']].mean(axis=0), lvl2_train.loc[:, idx[:, 'mean']].std(axis=0)

lvl2_train.loc[:, idx[:, 'mean']] = (lvl2_train.loc[:, idx[:, 'mean']] - lvl2_means) / lvl2_stds
lvl2_dev.loc[:, idx[:, 'mean']] = (lvl2_dev.loc[:, idx[:, 'mean']] - lvl2_means) / lvl2_stds
lvl2_test.loc[:, idx[:, 'mean']] = (lvl2_test.loc[:, idx[:, 'mean']] - lvl2_means) / lvl2_stds

lvl2_train, lvl2_dev, lvl2_test = [
    simple_imputer(df) for df in (lvl2_train, lvl2_dev, lvl2_test)
]
lvl2_flat_train, lvl2_flat_dev, lvl2_flat_test = [
    df.pivot_table(index=['subject_id', 'hadm_id', 'icustay_id'], columns=['hours_in']) for df in (
        lvl2_train, lvl2_dev, lvl2_test
    )
]

for df in lvl2_train, lvl2_dev, lvl2_test: assert not df.isnull().any().any()

[(Ys_train, Ys_dev, Ys_test)] = [
    [df[df.index.get_level_values('subject_id').isin(s)] for s in (train_subj, dev_subj, test_subj)] \
    for df in (Ys,)
]

pd.to_pickle(lvl2_train, "data/lvl2_imputer_train.pkl")
pd.to_pickle(lvl2_dev, "data/lvl2_imputer_dev.pkl")
pd.to_pickle(lvl2_test, "data/lvl2_imputer_test.pkl")

pd.to_pickle(Ys, "data/Ys.pkl")
pd.to_pickle(Ys_train, "data/Ys_train.pkl")
pd.to_pickle(Ys_dev, "data/Ys_dev.pkl")
pd.to_pickle(Ys_test, "data/Ys_test.pkl")

print("Shape of train, dev, test {}, {}, {}.".format(lvl2_train.shape, lvl2_dev.shape, lvl2_test.shape))

# Step 2. Select Sub Clinical Notes.
patient_ids = []  # store all patient ids
for each_entry in Ys.index:
    patient_ids.append(each_entry[0])

admission_df = pd.read_csv("data/ADMISSIONS.csv")
noteevents_df = pd.read_csv("data/NOTEEVENTS.csv")
icustays_df = pd.read_csv("data/ICUSTAYS.csv")
note_categories = noteevents_df.groupby(noteevents_df.CATEGORY).agg(['count']).index
selected_note_types = []
for each_cat in list(note_categories):
    if each_cat != 'Discharge summary':
        selected_note_types.append(each_cat)
# Select based on note category
sub_notes = noteevents_df[noteevents_df.CATEGORY.isin(selected_note_types)]
# Drop no char notes
missing_chardate_index = []
for each_note in sub_notes.itertuples():
    if isinstance(each_note.CHARTTIME, str):
        continue
    if np.isnan(each_note.CHARTTIME):
        missing_chardate_index.append(each_note.Index)
print("{} of notes does not charttime.".format(len(missing_chardate_index)))

sub_notes.drop(missing_chardate_index, inplace=True)
print("After dropping no notes, the note shape is {}".format(sub_notes.shape))

# Select based on patient id
sub_notes = sub_notes[sub_notes.SUBJECT_ID.isin(patient_ids)]
# Select based on time limit: 24hrs
MIMIC_EXTRACT_DATA = "data/all_hourly_data.h5"
stats = pd.read_hdf(MIMIC_EXTRACT_DATA, 'patients')
TIMELIMIT = 1  # 1day
new_stats = stats.reset_index()
new_stats.rename(columns={"subject_id": "SUBJECT_ID", "hadm_id": "HADM_ID"}, inplace=True)
print("New Stats shape is {}".format(new_stats.shape))
print("Sub note shape is {}".format(sub_notes.shape))
df_adm_notes = pd.merge(sub_notes[['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'CATEGORY', 'TEXT']],
                        new_stats[['SUBJECT_ID', 'HADM_ID', 'icustay_id', 'age', 'admittime', 'dischtime', 'deathtime',
                                   'intime', 'outtime', 'los_icu', 'mort_icu', 'mort_hosp', 'hospital_expire_flag',
                                   'hospstay_seq', 'max_hours']],
                        on=['SUBJECT_ID'],
                        how='left')

df_adm_notes['CHARTTIME'] = pd.to_datetime(df_adm_notes['CHARTTIME'])
df_less_n = df_adm_notes[
    ((df_adm_notes['CHARTTIME'] - df_adm_notes['intime']).dt.total_seconds() / (24 * 60 * 60)) < TIMELIMIT]
print("df_less_n shape is {}".format(df_less_n.shape))
pd.to_pickle(df_less_n, "data/sub_notes.p")

# Process clinical notes
sub_notes = df_less_n[df_less_n.SUBJECT_ID.notnull()]
sub_notes = sub_notes[sub_notes.CHARTTIME.notnull()]
sub_notes = sub_notes[sub_notes.TEXT.notnull()]
sub_notes = sub_notes[['SUBJECT_ID', 'HADM_ID_y', 'CHARTTIME', 'TEXT']]
sub_notes['preprocessed_text'] = None
for each_note in sub_notes.itertuples():
    text = each_note.TEXT
    sub_notes.at[each_note.Index, 'preprocessed_text'] = getSentences(text)
pd.to_pickle(sub_notes, "data/preprocessed_notes.p")

# Apply Med7 on Clinical Notes
med7 = spacy.load("en_core_med7_lg")
sub_notes = pd.read_pickle("data/preprocessed_notes.p")
sub_notes['ner'] = None

count = 0
preprocessed_index = {}
for i in sub_notes.itertuples():

    if count % 1000 == 0:
        print(count)

    count += 1
    ind = i.Index
    text = i.preprocessed_text

    all_pred = []
    for each_sent in text:
        try:
            doc = med7(each_sent)
            result = ([(ent.text, ent.label_) for ent in doc.ents])
            if len(result) == 0: continue
            all_pred.append(result)
        except:
            print("error..")
            continue
    sub_notes.at[ind, 'ner'] = all_pred

pd.to_pickle(sub_notes, "data/ner_df.p")