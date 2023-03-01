from config import TRAIN_PARAMS, DATA_FNAMES
from joblib import load
from data import GloveEmbedder
import pandas as pd
import numpy as np

glove_embedder = GloveEmbedder(glove_dim=TRAIN_PARAMS['glove_dim'])
sentence_cols = ['Conclusion', 'Premise']
words_to_remove = ['the', 'a', 'an', 'of']

train_dataset = glove_embedder.transform_dataset_from_file(DATA_FNAMES['train_arguments'], sentence_cols, words_to_remove)
train_labels = pd.read_table(DATA_FNAMES['train_labels'])
all_human_values = train_labels.columns[1:]

models_dir = './glove/saved_models/'
all_binary_preds = None
for human_value in all_human_values:
    clf_name = models_dir + human_value + '.joblib'
    clf = load(clf_name)
    binary_preds = clf.predict(train_dataset).reshape(-1, 1)

    if all_binary_preds is None:
        all_binary_preds = binary_preds
    else:
        all_binary_preds = np.concatenate((all_binary_preds, binary_preds), axis=1)

preds_fname = './glove/train_predictions/' + 'glove_binary_preds.npy'
np.save(preds_fname, all_binary_preds)

    


