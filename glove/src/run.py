from config import SEED, TRAIN_PARAMS, DATA_FNAMES
from data import GloveEmbedder
from trainer import Trainer
from sklearn import linear_model
from sklearn.utils import shuffle
from joblib import dump, load
import pandas as pd
import numpy as np
np.random.seed(SEED)

train_dataset = pd.read_table(DATA_FNAMES['train_arguments'])
train_stances = np.array(train_dataset['Stance'] == 'in favor of', dtype=int).reshape(-1, 1)
train_labels = pd.read_table(DATA_FNAMES['train_labels'])

val_dataset = pd.read_table(DATA_FNAMES['valid_arguments'])
val_stances = np.array(val_dataset['Stance'] == 'in favor of', dtype=int).reshape(-1, 1)
val_labels = pd.read_table(DATA_FNAMES['valid_labels'])

glove_embedder = GloveEmbedder(glove_dim=TRAIN_PARAMS['glove_dim'])

sentence_cols = ['Conclusion', 'Premise']
words_to_remove = ['the', 'a', 'an', 'of']

train_dataset = np.concatenate((train_stances, glove_embedder.transform_data(train_dataset, sentence_cols, sentence_cols)), axis=1)
val_dataset = np.concatenate((val_stances, glove_embedder.transform_data(val_dataset, sentence_cols, sentence_cols)), axis=1)

all_human_values = train_labels.columns[1:]
for label in all_human_values:
    train_label = train_labels[label].values
    val_label = val_labels[label].values

    ridge = linear_model.RidgeClassifier(class_weight = "balanced")
    trainer = Trainer(ridge, *shuffle(train_dataset, train_label), val_dataset, val_label)
    trainer.train()

    model_fname = "glove/saved_models/" + label + ".joblib"
    dump(ridge, model_fname)

    print("------------------")
    print(label)
    print("* Train F1: ", trainer.f1_score("train"))
    print("* Validation F1: ", trainer.f1_score("validation"))
