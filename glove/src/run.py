from config import SEED, TRAIN_PARAMS, DATA_FNAMES
from data import GloveEmbedder
from trainer import Trainer, Logger
from sklearn import linear_model
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
np.random.seed(SEED)

glove_embedder = GloveEmbedder(glove_dim=TRAIN_PARAMS['glove_dim'])
sentence_cols = ['Conclusion', 'Premise']
words_to_remove = ['the', 'a', 'an', 'of']

train_dataset = glove_embedder.transform_dataset_from_file(DATA_FNAMES['train_arguments'], sentence_cols, words_to_remove)
train_labels = pd.read_table(DATA_FNAMES['train_labels'])

val_dataset = glove_embedder.transform_dataset_from_file(DATA_FNAMES['valid_arguments'], sentence_cols, words_to_remove)
val_labels = pd.read_table(DATA_FNAMES['valid_labels'])

all_human_values = train_labels.columns[1:]
train_f1_scores = []
val_f1_scores = []
for label in all_human_values:
    train_label = train_labels[label].values
    val_label = val_labels[label].values

    ridge = linear_model.RidgeClassifier(class_weight='balanced')
    trainer = Trainer(ridge, *shuffle(train_dataset, train_label), val_dataset, val_label)
    trainer.train()

    logger = Logger(trainer, label)
    # save_path = "glove/saved_models/" + label + ".joblib"
    # logger.save_model(save_path)
    train_f1, val_f1 = logger.print_f1_scores()
    train_f1_scores.append(train_f1)
    val_f1_scores.append(val_f1)

mean_train_f1 = np.mean(train_f1_scores)
mean_val_f1 = np.mean(val_f1_scores)
print('Value-averaged train F1: ', mean_train_f1)
print('Value-averaged validation F1: ', mean_val_f1)

