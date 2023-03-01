from config import MODELS_DIR, HUMAN_VALUES, DATA_FNAMES
from ensemble import EnsembledModel
from model import BERTPredictor, GlovePredictor
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score


# bert_16 = torch.load(TRAIN_PREDS_DIR['bert'] + 'train_preds_bert_b16-w40-e5.pt', map_location=torch.device('cpu')).numpy()
# bert_64 = torch.load(TRAIN_PREDS_DIR['bert'] +  'train_preds_bert_b64-w50-e9.pt', map_location=torch.device('cpu')).numpy()
# glove_binary = np.load(TRAIN_PREDS_DIR['glove'] + 'glove_binary_preds.npy')

bert_16 = BERTPredictor(MODELS_DIR['bert'] + '')

# train_labels = pd.read_table(DATA_FNAMES['train_labels'])
# log_reg_clfs = []
# f1_scores = []
# accuracies = []
# for value_idx, human_value in enumerate(HUMAN_VALUES):    
#     ensemble_train = np.column_stack((bert_16[:, value_idx], bert_64[:, value_idx], glove_binary[:, value_idx]))
#     ensemble_targets = train_labels[human_value].to_numpy()

#     log_reg = LogisticRegression(class_weight='balanced')
#     log_reg.fit(ensemble_train, ensemble_targets)

#     pred_targets = log_reg.predict(ensemble_train)
#     acc = log_reg.score(ensemble_train, ensemble_targets)
#     f1 = f1_score(ensemble_targets, pred_targets)

#     f1_scores.append(f1)
#     accuracies.append(acc)
#     log_reg_clfs.append(log_reg)

#     print("---------------------")
#     print("Human value: ", human_value)
#     print("Accuracy: ",  acc)
#     print("F1 Score: ", f1)

# print("---------------------")
# print("Value-averaged train F1 score: ", np.mean(f1_scores))
# print("Value-averaged train accuracy: ", np.mean(accuracies))


    








