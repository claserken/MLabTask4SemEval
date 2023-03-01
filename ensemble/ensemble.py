from typing import List
from model import Predictor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import pandas as pd

def get_labels(label_fname):
    return pd.read_table(label_fname)

class EnsembledModel:
    def __init__(self, models: List[List[Predictor]], human_values):
        self.models = models
        self.human_values = human_values
        self.log_regs = []

    def train_log_regs(self, data_fname, labels_fname):
        labels = get_labels(labels_fname)
        if self.log_regs:
            self.log_regs = []
            print("Resetting logistic regressors")

        value_preds = self.value_organized_preds(data_fname)
        for human_value, ensemble_train in zip(self.human_values, value_preds):
            log_reg = LogisticRegression(class_weight='balanced')
            log_reg.fit(ensemble_train, labels[human_value])
            self.log_regs.append(log_reg)
    
    def individual_predict(self, data_fname):
        preds = []
        for model in self.models:
            pred = None
            # BERT will iterate once whereas Glove will run 20 times for each value
            for value_predictor in model:
                value_preds = value_predictor.predict_from_file(data_fname)
                if pred is None:
                    pred = value_preds
                else:
                    pred = np.column_stack((pred, value_preds))
            preds.append(pred)
        return preds
    
    def value_organized_preds(self, data_fname):
        model_preds = self.individual_predict(data_fname)
        value_preds = []
        for value_idx in range(len(self.human_values)):
            value_pred = np.column_stack(tuple([pred[:, value_idx] for pred in model_preds]))
            value_preds.append(value_pred)
        return value_preds

    def ensemble_predict(self, data_fname):
        if not self.log_regs:
            raise Exception("Logistic regressors not trained yet.")
    
        value_preds = self.value_organized_preds(data_fname)
        final_preds = None
        for value_idx, value_pred in enumerate(value_preds):
            value_log_reg = self.log_regs[value_idx]
            log_reg_preds = value_log_reg.predict(value_pred)

            if final_preds is None:
                final_preds = log_reg_preds
            else:
                final_preds = np.column_stack((final_preds, log_reg_preds))
        return final_preds

class EnsembleStatistics:
    def __init__(self, ensemble: EnsembledModel):
        self.ensemble = ensemble
    
    def score(self, metric, metric_name, data_fname, labels_fname, verbose):
        scores = []
        labels = get_labels(labels_fname)
        preds = self.ensemble.ensemble_predict(data_fname)
        for value_idx, human_value in enumerate(self.ensemble.human_values):
            # print(preds)
            # print(labels[human_value])
            score = metric(labels[human_value], preds[:, value_idx])
            scores.append(score)
            if verbose:
                print("---------------------")
                print(f'Human value: {human_value}')
                print(f'{metric_name} score: {score}')
        return scores

    def f1_score(self, data_fname, labels_fname, verbose):
        return self.score(f1_score, 'F1', data_fname, labels_fname, verbose)

    def accuracy(self, data_fname, labels_fname, verbose):
        return self.score(accuracy_score, 'Accuracy', data_fname, labels_fname, verbose)

    
            
        

