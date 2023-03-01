from typing import List
from model import Predictor
from sklearn.linear_model import LogisticRegression
import numpy as np

class EnsembledModel:
    def __init__(self, models: List[List[Predictor]], human_values):
        self.models = models
        self.human_values = human_values
        self.log_regs = []
    
    def train_log_regs(self, data_fname, labels):
        if self.log_regs:
            self.log_regs = []
            print("Resetting logistic regressors")

        value_preds = self.value_organized_preds(data_fname)
        for ensemble_train in value_preds:
            log_reg = LogisticRegression(class_weight='balanced')
            log_reg.fit(ensemble_train, labels)
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



