from sklearn.metrics import f1_score
from joblib import dump

class Trainer:
    def __init__(self, model, X_train, y_train, X_val, y_val):
        self.model = model
        self.X_train, self.y_train, self.X_val, self.y_val = X_train, y_train, X_val, y_val

    def train(self):
        self.model.fit(self.X_train, self.y_train)
    
    def f1_score(self, data_name):
        if data_name == "train":
            preds = self.model.predict(self.X_train)
            return f1_score(self.y_train, preds)
        elif data_name == "validation":
            preds = self.model.predict(self.X_val)
            return f1_score(self.y_val, preds)
        else:
            raise Exception("Invalid data name specified.")

class Logger:
    def __init__(self, trainer, value_name):
        self.trainer = trainer
        self.value_name = value_name
    
    def save_model(self, save_path):
        dump(self.trainer.model, save_path)

    def print_f1_scores(self):
        print("------------------")
        print(self.value_name)
        print("* Train F1: ", self.trainer.f1_score("train"))
        print("* Validation F1: ", self.trainer.f1_score("validation"))

        