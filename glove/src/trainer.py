from sklearn.metrics import f1_score

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
    
    
    


        