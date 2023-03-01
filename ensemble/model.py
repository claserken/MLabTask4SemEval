from abc import ABC, abstractmethod
from joblib import load
import numpy as np
import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

from BERT.src.model import ValueClassifier
from BERT.src.stats import ModelPredictions
from BERT.src.data import ArgumentsDataset
from glove import glove_embedder

class Predictor(ABC):
    def __init__(self, model_fname, params):
        self.params = params
        self.model = self.load_model(model_fname)
    
    @abstractmethod
    def load_model(self, model_fname):
        pass
    
    @abstractmethod
    def predict(self, data):
        pass

    @abstractmethod
    def load_data_from_file(self, data_fname):
        pass
    
    def predict_from_file(self, data_fname):
        return self.predict(self.load_data_from_file(data_fname))

class BERTPredictor(Predictor):
    def __init__(self, model_fname, params):
        super().__init__(model_fname, params)
    
    def load_model(self, model_fname):
        model = ValueClassifier(hidden_dim=self.params['hidden_dim'], output_dim=self.params['num_classes']).to(device)
        model.load_state_dict(torch.load(model_fname))
        return model

    def predict(self, dataloader):
        model_predictor = ModelPredictions(self.model, dataloader, self.params['pad_length'], self.params['num_classes'])
        probs = model_predictor.get_probs().cpu()
        return probs
    
    def load_data_from_file(self, data_fname):
        train_dataset = ArgumentsDataset(data_fname, pad_length=self.params['pad_length'], training=False)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.params['val_batch_size'], num_workers=0)
        return train_loader

class GlovePredictor(Predictor):
    def __init__(self, model_fname, params):
        super().__init__(model_fname, params)
    
    def load_model(self, model_fname):
        return load(model_fname)

    def predict(self, data):
        binary_preds = self.model.predict(data).reshape(-1, 1)
        return binary_preds
    
    def load_data_from_file(self, data_fname):
        data = glove_embedder.transform_dataset_from_file(data_fname, self.params['sentence_cols'], self.params['words_to_remove'])
        return data




