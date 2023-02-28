from model import ValueClassifier
from config import TRAIN_PARAMS, DATA_FNAMES, MODEL_URI
from data import ArgumentsDataset
from stats import ModelPredictions
import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model_name = 'bert_b64-w50-e9'
model_path = MODEL_URI + model_name
model = ValueClassifier(hidden_dim=TRAIN_PARAMS['hidden_dim'], output_dim=TRAIN_PARAMS['num_classes']).to(device)
model.load_state_dict(torch.load(model_path))

train_dataset = ArgumentsDataset(DATA_FNAMES['train_arguments'], DATA_FNAMES['train_labels'], pad_length=TRAIN_PARAMS['pad_length'])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=TRAIN_PARAMS['val_batch_size'], num_workers=0)

model_predictor = ModelPredictions(model, train_loader, TRAIN_PARAMS['pad_length'], TRAIN_PARAMS['num_classes'])
probs = model_predictor.get_probs()

probs_fname = './BERT/train_predictions/' + 'train_preds_' + model_name + '.pt'
torch.save(probs, probs_fname)

