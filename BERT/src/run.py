from config import SEED, TRAIN_PARAMS, DATA_FNAMES, SAVE_OPTIONS
from trainer import Trainer
from model import ValueClassifier
import torch
torch.manual_seed(SEED)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = ValueClassifier(hidden_dim=TRAIN_PARAMS['hidden_dim'], output_dim=TRAIN_PARAMS['num_classes']).to(device)
trainer = Trainer(model, TRAIN_PARAMS, DATA_FNAMES)

for epoch in range(TRAIN_PARAMS['num_epochs']):
    print("--------------------- Epoch ", epoch, " ---------------------")
    trainer.train_epoch(verbose=True, iter_print=35)

if SAVE_OPTIONS['save']:
    torch.save(model.state_dict(), SAVE_OPTIONS['path'])


