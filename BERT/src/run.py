from trainer import Trainer
from config import TRAIN_PARAMS, DATA_FNAMES
from model import StanceClassifier
import torch
torch.manual_seed(0)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = StanceClassifier(hidden_dim=TRAIN_PARAMS['hidden_dim'], output_dim=TRAIN_PARAMS['num_classes']).to(device)
trainer = Trainer(model, TRAIN_PARAMS, DATA_FNAMES)

for epoch in range(TRAIN_PARAMS['num_epochs']):
    print("--------------------- Epoch ", epoch, " ---------------------")
    trainer.train_epoch(verbose=True, iter_print=35)

