from data import ArgumentsDataset
from stats import LossConfusionStats
import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Trainer:
    def __init__(self, model, train_params, data_fnames):
        self.model = model
        self.train_params = train_params
        self.train_loader, self.valid_loader = self.get_dataloaders(data_fnames)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.train_params['lr'])

    def get_dataloaders(self, data_fnames):
        train_dataset = ArgumentsDataset(data_fnames['train_arguments'], data_fnames['train_labels'], pad_length=self.train_params['pad_length'])
        valid_dataset = ArgumentsDataset(data_fnames['valid_arguments'], data_fnames['valid_labels'], pad_length=self.train_params['pad_length'])

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.train_params['train_batch_size'], shuffle=True, num_workers=0)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=self.train_params['val_batch_size'], num_workers=0) 
        return train_loader, valid_loader

    def train_epoch(self, verbose, iter_print):
        self.model.train()
        for iter_num, batch in enumerate(self.train_loader):
            premise_tensor, attention_mask, labels, stance = batch
            premise_tensor = premise_tensor.reshape(-1, self.train_params['pad_length']).to(device)
            attention_mask = attention_mask.reshape(-1, self.train_params['pad_length']).to(device)
            labels = labels.to(device).float().flatten()
            stance = stance.to(device)

            probs = self.model(premise_tensor, stance, attention_mask)

            pos_count = (labels == 1).sum()
            neg_count = (labels == 0).sum()
            pos_weight = self.train_params['pos_samples_weight'] / (pos_count / (pos_count + neg_count))
            neg_weight = (1 - self.train_params['pos_samples_weight']) / (neg_count / (pos_count + neg_count))

            weights = torch.ones_like(labels)
            weights[torch.where(labels == 0)] = neg_weight
            weights[torch.where(labels == 1)] = pos_weight 
                          
            loss_fn = torch.nn.BCELoss(weights)
            loss = loss_fn(probs.flatten(), labels)  
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if verbose:
              print("Training Loss: " + str(loss))
              if iter_num % iter_print == 0:
                stats = LossConfusionStats(self.model, self.valid_loader, self.train_params['pad_length'], self.train_params['num_classes'])
                stats.print_stats(data_name="VALIDATION")


         