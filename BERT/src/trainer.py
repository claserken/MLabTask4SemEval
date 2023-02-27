from data import ArgumentsDataset
from torchmetrics.classification import MultilabelConfusionMatrix
import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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
        self.optimizer.zero_grad()
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
            loss.backward()
            self.optimizer.step()

            if verbose:
              print("Training Loss: " + str(loss))
              if iter_num % iter_print == 0:
                stats = ModelStatistics(self.model, self.valid_loader, self.train_params['pad_length'], self.train_params['num_classes'])
                stats.print_stats("Validation")

class ModelStatistics:
    def __init__(self, model, dataloader, pad_length, num_classes):
       self.model = model
       self.dataloader = dataloader
       self.pad_length = pad_length
       self.num_classes = num_classes
       self.loss_fn = torch.nn.BCELoss(reduction='sum')

    def loss_and_confusion_matrix(self):
      confusion_mat = None
      total_loss = 0
      len_data = 0

      metric = MultilabelConfusionMatrix(num_labels=self.num_classes).to(device)
      self.model.eval()
      with torch.no_grad():
        for batch in self.dataloader:
          premise_tensor, attention_mask, labels, stance = batch
          premise_tensor = premise_tensor.reshape(-1, self.pad_length).to(device)
          attention_mask = attention_mask.reshape(-1, self.pad_length).to(device)
          labels = labels.to(device).int()
          stance = stance.to(device)
          
          probs = self.model(premise_tensor, stance, attention_mask)
          batch_confusion_mat = metric(probs, labels)
          if confusion_mat is not None:
            confusion_mat += batch_confusion_mat
          else:
            confusion_mat = batch_confusion_mat

          total_loss += self.loss_fn(probs.flatten(), labels.flatten().float())  
          len_data += len(probs.flatten())

      avg_loss = total_loss / len_data
      return avg_loss, confusion_mat
    
    def confusion_to_f1(self, confusion_mat):
        sum_f1 = 0
        for value_confusion_mat in confusion_mat:
          tn, fp, fn, tp = value_confusion_mat.flatten()
          f1 = tp / (tp + 0.5 * (fp + fn))
          sum_f1 += f1
        avg_f1 = sum_f1 / self.num_classes
        return avg_f1

    def print_stats(self, data_name):
       loss, confusion_mat = self.loss_and_confusion_matrix()
       f1_score = self.confusion_to_f1(confusion_mat)

       print(data_name + ": " + loss)
       print("Value-averaged F1 score: " + str(f1_score))
       

# # Hyperparameters
# model = StanceClassifier(hidden_dim=TRAIN_PARAMS['hidden_dim'], output_dim=TRAIN_PARAMS['num_classes']).to(device)

# loss_BCE = torch.nn.BCELoss()
# loss_BCE_sum = torch.nn.BCELoss(reduction='sum')

# # Training Loop

# for epoch in range(num_epochs):
#   for iter_num, batch in enumerate(train_loader):
#     # Obtain batch cross-entropy loss
#     model.train()
#     premise_tensor, attention_mask, labels, stance = batch
#     premise_tensor = premise_tensor.reshape(-1, 256).to(device)
#     attention_mask = attention_mask.reshape(-1, 256).to(device)
#     labels = labels.to(device).float().flatten()
#     stance = stance.to(device)

#     optimizer.zero_grad()
    
#     probs = model(premise_tensor, stance, attention_mask)
#     #print("Prob and label shapes")
#     #print(probs.flatten().shape)
#     #print(labels.flatten().shape)
#     pos_count = (labels == 1).sum()
#     neg_count = (labels == 0).sum()

#     pos_weight = 0.5 / (pos_count / (pos_count + neg_count))
#     neg_weight = 0.5 / (neg_count / (pos_count + neg_count))

#     weights = torch.ones_like(labels)
#     weights[torch.where(labels == 0)] = neg_weight
#     weights[torch.where(labels == 1)] = pos_weight 
                  
#     loss_fn = torch.nn.BCELoss(weights)
#     loss = loss_fn(probs.flatten(), labels)  
#     print("Loss: " + str(loss)) 
#     loss.backward()
#     optimizer.step()


#     if iter_num % 35 == 0:
#       # Obtain confusion matrix and loss on validation
      
  



#   print(f'Epoch {epoch + 1}')



