from data import ArgumentsDataset
from config import TRAIN_PARAMS, DATA_FNAMES
from model import StanceClassifier
from torchmetrics.classification import MultilabelConfusionMatrix
import torch
torch.manual_seed(0)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class Trainer:
    def __init__(self, model, batch_size, train_params, data_fnames):
        self.model = model
        self.train_loader, self.valid_loader = self.get_dataloaders(batch_size, data_fnames)
        self.num_epochs = train_params['num_epochs']
        self.lr = train_params['lr']



    def get_dataloaders(batch_size, data_fnames):
        train_dataset = ArgumentsDataset(data_fnames['train_arguments'], data_fnames['train_labels'])
        valid_dataset = ArgumentsDataset(data_fnames['valid_arguments'], data_fnames['valid_labels'])

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=512, shuffle=True, num_workers=0)
        return train_loader, valid_loader

    
# Hyperparameters
model = StanceClassifier(hidden_dim=769, output_dim=20).to(device)

loss_BCE = torch.nn.BCELoss()
loss_BCE_sum = torch.nn.BCELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr)

# Training Loop
metric = MultilabelConfusionMatrix(num_labels=20).to(device)

for epoch in range(num_epochs):
  for iter_num, batch in enumerate(train_loader):
    # Obtain batch cross-entropy loss
    model.train()
    premise_tensor, attention_mask, labels, stance = batch
    premise_tensor = premise_tensor.reshape(-1, 256).to(device)
    attention_mask = attention_mask.reshape(-1, 256).to(device)
    labels = labels.to(device).float().flatten()
    stance = stance.to(device)

    optimizer.zero_grad()
    
    probs = model(premise_tensor, stance, attention_mask)
    #print("Prob and label shapes")
    #print(probs.flatten().shape)
    #print(labels.flatten().shape)
    pos_count = (labels == 1).sum()
    neg_count = (labels == 0).sum()

    pos_weight = 0.5 / (pos_count / (pos_count + neg_count))
    neg_weight = 0.5 / (neg_count / (pos_count + neg_count))

    weights = torch.ones_like(labels)
    weights[torch.where(labels == 0)] = neg_weight
    weights[torch.where(labels == 1)] = pos_weight 
                  
    loss_fn = torch.nn.BCELoss(weights)
    loss = loss_fn(probs.flatten(), labels)  
    print("Loss: " + str(loss)) 
    loss.backward()
    optimizer.step()


    if iter_num % 35 == 0:
      # Obtain confusion matrix and loss on validation
      validation_confusion_mat = None
      loss_val_tot = 0
      loss_count = 0
      model.eval()
      with torch.no_grad():
        for val_batch in valid_loader:
          premise_tensor, attention_mask, labels, stance = val_batch
          premise_tensor = premise_tensor.reshape(-1, 256).to(device)
          attention_mask = attention_mask.reshape(-1, 256).to(device)
          labels = labels.to(device).int()
          stance = stance.to(device)
          optimizer.zero_grad()
          
          probs = model(premise_tensor, stance, attention_mask)

          batched_confusion_mat = metric(probs, labels)
          if validation_confusion_mat is not None:
            validation_confusion_mat += batched_confusion_mat
          else:
            validation_confusion_mat = batched_confusion_mat

          loss_val_tot += loss_BCE_sum(probs.flatten(), labels.flatten().float())  
          loss_count += len(probs.flatten())

        print("Validation Loss: " + str(loss_val_tot / loss_count)) 
        # print("Validation confusion:")
        # print(validation_confusion_mat)

        sum_f1 = 0

        for confusionMatrix in validation_confusion_mat:

          tn, fp, fn, tp = confusionMatrix.flatten()
          # print(tn, fp, fn, tp)

          f1 = tp / (tp + 0.5 * (fp + fn))
          sum_f1 += f1
          print("f1: ", f1)

        print("average val f1: ", sum_f1/20)
  



  print(f'Epoch {epoch + 1}')



