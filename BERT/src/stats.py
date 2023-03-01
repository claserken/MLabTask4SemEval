from abc import ABC, abstractmethod
from torchmetrics.classification import MultilabelConfusionMatrix
import numpy as np
import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class ModelStatistics(ABC):
    def __init__(self, model, dataloader, pad_length, num_classes):
      self.model = model
      self.dataloader = dataloader
      self.pad_length = pad_length
      self.num_classes = num_classes
    
    @abstractmethod
    def metrics(self, probs, labels):
       pass

    def compute_statistics(self, with_labels):
      batch_stats = []
      self.model.eval()
      with torch.no_grad():
        for batch in self.dataloader:
          premise_tensor = attention_mask = labels = stance = None
          if with_labels:
             premise_tensor, attention_mask, labels, stance = batch
             labels = labels.to(device).int()
          else:
             premise_tensor, attention_mask, stance = batch

          premise_tensor = premise_tensor.reshape(-1, self.pad_length).to(device)
          attention_mask = attention_mask.reshape(-1, self.pad_length).to(device)
          stance = stance.to(device)
          
          probs = self.model(premise_tensor, stance, attention_mask)
          batch_stats.append(self.metrics(probs, labels))
      return batch_stats

class LossConfusionStats(ModelStatistics):
    def __init__(self, model, dataloader, pad_length, num_classes):
        super().__init__(model, dataloader, pad_length, num_classes)
        self.loss_fn = torch.nn.BCELoss(reduction='sum')

    def metrics(self, probs, labels):
       conf = MultilabelConfusionMatrix(num_labels=self.num_classes).to(device)
       confusion_mat = conf(probs, labels)
       loss = self.loss_fn(probs.flatten(), labels.flatten().float())  
       return [len(probs.flatten()), loss, confusion_mat]

    def loss_and_confusion_matrix(self):
       batch_stats = self.compute_statistics(with_labels=True)
       confusion_mat = None
       total_loss = 0
       len_data = 0

       for batch_stat in batch_stats:
           batch_len, batch_loss, batch_confusion_mat = batch_stat

           total_loss += batch_loss
           if confusion_mat is not None:
               confusion_mat += batch_confusion_mat
           else:
               confusion_mat = batch_confusion_mat
           len_data += batch_len
       
       avg_loss = total_loss / len_data
       return avg_loss, confusion_mat
    
    def confusion_based_stats(self, confusion_mat):
        sum_f1 = 0
        sum_precision = 0
        sum_recall = 0

        for value_confusion_mat in confusion_mat:
          tn, fp, fn, tp = value_confusion_mat.flatten()
          f1 = tp / (tp + 0.5 * (fp + fn))
          precision = tp / (tp + fp)
          recall = tp / (tp + fn)

          sum_f1 += f1
          sum_precision += precision if not torch.isnan(precision) else 0 # We may not classify any argument as positive in a batch
          sum_recall += recall

        avg_f1 = sum_f1 / self.num_classes
        avg_precision = sum_precision / self.num_classes
        avg_recall = sum_recall / self.num_classes
        return avg_f1, avg_precision, avg_recall

    def print_stats(self, data_name):
       loss, confusion_mat = self.loss_and_confusion_matrix()
       f1_score, precision, recall = self.confusion_based_stats(confusion_mat)
       
       print(data_name)
       print("Loss" + ": " + str(loss))
       print("Average F1 score: " + str(f1_score))
       print("Average precision: " + str(precision))
       print("Average recall: " + str(recall))

class ModelPredictions(ModelStatistics):
    def __init__(self, model, dataloader, pad_length, num_classes):
        super().__init__(model, dataloader, pad_length, num_classes)
    
    def metrics(self, probs, labels):
       return [probs]
    
    def get_probs(self):
       batch_probs = self.compute_statistics(with_labels=False)
       all_probs = None
       for batch_prob in batch_probs:
          if all_probs is None:
             all_probs = batch_prob[0]
          else:
             all_probs = torch.cat((all_probs, batch_prob[0]), axis=0)

       return all_probs