import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from typing import Union
import numpy as np
import pandas as pd

class ArgumentsDataset(Dataset):
  """Dataset for storing arguments (conclusion-stance-premise)."""
  def __init__(self, arguments_fname: str, label_fname: Union[str, None] = None, training=True, transform=None, pad_length=256):
    """
    Args:
    arguments_fname (string): filename pointing to csv containing arguments
    label_fname (string, optional): filename pointing to csv of labels behind supplied arguments.
    training (bool): if false, corresponding human values are not be provided for each sample
    transform (callable, optional): any transform to be applied to each sample
    """
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    self.arguments_frame = pd.read_csv(arguments_fname, sep='\t')
    self.pad_length = pad_length
    self.labels_frame = None

    if training:
      assert label_fname is not None
      self.labels_frame = pd.read_csv(label_fname, sep='\t')
      assert len(self.arguments_frame) == len(self.labels_frame)
      premises = self.arguments_frame["Premise"]
      max_length = 0
      for premise in premises:
        if len(premise.split(" ")) > max_length:
          max_length = len(premise.split(" "))
      print(max_length)
      self.pad_length = 1
      while self.pad_length < max_length:
        self.pad_length *= 2
    self.transform = transform

  def __len__(self):
    return len(self.arguments_frame)
  
  def __getitem__(self, index: Union[int, torch.Tensor]):
    if torch.is_tensor(index):
      index = index.tolist()
    argument = self.arguments_frame.iloc[index, :]
    # unpacking argument entry in csv
    arg_id = argument[0]
    conclusion = argument[1]
    stance = argument[2]
    premise = argument[3]

    tokenized_info = self.tokenizer.encode_plus(
          premise,
          add_special_tokens=True,
          max_length = self.pad_length,
          pad_to_max_length=True,
          return_attention_mask = True,
          return_tensors ='pt',
      )
    
    sample = [tokenized_info['input_ids'], tokenized_info['attention_mask']]
    bin_stance = 0
    if stance == "in favor of":
      bin_stance = 1
    # get corresponding values from label_fname if this is a training dataset
    if self.labels_frame is not None:
      labels = np.asarray(self.labels_frame.columns)[1:]
      values_for_argument = self.labels_frame.loc[self.labels_frame["Argument ID"] == arg_id]
      values_for_argument = values_for_argument.to_numpy().flatten()[1:].astype('int')
      sample.append(torch.from_numpy(values_for_argument))
    sample.append(bin_stance)
    return sample
