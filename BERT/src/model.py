import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class StanceClassifier(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        bert_output_dim = 768
        bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
        bert_model.train()

        self.bert_model = bert_model
        self.linear1 = nn.Linear(bert_output_dim + 1, hidden_dim) # Add one for the stance
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = torch.sigmoid

    def forward(self, premise, stance, attention_mask):
        bert_output = self.bert_model(premise, attention_mask=attention_mask).pooler_output
        combined_output_stance = torch.cat([bert_output, stance.unsqueeze(1)], dim=1)

        hidden_layer = self.linear1(combined_output_stance)
        hidden_layer = F.relu(hidden_layer)
        
        logits = self.linear2(hidden_layer)
        output_probs = self.sigmoid(logits)

        return output_probs