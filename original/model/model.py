from transformers import BertPreTrainedModel, BertModel
import torch.nn as nn
from torch.nn import BCELoss

class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.softmax = nn.Softmax(dim=-1)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[0]

        cls_mask = input_ids == 101
        cls_output = pooled_output[cls_mask]
        pooled_output = self.dropout(cls_output)
        logits = self.classifier(pooled_output)

        #predicts = logits.argmax(dim=-1).cpu().detach().numpy().tolist()
        #logits = logits.argmax(dim=-1)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

            return loss, logits
        else:
            return logits
