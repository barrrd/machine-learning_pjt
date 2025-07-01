import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import BertModel

label_mapping = {
    "공포": 0, "놀람": 1, "분노": 2,
    "슬픔": 3, "중립": 4, "행복": 5, "혐오": 6
}

class AIHubEmotionDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.texts = df['Sentence'].values
        self.labels = df['EmotionVec'].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float)
        }

class EmotionClassifier(nn.Module):
    def __init__(self, pretrained_model_name, num_classes):
        super(EmotionClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = self.dropout(outputs.pooler_output)
        return self.classifier(x)