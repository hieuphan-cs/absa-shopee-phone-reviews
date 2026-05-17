import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Import toàn bộ cấu hình từ file config.py
import config

# ==========================================
# 1. TIỀN XỬ LÝ DỮ LIỆU
# ==========================================
def load_and_group_data(file_path):
    df = pd.read_json(file_path, lines=True)
    unique_aspects = sorted(df['aspect'].dropna().unique().tolist())
    
    grouped = df.groupby('text').apply(
        lambda x: dict(zip(x['aspect'], x['sentiment']))
    ).reset_index(name='aspect_sentiments')
    
    texts = grouped['text'].tolist()
    aspect_dicts = grouped['aspect_sentiments'].tolist()
    
    labels = []
    for d in aspect_dicts:
        label_row = []
        for asp in unique_aspects:
            sent = d.get(asp, 'none')
            # Sử dụng biến từ config
            label_row.append(config.SENTIMENT_MAP.get(sent, 0)) 
        labels.append(label_row)
        
    return texts, labels, unique_aspects

class ABSADataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.max_len,
            padding='max_length', truncation=True,
            return_attention_mask=True, return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ==========================================
# 2. KIẾN TRÚC MÔ HÌNH
# ==========================================
# (Class HighwayNetwork giữ nguyên như cũ)
class HighwayNetwork(nn.Module):
    def __init__(self, size):
        super(HighwayNetwork, self).__init__()
        self.transform = nn.Linear(size, size)
        self.gate = nn.Linear(size, size)

    def forward(self, x):
        t = torch.sigmoid(self.gate(x))
        h = torch.relu(self.transform(x))
        return t * h + (1.0 - t) * x

class PhoBERT_BiLSTM_Highway_ABSA(nn.Module):
    def __init__(self, num_aspects, num_sentiments=4):
        super(PhoBERT_BiLSTM_Highway_ABSA, self).__init__()
        self.num_aspects = num_aspects
        self.num_sentiments = num_sentiments
        
        # Đọc tên mô hình từ config
        self.phobert = AutoModel.from_pretrained(config.PHOBERT_VERSION)
        
        # Kiểm tra config xem có đóng băng PhoBERT không
        if config.FREEZE_PHOBERT:
            for param in self.phobert.parameters():
                param.requires_grad = False
            
        # Đọc tham số LSTM từ config
        self.bilstm = nn.LSTM(input_size=768, hidden_size=config.LSTM_HIDDEN_SIZE, 
                              num_layers=config.LSTM_LAYERS, bidirectional=True, batch_first=True)
        
        # Highway (Hidden size * 2 vì là Bi-directional)
        highway_size = config.LSTM_HIDDEN_SIZE * 2
        self.highway = HighwayNetwork(size=highway_size)
        
        self.classifier = nn.Linear(highway_size, num_aspects * num_sentiments)

    def forward(self, input_ids, attention_mask):
        phobert_out = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = phobert_out.last_hidden_state
        lstm_out, _ = self.bilstm(sequence_output)
        pooled_out, _ = torch.max(lstm_out, dim=1)
        highway_out = self.highway(pooled_out)
        logits = self.classifier(highway_out) 
        logits = logits.view(-1, self.num_aspects, self.num_sentiments)
        return logits

# (Các hàm train_epoch, eval_model giữ nguyên, tôi rút gọn ở đây để bạn dễ nhìn)
# ...

# ==========================================
# 3. CHẠY PIPELINE
# ==========================================
if __name__ == "__main__":
    print(f"Using device: {config.DEVICE}")

    print("Loading data...")
    texts, labels, unique_aspects = load_and_group_data(config.DATA_FILE)
    
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
    val_texts, test_texts, val_labels, test_labels = train_test_split(temp_texts, temp_labels, test_size=0.5, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(config.PHOBERT_VERSION)
    
    # Dùng MAX_LEN từ config
    train_dataset = ABSADataset(train_texts, train_labels, tokenizer, config.MAX_LEN)
    val_dataset = ABSADataset(val_texts, val_labels, tokenizer, config.MAX_LEN)

    # Dùng BATCH_SIZE từ config
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)

    model = PhoBERT_BiLSTM_Highway_ABSA(num_aspects=len(unique_aspects)).to(config.DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    
    # Dùng LEARNING_RATE từ config
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    best_val_acc = 0
    # Dùng EPOCHS từ config
    for epoch in range(config.EPOCHS):
        # Gọi hàm train và eval của bạn...
        # ...
        
        # Nếu mô hình tốt nhất, lưu vào MODEL_SAVE_PATH
        val_acc = 0.9 # Ví dụ
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)