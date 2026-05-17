import random
import os
# ... (các import cũ như torch, numpy, pandas...)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import cấu hình
import config

def set_seed(seed_value):
    """Cố định seed cho toàn bộ môi trường để tái lập kết quả"""
    # 1. Cố định cho Python và Hashing
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    # 2. Cố định cho Numpy
    np.random.seed(seed_value)
    
    # 3. Cố định cho PyTorch (CPU)
    torch.manual_seed(seed_value)
    
    # 4. Cố định cho PyTorch (GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # Dành cho multi-GPU
        
    # 5. Cố định thuật toán của CUDNN
    # Lưu ý: deterministic=True có thể làm model train chậm hơn một chút
    # nhưng đảm bảo kết quả 100% giống nhau giữa các lần chạy.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==========================================
# 1. TIỀN XỬ LÝ DỮ LIỆU
# ==========================================
def load_and_group_data(file_path):
    print("Reading data...")
    df = pd.read_json(file_path, lines=True)
    unique_aspects = sorted(df['aspect'].dropna().unique().tolist())
    
    # Thêm include_groups=False để tắt cảnh báo Pandas
    grouped = df.groupby('text').apply(
        lambda x: dict(zip(x['aspect'], x['sentiment'])),
        include_groups=False
    ).reset_index(name='aspect_sentiments')
    
    # ... (phần code bên dưới giữ nguyên)
    
    texts = grouped['text'].tolist()
    aspect_dicts = grouped['aspect_sentiments'].tolist()
    
    labels = []
    for d in aspect_dicts:
        label_row = []
        for asp in unique_aspects:
            sent = d.get(asp, 'none')
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

        # Gọi trực tiếp tokenizer thay vì dùng encode_plus
        encoding = self.tokenizer(
            text, 
            add_special_tokens=True, 
            max_length=self.max_len,
            padding='max_length', 
            truncation=True,
            return_attention_mask=True, 
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ==========================================
# 2. KIẾN TRÚC MÔ HÌNH
# ==========================================
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
        
        self.phobert = AutoModel.from_pretrained(config.PHOBERT_VERSION)
        
        if config.FREEZE_PHOBERT:
            for param in self.phobert.parameters():
                param.requires_grad = False
            
        self.bilstm = nn.LSTM(input_size=768, hidden_size=config.LSTM_HIDDEN_SIZE, 
                              num_layers=config.LSTM_LAYERS, bidirectional=True, batch_first=True)
        
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

# ==========================================
# 3. HÀM TRAIN & EVALUATE
# ==========================================
def train_epoch(model, data_loader, loss_fn, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    # Format TQDM đẹp hơn, hiện tiến trình chạy
    loop = tqdm(data_loader, leave=False, bar_format="{l_bar}{bar:30}{r_bar}")
    loop.set_description(f"Epoch [{epoch}/{total_epochs}] Train")
    
    for batch in loop:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)

        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)
        
        loss = loss_fn(logits_flat, labels_flat)
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits_flat, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels_flat.cpu().numpy())
        
        # Hiển thị loss realtime trên thanh TQDM
        loop.set_postfix(loss=f"{loss.item():.4f}")

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return total_loss / len(data_loader), acc, f1

def eval_model(model, data_loader, loss_fn, device, phase="Val  ", epoch=None, total_epochs=None):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    loop = tqdm(data_loader, leave=False, bar_format="{l_bar}{bar:30}{r_bar}")
    if epoch:
        loop.set_description(f"Epoch [{epoch}/{total_epochs}] {phase.strip()}")
    else:
        loop.set_description(f"{phase}")

    with torch.no_grad():
        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask)
            
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)
            
            loss = loss_fn(logits_flat, labels_flat)
            total_loss += loss.item()

            preds = torch.argmax(logits_flat, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_flat.cpu().numpy())
            
            loop.set_postfix(loss=f"{loss.item():.4f}")

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return total_loss / len(data_loader), acc, f1, all_preds, all_labels

def plot_history(history):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_f1'], 'b-', label='Train F1')
    plt.plot(epochs, history['val_f1'], 'r-', label='Val F1')
    plt.title('F1-Macro Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

# ==========================================
# 4. CHƯƠNG TRÌNH CHÍNH
# ==========================================
if __name__ == "__main__":
    # --- GỌI HÀM SET SEED Ở ĐÂY ---
    set_seed(config.SEED)
    print(f"Set SEED = {config.SEED} for reproducibility.")
    # ------------------------------

    print(f"Using device: {config.DEVICE}")

    # 1. Load Data
    texts, labels, unique_aspects = load_and_group_data(config.DATA_FILE)
    
    # Ở hàm train_test_split, bạn cũng nên dùng chung config.SEED
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=config.SEED
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=config.SEED
    )

    # ... (phần code bên dưới giữ nguyên)

    # 2. Tokenizer & DataLoader
    tokenizer = AutoTokenizer.from_pretrained(config.PHOBERT_VERSION)
    
    train_dataset = ABSADataset(train_texts, train_labels, tokenizer, config.MAX_LEN)
    val_dataset = ABSADataset(val_texts, val_labels, tokenizer, config.MAX_LEN)
    test_dataset = ABSADataset(test_texts, test_labels, tokenizer, config.MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)

    # 3. Khởi tạo Model
    model = PhoBERT_BiLSTM_Highway_ABSA(num_aspects=len(unique_aspects)).to(config.DEVICE)
    
    # THÊM CLASS WEIGHT ĐỂ TRỊ MẤT CÂN BẰNG DỮ LIỆU
    # Trọng số: None (0.1) - ít quan tâm | Positive (1.0), Neutral (1.5), Negative (1.5) - bắt buộc phải học
    class_weights = torch.tensor([0.1, 1.0, 2.0, 1.5]).to(config.DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    # 4. Training Loop
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'train_f1': [], 'val_f1': []}
    best_val_f1 = 0 

    for epoch in range(config.EPOCHS):
        # Truyền thêm epoch+1 và config.EPOCHS vào hàm
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, loss_fn, optimizer, config.DEVICE, epoch + 1, config.EPOCHS)
        val_loss, val_acc, val_f1, _, _ = eval_model(model, val_loader, loss_fn, config.DEVICE, phase="Validation", epoch=epoch + 1, total_epochs=config.EPOCHS)
        
        # Các dòng append history và in kết quả giữ nguyên
        # ...
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        
        print(f"Train | Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"Val   | Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
        
        # Lưu model tốt nhất theo F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(">> Saved Best Model (F1 improved)!")

    # 5. Plot Biểu Đồ
    print("\nTraining completed. Plotting history...")
    plot_history(history)

    # 6. Đánh Giá Tập Test (Metric Cuối Cùng)
    print("\n=========================================")
    print("       FINAL EVALUATION ON TEST SET      ")
    print("=========================================")
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    test_loss, test_acc, test_f1, test_preds, test_labels = eval_model(model, test_loader, loss_fn, config.DEVICE, phase="Testing")
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1-Macro: {test_f1:.4f}\n")
    
    # Báo cáo chi tiết từng nhãn
    target_names = []
    for id_val in range(4): # 0, 1, 2, 3
        target_names.append(config.ID_TO_SENTIMENT.get(id_val, f"Class {id_val}"))
        
    print("Detailed Classification Report (For all Aspects & Sentiments):")
    print(classification_report(test_labels, test_preds, target_names=target_names, zero_division=0))
    
    
    # ==========================================
    # 7. INFERENCE (TEST THỬ VỚI CÂU MỚI)
    # ==========================================
    def predict_sentiment(text, model, tokenizer, unique_aspects, device, max_len=128):
        model.eval()
        encoding = tokenizer(
            text, add_special_tokens=True, max_length=max_len,
            padding='max_length', truncation=True,
            return_attention_mask=True, return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=2).squeeze(0) # Lấy nhãn có xác suất cao nhất
            
        result = {}
        for i, asp in enumerate(unique_aspects):
            sentiment_id = preds[i].item()
            if sentiment_id != 0: # 0 là 'none' (không nhắc đến)
                result[asp] = config.ID_TO_SENTIMENT[sentiment_id]
                
        return result

    print("\n=========================================")
    print("         INFERENCE (TEST CASES)          ")
    print("=========================================")
    
    # Load lại model tốt nhất trước khi test (phòng hờ)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    
    test_cases = [
        # Trường hợp 1: Tích cực đa khía cạnh
        "Điện thoại thiết kế rất đẹp và sang trọng, chụp ảnh sắc nét, shop giao hàng hỏa tốc luôn. Rất đáng tiền!",
        
        # Trường hợp 2: Tiêu cực đa khía cạnh
        "Pin tụt nhanh kinh khủng, mới chơi game xíu đã nóng ran. Gọi điện bảo hành thì nhân viên trả lời thái độ vô cùng lồi lõm, thất vọng thực sự.",
        
        # Trường hợp 3: Hỗn hợp (Vừa khen vừa chê)
        "Máy cầm đầm tay, màn hình hiển thị khá ổn nhưng loa nghe hơi nhỏ. Giá cả như vậy thì cũng tạm chấp nhận được.",
        
        # Trường hợp 4: Ngắn gọn, chỉ 1 khía cạnh
        "Shipper nhiệt tình thân thiện.",
        
        # Trường hợp 5: Không rõ ràng (Challenging)
        "Nói chung là xài cũng được, không có gì nổi bật."
    ]

    for i, text in enumerate(test_cases, 1):
        print(f"\n[Test Case {i}]: {text}")
        predictions = predict_sentiment(text, model, tokenizer, unique_aspects, config.DEVICE, config.MAX_LEN)
        
        if not predictions:
            print("  => Dự đoán: Không phát hiện khía cạnh nào (None).")
        else:
            for aspect, sentiment in predictions.items():
                # In ra có màu sắc cho dễ nhìn (Tùy chọn)
                if sentiment == 'positive':
                    sent_str = "🟢 POSITIVE"
                elif sentiment == 'negative':
                    sent_str = "🔴 NEGATIVE"
                else:
                    sent_str = "🟡 NEUTRAL"
                    
                print(f"  => Khía cạnh: {aspect.ljust(15)} | Sentiment: {sent_str}")