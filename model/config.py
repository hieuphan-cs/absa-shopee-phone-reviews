import torch

# ==========================================
# CẤU HÌNH DỰ ÁN ABSA (config.py)
# ==========================================

# 1. Đường dẫn file (Paths)
DATA_FILE = "ABSA/data/outputs/absa_training_flat.jsonl"
MODEL_SAVE_PATH = "best_absa_model.pth"

# 2. Tham số mô hình (Model Parameters)
PHOBERT_VERSION = "vinai/phobert-base"
MAX_LEN = 128               # Độ dài tối đa của câu sau khi tokenize
LSTM_HIDDEN_SIZE = 256      # Kích thước hidden state của Bi-LSTM
LSTM_LAYERS = 1             # Số lớp LSTM
FREEZE_PHOBERT = False      # Đặt True nếu muốn đóng băng PhoBERT để train nhanh hơn

# 3. Tham số huấn luyện (Training Hyperparameters)
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 2e-5        # Khuyến nghị: 2e-5 đến 5e-5 nếu Fine-tune PhoBERT; 1e-3 nếu Freeze PhoBERT

# 4. Nhãn phân loại (Labels)
SENTIMENT_MAP = {'none': 0, 'positive': 1, 'neutral': 2, 'negative': 3}
ID_TO_SENTIMENT = {v: k for k, v in SENTIMENT_MAP.items()}

# 5. Thiết bị (Cấu hình tự động nhận diện GPU/CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 6. Cấu hình Seed (Đảm bảo tái lập kết quả)
SEED = 42