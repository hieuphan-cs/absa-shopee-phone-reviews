"""
=============================================================================
ABSA - Aspect-Based Sentiment Analysis  |  BiLSTM + Attention  |  PyTorch
=============================================================================

BẢN CHẤT BÀI TOÁN (đúng):
  - Dataset có 92 reviews, mỗi review chứa NHIỀU (aspect, sentiment) pairs
  - Ví dụ: 1 review → [giao_hang: positive, camera: neutral, pin: negative]
  - Formulation đúng: Aspect Category Sentiment Analysis (ACSA)

  Input  : text (câu review)
  Output : với mỗi aspect trong 14 aspects →
             4 khả năng: none | negative | neutral | positive

  Tức là: 14 bài toán phân loại 4 lớp, dùng chung 1 BiLSTM backbone.
  (Đây là multi-task learning với 14 đầu ra song song)

CẤU TRÚC DỮ LIỆU (absa_training_flat.jsonl):
  - Mỗi dòng = 1 (review, aspect, sentiment, opinion_phrase) pair
  - Cùng review_id → cùng text, nhiều dòng khác nhau về aspect/sentiment
  - Aggregate: nếu cùng aspect xuất hiện nhiều lần → lấy sentiment đa số

LABEL SCHEMA:
  Mỗi aspect → {0: none, 1: negative, 2: neutral, 3: positive}
  none = aspect đó không được đề cập trong review

=============================================================================
"""

import json
import re
import random
import numpy as np
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0. SEED & DEVICE
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device: {DEVICE}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. CẤU HÌNH
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    "data_path"   : "ABSA/data/outputs/absa_training_flat.jsonl",
    "max_len"     : 128,      # token tối đa mỗi câu
    "min_freq"    : 1,        # ngưỡng vocab

    # Model
    "embed_dim"   : 64,
    "hidden_dim"  : 256,      # BiLSTM hidden (mỗi chiều) → output 512
    "num_layers"  : 1,
    "dropout"     : 0.3,

    # Training
    "batch_size"  : 8,        # nhỏ vì chỉ có 92 reviews
    "epochs"      : 60,
    "lr"          : 1e-3,
    "weight_decay": 1e-4,
    "patience"    : 10,       # early stopping

    # Loss: trọng số cho "none" thấp hơn (class chiếm đa số)
    "none_weight" : 0.3,
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. DANH SÁCH ASPECTS CỐ ĐỊNH (14 aspects)
# ─────────────────────────────────────────────────────────────────────────────
ALL_ASPECTS = [
    'am_thanh', 'bao_hanh', 'camera', 'dich_vu',
    'gia_tri',  'giao_hang', 'hieu_nang', 'ket_noi',
    'man_hinh', 'phu_kien',  'pin',      'pin_sac',
    'thiet_ke', 'tong_the'
]
ASP2IDX = {a: i for i, a in enumerate(ALL_ASPECTS)}
NUM_ASPECTS = len(ALL_ASPECTS)

# Label cho mỗi slot: 0=none, 1=negative, 2=neutral, 3=positive
SENT2IDX = {"none": 0, "negative": 1, "neutral": 2, "positive": 3}
IDX2SENT = {v: k for k, v in SENT2IDX.items()}
NUM_SENT_CLASSES = 4  # none / neg / neu / pos

# ─────────────────────────────────────────────────────────────────────────────
# 3. TIỀN XỬ LÝ VĂN BẢN
# ─────────────────────────────────────────────────────────────────────────────
NORMALIZE_MAP = {
    "ko": "không", "k": "không", "kh": "không",
    "dc": "được",  "đc": "được",
    "vs": "với",   "ntn": "như thế nào",
    "sp": "sản phẩm", "sdt": "số điện thoại",
    "bh": "bảo hành",
    "nv": "nhân viên",
    "ok": "ổn",
    "ship": "giao hàng",
    "shipper": "người giao hàng",
}

def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [NORMALIZE_MAP.get(t, t) for t in text.split()]
    return " ".join(tokens)

def tokenize(text: str) -> list:
    return normalize(text).split()

# ─────────────────────────────────────────────────────────────────────────────
# 4. ĐỌC & XÂY DỰNG DỮ LIỆU ĐÚNG BẢN CHẤT
# ─────────────────────────────────────────────────────────────────────────────
def load_data(path: str) -> list:
    """
    Đọc JSONL, group theo review_id.
    Mỗi review → {
        text: str,
        labels: list[int] shape (14,)
            labels[i] = 0 nếu aspect i không xuất hiện (none)
            labels[i] = 1/2/3 nếu aspect i có sentiment neg/neu/pos
    }
    Nếu cùng aspect xuất hiện nhiều lần trong 1 review →
        lấy sentiment đa số (majority vote).
    """
    # Bước 1: đọc raw
    raw = defaultdict(lambda: {"text": "", "aspect_sents": defaultdict(list)})
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if not d.get("aspect") or not d.get("sentiment"):
                continue
            rid = d["review_id"]
            raw[rid]["text"] = d["text"]
            raw[rid]["aspect_sents"][d["aspect"]].append(d["sentiment"])

    # Bước 2: xây dựng label vector cho mỗi review
    samples = []
    for rid, info in raw.items():
        text   = info["text"]
        # label vector: mặc định = 0 (none)
        labels = [0] * NUM_ASPECTS

        for asp, sents in info["aspect_sents"].items():
            if asp not in ASP2IDX:
                continue
            # Majority vote
            dominant_sent = Counter(sents).most_common(1)[0][0]
            labels[ASP2IDX[asp]] = SENT2IDX[dominant_sent]

        samples.append({"text": text, "labels": labels})

    return samples

def analyze_data(samples: list):
    print(f"\n{'='*65}")
    print(f"📊 PHÂN TÍCH DỮ LIỆU")
    print(f"{'='*65}")
    print(f"  Số reviews (samples) : {len(samples)}")

    # Đếm phân phối label cho từng aspect
    print(f"\n  {'Aspect':<15}  {'none':>6}  {'neg':>6}  {'neu':>6}  {'pos':>6}")
    print(f"  {'-'*50}")
    for i, asp in enumerate(ALL_ASPECTS):
        counts = Counter(s["labels"][i] for s in samples)
        print(f"  {asp:<15}  "
              f"{counts[0]:>6}  {counts[1]:>6}  {counts[2]:>6}  {counts[3]:>6}")

    # Số aspect trung bình mỗi review
    avg_asp = np.mean([sum(1 for l in s["labels"] if l > 0) for s in samples])
    print(f"\n  Trung bình số aspect/review: {avg_asp:.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. VOCABULARY
# ─────────────────────────────────────────────────────────────────────────────
class Vocabulary:
    PAD, UNK = "<PAD>", "<UNK>"

    def __init__(self, min_freq=1):
        self.min_freq  = min_freq
        self.token2idx = {self.PAD: 0, self.UNK: 1}
        self.idx2token = {0: self.PAD, 1: self.UNK}

    def build(self, texts: list):
        freq = Counter()
        for t in texts:
            freq.update(tokenize(t))
        for tok, cnt in freq.items():
            if cnt >= self.min_freq and tok not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[tok] = idx
                self.idx2token[idx] = tok
        print(f"\n  📖 Vocabulary: {len(self.token2idx):,} tokens")

    def encode(self, text: str, max_len: int) -> list:
        toks = tokenize(text)[:max_len]
        ids  = [self.token2idx.get(t, 1) for t in toks]
        ids += [0] * (max_len - len(ids))
        return ids

    def __len__(self):
        return len(self.token2idx)

# ─────────────────────────────────────────────────────────────────────────────
# 6. PYTORCH DATASET
# ─────────────────────────────────────────────────────────────────────────────
class ABSADataset(Dataset):
    """
    Mỗi item: (input_ids, labels)
      input_ids : LongTensor (max_len,)
      labels    : LongTensor (num_aspects,)  — giá trị 0..3 mỗi slot
    """
    def __init__(self, samples, vocab, max_len):
        self.items = []
        for s in samples:
            ids    = vocab.encode(s["text"], max_len)
            labels = s["labels"]
            self.items.append((
                torch.tensor(ids,    dtype=torch.long),
                torch.tensor(labels, dtype=torch.long),
            ))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

# ─────────────────────────────────────────────────────────────────────────────
# 7. MÔ HÌNH BiLSTM ACSA
# ─────────────────────────────────────────────────────────────────────────────
class ABSABiLSTM(nn.Module):
    """
    Kiến trúc Aspect Category Sentiment Analysis:

    Embedding → BiLSTM → Aspect-Specific Attention → Context Vector (per aspect)
                                                              ↓
                                    [Head_0, Head_1, ..., Head_13]  (14 heads)
                                    Mỗi head: 4 classes (none/neg/neu/pos)

    Key design: mỗi aspect có attention riêng → học phần text liên quan đến
    aspect đó → tốt hơn dùng chung 1 attention cho tất cả.
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers,
                 num_aspects, num_sent_classes, dropout, pad_idx=0):
        super().__init__()

        # ── Embedding ─────────────────────────────────────────────────
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=pad_idx
        )

        # ── BiLSTM backbone ───────────────────────────────────────────
        # bidirectional=True → output dim = hidden_dim * 2
        self.bilstm = nn.LSTM(
            input_size    = embed_dim,
            hidden_size   = hidden_dim,
            num_layers    = num_layers,
            bidirectional = True,
            batch_first   = True,
            dropout       = dropout if num_layers > 1 else 0.0,
        )

        context_dim = hidden_dim * 2   # 512

        # ── Aspect-Specific Attention (14 attention layers) ───────────
        # Mỗi aspect học attention weight riêng trên các token
        # → "camera" chú ý token "chụp", "ảnh"; "pin" chú ý "sạc", "trâu"
        self.aspect_attention = nn.ModuleList([
            nn.Linear(context_dim, 1, bias=False)
            for _ in range(num_aspects)
        ])

        self.dropout = nn.Dropout(dropout)

        # ── 14 Classifier Heads ───────────────────────────────────────
        # Mỗi head: context_dim → hidden → 4 (none/neg/neu/pos)
        self.aspect_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(context_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_sent_classes),
            )
            for _ in range(num_aspects)
        ])

    def forward(self, input_ids):
        """
        input_ids : (B, T)
        Returns:
          logits_list: list of 14 tensors, mỗi tensor shape (B, 4)
        """
        # Embedding + dropout
        emb = self.dropout(self.embedding(input_ids))   # (B, T, E)

        # BiLSTM
        lstm_out, _ = self.bilstm(emb)                  # (B, T, H*2)

        # Mask padding để không attend vào PAD token
        pad_mask = (input_ids == 0)                     # (B, T) True = PAD

        logits_list = []
        for i in range(len(self.aspect_heads)):
            # Aspect-specific attention score
            attn = self.aspect_attention[i](lstm_out).squeeze(-1)  # (B, T)
            attn = attn.masked_fill(pad_mask, -1e9)
            attn = torch.softmax(attn, dim=-1)                     # (B, T)

            # Weighted context vector riêng cho aspect i
            ctx = torch.bmm(
                attn.unsqueeze(1),   # (B, 1, T)
                lstm_out             # (B, T, H*2)
            ).squeeze(1)             # (B, H*2)

            ctx = self.dropout(ctx)
            logits_list.append(self.aspect_heads[i](ctx))  # (B, 4)

        return logits_list   # list of 14 × (B, 4)

# ─────────────────────────────────────────────────────────────────────────────
# 8. TÍNH CLASS WEIGHTS (xử lý imbalance)
# ─────────────────────────────────────────────────────────────────────────────
def build_aspect_loss_fns(samples, none_weight, device):
    """
    Mỗi aspect có 1 CrossEntropyLoss với class weights riêng.
    Class 0 (none) chiếm đa số → giảm weight xuống.
    """
    loss_fns = []
    for i in range(NUM_ASPECTS):
        labels = [s["labels"][i] for s in samples]
        counts = Counter(labels)
        total  = len(labels)
        weights = []
        for c in range(NUM_SENT_CLASSES):
            if c == 0:  # none
                w = none_weight
            else:
                freq = counts.get(c, 1)
                w    = total / (NUM_SENT_CLASSES * max(freq, 1))
            weights.append(w)
        wt = torch.tensor(weights, dtype=torch.float32).to(device)
        loss_fns.append(nn.CrossEntropyLoss(weight=wt))
    return loss_fns

# ─────────────────────────────────────────────────────────────────────────────
# 9. TRAIN 1 EPOCH
# ─────────────────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, loss_fns, device):
    model.train()
    total_loss = 0.0
    total      = 0

    for input_ids, labels in loader:
        input_ids = input_ids.to(device)   # (B, T)
        labels    = labels.to(device)       # (B, 14)

        optimizer.zero_grad()
        logits_list = model(input_ids)      # 14 × (B, 4)

        # Tổng loss = trung bình loss của 14 aspect heads
        loss = sum(
            loss_fns[i](logits_list[i], labels[:, i])
            for i in range(NUM_ASPECTS)
        ) / NUM_ASPECTS

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)
        total      += input_ids.size(0)

    return total_loss / total

# ─────────────────────────────────────────────────────────────────────────────
# 10. ĐÁNH GIÁ
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(model, loader, loss_fns, device):
    """
    Trả về:
      val_loss  : float
      all_preds : np.array (N, 14)
      all_labels: np.array (N, 14)
    """
    model.eval()
    total_loss = 0.0
    total      = 0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for input_ids, labels in loader:
            input_ids = input_ids.to(device)
            labels    = labels.to(device)

            logits_list = model(input_ids)

            loss = sum(
                loss_fns[i](logits_list[i], labels[:, i])
                for i in range(NUM_ASPECTS)
            ) / NUM_ASPECTS

            total_loss += loss.item() * input_ids.size(0)
            total      += input_ids.size(0)

            # Predicted class cho mỗi aspect
            preds = torch.stack(
                [l.argmax(dim=-1) for l in logits_list], dim=1
            )  # (B, 14)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds  = torch.cat(all_preds,  dim=0).numpy()  # (N, 14)
    all_labels = torch.cat(all_labels, dim=0).numpy()  # (N, 14)

    return total_loss / total, all_preds, all_labels

# ─────────────────────────────────────────────────────────────────────────────
# 11. METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(preds, labels):
    """
    Tính các metric chuẩn cho ABSA:
    1. Aspect Detection F1 (binary: none vs có aspect)
    2. Sentiment Macro-F1  (chỉ tính slot có aspect thực sự)
    3. Exact Match         (tất cả 14 slot đúng)
    """
    results = {}

    # 1. Aspect Detection: predicted != 0 vs true != 0
    asp_pred_bin  = (preds  != 0).astype(int).flatten()
    asp_label_bin = (labels != 0).astype(int).flatten()
    results["asp_detect_f1"] = f1_score(
        asp_label_bin, asp_pred_bin, average="binary", zero_division=0
    )

    # 2. Sentiment F1: chỉ trên slot mà true label != none
    mask = (labels != 0).flatten()
    if mask.sum() > 0:
        sent_pred  = preds.flatten()[mask]
        sent_label = labels.flatten()[mask]
        results["sent_f1_macro"] = f1_score(
            sent_label, sent_pred, average="macro", zero_division=0
        )
        results["sent_acc"] = (sent_pred == sent_label).mean()
    else:
        results["sent_f1_macro"] = 0.0
        results["sent_acc"]      = 0.0

    # 3. Exact Match per review
    results["exact_match"] = np.all(preds == labels, axis=1).mean()

    return results

# ─────────────────────────────────────────────────────────────────────────────
# 12. VÒNG HUẤN LUYỆN CHÍNH
# ─────────────────────────────────────────────────────────────────────────────
def train(model, train_loader, val_loader, loss_fns, optimizer, scheduler,
          cfg, device):

    best_val_loss = float("inf")
    patience_cnt  = 0

    print(f"\n{'='*80}")
    print(f"🚀 HUẤN LUYỆN  ({cfg['epochs']} epochs max, patience={cfg['patience']})")
    print(f"{'='*80}")
    print(f"{'Ep':>4}  {'TrLoss':>8}  {'VaLoss':>8}  "
          f"{'Asp-F1':>8}  {'Sent-F1':>8}  {'ExMatch':>8}  Note")

    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, loss_fns, device)
        va_loss, va_preds, va_labels = evaluate(
            model, val_loader, loss_fns, device
        )
        metrics = compute_metrics(va_preds, va_labels)

        scheduler.step(va_loss)

        note = ""
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            patience_cnt  = 0
            torch.save(model.state_dict(), "best_absa_model.pt")
            note = "✅ best"
        else:
            patience_cnt += 1
            if patience_cnt >= cfg["patience"]:
                print(f"  ⏹️  Early stopping tại epoch {epoch}")
                break

        print(f"{epoch:>4}  {tr_loss:>8.4f}  {va_loss:>8.4f}  "
              f"{metrics['asp_detect_f1']:>8.3f}  "
              f"{metrics['sent_f1_macro']:>8.3f}  "
              f"{metrics['exact_match']:>8.3f}  {note}")

# ─────────────────────────────────────────────────────────────────────────────
# 13. BÁO CÁO KẾT QUẢ CHI TIẾT TRÊN TEST SET
# ─────────────────────────────────────────────────────────────────────────────
def print_report(model, test_loader, loss_fns, device):
    print(f"\n{'='*65}")
    print(f"📈 KẾT QUẢ TRÊN TẬP TEST")
    print(f"{'='*65}")

    _, preds, labels = evaluate(model, test_loader, loss_fns, device)
    metrics = compute_metrics(preds, labels)

    print(f"\n  🎯 Aspect Detection F1  : {metrics['asp_detect_f1']:.4f}")
    print(f"  💬 Sentiment Macro-F1   : {metrics['sent_f1_macro']:.4f}")
    print(f"  ✅ Sentiment Accuracy   : {metrics['sent_acc']:.4f}")
    print(f"  🏆 Exact Match          : {metrics['exact_match']:.4f}")

    # Báo cáo per-aspect
    print(f"\n  {'Aspect':<15}  {'#True':>6}  {'#Pred':>6}  "
          f"{'SentAcc':>8}  {'SentF1':>8}")
    print(f"  {'-'*55}")
    for i, asp in enumerate(ALL_ASPECTS):
        true_asp = (labels[:, i] != 0)
        pred_asp = (preds[:, i]  != 0)

        if true_asp.sum() > 0:
            s_acc = (preds[true_asp, i] == labels[true_asp, i]).mean()
            s_f1  = f1_score(
                labels[true_asp, i], preds[true_asp, i],
                average="macro", zero_division=0
            )
        else:
            s_acc = s_f1 = float("nan")

        print(f"  {asp:<15}  {int(true_asp.sum()):>6}  {int(pred_asp.sum()):>6}  "
              f"{s_acc:>8.3f}  {s_f1:>8.3f}")

    # Sentiment classification report (chỉ slot có aspect)
    mask = (labels != 0).flatten()
    if mask.sum() > 0:
        print(f"\n  Sentiment Report (chỉ slot có aspect):")
        print(classification_report(
            labels.flatten()[mask],
            preds.flatten()[mask],
            labels=[1, 2, 3],
            target_names=["negative", "neutral", "positive"],
            zero_division=0
        ))

# ─────────────────────────────────────────────────────────────────────────────
# 14. INFERENCE — PREDICT ĐÚNG BẢN CHẤT MULTI-ASPECT
# ─────────────────────────────────────────────────────────────────────────────
def predict(texts, model, vocab, cfg, device):
    """
    Dự đoán TẤT CẢ aspects và sentiments cho danh sách câu.

    Với mỗi câu:
      - Chạy qua 14 heads
      - Chỉ báo cáo aspect có predicted class != none (0)
      - Nếu tất cả = none → câu không rõ aspect

    Returns:
      list of dict: {"text", "detected": [{"aspect", "sentiment", "conf"}]}
    """
    model.eval()
    results = []

    with torch.no_grad():
        for text in texts:
            ids = vocab.encode(text, cfg["max_len"])
            t   = torch.tensor([ids], dtype=torch.long).to(device)  # (1, T)

            logits_list = model(t)   # 14 × (1, 4)

            detected = []
            for i, asp in enumerate(ALL_ASPECTS):
                probs   = torch.softmax(logits_list[i], dim=-1).squeeze()  # (4,)
                pred_id = probs.argmax().item()
                conf    = probs[pred_id].item()

                if pred_id != 0:   # 0 = none → bỏ qua
                    detected.append({
                        "aspect"    : asp,
                        "sentiment" : IDX2SENT[pred_id],
                        "conf"      : conf,
                    })

            results.append({"text": text, "detected": detected})

    return results

def print_predictions(predictions):
    icon_map = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}
    print(f"\n{'='*65}")
    print(f"🔮 DEMO INFERENCE — Multi-Aspect Output (đúng bản chất ABSA)")
    print(f"{'='*65}")
    for pred in predictions:
        print(f"\n📝 {pred['text']}")
        if pred["detected"]:
            for d in pred["detected"]:
                icon = icon_map.get(d["sentiment"], "⚪")
                print(f"   {icon} {d['aspect']:<15} → {d['sentiment']:<10} "
                      f"(conf: {d['conf']:.3f})")
        else:
            print(f"   ⚠️  Không phát hiện aspect rõ ràng")

# ─────────────────────────────────────────────────────────────────────────────
# 15. MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # 15.1 Load & phân tích ────────────────────────────────────────
    samples = load_data(CONFIG["data_path"])
    analyze_data(samples)

    # 15.2 Build vocab ─────────────────────────────────────────────
    vocab = Vocabulary(min_freq=CONFIG["min_freq"])
    vocab.build([s["text"] for s in samples])

    # 15.3 Train / Val / Test split (70 / 15 / 15) ─────────────────
    # 92 reviews → train≈64, val≈14, test≈14
    tr_idx, tmp_idx = train_test_split(
        range(len(samples)), test_size=0.20, random_state=SEED
    )
    va_idx, te_idx  = train_test_split(
        tmp_idx, test_size=0.50, random_state=SEED
    )

    tr_samples = [samples[i] for i in tr_idx]
    va_samples = [samples[i] for i in va_idx]
    te_samples = [samples[i] for i in te_idx]

    print(f"\n  📦 Train: {len(tr_samples)} | Val: {len(va_samples)} | Test: {len(te_samples)}")

    def make_loader(smpls, shuffle=False):
        ds = ABSADataset(smpls, vocab, CONFIG["max_len"])
        return DataLoader(ds, batch_size=CONFIG["batch_size"], shuffle=shuffle)

    train_loader = make_loader(tr_samples, shuffle=True)
    val_loader   = make_loader(va_samples)
    test_loader  = make_loader(te_samples)

    # 15.4 Loss functions (per-aspect weighted CE) ─────────────────
    loss_fns = build_aspect_loss_fns(
        tr_samples, CONFIG["none_weight"], DEVICE
    )

    # 15.5 Khởi tạo model ──────────────────────────────────────────
    model = ABSABiLSTM(
        vocab_size       = len(vocab),
        embed_dim        = CONFIG["embed_dim"],
        hidden_dim       = CONFIG["hidden_dim"],
        num_layers       = CONFIG["num_layers"],
        num_aspects      = NUM_ASPECTS,
        num_sent_classes = NUM_SENT_CLASSES,
        dropout          = CONFIG["dropout"],
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  🧠 Tổng tham số: {total_params:,}")
    print(model)

    # 15.6 Optimizer & Scheduler ───────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"]
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    # 15.7 Train ───────────────────────────────────────────────────
    train(
        model, train_loader, val_loader,
        loss_fns, optimizer, scheduler,
        CONFIG, DEVICE
    )

    # 15.8 Evaluate với best model ─────────────────────────────────
    model.load_state_dict(torch.load("best_absa_model.pt", map_location=DEVICE))
    print_report(model, test_loader, loss_fns, DEVICE)

    # 15.9 Demo Inference ──────────────────────────────────────────
    demo_texts = [
        "Pin thì rất bền nhưng camera chụp đêm lại quá tệ.",

        "Màn hình hiển thị sắc nét, loa ngoài to rõ nhưng thiết kế hơi cồng kềnh.",

        "Hiệu năng chơi game mượt mà, tốc độ sạc nhanh nhưng máy lại khá nóng.",

        "Giao hàng chậm quá sản phẩm thì quá ok",

        "Thiết kế sang trọng, trọng lượng nhẹ nhưng độ bền vỏ ngoài không cao.",

        "Camera chính chụp đẹp, kết nối 5G ổn định nhưng pin tụt nhanh khi dùng lâu.",

        "Âm thanh nghe nhạc rất hay, màn hình OLED rực rỡ nhưng độ sáng ngoài trời chưa đủ.",

        "Tốc độ xử lý ứng dụng nhanh, giao diện dễ dùng nhưng loa thoại hơi nhỏ.",

        "Pin sạc đầy nhanh, bộ nhớ trong rộng rãi nhưng giá lại quá cao.",

        "Thiết kế đẹp mắt, camera góc rộng tiện lợi nhưng máy dễ bị trầy xước."
    ]

    preds = predict(demo_texts, model, vocab, CONFIG, DEVICE)
    print_predictions(preds)

    # 15.10 Save artifacts ─────────────────────────────────────────
    import pickle
    with open("absa_vocab.pkl",  "wb") as f: pickle.dump(vocab, f)
    with open("absa_config.pkl", "wb") as f: pickle.dump(CONFIG, f)
    print(f"\n  💾 Saved: best_absa_model.pt | absa_vocab.pkl | absa_config.pkl")

    # 15.11 Đề xuất cải tiến ───────────────────────────────────────
    # print_improvements()


# ─────────────────────────────────────────────────────────────────────────────
# 16. ĐỀ XUẤT CẢI TIẾN
# ─────────────────────────────────────────────────────────────────────────────
# def print_improvements():
#     print(f"""
# {'='*70}
# 💡 ĐỀ XUẤT CẢI TIẾN (theo thứ tự ưu tiên)
# {'='*70}

# ⚠️  VẤN ĐỀ CỦA BASELINE NÀY:
#    • 92 reviews → cực kỳ ít cho 14 aspects × 4 classes = 56 outputs
#    • Các aspect hiếm (am_thanh: 1, pin_sac: 1) không học được
#    • BiLSTM với random embedding chưa hiểu ngữ nghĩa tiếng Việt

# ─────────────────────────────────────────────────────────────────────────
# 1. 📚 TĂNG DỮ LIỆU (ưu tiên #1)
# ─────────────────────────────────────────────────────────────────────────
#    • Back-translation VI → EN → VI (Google/DeepL API)         [x2 data]
#    • EDA: random swap, random delete, synonym replacement       [x1.5 data]
#    • GPT-4 sinh thêm review theo template từng aspect+sentiment [x3 data]
#    • Thu thập Shopee API / web scraping Tiki, Lazada

# ─────────────────────────────────────────────────────────────────────────
# 2. 🔤 PRETRAINED EMBEDDINGS TIẾNG VIỆT (ưu tiên #2)
# ─────────────────────────────────────────────────────────────────────────
#    Option A — Nhẹ, dễ tích hợp:
#      FastText cc.vi.300.bin → load pretrained, thay random init
#      model.embedding.weight.data.copy_(pretrained_vectors)

#    Option B — Mạnh nhất:
#      PhoBERT (vinai/phobert-base) fine-tuning toàn bộ
#      from transformers import AutoModel
#      bert = AutoModel.from_pretrained("vinai/phobert-base")
#      → Kỳ vọng tăng +15-25% F1

# ─────────────────────────────────────────────────────────────────────────
# 3. 🎯 CẢI TIẾN LOSS (ưu tiên #3)
# ─────────────────────────────────────────────────────────────────────────
#    • Focal Loss — tốt hơn với "none" chiếm 80% slot:
#      loss = (1 - p_t)^gamma * CE_loss   (gamma=2 thường dùng)

#    • Label Smoothing giảm overfit:
#      nn.CrossEntropyLoss(label_smoothing=0.1)

# ─────────────────────────────────────────────────────────────────────────
# 4. ⚙️ TRAINING TRICKS
# ─────────────────────────────────────────────────────────────────────────
#    • 5-Fold Cross Validation thay split cố định
#      (92 mẫu quá ít → kết quả test không ổn định)

#    • Threshold tuning per-aspect:
#      Với mỗi aspect, tìm threshold tối ưu để quyết định "none vs có"
#      thay vì dùng argmax cứng

#    • Ensemble nhiều seed → vote majority

# ─────────────────────────────────────────────────────────────────────────
# 5. 🏗️ KIẾN TRÚC NÂNG CAO
# ─────────────────────────────────────────────────────────────────────────
#    • Aspect Interaction Layer:
#      Các aspect không độc lập → "giao_hang tốt" thường đi với "dich_vu tốt"
#      Dùng cross-attention giữa 14 aspect context vectors

#    • PhoBERT + Linear Heads (state-of-the-art):
#      class ABSA_PhoBERT(nn.Module):
#          self.bert  = AutoModel.from_pretrained("vinai/phobert-base")
#          self.heads = nn.ModuleList([nn.Linear(768, 4) for _ in range(14)])
#          # Chỉ fine-tune heads + last 2 BERT layers để tiết kiệm RAM

# ─────────────────────────────────────────────────────────────────────────
# 6. 📏 METRIC CHUẨN CHO ABSA
# ─────────────────────────────────────────────────────────────────────────
#    Theo chuẩn SemEval ABSA:
#    • Aspect Detection F1       (đang dùng ✓)
#    • Sentiment Macro-F1        (đang dùng ✓)
#    • Joint F1: aspect VÀ sentiment đúng  ← strictest, nên thêm

# 🎖️  LỘ TRÌNH ĐỀ XUẤT:
#    Bước 1  Thu thêm data (≥200 reviews/aspect)
#    Bước 2  FastText embedding                    → +5-10% Asp-F1
#    Bước 3  Focal Loss + Label Smoothing          → +3-8%  Asp-F1
#    Bước 4  PhoBERT fine-tuning                   → +15-25% toàn bộ
#    Bước 5  Aspect Interaction + Threshold tuning → +3-5%  Asp-F1
# {'='*70}
# """)


if __name__ == "__main__":
    main()