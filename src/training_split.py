"""
Script thống kê kết quả ABSA và xuất dữ liệu sang định dạng phẳng (flat) 
để huấn luyện mô hình Machine Learning/Deep Learning.
"""

import json
from collections import Counter
from pathlib import Path

# ──────────────────────────────────────────────
# 1. THỐNG KÊ KẾT QUẢ
# ──────────────────────────────────────────────
def analyze_results(output_path: str):
    """Thống kê phân phối aspect & sentiment từ kết quả ABSA."""
    records = []
    
    if not Path(output_path).exists():
        print(f"❌ Không tìm thấy file kết quả: {output_path}")
        return records

    with open(output_path, encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except:
                pass

    print(f"\n{'='*50}")
    print(f"📊 THỐNG KÊ ABSA - {len(records)} reviews")
    print(f"{'='*50}")

    all_aspects = []
    aspect_sentiment = []
    
    for r in records:
        for a in r.get("absa", []):
            # Kiểm tra an toàn: đảm bảo 'a' là dictionary và có chứa key 'aspect'
            if isinstance(a, dict):
                aspect = a.get("aspect")
                sentiment = a.get("sentiment", "unknown") # Nếu mất sentiment thì để unknown
                
                # Chỉ đưa vào thống kê nếu AI thực sự sinh ra aspect
                if aspect:
                    all_aspects.append(aspect)
                    aspect_sentiment.append((aspect, sentiment))

    print(f"\n🔢 Tổng số aspect mentions: {len(all_aspects)}")
    
    print(f"\n📌 Top aspects:")
    from collections import Counter
    for asp, count in Counter(all_aspects).most_common():
        print(f"  {asp:<20} {count:>4}")

    print(f"\n😊 Sentiment distribution:")
    sentiment_counts = Counter(s for _, s in aspect_sentiment)
    for s, c in sentiment_counts.most_common():
        if len(aspect_sentiment) > 0:
            pct = c / len(aspect_sentiment) * 100
            print(f"  {s:<12} {c:>4} ({pct:.1f}%)")

    # Pivot: aspect × sentiment
    print(f"\n📋 Aspect × Sentiment:")
    aspects_unique = sorted(set(a for a, _ in aspect_sentiment))
    sentiments = ["positive", "neutral", "negative", "unknown"]
    header = f"{'Aspect':<22}" + "".join(f"{s:>12}" for s in sentiments)
    print(header)
    print("-" * (22 + 12 * len(sentiments)))
    
    for asp in aspects_unique:
        row = f"{asp:<22}"
        for sent in sentiments:
            count = sum(1 for a, s in aspect_sentiment if a == asp and s == sent)
            row += f"{count:>12}"
        # Chỉ in những dòng có dữ liệu (lọc bỏ Unknown nếu bằng 0)
        if sum(1 for a, s in aspect_sentiment if a == asp) > 0:
            print(row)

    return records

# ──────────────────────────────────────────────
# 2. XUẤT FILE TRAINING
# ──────────────────────────────────────────────
def export_for_training(output_path: str, train_path: str):
    """
    Export sang định dạng phẳng để train model ABSA:
    mỗi dòng = 1 (review, aspect, sentiment) triplet.
    """
    if not Path(output_path).exists():
        return

    records = []
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except:
                pass

    # Tạo thư mục chứa file train nếu chưa có
    Path(train_path).parent.mkdir(parents=True, exist_ok=True)

    with open(train_path, "w", encoding="utf-8") as out:
        for r in records:
            for a in r.get("absa", []):
                flat = {
                    "review_id": r.get("review_id", ""),
                    "product_name": r.get("product_name", ""),
                    "rating_star": r.get("rating_star", ""),
                    "text": r.get("comment_clean", ""),
                    "aspect": a.get("aspect", ""),
                    "sentiment": a.get("sentiment", ""),
                    "opinion_phrase": a.get("opinion_phrase", ""),
                }
                out.write(json.dumps(flat, ensure_ascii=False) + "\n")

    print(f"\n✅ Đã xuất file training data thành công tại: {train_path}")


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # Đường dẫn file đầu vào (kết quả từ API) và đầu ra (file train)
    OUTPUT_FILE = "ABSA/data/outputs/absa_results.jsonl"
    TRAIN_FILE  = "ABSA/data/outputs/absa_training_flat.jsonl"

    # 1. Chạy thống kê xem data hiện tại phân bổ thế nào
    analyze_results(OUTPUT_FILE)

    # 2. Xuất file training phẳng
    export_for_training(OUTPUT_FILE, TRAIN_FILE)