"""
ABSA Processor cho dữ liệu review điện thoại tiếng Việt
Pipeline: Raw JSONL → Clean → LLM Extraction → ABSA Dataset
(Đã cập nhật SDK google-genai mới và xử lý Rate Limit)
"""

import json
import re
import time
from pathlib import Path
from tqdm import tqdm

# CÀI ĐẶT THÊM THƯ VIỆN NÀY: pip install google-genai
# (Hãy nhớ chạy: pip uninstall google-generativeai -y trước)
from google import genai

# ==========================================
# CẤU HÌNH API GOOGLE GEMINI (MIỄN PHÍ)
# ==========================================
GENAI_API_KEY = "" 

# Khởi tạo client theo chuẩn SDK mới
client = genai.Client(api_key=GENAI_API_KEY)

# ──────────────────────────────────────────────
# BƯỚC 1: ĐỌC & LÀM SẠCH DỮ LIỆU
# ──────────────────────────────────────────────

def load_reviews(path: str) -> list[dict]:
    reviews = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                reviews.append(json.loads(line))
    return reviews


def clean_text(text: str) -> str:
    """
    Làm sạch cơ bản cho review tiếng Việt:
    - Chuẩn hoá khoảng trắng
    - Xoá emoji/ký tự đặc biệt không cần thiết (giữ dấu tiếng Việt)
    - Chuẩn hoá dấu câu lặp
    """
    # Xoá emoji
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(" ", text)

    # Chuẩn hoá dấu lặp: "ok ok ok" → "ok ok ok" (giữ nguyên nhưng xoá dấu lặp)
    text = re.sub(r"([!?.]{2,})", lambda m: m.group(0)[0], text)

    # Chuẩn hoá khoảng trắng
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ──────────────────────────────────────────────
# BƯỚC 2: EXTRACT ASPECT + SENTIMENT BẰNG LLM (GEMINI)
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """Bạn là chuyên gia phân tích cảm xúc (ABSA) cho reviews điện thoại tiếng Việt.

Với mỗi review được cung cấp, hãy trích xuất tất cả các cặp (aspect, sentiment, opinion_phrase).

Các aspect được phép (dùng đúng nhãn này):
- camera
- pin
- hieu_nang (hiệu năng/tốc độ xử lý)
- man_hinh (màn hình)
- thiet_ke (thiết kế/ngoại hình/màu sắc/trọng lượng)
- gia_tri (giá cả/khuyến mãi/sale/giá trị tiền)
- giao_hang (giao hàng/vận chuyển/đóng gói/tốc độ giao)
- dich_vu (dịch vụ shop/tư vấn/hỗ trợ/thái độ nhân viên)
- bao_hanh (bảo hành/seal/chính hãng)
- phu_kien (phụ kiện đi kèm: tai nghe, củ sạc, ốp lưng...)
- am_thanh (loa/âm thanh)
- ket_noi (5G/wifi/bluetooth/SIM)
- pin_sac (sạc nhanh/sạc không dây)
- tong_the (nhận xét tổng thể không rõ aspect cụ thể)

Sentiment:
- positive
- negative
- neutral

Trả về JSON duy nhất (KHÔNG có markdown, KHÔNG có giải thích), định dạng:
{
  "aspects": [
    {
      "aspect": "<nhãn aspect>",
      "sentiment": "<positive|negative|neutral>",
      "opinion_phrase": "<cụm từ gốc trong review thể hiện ý kiến>"
    }
  ]
}

Nếu review quá ngắn/không rõ ý kiến cụ thể, trả về {"aspects": []}.
"""


def call_gemini_api(comment: str, max_retries: int = 5) -> dict | None:
    """Gọi Google Gemini API (Miễn phí) để extract aspect-sentiment."""
    
    # Ghép System Prompt và Comment cần xử lý lại với nhau
    full_prompt = f"{SYSTEM_PROMPT}\n\nReview cần phân tích:\n{comment}"

    for attempt in range(max_retries):
        try:
            # Gọi API theo chuẩn SDK mới
            response = client.models.generate_content(
                model='gemini-3.1-flash-lite-preview',
                contents=full_prompt,
            )
            raw_text = response.text.strip()
            
            # Xoá markdown (```json ... ```) nếu model có trả về
            raw_text = re.sub(r"^```json|^```|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()
            
            return json.loads(raw_text)
        except Exception as e:
            # Nếu gặp lỗi Rate Limit, API thường yêu cầu đợi 14-15s
            wait = 15 + (attempt * 5)
            print(f"  [attempt {attempt+1}] Error: {e}. Đang làm mát API trong {wait}s...")
            time.sleep(wait)

    return None


# ──────────────────────────────────────────────
# BƯỚC 3: PIPELINE CHÍNH
# ──────────────────────────────────────────────

def process_reviews(
    input_path: str,
    output_path: str,
    batch_size: int = 5,
    max_reviews: int | None = None,
    resume: bool = True,
):
    """
    Xử lý toàn bộ dataset:
    - Load + clean reviews
    - Gọi LLM extract aspect-sentiment từng review
    - Lưu kết quả dạng JSONL (hỗ trợ resume nếu bị gián đoạn)
    """
    reviews = load_reviews(input_path)
    if max_reviews:
        reviews = reviews[:max_reviews]

    print(f"📦 Tổng số review: {len(reviews)}")

    # Tạo thư mục output nếu chưa có
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Resume: đọc các review đã xử lý
    done_ids = set()
    if resume and Path(output_path).exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    done_ids.add(obj["review_id"])
                except:
                    pass
        print(f"✅ Đã xử lý trước đó: {len(done_ids)} reviews. Tiếp tục từ đó...")

    remaining = [r for r in reviews if r["review_id"] not in done_ids]
    print(f"🔄 Cần xử lý thêm: {len(remaining)} reviews\n")

    out_file = open(output_path, "a", encoding="utf-8")
    stats = {"success": 0, "empty": 0, "error": 0}

    try:
        for review in tqdm(remaining, desc="Extracting ABSA"):
            review_id = review["review_id"]
            comment_clean = clean_text(review["comment"])

            result = call_gemini_api(comment_clean)

            if result is None:
                stats["error"] += 1
                aspects = []
            elif not result.get("aspects"):
                stats["empty"] += 1
                aspects = []
            else:
                stats["success"] += 1
                aspects = result["aspects"]

            output_record = {
                "review_id": review_id,
                "product_name": review["product_name"],
                "rating_star": review["rating_star"],
                "comment_original": review["comment"],
                "comment_clean": comment_clean,
                "absa": aspects,
            }

            out_file.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            out_file.flush()

            # Nghỉ 4.5 giây để khớp hoàn toàn với giới hạn 15 RPM của Google
            time.sleep(4.5)

    finally:
        out_file.close()

    print(f"\n📊 Kết quả:")
    print(f"  ✅ Có aspect: {stats['success']}")
    print(f"  ⚪ Không có aspect: {stats['empty']}")
    print(f"  ❌ Lỗi API:   {stats['error']}")
    print(f"  💾 Output lưu tại: {output_path}")


# ──────────────────────────────────────────────
# BƯỚC 4: THỐNG KÊ & EXPORT
# ──────────────────────────────────────────────

def analyze_results(output_path: str):
    """Thống kê phân phối aspect & sentiment từ kết quả ABSA."""
    from collections import Counter

    records = []
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
            all_aspects.append(a["aspect"])
            aspect_sentiment.append((a["aspect"], a["sentiment"]))

    print(f"\n🔢 Tổng số aspect mentions: {len(all_aspects)}")
    print(f"\n📌 Top aspects:")
    for asp, count in Counter(all_aspects).most_common():
        print(f"  {asp:<20} {count:>4}")

    print(f"\n😊 Sentiment distribution:")
    sentiment_counts = Counter(s for _, s in aspect_sentiment)
    for s, c in sentiment_counts.most_common():
        pct = c / len(aspect_sentiment) * 100
        print(f"  {s:<12} {c:>4} ({pct:.1f}%)")

    # Pivot: aspect × sentiment
    print(f"\n📋 Aspect × Sentiment:")
    aspects_unique = sorted(set(a for a, _ in aspect_sentiment))
    sentiments = ["positive", "neutral", "negative"]
    header = f"{'Aspect':<22}" + "".join(f"{s:>12}" for s in sentiments)
    print(header)
    print("-" * (22 + 12 * len(sentiments)))
    for asp in aspects_unique:
        row = f"{asp:<22}"
        for sent in sentiments:
            count = sum(1 for a, s in aspect_sentiment if a == asp and s == sent)
            row += f"{count:>12}"
        print(row)

    return records


def export_for_training(output_path: str, train_path: str):
    """
    Export sang định dạng phẳng để train model ABSA:
    mỗi dòng = 1 (review, aspect, sentiment) triplet.
    """
    records = []
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except:
                pass

    with open(train_path, "w", encoding="utf-8") as out:
        for r in records:
            for a in r.get("absa", []):
                flat = {
                    "review_id": r["review_id"],
                    "product_name": r["product_name"],
                    "rating_star": r["rating_star"],
                    "text": r["comment_clean"],
                    "aspect": a["aspect"],
                    "sentiment": a["sentiment"],
                    "opinion_phrase": a.get("opinion_phrase", ""),
                }
                out.write(json.dumps(flat, ensure_ascii=False) + "\n")

    print(f"✅ Export training data → {train_path}")


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    INPUT  = "ABSA/data/parsed_reviews_V2.jsonl"
    OUTPUT = "ABSA/data/outputs/absa_results.jsonl"
    TRAIN  = "ABSA/data/outputs/absa_training_flat.jsonl"

    # Chạy full pipeline
    # Đặt max_reviews=20 để test thử, bỏ đi để chạy toàn bộ
    process_reviews(INPUT, OUTPUT, max_reviews=None, resume=True)

    # Phân tích kết quả
    analyze_results(OUTPUT)

    # Export để train
    export_for_training(OUTPUT, TRAIN)