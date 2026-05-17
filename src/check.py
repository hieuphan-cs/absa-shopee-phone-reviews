import json

def print_empty_labels(output_path: str):
    empty_count = 0
    total_count = 0
    
    print(f"🔎 Đang quét kiểm tra file: {output_path}")
    print("-" * 60)
    
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                
                total_count += 1
                try:
                    data = json.loads(line)
                    # Kiểm tra nếu key 'absa' bị trống (rỗng hoặc không tồn tại)
                    if not data.get("absa"):
                        empty_count += 1
                        review_id = data.get("review_id", "Unknown")
                        # Lấy comment đã clean (nếu có), không thì lấy comment gốc
                        comment = data.get("comment_clean", data.get("comment_original", ""))
                        
                        print(f"🔸 Dòng {line_idx} | ID: {review_id}")
                        print(f"   Text: {comment}\n")
                        
                except json.JSONDecodeError:
                    print(f"❌ Lỗi format JSON ở dòng {line_idx}")
                    
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file: {output_path}. Hãy kiểm tra lại đường dẫn!")
        return

    print("-" * 60)
    print(f"📊 Thống kê nhanh:")
    print(f"   - Tổng số sample đã duyệt: {total_count}")
    print(f"   - Số sample bị rỗng nhãn (absa = []): {empty_count}")

if __name__ == "__main__":
    # Nhớ dùng r"..." (raw string) để tránh lỗi ký tự \ trong Windows path
    OUTPUT_FILE = r"ABSA\data\outputs\absa_results.jsonl"
    
    print_empty_labels(OUTPUT_FILE)