import json
import time
import requests
from config import Config

class CrawlData:
    def __init__(self, config: Config):
        self.config = config

        # Giữ nguyên cấu trúc lấy dữ liệu từ config
        self.cookies = self.config.cookies
        self.headers = self.config.headers
        self.params = self.config.params
        self.response = getattr(self.config, 'response', None) # Dùng getattr để tránh lỗi nếu config không có sẵn response
        
        # Thêm URL gốc của API Shopee để tự động gọi trong vòng lặp
        self.url = 'https://shopee.vn/api/v2/item/get_ratings'

    def save_response_to_file(self, out_file="output.jsonl"):
        # Lấy offset và limit từ params ban đầu (ép kiểu int để tính toán)
        offset = int(self.params.get('offset', 0))
        limit = int(self.params.get('limit', 6))

        # Mở file một lần ở chế độ append "a" giống hệt code cũ của bạn
        with open(out_file, "a", encoding="utf-8") as f:
            
            while True:
                print(f"Đang cào dữ liệu ở offset: {offset}...")
                
                # Cập nhật tham số trang mới vào params
                self.params['offset'] = str(offset)
                self.params['limit'] = str(limit)

                # Thực hiện gọi API trực tiếp trong class
                current_response = requests.get(
                    self.url,
                    params=self.params,
                    cookies=self.cookies,
                    headers=self.headers
                )

                if current_response.status_code == 200:
                    data = current_response.json()
                    ratings = data.get('data', {}).get('ratings')

                    # Nếu trả về mảng rỗng -> hết bình luận -> Thoát vòng lặp
                    if not ratings:
                        print("Đã cào xong toàn bộ bình luận! Hoặc bị Shopee chặn ngầm.")
                        print(f"DỮ LIỆU THỰC TẾ TỪ SHOPEE: {current_response.text[:200]}") # In ra 200 ký tự đầu để xem lỗi
                        break

                    # Cập nhật self.response hiện tại để hàm test() vẫn hoạt động đúng
                    self.response = current_response

                    # Ghi kết quả vào file JSON y chang code cũ
                    json.dump(data, f, indent=4, ensure_ascii=False)
                    f.write("\n") # Thêm một dấu xuống dòng để ngăn cách các cục JSON cho dễ nhìn

                    # Cộng dồn offset để sang trang kế tiếp
                    offset += limit
                    
                    # Nghỉ 2 giây chống block IP
                    time.sleep(5)
                else:
                    print(f"Lỗi kết nối API: HTTP {current_response.status_code}")
                    break

    def test(self):
        # Vẫn in ra response cuối cùng như bình thường
        print(self.response)