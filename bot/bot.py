import json
import time
import undetected_chromedriver as uc

import json
import time
from selenium import webdriver # Dùng selenium chuẩn thay vì uc

class CrawlData:
    def __init__(self, shop_id: str, item_id: str):
        self.shop_id = shop_id
        self.item_id = item_id
        self.response = None
        
        print("Đang kết nối với Chrome thật...")
        options = webdriver.ChromeOptions()
        
        # CHÌA KHÓA Ở ĐÂY: Móc code vào cái Chrome bạn vừa mở ở cổng 9222
        options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
        
        # Khởi tạo webdriver nối thẳng vào Chrome thật
        self.driver = webdriver.Chrome(options=options)

    # Toàn bộ phần def save_response_to_file(...) và test() GIỮ NGUYÊN KHÔNG ĐỔI

    def save_response_to_file(self, out_file="output.jsonl"):
        url_product = f"https://shopee.vn/product/{self.shop_id}/{self.item_id}"
        print(f"Đang vào trang sản phẩm: {url_product}")
        self.driver.get(url_product)
        
        # Chờ 3 giây cho trang tải giao diện
        time.sleep(3)
        
        # Tự động cuộn chuột xuống giữa trang để lừa Shopee kích hoạt mã chống bot
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight / 2);")
        print("Đã tự động cuộn trang để kích hoạt Token Shopee...")
        
        input("LƯU Ý QUAN TRỌNG: Hãy qua trình duyệt, lăn chuột xuống tới phần BÌNH LUẬN để Shopee xác nhận là người thật. Xong thì nhấn ENTER ở đây...")

        offset = 0
        limit = 6
        # ... (phần code while True bên dưới giữ nguyên, chỉ thay đoạn js_fetch như Bước 1)

        # Mở file chuẩn bị ghi data (y hệt code cũ của bạn)
        with open(out_file, "a", encoding="utf-8") as f:
            while True:
                print(f"Đang cào dữ liệu ở offset: {offset}...")
                
                # Link API chuẩn
                api_url = f"https://shopee.vn/api/v2/item/get_ratings?filter=0&flag=1&itemid={self.item_id}&limit={limit}&offset={offset}&shopid={self.shop_id}&type=0"
                
                # 2. Tiêm JavaScript vào trình duyệt để gọi API
                # Việc này đảm bảo lấy được dữ liệu JSON thô (raw JSON) thay vì phải đọc HTML
                # 2. Tiêm JavaScript với Headers ngụy trang y hệt luồng thực tế của Shopee
                js_fetch = """
                var callback = arguments[arguments.length - 1];
                fetch(arguments[0], {
                    method: 'GET',
                    credentials: 'same-origin',
                    headers: {
                        'accept': 'application/json',
                        'x-api-source': 'pc',
                        'x-shopee-language': 'vi'
                    }
                })
                    .then(response => response.json())
                    .then(data => callback(data))
                    .catch(error => callback({"error": error.toString()}));
                """
                
                # Chạy script bất đồng bộ và trả cục JSON từ trình duyệt về lại Python
                current_data = self.driver.execute_async_script(js_fetch, api_url)

                # Kiểm tra lỗi chặn bot của Shopee (nếu có)
                if current_data.get('error') == 90309999:
                    print("Lỗi 90309999: Trình duyệt bị bắt làm Captcha hoặc văng Cookie. Hãy thử chạy lại.")
                    break

                # Lấy danh sách bình luận
                ratings = current_data.get('data', {}).get('ratings')

                if not ratings:
                    print("Đã cào xong toàn bộ bình luận!")
                    break
                
                # Lưu response để hàm test() gọi ra y như cũ
                self.response = current_data

                # 3. Ghi dữ liệu JSON ra file y chang cấu trúc code trước đó của bạn
                json.dump(current_data, f, indent=4, ensure_ascii=False)
                f.write("\n")

                # Tăng trang và nghỉ ngơi
                offset += limit
                time.sleep(3) # Nghỉ 3 giây để tránh bị Shopee Rate Limit (giới hạn tần suất)
                
        # Đóng trình duyệt sau khi hoàn thành nhiệm vụ
        self.driver.quit()
        print("Đã đóng trình duyệt ảo.")

    def test(self):
        # Vì ta lấy raw JSON thẳng từ JS, nên response giờ là một Dict (từ điển) thay vì object Requests
        # Bạn có thể in ra bình thường
        print(self.response)

# --- CÁCH SỬ DỤNG CHÍNH THỨC ---
if __name__ == "__main__":
    # Điền ID thực tế của bạn vào đây
    SHOP_ID = "301723517" 
    ITEM_ID = "28930021325"
    
    crawler = CrawlData(shop_id=SHOP_ID, item_id=ITEM_ID)
    crawler.save_response_to_file("ABSA/data/output.jsonl")
    
    print("\nTest thử dữ liệu trang cuối cùng:")
    crawler.test()