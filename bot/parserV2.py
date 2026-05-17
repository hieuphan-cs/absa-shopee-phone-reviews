import json
import csv
import os
import re
from typing import Any, Dict, List, Optional


class ShopeeReviewParser:
    def __init__(self, input_file: str, output_csv: str = "parsed_reviews.csv", output_jsonl: str = "parsed_reviews.jsonl"):
        self.input_file = input_file
        self.output_csv = output_csv
        self.output_jsonl = output_jsonl
        self.parsed_reviews: List[Dict[str, Any]] = []
        self.seen_ids = set()

    def _safe_get(self, obj: Any, path: List[Any], default=None):
        cur = obj
        for key in path:
            try:
                if isinstance(cur, dict):
                    cur = cur.get(key, default)
                elif isinstance(cur, list) and isinstance(key, int) and 0 <= key < len(cur):
                    cur = cur[key]
                else:
                    return default
            except Exception:
                return default
        return default if cur is None else cur

    def _clean_comment(self, text: Optional[str]) -> str:
        """
        Giữ lại câu comment chính, bỏ các dòng dạng:
        Chất lượng sản phẩm: ...
        Tính năng nổi bật: ...
        """
        if not text:
            return ""

        text = str(text)

        # split theo dòng
        lines = text.split("\n")

        cleaned_lines = []

        for line in lines:
            line = line.strip()

            if not line:
                continue

            # loại bỏ pattern "something: something"
            if re.match(r"^[^:]{2,40}:\s*", line):
                continue

            cleaned_lines.append(line)

        comment = " ".join(cleaned_lines)

        # normalize whitespace
        comment = re.sub(r"\s+", " ", comment).strip()

        return comment

    def _extract_one_review(self, r: Dict[str, Any]) -> Optional[Dict[str, Any]]:

        comment_raw = r.get("comment") or r.get("comment_text") or r.get("message")
        comment = self._clean_comment(comment_raw)

        if not comment:
            return None

        review_id = r.get("cmtid") or r.get("comment_id") or r.get("rating_id") or r.get("id")

        if review_id is not None:
            review_id = str(review_id)

            if review_id in self.seen_ids:
                return None

            self.seen_ids.add(review_id)

        product_name = ""
        product_items = r.get("product_items") or []

        if isinstance(product_items, list) and product_items:
            first_item = product_items[0] if isinstance(product_items[0], dict) else {}
            product_name = first_item.get("name") or first_item.get("model_name") or ""

        parsed = {
            "review_id": review_id,
            "product_name": product_name,
            "rating_star": r.get("rating_star"),
            "comment": comment
        }

        return parsed

    def _parse_ratings_list(self, ratings: Any):

        if not isinstance(ratings, list):
            return

        for r in ratings:
            if isinstance(r, dict):

                parsed = self._extract_one_review(r)

                if parsed:
                    self.parsed_reviews.append(parsed)

    def _handle_object(self, obj: Any):

        if not isinstance(obj, dict):
            return

        ratings = self._safe_get(obj, ["data", "ratings"], default=None)

        if ratings is not None:
            self._parse_ratings_list(ratings)
            return

        if "ratings" in obj and isinstance(obj["ratings"], list):
            self._parse_ratings_list(obj["ratings"])
            return

        if "comment" in obj or "comment_text" in obj or "message" in obj:

            parsed = self._extract_one_review(obj)

            if parsed:
                self.parsed_reviews.append(parsed)

    def _load_json_objects_from_text(self, text: str):

        text = text.strip()

        if not text:
            return []

        try:
            return [json.loads(text)]
        except json.JSONDecodeError:
            pass

        objs = []

        lines = [line.strip() for line in text.splitlines() if line.strip()]

        jsonl_ok = True

        for line in lines:
            try:
                objs.append(json.loads(line))
            except json.JSONDecodeError:
                jsonl_ok = False
                break

        if jsonl_ok and objs:
            return objs

        decoder = json.JSONDecoder()

        idx = 0
        n = len(text)

        while idx < n:

            while idx < n and text[idx].isspace():
                idx += 1

            if idx >= n:
                break

            obj, end = decoder.raw_decode(text, idx)

            objs.append(obj)

            idx = end

        return objs

    def parse(self):

        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"Không tìm thấy file: {self.input_file}")

        with open(self.input_file, "r", encoding="utf-8") as f:
            text = f.read()

        objects = self._load_json_objects_from_text(text)

        for obj in objects:
            self._handle_object(obj)

        return self.parsed_reviews

    def save_csv(self):

        if not self.parsed_reviews:
            return

        fieldnames = list(self.parsed_reviews[0].keys())

        with open(self.output_csv, "w", newline="", encoding="utf-8-sig") as f:

            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()

            writer.writerows(self.parsed_reviews)

    def save_jsonl(self):

        if not self.parsed_reviews:
            return

        with open(self.output_jsonl, "w", encoding="utf-8") as f:

            for row in self.parsed_reviews:

                json.dump(row, f, ensure_ascii=False)

                f.write("\n")

    def run(self):

        self.parse()

        self.save_csv()

        self.save_jsonl()

        print(f"Parsed reviews: {len(self.parsed_reviews)}")
        print(f"Saved CSV: {self.output_csv}")
        print(f"Saved JSONL: {self.output_jsonl}")


if __name__ == "__main__":

    parser = ShopeeReviewParser(
        input_file="ABSA/data/output.jsonl",
        output_csv="ABSA/data/parsed_reviews_V2.csv",
        output_jsonl="ABSA/data/parsed_reviews_V2.jsonl"
    )

    parser.run()