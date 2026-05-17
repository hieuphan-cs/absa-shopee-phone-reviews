import json

file_path = "ABSA/data/outputs/absa_results.jsonl"  # đổi thành tên file của bạn

seen_ids = set()
duplicates = []

with open(file_path, "r", encoding="utf-8") as f:
    for line_number, line in enumerate(f, 1):
        data = json.loads(line)
        review_id = data.get("review_id")

        if review_id in seen_ids:
            print(f"Duplicate review_id found at line {line_number}:")
            print(data)
            duplicates.append(data)
        else:
            seen_ids.add(review_id)

print(f"\nTotal duplicates found: {len(duplicates)}")