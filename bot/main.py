from config import Config
from crawler import CrawlData
from parser import ShopeeReviewParser

res = CrawlData(Config()).save_response_to_file("ABSA/data/output.jsonl")

parser = ShopeeReviewParser(
    input_file="ABSA/data/output.jsonl", 
    output_jsonl="ABSA/data/parsed_reviews.jsonl", 
    output_csv="ABSA/data/parsed_reviews.csv")
parser.run()