# Aspect-Based Sentiment Analysis (ABSA) on Shopee Phone Reviews

## 🎯 Introduction
This project applies **Aspect-Based Sentiment Analysis (ABSA)** to smartphone reviews collected from Shopee.  
Unlike general sentiment analysis, ABSA identifies **specific aspects** (e.g., battery, camera, delivery, price) and determines the sentiment expressed toward each aspect.

Objectives:
- Extract key aspects from user reviews.
- Classify sentiment polarity (positive, negative, neutral) for each aspect.
- Provide actionable insights for buyers and sellers.

---

## 📂 Project Structure
├── data/              # Raw Shopee reviews

├── notebooks/         # Jupyter notebooks for ABSA experiments

├── src/               # Source code for preprocessing & modeling

├── results/           # Aspect-level sentiment results

└── README.md          # Project documentation


---

## Methodology
1. **Data Collection**: Web scraping Shopee smartphone reviews.
2. **Preprocessing**: Tokenization, stopword removal, normalization.
3. **Aspect Extraction**: Identify aspects using rule-based methods or deep learning (e.g., dependency parsing, transformer models).
4. **Sentiment Classification**: Train models to classify sentiment for each aspect.
5. **Evaluation**: Use metrics such as accuracy, F1-score, and aspect-level precision/recall.

Example formula for **aspect sentiment score**:



\[
\text{Sentiment}_{aspect} = \frac{\sum_{i=1}^{n} s_i}{n}
\]



Where:
- \( s_i \): sentiment polarity score of review \( i \) for a given aspect.
- \( n \): total number of reviews mentioning that aspect.

---

## Results
- **Battery**: Mostly positive (\~85% reviews highlight long-lasting battery).  
- **Camera**: Mixed sentiment (good quality but struggles in low light).  
- **Delivery**: Negative sentiment in \~30% reviews due to delays.  
- **Price**: Strongly positive sentiment, users value affordability.

---

## How to Run
1. Clone the repository:
```bash```
git clone https://github.com/hieuphan-cs/shopee-absa-phone-reviews.git
2. Install dependencies:
pip install -r requirements.txt

3. Run ABSA notebook:
jupyter notebook notebooks/absa_analysis.ipynb

## Future work
- Expand ABSA to multiple product categories beyond smartphones.

- Apply transformer-based ABSA models (e.g., BERT, RoBERTa).

- Build an interactive dashboard for aspect-level insights.

## Authors
- Team/Project Name: Shopee ABSA Analyzer

- Contact: trihieugocong85@gmail.com


This version emphasizes **ABSA methodology and results**, making it clear that your project isn’t just about overall ratings but about *fine-grained sentiment analysis*.  

Would you like me to also sketch out a **sample visualization idea** (like an aspect-sentiment bar chart in LaTeX PGFPlots) to include in your README for extra clarity?
