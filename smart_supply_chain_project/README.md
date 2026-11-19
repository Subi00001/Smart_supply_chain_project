# Smart Supply Chain Risk Prediction and Supplier Recommendation

**Course:** Big Data and Analytics  
**Assignment:** 2 – Question 12

## 1. Problem statement
A retail company experiences delays in receiving goods from suppliers. This project analyses purchase-order and supplier data to predict delayed deliveries, discover recurring risk patterns, analyse supplier reviews, and rank alternative suppliers.

## 2. Objectives
- Predict whether a purchase order will be delivered on-time or delayed.
- Identify factors and combinations associated with delivery delays.
- Analyse supplier review sentiment as positive, negative, or neutral.
- Recommend suppliers using delivery reliability, cost, and customer feedback.
- Display findings through an interactive dashboard.

## 3. Dataset
The `data/` folder contains documented synthetic datasets:
- `supply_chain_orders.csv` – 1,200 purchase orders, delivery details, costs, supplier ratings, reviews, and delivery status.
- `supplier_details.csv` – supplier reference information.

The data is synthetic and generated for this academic demonstration. It is not a real company's operational data. The target `delivery_status` is generated from supplier reliability, shipping mode, and product category to create a reproducible demonstration dataset.

## 4. Algorithms and techniques
- **Preprocessing:** date conversion, missing-value imputation, categorical encoding, numerical scaling.
- **Feature engineering:** order month, weekday, order value, promised duration, supplier and product attributes.
- **Classification:** Random Forest classifier to predict delivery delay.
- **Baseline:** majority-class DummyClassifier for comparison.
- **Pattern mining:** recurring supplier + product category + shipping mode combinations, ranked by delay frequency/rate.
- **Sentiment analysis:** transparent keyword-based sentiment baseline for supplier reviews.
- **Supplier recommendation:** weighted ranking: delivery on-time rate (50%), cost score (20%), positive review sentiment (30%).

## 5. Installation
Install Python 3.10 or later. Open a terminal in the project folder and run:

```bash
python -m venv venv
```

Activate the environment:

**Windows**
```bash
venv\Scripts\activate
```

**macOS/Linux**
```bash
source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## 6. Run the application
```bash
streamlit run app.py
```
The dashboard opens in a browser. Use the left sidebar to navigate between:
1. Project Overview
2. Delivery Prediction
3. Risk Pattern Mining
4. Supplier Sentiment
5. Supplier Recommendation
6. Model Evaluation

## 7. Expected outputs
- On-time or delayed prediction with delay-risk probability.
- Risk patterns grouped by supplier, product category, and shipping mode.
- Positive, negative, and neutral review summaries.
- Ranked supplier recommendations.
- Accuracy, precision, recall, F1-score, baseline comparison, and confusion matrix.
- Interactive charts and analytical summaries.

## 8. Limitations
- Synthetic data is used; results are for demonstration and are not evidence of actual supplier performance.
- Keyword-based sentiment analysis is a simple baseline and may not understand sarcasm or context.
- Recommendation weights are illustrative and can be tuned with business feedback.
- The model should be retrained and validated on real, representative historical data before operational use.

## 9. GitHub submission
Upload the complete project folder to a GitHub repository. Include the source code, datasets, README, and screenshots of the running dashboard.
