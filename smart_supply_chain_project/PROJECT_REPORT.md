# Project Report – Smart Supply Chain Risk Prediction and Supplier Recommendation

## Abstract
This project develops a Python-based supply chain analytics application. It uses purchase-order records, supplier details, delivery information, product categories, costs, and supplier reviews to predict delivery delays, discover recurring risk combinations, analyse review sentiment, and recommend alternative suppliers.

## Dataset summary
- Number of purchase orders: 1200
- Number of suppliers: 8
- Product categories: 5
- Overall delayed-order rate in the synthetic dataset: 15.2%
- Highest on-time supplier in this generated sample: GreenLine Supply

## Methodology
1. Load and inspect the purchase-order dataset.
2. Convert date fields and create order-month and order-weekday features.
3. Encode categorical features and impute missing values.
4. Train a Random Forest classification model with a stratified train-test split.
5. Compare the model with a majority-class baseline using accuracy, precision, recall, and F1-score.
6. Mine recurring supplier, product-category, and shipping-mode combinations by their delay rates.
7. Classify supplier reviews using a keyword-based sentiment baseline.
8. Rank suppliers using delivery reliability, cost, and positive feedback.

## Modules
- Project overview dashboard
- Delivery status prediction
- Supply-chain risk pattern analysis
- Supplier review sentiment analysis
- Supplier recommendation and ranking
- Model evaluation

## Conclusion
The application demonstrates how predictive modelling and analytics can support supply-chain decisions. It identifies orders at risk of delay and provides a ranked supplier shortlist. The synthetic dataset is suitable for demonstrating the complete workflow; real operational decisions require validation with genuine historical data.
