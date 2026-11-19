import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance

st.set_page_config(page_title="Smart Supply Chain Analytics", page_icon="📦", layout="wide")

st.title("📦 Smart Supply Chain Risk Prediction & Supplier Recommendation")
st.caption("Big Data and Analytics | Assignment 2 – Question 12")

DATA_PATH = "data/supply_chain_orders.csv"

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    for col in ["order_date", "promised_delivery_date", "actual_delivery_date"]:
        data[col] = pd.to_datetime(data[col], errors="coerce")
    return data

def sentiment_label(review):
    positive = {"fast", "excellent", "good", "reliable", "helpful", "quick", "satisfied",
                "quality", "on", "time", "acceptable", "without"}
    negative = {"late", "poor", "damaged", "slow", "unreliable", "delayed", "disappointing",
                "very", "issues"}
    words = str(review).lower().replace(".", "").replace(",", "").split()
    score = sum(w in positive for w in words) - sum(w in negative for w in words)
    if score > 0:
        return "Positive"
    if score < 0:
        return "Negative"
    return "Neutral"

def prepare_features(data):
    x = data[["supplier_id", "product_category", "region", "shipping_mode",
              "order_quantity", "unit_cost", "total_order_value",
              "promised_duration_days", "supplier_rating"]].copy()
    x["order_month"] = data["order_date"].dt.month
    x["order_weekday"] = data["order_date"].dt.dayofweek
    y = (data["delivery_status"] == "Delayed").astype(int)
    return x, y

@st.cache_resource
def train_model(data):
    x, y = prepare_features(data)
    categorical = ["supplier_id", "product_category", "region", "shipping_mode"]
    numeric = [c for c in x.columns if c not in categorical]
    preprocessor = ColumnTransformer([
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric)
    ])
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y)
    model = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=180, max_depth=10,
                                               min_samples_split=4, class_weight="balanced",
                                               random_state=42))
    ])
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(x_train, y_train)
    base_pred = baseline.predict(x_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1-score": f1_score(y_test, pred, zero_division=0),
        "Baseline accuracy": accuracy_score(y_test, base_pred),
        "Baseline F1-score": f1_score(y_test, base_pred, zero_division=0),
    }
    return model, x_test, y_test, pred, metrics

data = load_data()
data["sentiment"] = data["supplier_review"].apply(sentiment_label)
model, x_test, y_test, pred, metrics = train_model(data)

page = st.sidebar.radio("Navigate", [
    "Project Overview", "Delivery Prediction", "Risk Pattern Mining",
    "Supplier Sentiment", "Supplier Recommendation", "Model Evaluation"
])

if page == "Project Overview":
    st.subheader("Project overview")
    st.write("This application predicts delivery delays, discovers recurring supply-chain risk patterns, analyses supplier reviews, and ranks alternative suppliers.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total orders", f"{len(data):,}")
    c2.metric("Delayed orders", f"{(data.delivery_status == 'Delayed').sum():,}")
    c3.metric("Delay rate", f"{(data.delivery_status == 'Delayed').mean()*100:.1f}%")
    c4.metric("Suppliers", data.supplier_id.nunique())
    a, b = st.columns(2)
    with a:
        status = data["delivery_status"].value_counts().rename_axis("Status").reset_index(name="Orders")
        st.plotly_chart(px.pie(status, names="Status", values="Orders", title="On-time vs delayed orders"), use_container_width=True)
    with b:
        monthly = data.assign(month=data.order_date.dt.to_period("M").astype(str)).groupby("month").size().reset_index(name="Orders")
        st.plotly_chart(px.line(monthly, x="month", y="Orders", markers=True, title="Monthly order volume"), use_container_width=True)
    st.subheader("Sample purchase-order data")
    st.dataframe(data.head(12), use_container_width=True)

elif page == "Delivery Prediction":
    st.subheader("Predict whether a new order will be delayed")
    left, right = st.columns(2)
    with left:
        supplier = st.selectbox("Supplier ID", sorted(data.supplier_id.unique()))
        category = st.selectbox("Product category", sorted(data.product_category.unique()))
        region = st.selectbox("Delivery region", sorted(data.region.unique()))
        mode = st.selectbox("Shipping mode", ["Standard", "Express", "Economy"])
        qty = st.number_input("Order quantity", min_value=1, max_value=10000, value=100)
    with right:
        unit_cost = st.number_input("Unit cost", min_value=0.1, value=50.0, step=1.0)
        promised = st.number_input("Promised delivery duration (days)", min_value=1, max_value=60, value={"Express":2,"Standard":5,"Economy":8}[mode])
        rating = float(data.loc[data.supplier_id == supplier, "supplier_rating"].mean())
        st.metric("Supplier average rating", f"{rating:.1f}/10")
        st.caption("The model uses historical supplier, product, order and delivery features.")
    if st.button("Predict delivery status", type="primary"):
        sample = pd.DataFrame([{
            "supplier_id": supplier, "product_category": category, "region": region,
            "shipping_mode": mode, "order_quantity": qty, "unit_cost": unit_cost,
            "total_order_value": qty*unit_cost, "promised_duration_days": promised,
            "supplier_rating": rating, "order_month": pd.Timestamp.today().month,
            "order_weekday": pd.Timestamp.today().dayofweek
        }])
        probability = model.predict_proba(sample)[0][1]
        label = "Delayed" if probability >= 0.5 else "On-time"
        if label == "Delayed":
            st.error(f"Prediction: {label} | Delay risk: {probability*100:.1f}%")
        else:
            st.success(f"Prediction: {label} | Delay risk: {probability*100:.1f}%")
        st.progress(float(probability))

elif page == "Risk Pattern Mining":
    st.subheader("Recurring combinations associated with delivery delays")
    st.write("Pattern mining groups supplier, product category and shipping mode combinations, then calculates how often they are delayed.")
    patterns = data.groupby(["supplier_name", "product_category", "shipping_mode"]).agg(
        Orders=("order_id", "count"),
        Delayed=("delivery_status", lambda s: (s == "Delayed").sum()),
        Delay_Rate=("delivery_status", lambda s: (s == "Delayed").mean())
    ).reset_index()
    patterns = patterns[patterns.Orders >= 5].sort_values(["Delay_Rate", "Orders"], ascending=[False, False])
    st.dataframe(patterns.head(20).style.format({"Delay_Rate": "{:.1%}"}), use_container_width=True)
    if len(patterns):
        st.plotly_chart(px.bar(patterns.head(12), x="supplier_name", y="Delay_Rate",
                               color="product_category", hover_data=["shipping_mode", "Orders"],
                               title="High-risk supplier and product combinations"), use_container_width=True)

elif page == "Supplier Sentiment":
    st.subheader("Supplier review sentiment analysis")
    counts = data["sentiment"].value_counts().rename_axis("Sentiment").reset_index(name="Reviews")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.pie(counts, names="Sentiment", values="Reviews", title="Review sentiment distribution"), use_container_width=True)
    with c2:
        st.plotly_chart(px.histogram(data, x="sentiment", color="delivery_status",
                                     barmode="group", title="Sentiment vs delivery status"), use_container_width=True)
    st.dataframe(data[["supplier_name", "supplier_review", "sentiment", "delivery_status"]].head(25), use_container_width=True)
    st.caption("Sentiment is assigned using a simple, transparent keyword-based baseline suitable for this demonstration.")

elif page == "Supplier Recommendation":
    st.subheader("Rank alternative suppliers")
    selected_category = st.selectbox("Select product category", sorted(data.product_category.unique()))
    selected_region = st.selectbox("Select delivery region", sorted(data.region.unique()))
    supplier_stats = data.groupby(["supplier_id", "supplier_name"]).agg(
        Orders=("order_id", "count"),
        On_Time_Rate=("delivery_status", lambda s: (s == "On-time").mean()),
        Average_Cost=("unit_cost", "mean"),
        Average_Rating=("supplier_rating", "mean"),
        Positive_Sentiment=("sentiment", lambda s: (s == "Positive").mean())
    ).reset_index()
    regional = data[(data.product_category == selected_category) & (data.region == selected_region)]
    available = set(regional.supplier_id.unique())
    ranked = supplier_stats[supplier_stats.supplier_id.isin(available)].copy()
    if ranked.empty:
        ranked = supplier_stats.copy()
    # Higher reliability and feedback are better; lower average cost is better.
    cost_range = ranked.Average_Cost.max() - ranked.Average_Cost.min()
    ranked["Cost_Score"] = 1.0 if cost_range == 0 else 1 - (ranked.Average_Cost - ranked.Average_Cost.min()) / cost_range
    ranked["Recommendation_Score"] = (
        0.50 * ranked.On_Time_Rate +
        0.20 * ranked.Cost_Score +
        0.30 * ranked.Positive_Sentiment
    ) * 100
    ranked = ranked.sort_values("Recommendation_Score", ascending=False)
    ranked.insert(0, "Rank", range(1, len(ranked)+1))
    st.write("Score weights: delivery reliability 50%, cost 20%, positive feedback 30%.")
    st.dataframe(ranked[["Rank", "supplier_name", "Orders", "On_Time_Rate",
                         "Average_Cost", "Positive_Sentiment", "Recommendation_Score"]]
                 .style.format({"On_Time_Rate":"{:.1%}", "Average_Cost":"{:.2f}",
                                "Positive_Sentiment":"{:.1%}", "Recommendation_Score":"{:.1f}"}),
                 use_container_width=True)
    if len(ranked):
        st.success(f"Recommended supplier: {ranked.iloc[0].supplier_name}")

elif page == "Model Evaluation":
    st.subheader("Classification model performance")
    metric_df = pd.DataFrame({"Metric": list(metrics.keys()), "Score": list(metrics.values())})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
    c2.metric("Precision", f"{metrics['Precision']:.3f}")
    c3.metric("Recall", f"{metrics['Recall']:.3f}")
    c4.metric("F1-score", f"{metrics['F1-score']:.3f}")
    st.dataframe(metric_df.style.format({"Score":"{:.3f}"}), use_container_width=True)
    comparison = pd.DataFrame({
        "Model": ["Random Forest", "Majority-class baseline"],
        "Accuracy": [metrics["Accuracy"], metrics["Baseline accuracy"]],
        "F1-score": [metrics["F1-score"], metrics["Baseline F1-score"]]
    }).melt("Model", var_name="Metric", value_name="Score")
    st.plotly_chart(px.bar(comparison, x="Metric", y="Score", color="Model",
                           barmode="group", title="Model vs baseline comparison",
                           range_y=[0,1]), use_container_width=True)
    cm = confusion_matrix(y_test, pred)
    st.plotly_chart(px.imshow(cm, text_auto=True, x=["Predicted on-time","Predicted delayed"],
                              y=["Actual on-time","Actual delayed"], title="Confusion matrix",
                              labels=dict(x="Prediction", y="Actual", color="Orders")),
                    use_container_width=True)
    st.write("Classification report")
    st.code(classification_report(y_test, pred, target_names=["On-time", "Delayed"], zero_division=0))
    st.info("The dataset is synthetic for academic demonstration. Metrics demonstrate the workflow and should not be presented as real-world supplier performance.")
