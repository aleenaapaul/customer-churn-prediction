import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Telecom Churn Analytics",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .section-title {
        font-size: 24px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    .insight-box {
        padding: 18px;
        border-radius: 12px;
        background-color: #f8f9fa;
        border-left: 5px solid #4f46e5;
        margin-bottom: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.title("📊 Telecom Customer Churn Analytics")

st.markdown(
    """
    ### Predicting customer churn and identifying retention opportunities

    An interactive machine learning dashboard built using
    **Random Forest, SMOTE and customer-level churn probability scoring.**
    """
)

st.divider()


# ============================================================
# CHECK FILES
# ============================================================

DATA_FILE = "Telco-Customer-Churn.csv"
MODEL_FILE = "churn_model.pkl"


if not os.path.exists(DATA_FILE):

    st.error(
        f"Dataset not found: {DATA_FILE}"
    )

    st.stop()


if not os.path.exists(MODEL_FILE):

    st.error(
        "Model file not found. Please run train_model.py first."
    )

    st.stop()


# ============================================================
# LOAD DATA + MODEL
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        DATA_FILE
    )

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["TotalCharges"]
    ).copy()

    return df


@st.cache_resource
def load_model():

    return joblib.load(
        MODEL_FILE
    )


df = load_data()

model = load_model()


# ============================================================
# CUSTOMER PREDICTIONS
# ============================================================

customer_ids = df["customerID"].copy()

customer_features = df.drop(
    columns=[
        "customerID",
        "Churn"
    ]
)


customer_probabilities = model.predict_proba(
    customer_features
)[:, 1]


customer_predictions = (
    customer_probabilities >= 0.50
).astype(int)


customer_risk = pd.DataFrame({

    "Customer ID": customer_ids,

    "Churn Probability":
        customer_probabilities,

    "Predicted Churn":
        customer_predictions
})


customer_risk["Risk Level"] = pd.cut(
    customer_risk["Churn Probability"],

    bins=[
        -0.01,
        0.30,
        0.60,
        1.00
    ],

    labels=[
        "Low Risk",
        "Medium Risk",
        "High Risk"
    ]
)


# ============================================================
# KPI SECTION
# ============================================================

total_customers = len(df)

total_churned = (
    df["Churn"] == "Yes"
).sum()

churn_rate = (
    total_churned /
    total_customers
)

high_risk_count = (
    customer_risk["Risk Level"]
    == "High Risk"
).sum()


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Customers",
        f"{total_customers:,}"
    )


with col2:

    st.metric(
        "Churned Customers",
        f"{total_churned:,}"
    )


with col3:

    st.metric(
        "Overall Churn Rate",
        f"{churn_rate:.1%}"
    )


with col4:

    st.metric(
        "High-Risk Customers",
        f"{high_risk_count:,}"
    )


st.divider()


# ============================================================
# CHURN OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-title">📈 Churn Overview</div>',
    unsafe_allow_html=True
)


col1, col2 = st.columns(2)


with col1:

    churn_data = (
        df["Churn"]
        .value_counts()
        .reset_index()
    )

    churn_data.columns = [
        "Churn",
        "Customers"
    ]

    churn_data["Churn"] = (
        churn_data["Churn"]
        .replace({
            "No": "Stayed",
            "Yes": "Churned"
        })
    )

    fig = px.pie(
        churn_data,
        names="Churn",
        values="Customers",
        hole=0.45,
        title="Customer Churn Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


with col2:

    contract_data = (
        pd.crosstab(
            df["Contract"],
            df["Churn"],
            normalize="index"
        ) * 100
    ).reset_index()

    contract_data = contract_data.melt(
        id_vars="Contract",
        var_name="Churn",
        value_name="Percentage"
    )

    contract_data["Churn"] = (
        contract_data["Churn"]
        .replace({
            "No": "Stayed",
            "Yes": "Churned"
        })
    )

    fig = px.bar(
        contract_data,
        x="Contract",
        y="Percentage",
        color="Churn",
        barmode="group",
        title="Churn Rate by Contract Type",
        text_auto=".1f"
    )

    fig.update_layout(
        yaxis_title="Percentage (%)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# CUSTOMER BEHAVIOR
# ============================================================

st.markdown(
    '<div class="section-title">👥 Customer Behavior</div>',
    unsafe_allow_html=True
)


col1, col2 = st.columns(2)


with col1:

    fig = px.box(
        df,
        x="Churn",
        y="tenure",
        color="Churn",
        title="Tenure by Churn Status",
        labels={
            "tenure": "Tenure (Months)"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


with col2:

    fig = px.box(
        df,
        x="Churn",
        y="MonthlyCharges",
        color="Churn",
        title="Monthly Charges by Churn",
        labels={
            "MonthlyCharges":
                "Monthly Charges"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.markdown(
    '<div class="section-title">🔍 Key Churn Drivers</div>',
    unsafe_allow_html=True
)


rf_model = model.named_steps[
    "random_forest"
]

preprocessor = model.named_steps[
    "preprocessor"
]


feature_names = (
    preprocessor
    .get_feature_names_out()
)


importance_df = pd.DataFrame({

    "Feature":
        feature_names,

    "Importance":
        rf_model.feature_importances_
})


importance_df["Feature"] = (
    importance_df["Feature"]
    .str.replace(
        "numeric__",
        "",
        regex=False
    )
    .str.replace(
        "categorical__",
        "",
        regex=False
    )
)


importance_df = (
    importance_df
    .sort_values(
        "Importance",
        ascending=False
    )
)


top_features = (
    importance_df
    .head(10)
    .sort_values(
        "Importance"
    )
)


fig = px.bar(
    top_features,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Top 10 Predictive Features",
    text_auto=".3f"
)

fig.update_layout(
    xaxis_title="Feature Importance",
    yaxis_title="Feature"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown(
    '<div class="section-title">🤖 Model Performance</div>',
    unsafe_allow_html=True
)


# Recreate the same test split used during training

df_model = df.copy()

df_model["Churn"] = (
    df_model["Churn"]
    .map({
        "No": 0,
        "Yes": 1
    })
)


X = df_model.drop(
    columns=[
        "customerID",
        "Churn"
    ]
)

y = df_model["Churn"]


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
)


y_test_pred = model.predict(
    X_test
)

y_test_prob = model.predict_proba(
    X_test
)[:, 1]


accuracy = accuracy_score(
    y_test,
    y_test_pred
)

precision = precision_score(
    y_test,
    y_test_pred
)

recall = recall_score(
    y_test,
    y_test_pred
)

f1 = f1_score(
    y_test,
    y_test_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_test_prob
)


col1, col2, col3, col4, col5 = st.columns(5)


with col1:
    st.metric(
        "Accuracy",
        f"{accuracy:.2%}"
    )

with col2:
    st.metric(
        "Precision",
        f"{precision:.2%}"
    )

with col3:
    st.metric(
        "Recall",
        f"{recall:.2%}"
    )

with col4:
    st.metric(
        "F1 Score",
        f"{f1:.2%}"
    )

with col5:
    st.metric(
        "ROC-AUC",
        f"{roc_auc:.2%}"
    )


# ============================================================
# CONFUSION MATRIX + ROC
# ============================================================

col1, col2 = st.columns(2)


with col1:

    cm = confusion_matrix(
        y_test,
        y_test_pred
    )

    fig = px.imshow(
        cm,
        text_auto=True,
        x=[
            "Predicted No Churn",
            "Predicted Churn"
        ],
        y=[
            "Actual No Churn",
            "Actual Churn"
        ],
        title="Confusion Matrix"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


with col2:

    fpr, tpr, _ = roc_curve(
        y_test,
        y_test_prob
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"Random Forest (AUC={roc_auc:.3f})"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(
                dash="dash"
            ),
            name="Random Guess"
        )
    )

    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# RISK SEGMENTATION
# ============================================================

st.markdown(
    '<div class="section-title">🎯 Customer Risk Segmentation</div>',
    unsafe_allow_html=True
)


risk_counts = (
    customer_risk["Risk Level"]
    .value_counts()
    .reset_index()
)

risk_counts.columns = [
    "Risk Level",
    "Customers"
]


fig = px.bar(
    risk_counts,
    x="Risk Level",
    y="Customers",
    title="Customers by Risk Level",
    text_auto=True
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# CUSTOMER FILTER
# ============================================================

risk_filter = st.selectbox(
    "Select Risk Level",
    [
        "All",
        "High Risk",
        "Medium Risk",
        "Low Risk"
    ]
)


if risk_filter == "All":

    filtered = customer_risk.copy()

else:

    filtered = customer_risk[
        customer_risk["Risk Level"]
        == risk_filter
    ].copy()


filtered = filtered.sort_values(
    "Churn Probability",
    ascending=False
)


filtered["Churn Probability"] = (
    filtered["Churn Probability"] * 100
).round(2)


st.dataframe(
    filtered.head(100),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# BUSINESS INSIGHTS
# ============================================================

st.markdown(
    '<div class="section-title">💼 Business Insights</div>',
    unsafe_allow_html=True
)


insights = [

    (
        "Month-to-month contracts",
        "Month-to-month customers show substantially higher churn and should be prioritized for retention campaigns and long-term contract incentives."
    ),

    (
        "Early-tenure customers",
        "Churned customers have substantially lower tenure, indicating that the first year is an important retention period."
    ),

    (
        "Technical support",
        "Customers without technical support show elevated churn and could be targeted with proactive support offers."
    ),

    (
        "Online security",
        "Lack of online security is a strong predictive signal and may represent an opportunity for targeted service bundles."
    ),

    (
        "Payment method",
        "Electronic-check customers exhibit elevated churn and could be targeted for payment-method migration."
    ),

    (
        "Fiber optic service",
        "Fiber customers show elevated churn and warrant further investigation into pricing, service quality and customer expectations."
    )

]


for title, description in insights:

    st.markdown(
        f"""
        <div class="insight-box">
            <strong>{title}</strong><br>
            {description}
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Telecom Customer Churn Analytics | "
    "Tuned Random Forest | Python + Streamlit"
)