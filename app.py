```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="European Banking | Churn Analytics",
    page_icon="🏦",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
    .main {
        background-color: #f5f7fb;
    }

    .block-container {
        padding-top: 1.5rem;
    }

    .main-title {
        font-size: 32px;
        font-weight: 700;
        color: #16325c;
    }

    .subtitle {
        font-size: 15px;
        color: #65758b;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🏦 Customer Segmentation & Churn Analytics</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">European Banking | Customer Behaviour | '
    'Financial Risk | Retention Analytics</div>',
    unsafe_allow_html=True
)

st.divider()

# --------------------------------------------------
# DATASET UPLOAD
# --------------------------------------------------

uploaded_file = st.sidebar.file_uploader(
    "Upload Banking Dataset (CSV)",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload your European Banking customer CSV dataset.")
    st.markdown("""
    ### Expected columns

    - CustomerId
    - Surname
    - CreditScore
    - Geography
    - Gender
    - Age
    - Tenure
    - Balance
    - NumOfProducts
    - HasCrCard
    - IsActiveMember
    - EstimatedSalary
    - Exited

    **Exited:** 1 = Churned, 0 = Retained
    """)
    st.stop()

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Unable to read CSV: {e}")
    st.stop()

df.columns = df.columns.str.strip()

required_columns = [
    "CustomerId", "CreditScore", "Geography", "Gender",
    "Age", "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    st.error(f"Missing required columns: {missing_columns}")
    st.stop()

# --------------------------------------------------
# DATA VALIDATION
# --------------------------------------------------

st.sidebar.header("Dashboard Filters")

with st.expander("📋 Data Quality & Validation", expanded=False):

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Rows", f"{len(df):,}")
    col2.metric("Total Columns", len(df.columns))
    col3.metric("Duplicate Rows", int(df.duplicated().sum()))

    st.subheader("Missing Values")

    missing_data = df.isnull().sum().reset_index()
    missing_data.columns = ["Column", "Missing Values"]
    missing_data = missing_data[missing_data["Missing Values"] > 0]

    if missing_data.empty:
        st.success("No missing values found.")
    else:
        st.dataframe(missing_data, use_container_width=True)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

# --------------------------------------------------
# DATA PREPARATION
# --------------------------------------------------

data = df.copy()

numeric_columns = [
    "CreditScore", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember",
    "EstimatedSalary", "Exited"
]

for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")

data = data.dropna(subset=required_columns)

# Validate binary target and binary indicator columns.
data = data[
    data["Exited"].isin([0, 1]) &
    data["HasCrCard"].isin([0, 1]) &
    data["IsActiveMember"].isin([0, 1])
].copy()

if data.empty:
    st.error("No valid customer records remain after data validation.")
    st.stop()

data["Exited"] = data["Exited"].astype(int)
data["HasCrCard"] = data["HasCrCard"].astype(int)
data["IsActiveMember"] = data["IsActiveMember"].astype(int)

# --------------------------------------------------
# CUSTOMER SEGMENTATION
# --------------------------------------------------

data["Age_Group"] = pd.cut(
    data["Age"],
    bins=[-np.inf, 30, 45, 60, np.inf],
    labels=["Under 30", "30–45", "46–60", "60+"],
    right=True
)

data["Credit_Score_Band"] = pd.cut(
    data["CreditScore"],
    bins=[-np.inf, 579, 669, np.inf],
    labels=["Low", "Medium", "High"]
)

data["Tenure_Group"] = pd.cut(
    data["Tenure"],
    bins=[-np.inf, 2, 6, np.inf],
    labels=["New", "Mid-term", "Long-term"]
)

balance_75 = data["Balance"].quantile(0.75)

data["Balance_Segment"] = np.select(
    [
        data["Balance"].eq(0),
        data["Balance"].gt(0) & data["Balance"].le(balance_75),
        data["Balance"].gt(balance_75)
    ],
    [
        "Zero Balance",
        "Low Balance",
        "High Balance"
    ],
    default="Zero Balance"
)

# High-value proxy: upper quartile of positive account balances.
positive_balances = data.loc[data["Balance"] > 0, "Balance"]

if len(positive_balances) > 0:
    high_value_threshold = positive_balances.quantile(0.75)
    data["High_Value"] = data["Balance"] >= high_value_threshold
    data.loc[data["Balance"] <= 0, "High_Value"] = False
else:
    high_value_threshold = 0
    data["High_Value"] = False

data["Customer_Status"] = data["Exited"].map({
    0: "Retained",
    1: "Churned"
})

# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------

geographies = sorted(data["Geography"].dropna().unique().tolist())
genders = sorted(data["Gender"].dropna().unique().tolist())

selected_geo = st.sidebar.multiselect(
    "Geography",
    geographies,
    default=geographies
)

selected_gender = st.sidebar.multiselect(
    "Gender",
    genders,
    default=genders
)

selected_age = st.sidebar.multiselect(
    "Age Group",
    data["Age_Group"].dropna().astype(str).unique().tolist(),
    default=data["Age_Group"].dropna().astype(str).unique().tolist()
)

selected_status = st.sidebar.multiselect(
    "Customer Status",
    ["Retained", "Churned"],
    default=["Retained", "Churned"]
)

filtered = data[
    data["Geography"].isin(selected_geo) &
    data["Gender"].isin(selected_gender) &
    data["Age_Group"].astype(str).isin(selected_age) &
    data["Customer_Status"].isin(selected_status)
].copy()

if filtered.empty:
    st.warning("No customers match the selected filters.")
    st.stop()

# --------------------------------------------------
# KPI CALCULATIONS
# --------------------------------------------------

total_customers = len(filtered)
churned_customers = int(filtered["Exited"].sum())
retained_customers = total_customers - churned_customers

churn_rate = filtered["Exited"].mean() * 100

high_value_customers = filtered[filtered["High_Value"]]
high_value_churn_rate = (
    high_value_customers["Exited"].mean() * 100
    if len(high_value_customers) > 0 else 0
)

inactive_customers = filtered[filtered["IsActiveMember"] == 0]
inactive_churn_rate = (
    inactive_customers["Exited"].mean() * 100
    if len(inactive_customers) > 0 else 0
)

# --------------------------------------------------
# NAVIGATION
# --------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🌍 Geography",
    "👥 Segmentation",
    "💰 Financial Risk",
    "📋 Customer Explorer"
])

# --------------------------------------------------
# TAB 1: OVERVIEW
# --------------------------------------------------

with tab1:

    st.subheader("Executive Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Customers", f"{total_customers:,}")
    c2.metric("Churned Customers", f"{churned_customers:,}")
    c3.metric("Churn Rate", f"{churn_rate:.2f}%")
    c4.metric("Retained Customers", f"{retained_customers:,}")

    st.divider()

    left, right = st.columns(2)

    with left:
        status_counts = (
            filtered["Customer_Status"]
            .value_counts()
            .rename_axis("Status")
            .reset_index(name="Customers")
        )

        fig = px.pie(
            status_counts,
            names="Status",
            values="Customers",
            hole=0.55,
            title="Customer Churn Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        churn_by_gender = (
            filtered.groupby("Gender", observed=True)
            .agg(
                Customers=("Exited", "size"),
                Churn_Rate=("Exited", "mean")
            )
            .reset_index()
        )

        churn_by_gender["Churn_Rate"] *= 100

        fig = px.bar(
            churn_by_gender,
            x="Gender",
            y="Churn_Rate",
            color="Gender",
            text_auto=".2f",
            title="Churn Rate by Gender (%)"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Key Performance Indicators")

    k1, k2, k3 = st.columns(3)

    k1.metric(
        "High-Value Customer Churn",
        f"{high_value_churn_rate:.2f}%"
    )

    k2.metric(
        "Inactive Customer Churn",
        f"{inactive_churn_rate:.2f}%"
    )

    k3.metric(
        "Average Customer Tenure",
        f"{filtered['Tenure'].mean():.2f} years"
    )

# --------------------------------------------------
# TAB 2: GEOGRAPHY
# --------------------------------------------------

with tab2:

    st.subheader("Geographic Churn Analysis")

    geo_analysis = (
        filtered.groupby("Geography", observed=True)
        .agg(
            Total_Customers=("Exited", "size"),
            Churned_Customers=("Exited", "sum"),
            Churn_Rate=("Exited", "mean"),
            Average_Balance=("Balance", "mean")
        )
        .reset_index()
    )

    geo_analysis["Churn_Rate"] *= 100

    left, right = st.columns(2)

    with left:
        fig = px.bar(
            geo_analysis,
            x="Geography",
            y="Churn_Rate",
            color="Geography",
            text_auto=".2f",
            title="Churn Rate by Country (%)"
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.bar(
            geo_analysis,
            x="Geography",
            y="Churned_Customers",
            color="Geography",
            text_auto=True,
            title="Number of Churned Customers"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Country-Level Summary")

    st.dataframe(
        geo_analysis.round(2),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Geographic risk should be interpreted using both churn rate "
        "and the number of churned customers."
    )

# --------------------------------------------------
# TAB 3: SEGMENTATION
# --------------------------------------------------

with tab3:

    st.subheader("Customer Segment Analysis")

    segment_column = st.selectbox(
        "Select a segmentation dimension",
        [
            "Age_Group",
            "Credit_Score_Band",
            "Tenure_Group",
            "Balance_Segment",
            "NumOfProducts",
            "IsActiveMember",
            "HasCrCard"
        ]
    )

    segment_analysis = (
        filtered.groupby(segment_column, observed=True)
        .agg(
            Customers=("Exited", "size"),
            Churned=("Exited", "sum"),
            Churn_Rate=("Exited", "mean")
        )
        .reset_index()
    )

    segment_analysis["Churn_Rate"] *= 100
    segment_analysis["Churn_Contribution"] = (
        segment_analysis["Churned"] /
        max(churned_customers, 1) * 100
    )

    left, right = st.columns(2)

    with left:
        fig = px.bar(
            segment_analysis,
            x=segment_column,
            y="Churn_Rate",
            color="Churn_Rate",
            text_auto=".2f",
            title="Churn Rate by Segment (%)"
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.bar(
            segment_analysis,
            x=segment_column,
            y="Churn_Contribution",
            text_auto=".2f",
            title="Share of All Churned Customers (%)"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Segment Performance Table")

    st.dataframe(
        segment_analysis.round(2),
        use_container_width=True,
        hide_index=True
    )

# --------------------------------------------------
# TAB 4: FINANCIAL RISK
# --------------------------------------------------

with tab4:

    st.subheader("High-Value Customer & Financial Risk")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "High-Value Customers",
        f"{len(high_value_customers):,}"
    )

    c2.metric(
        "High-Value Churners",
        f"{int(high_value_customers['Exited'].sum()):,}"
    )

    c3.metric(
        "High-Value Churn Rate",
        f"{high_value_churn_rate:.2f}%"
    )

    st.caption(
        "High-value is defined here as customers whose positive account "
        "balance is at or above the 75th percentile of positive balances. "
        "Balance alone is a proxy, not a complete measure of customer value."
    )

    left, right = st.columns(2)

    with left:
        fig = px.box(
            filtered,
            x="Customer_Status",
            y="Balance",
            color="Customer_Status",
            title="Account Balance: Churned vs Retained"
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.box(
            filtered,
            x="Customer_Status",
            y="EstimatedSalary",
            color="Customer_Status",
            title="Estimated Salary: Churned vs Retained"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Balance and Churn Relationship")

    financial_summary = (
        filtered.groupby("Customer_Status", observed=True)
        .agg(
            Customers=("CustomerId", "size"),
            Average_Balance=("Balance", "mean"),
            Average_Salary=("EstimatedSalary", "mean"),
            Average_Credit_Score=("CreditScore", "mean"),
            Average_Tenure=("Tenure", "mean")
        )
        .reset_index()
    )

    st.dataframe(
        financial_summary.round(2),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Financial Profile Explorer")

    financial_view = filtered[
        [
            "CustomerId", "Geography", "Age",
            "Balance", "EstimatedSalary", "NumOfProducts",
            "IsActiveMember", "Exited", "Customer_Status",
            "High_Value"
        ]
    ]

    fig = px.scatter(
        financial_view,
        x="Balance",
        y="EstimatedSalary",
        color="Customer_Status",
        size="NumOfProducts",
        hover_data=["Geography", "Age", "High_Value"],
        title="Customer Financial Profile"
    )

    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# TAB 5: CUSTOMER EXPLORER
# --------------------------------------------------

with tab5:

    st.subheader("Customer-Level Explorer")

    explorer_status = st.selectbox(
        "Choose customer status",
        ["All", "Churned", "Retained"]
    )

    explorer = filtered.copy()

    if explorer_status != "All":
        explorer = explorer[
            explorer["Customer_Status"] == explorer_status
        ]

    display_columns = [
        "CustomerId",
        "Geography",
        "Gender",
        "Age",
        "Age_Group",
        "CreditScore",
        "Credit_Score_Band",
        "Tenure",
        "Tenure_Group",
        "Balance",
        "Balance_Segment",
        "EstimatedSalary",
        "NumOfProducts",
        "IsActiveMember",
        "Exited",
        "Customer_Status",
        "High_Value"
    ]

    st.write(f"Customers displayed: {len(explorer):,}")

    st.dataframe(
        explorer[display_columns],
        use_container_width=True,
        hide_index=True
    )

    csv_data = explorer[display_columns].to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Customer Analysis CSV",
        data=csv_data,
        file_name="customer_churn_analysis.csv",
        mime="text/csv"
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Customer Segmentation & Churn Pattern Analytics in European Banking | "
    "Exploratory analytics for decision support. "
    "Results describe historical patterns and do not establish causation."
)
```
