"""Customer Segmentation & Churn Pattern Analytics in European Banking
Run:  pip install -r requirements.txt  ->  streamlit run app.py
"""
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="European Bank Churn Analytics", page_icon="🏦", layout="wide")


# ---------------------------------------------------------------- data
@st.cache_data
def load_data(path="European_Bank.csv"):
    df = pd.read_csv(path)
    # validation
    for c in ["HasCrCard", "IsActiveMember", "Exited"]:
        assert set(df[c].unique()) <= {0, 1}, f"{c} is not binary"
    df = df.drop(columns=["Surname"])  # non-analytical
    df["Age_Group"] = pd.cut(df.Age, [0, 29, 45, 60, 200], labels=["<30", "30-45", "46-60", "60+"])
    df["Tenure_Group"] = pd.cut(df.Tenure, [-1, 2, 6, 20], labels=["New (0-2)", "Mid-term (3-6)", "Long-term (7+)"])
    df["Credit_Band"] = pd.cut(df.CreditScore, [0, 579, 669, 900], labels=["Low", "Medium", "High"])
    med = df.loc[df.Balance > 0, "Balance"].median()
    df["Balance_Segment"] = pd.cut(df.Balance, [-1, 0, med, 1e12], labels=["Zero", "Low", "High"])
    df["Status"] = df.Exited.map({1: "Churned", 0: "Retained"})
    return df


df = load_data()
HV_CUT = df.Balance.quantile(0.75)  # high-value = top 25% by balance

# ------------------------------------------------------------- filters
st.sidebar.header("Segment filters")
geo = st.sidebar.multiselect("Geography", df.Geography.unique(), df.Geography.unique())
gen = st.sidebar.multiselect("Gender", df.Gender.unique(), df.Gender.unique())
age = st.sidebar.multiselect("Age group", df.Age_Group.cat.categories, df.Age_Group.cat.categories)
ten = st.sidebar.multiselect("Tenure group", df.Tenure_Group.cat.categories, df.Tenure_Group.cat.categories)
cb = st.sidebar.multiselect("Credit band", df.Credit_Band.cat.categories, df.Credit_Band.cat.categories)
bal = st.sidebar.multiselect("Balance segment", df.Balance_Segment.cat.categories, df.Balance_Segment.cat.categories)

f = df[
    df.Geography.isin(geo) & df.Gender.isin(gen) & df.Age_Group.isin(age)
    & df.Tenure_Group.isin(ten) & df.Credit_Band.isin(cb) & df.Balance_Segment.isin(bal)
]
if f.empty:
    st.warning("No customers match the selected filters.")
    st.stop()


def churn_by(col, data=f):
    g = data.groupby(col, observed=True).Exited.agg(Customers="count", Churned="sum").reset_index()
    g["Churn Rate %"] = (g.Churned / g.Customers * 100).round(1)
    g["Share of All Churn %"] = (g.Churned / max(g.Churned.sum(), 1) * 100).round(1)
    return g


# ----------------------------------------------------------------- KPIs
overall = f.Exited.mean() * 100
hv = f[f.Balance >= HV_CUT]
hv_churn = hv.Exited.mean() * 100 if len(hv) else float("nan")
act = f[f.IsActiveMember == 1].Exited.mean() * 100 if (f.IsActiveMember == 1).any() else float("nan")
inact = f[f.IsActiveMember == 0].Exited.mean() * 100 if (f.IsActiveMember == 0).any() else float("nan")

st.title("🏦 Customer Segmentation & Churn Analytics – European Banking")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Overall Churn Rate", f"{overall:.1f}%", f"{overall - df.Exited.mean()*100:+.1f} pts vs all")
k2.metric("Customers", f"{len(f):,}")
k3.metric("High-Value Churn Ratio", f"{hv_churn:.1f}%")
k4.metric("Engagement Drop (inactive ÷ active)", f"{inact/act:.2f}x" if act else "n/a")
k5.metric("Balance Lost to Churn", f"€{f.loc[f.Exited==1,'Balance'].sum()/1e6:,.1f}M")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Geography", "Age & Tenure", "High-Value Explorer", "Segments & Drill-down"])

# ------------------------------------------------------------ overview
with tab1:
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.pie(f, names="Status", title="Churned vs Retained", hole=.5,
                           color="Status", color_discrete_map={"Churned": "#d62728", "Retained": "#1f77b4"}),
                    width='stretch')
    p = churn_by("Gender")
    c2.plotly_chart(px.bar(p, x="Gender", y="Churn Rate %", text="Churn Rate %", title="Churn by Gender"),
                    width='stretch')
    c3, c4 = st.columns(2)
    n = churn_by("NumOfProducts")
    c3.plotly_chart(px.bar(n, x="NumOfProducts", y="Churn Rate %", text="Churn Rate %",
                           title="Churn by Number of Products"), width='stretch')
    a = churn_by("IsActiveMember")
    a["IsActiveMember"] = a.IsActiveMember.map({1: "Active", 0: "Inactive"})
    c4.plotly_chart(px.bar(a, x="IsActiveMember", y="Churn Rate %", text="Churn Rate %",
                           title="Engagement: Active vs Inactive"), width='stretch')
    st.subheader("Churned vs retained profile")
    prof = f.groupby("Status")[["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
                                "EstimatedSalary", "IsActiveMember"]].mean().round(1).T
    st.dataframe(prof, width='stretch')

# ----------------------------------------------------------- geography
with tab2:
    g = churn_by("Geography")
    g["Geographic Risk Index"] = (g["Churn Rate %"] / overall).round(2)
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(g, x="Geography", y="Churn Rate %", text="Churn Rate %",
                           color="Geography", title="Churn Rate by Country"), width='stretch')
    c2.plotly_chart(px.pie(g, names="Geography", values="Churned", title="Share of Total Churn by Country"),
                    width='stretch')
    st.caption("Geographic Risk Index = country churn rate ÷ overall churn rate (>1 means above-average risk).")
    st.dataframe(g, width='stretch', hide_index=True)
    ga = f.groupby(["Geography", "Age_Group"], observed=True).Exited.mean().mul(100).round(1).reset_index()
    st.plotly_chart(px.density_heatmap(ga, x="Age_Group", y="Geography", z="Exited", text_auto=True,
                                       color_continuous_scale="Reds", title="Geography × Age churn % (interaction)"),
                    width='stretch')

# ------------------------------------------------------- age & tenure
with tab3:
    c1, c2 = st.columns(2)
    ag = churn_by("Age_Group")
    c1.plotly_chart(px.bar(ag, x="Age_Group", y="Churn Rate %", text="Churn Rate %", title="Churn by Age Group"),
                    width='stretch')
    tg = churn_by("Tenure_Group")
    c2.plotly_chart(px.bar(tg, x="Tenure_Group", y="Churn Rate %", text="Churn Rate %", title="Churn by Tenure"),
                    width='stretch')
    st.plotly_chart(px.histogram(f, x="Age", color="Status", barmode="overlay", nbins=40, opacity=.65,
                                 title="Age distribution: churned vs retained"), width='stretch')
    st.dataframe(ag, width='stretch', hide_index=True)

# --------------------------------------------------- high-value explorer
with tab4:
    st.write(f"High-value customers = balance ≥ **€{HV_CUT:,.0f}** (top 25% of all customers).")
    thr = st.slider("Custom balance threshold (€)", 0, int(df.Balance.max()), int(HV_CUT), 5000)
    h = f[f.Balance >= thr]
    if h.empty:
        st.info("No customers above this threshold in the current filters.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("High-value customers", f"{len(h):,}")
        m2.metric("Churn rate", f"{h.Exited.mean()*100:.1f}%")
        m3.metric("Balance at risk", f"€{h.loc[h.Exited==1,'Balance'].sum()/1e6:,.1f}M")
        st.plotly_chart(px.scatter(h, x="EstimatedSalary", y="Balance", color="Status", opacity=.6,
                                   color_discrete_map={"Churned": "#d62728", "Retained": "#1f77b4"},
                                   title="Salary vs Balance (high-value customers)"), width='stretch')
        hg = churn_by("Geography", h)
        st.plotly_chart(px.bar(hg, x="Geography", y="Churn Rate %", text="Churn Rate %",
                               title="High-value churn by country"), width='stretch')
        st.subheader("Top high-balance churners")
        st.dataframe(h[h.Exited == 1].nlargest(25, "Balance")[
            ["CustomerId", "Geography", "Gender", "Age", "Tenure", "Balance", "EstimatedSalary",
             "NumOfProducts", "IsActiveMember"]], width='stretch', hide_index=True)

# --------------------------------------------------------- drill-down
with tab5:
    dim = st.selectbox("Segment dimension", ["Geography", "Age_Group", "Credit_Band", "Tenure_Group",
                                             "Balance_Segment", "Gender", "NumOfProducts"])
    t = churn_by(dim)
    st.plotly_chart(px.bar(t, x=dim, y="Churn Rate %", text="Churn Rate %", title=f"Churn by {dim}"),
                    width='stretch')
    st.dataframe(t, width='stretch', hide_index=True)
    sub = st.selectbox(f"Drill into a {dim} value", t[dim].astype(str).tolist())
    d2 = f[f[dim].astype(str) == sub]
    other = st.selectbox("…then break down by", [c for c in ["Geography", "Age_Group", "Credit_Band",
                         "Tenure_Group", "Balance_Segment", "Gender"] if c != dim])
    st.dataframe(churn_by(other, d2), width='stretch', hide_index=True)
    st.download_button("Download filtered data (CSV)", f.to_csv(index=False), "filtered_customers.csv")
