# 🏦 Customer Segmentation & Churn Pattern Analytics in European Banking

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SSuryapriya/european-bank-churn-analytics/blob/main/Churn_Dashboard_Colab.ipynb)

An interactive Streamlit dashboard that segments 10,000 bank customers (France, Spain, Germany) and shows where and why they churn.
Built as part of my Data Analyst internship project at Unified Mentor.

## Key findings
- Overall churn rate: **20.4%**; Germany is highest at **32.4%** (France and Spain ≈ 16%).
- Age is the strongest driver: **51%** of customers aged 46–60 churned vs 7.6% of those under 30.
- Female customers churn more (25.1% vs 16.5%); inactive members churn at 26.9% vs 14.3% for active.
- High-value customers (top 25% by balance) churn at 23.7%, putting about €88.7M of balances at risk.

## Features
- Sidebar segment filters (geography, gender, age, tenure, credit band, balance) with live KPI updates
- KPIs: Overall Churn Rate, Segment Churn Rate, High-Value Churn Ratio, Geographic Risk Index, Engagement Drop Indicator
- Tabs: Overview · Geography · Age & Tenure · High-Value Explorer · Segments & Drill-down

## Segment definitions
| Dimension | Groups |
|---|---|
| Age | <30, 30–45, 46–60, 60+ |
| Tenure | New (0–2), Mid-term (3–6), Long-term (7+) |
| Credit score | Low (<580), Medium (580–669), High (670+) |
| Balance | Zero, Low / High (split at median of non-zero balances) |
| High-value | Top 25% of customers by balance |

## Run locally
```bash
git clone https://github.com/SSuryapriya/european-bank-churn-analytics.git
cd european-bank-churn-analytics
pip install -r requirements.txt
streamlit run app.py
```

## Run in Google Colab
Click the **Open in Colab** badge above and run the cells in order.

## Tech stack
Python · Pandas · Plotly · Streamlit

## Files
- `app.py` – Streamlit dashboard
- `European_Bank.csv` – dataset
- `Churn_Dashboard_Colab.ipynb` – Colab launcher
- `requirements.txt` – dependencies
