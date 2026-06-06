# 🛒 Superstore Sales Analytics & Profit Prediction

> End-to-end data analytics and machine learning project on four years of US retail sales data (2014–2017), combining exploratory analysis, customer segmentation, profit prediction, and time-series forecasting.

---

## 📌 Project Overview

This project analyzes the [Superstore Sales Dataset](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final) — a widely-used retail benchmark containing **~10,000 orders** spanning four years (2014–2017) across US regions, product categories, and customer segments.

### Business Questions Addressed

- Which regions, categories, and sub-categories drive the most revenue and profit?
- How does seasonality affect monthly and yearly sales trends?
- What is the operational efficiency in terms of shipping time?
- Can we segment customers into actionable groups based on purchasing behavior?
- Can we reliably predict order-level profit? Which features matter most?
- Is monthly sales volume forecastable with time-series methods?

---

## 🔬 Techniques Used

### Data Quality & Preparation
- Missing value checks, duplicate detection, and type casting
- Feature engineering: `Shipping Duration`, `Order Month/Year`, RFM metrics

### Exploratory Data Analysis (EDA)
- Sales and profit trends by region, category, sub-category, and segment
- Discount impact analysis on profit margins
- Seasonality visualization (monthly and yearly aggregations)

### KPI Calculations
- Total Revenue, Total Profit, Profit Margin
- Average Shipping Time (operational KPI)
- Sub-category profitability rankings

### Customer Segmentation — RFM Analysis
- **Recency**, **Frequency**, **Monetary** scoring per customer
- Rule-based segmentation into: `Champions`, `Loyal`, `Potential Loyalists`, `At Risk`, and others
- Segment-level profiling for targeted marketing strategy

### Profit Prediction (Supervised ML)
- Target: `Profit` (continuous — regression task)
- **Time-based train/test split** to prevent data leakage (train: 2014–2016 / test: 2017)
- Models evaluated: **Random Forest**, **XGBoost**
- Metrics: R², RMSE, MAE
- **Feature importance analysis** to identify the top profit drivers

### Sales Forecasting (Exploratory)
- Monthly aggregated time series
- Models explored: **ARIMA**, **Prophet**
- Qualitative evaluation of trend and seasonality capture

---

## 📊 Results & Key Insights

### 📈 Sales Trends & Seasonality
- Clear **end-of-year seasonality**: December consistently ranks among the strongest months for sales, with a secondary peak in November (holiday effect).
- Sales grew year-over-year across the full 2014–2017 period.

### 🗺️ Regional & Category Performance
- The **West region** leads in total sales volume.
- **Technology** is the top-performing category by both revenue and profit.
- **Furniture** (notably Tables and Bookcases) and high-discount sub-categories are frequent loss contributors despite strong order volumes.

### 🚚 Operational KPI
- **Average shipping duration ≈ 3.96 days** — a useful baseline for logistics benchmarking and SLA definition.

### 👥 Customer Segmentation (RFM)
| Segment | Profile | Recommended Action |
|---|---|---|
| **Champions** | High R, F, M | Reward & retain; leverage as brand advocates |
| **Loyal** | High F & M, moderate R | Upsell, cross-sell premium products |
| **Potential Loyalists** | Recent, moderate frequency | Nurture with loyalty programs |
| **At Risk** | Previously active, declining | Re-engagement campaigns |

### 💸 Discount & Profitability Review
- High discounts are **strongly correlated with negative profit** — several sub-categories (Tables, Bookcases, Supplies) are structurally loss-making.
- **Volume does not equal profitability**: some high-order sub-categories consistently destroy margin.

### 🤖 Profit Prediction
| Model | Test R² | Test RMSE |
|---|---|---|
| Random Forest | 0.821 | 23.45 |
| **XGBoost** | **0.854** | **21.60** |

- **XGBoost achieves the best performance**, explaining ~85% of profit variance on held-out 2017 data.
- Top predictive features: `Sales`, `Discount`, `Category`, `Sub-Category`, `Quantity`.
- Discount is among the strongest predictors of **profit loss** — confirming the EDA findings.

### 📅 Sales Forecasting
- Both ARIMA and Prophet capture the general seasonal pattern (end-of-year peaks).
- Results are **exploratory only**: with just 4 years of monthly data (~48 points), models lack sufficient history for reliable production forecasting.
- Recommendation: integrate additional years of data and perform rigorous backtesting (walk-forward validation) before deploying any forecast in a business context.

---

## 💡 Business Takeaways

1. **Revisit discount policies** — particularly for structurally unprofitable sub-categories. A blanket discount strategy is eroding margins without proportional revenue benefit.
2. **Prioritize the West region and Technology category** for resource allocation and growth investment.
3. **Plan inventory and staffing around Q4 seasonality** — December and November demand predictable surges.
4. **Activate the RFM segments differently**: retention spend should focus on *At Risk* customers; upsell efforts on *Loyal* and *Champions*.
5. **Use the XGBoost model as a profit screening tool** for new orders or pricing decisions — an R² of 0.854 is a strong operational baseline.
6. **Do not deploy the sales forecast without further validation** — treat current ARIMA/Prophet results as directional signals, not production-grade predictions.

---

