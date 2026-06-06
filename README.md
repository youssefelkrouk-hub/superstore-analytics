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
# 📐 Model Evaluation Indicators —

---

## 1. CV_R² — Cross-Validation R²

**What it is:**  
The average R² score across all cross-validation folds, computed on the **training data only** (2014–2016).

**What R² means:**  
R² measures how much of the variance in `Profit` the model explains.  
- R² = 1.0 → perfect predictions  
- R² = 0.0 → model is no better than predicting the mean  
- R² < 0 → model is worse than the mean (bad)

**In our context:**  
| Model | CV_R² |
|---|---|
| XGBoost | 0.728 |
| RandomForest (Tuned) | 0.725 |
| RandomForest (Base) | 0.712 |

All three models explain ~71–73% of profit variance during cross-validation. The scores are close, meaning no model has a strong advantage on training data alone. CV_R² tells us how well the model **learns** — but not how well it **generalizes**.

---

## 2. CV_RMSE — Cross-Validation Root Mean Squared Error

**What it is:**  
The average prediction error (in dollars) across CV folds, still on **training data**.

**What RMSE means:**  
RMSE is the square root of the average squared difference between predicted and actual profit.  
- Same unit as the target → directly interpretable as **"on average, the model is off by X dollars"**  
- Penalizes **large errors** more heavily than small ones (because of the square)

**In our context:**  
| Model | CV_RMSE |
|---|---|
| RandomForest (Base) | 39.31 $ |
| XGBoost | 38.20 $ |
| RandomForest (Tuned) | — |

During cross-validation, models make an average error of ~38–39 $ per order. This is relatively high, which is expected — profit in retail is noisy and heavily influenced by discounts and sub-category, making it hard to predict perfectly. XGBoost has a slightly lower CV_RMSE, suggesting it handles the training data more efficiently.

> ⚠️ CV_RMSE for RandomForest (Tuned) is missing (`—`) because `GridSearchCV` was configured to optimize R² only, so RMSE was not collected during tuning.

---

## 3. Test_R² — Test Set R²

**What it is:**  
R² computed on the **held-out test set (year 2017)** — data the model has never seen during training.

**Why it matters more than CV_R²:**  
This is the **honest evaluation**. It tells us whether the model truly generalizes to new data, or if it just memorized patterns from the training period.

**In our context:**  
| Model | Test_R² |
|---|---|
| **XGBoost** | **0.872** ✅ |
| RandomForest (Base) | 0.857 |
| RandomForest (Tuned) | 0.806 |

XGBoost explains **87.2% of profit variance** on unseen 2017 orders — a strong result for a noisy retail target. Notice that RandomForest (Tuned) actually **drops** from CV to Test (0.725 → 0.806), while Base RF holds up better. This suggests the tuning **overfitted the CV folds** and did not improve real-world generalization.

---

## 4. Test_RMSE — Test Set Root Mean Squared Error

**What it is:**  
The average prediction error in **dollars** on the 2017 test set.

**Why it matters:**  
This is the metric that translates directly into **business impact**. An RMSE of 26$ means that on average, the model's profit prediction deviates by 26$ per order.

**In our context:**  
| Model | Test_RMSE |
|---|---|
| **XGBoost** | **26.37 $** ✅ |
| RandomForest (Base) | 27.84 $ |
| RandomForest (Tuned) | 32.49 $ |

XGBoost makes the smallest average error. The gap between XGBoost (26.37$) and Tuned RF (32.49$) is meaningful — the tuned model is **6$ worse per order**, confirming that tuning hurt generalization here.

---

## 🔑 Summary Table

| Indicator | Computed On | Tells Us | Higher is better? |
|---|---|---|---|
| **CV_R²** | Training folds | How well the model learns | ✅ Yes |
| **CV_RMSE** | Training folds | Average training error ($) | ❌ No (lower = better) |
| **Test_R²** | Held-out 2017 data | How well the model generalizes | ✅ Yes |
| **Test_RMSE** | Held-out 2017 data | Real-world prediction error ($) | ❌ No (lower = better) |

---

## ⚖️ How to Read Them Together

```
CV_R² ≈ Test_R²  →  model generalizes well, no overfitting
CV_R² >> Test_R² →  model overfitted the training period (e.g. RandomForest Tuned)

Low CV_RMSE + Low Test_RMSE  →  consistently accurate
Low CV_RMSE + High Test_RMSE →  good on paper, poor in practice
```

**In our project:** XGBoost is the best choice because it achieves the highest Test_R² (0.872) and lowest Test_RMSE (26.37$), meaning it is both accurate and generalizes well to new orders — making it reliable as a **profit screening tool** for business decisions.


