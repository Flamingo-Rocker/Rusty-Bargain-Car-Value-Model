# 🚗 Rusty Bargain Car Value Model  
*Fast, accurate used-car price prediction with production-minded trade-offs (quality vs. training/prediction speed).*

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)  
![pandas](https://img.shields.io/badge/pandas-EDA-green?logo=pandas)  
![numpy](https://img.shields.io/badge/numpy-Numerical-blue?logo=numpy)  
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)  
![matplotlib](https://img.shields.io/badge/matplotlib-Visualization-orange?logo=matplotlib)  
![seaborn](https://img.shields.io/badge/seaborn-EDA-blue?logo=python)  
![plotly](https://img.shields.io/badge/plotly-InteractiveViz-blue?logo=plotly)  
![tqdm](https://img.shields.io/badge/tqdm-ProgressBars-yellow)  
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)  
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 📌 Overview
**Rusty Bargain**, a used-car sales service, is building an app to estimate list prices from vehicle attributes. The modeling objective balances three criteria:

1. **Prediction quality** (low RMSE)  
2. **Prediction speed** (latency suitable for an app)  
3. **Training time** (practicality for iteration/refresh)

The project compares a **Linear Regression baseline**, a **Random Forest Regressor**, and an **LGBMRegressor** (LightGBM), with careful preprocessing for high-cardinality categorical features and a custom repeated CV routine. Final recommendations weigh **RMSE vs. latency** for production.

---

## 📊 Data
- Large historical listings with both **numeric** and **categorical** features (e.g., `price`, `power`, `kilometer`, `brand`, `model`, `vehicle_type`, `fuel_type`, `gearbox`, `not_repaired`).
- **Data quality actions** (per notebook):
  - Dropped clearly problematic/underrepresented brands (`sonstige_autos`, `trabant`) that skewed distributions.
  - Filled remaining categorical nulls with `"Unknown"`.
  - Filtered implausible prices (e.g., zero / extreme tails) to stabilize training targets.

---

## ⚙️ Methods
**Preprocessing & Feature Handling**
- Limited cardinality of `brand` and `model` for model stability.
- Encoders/scalers chosen per algorithm:
  - **Linear Regression:** `OneHotEncoder` (categoricals) + `MaxAbsScaler` (numericals); `positive=True` to avoid negative prices.
  - **Random Forest:** `OrdinalEncoder` for categoricals.
  - **LightGBM:** converted categoricals to `category` dtype to leverage native handling.

**Modeling & Validation**
- Models: `LinearRegression`, `RandomForestRegressor`, `LGBMRegressor`.
- **Custom CV:** `repeated_random_cv` (5 folds / repeated splits) to estimate generalization.
- Tracked **RMSE**, **training time**, and **prediction latency** for each candidate.

---

## 📈 Results
- **Linear Regression (baseline):**  
  - **RMSE ≈ 2971.95** — very fast train/predict, but accuracy not sufficient.

- **Random Forest Regressor (tuned):**  
  - Best RMSE configuration: **RMSE ≈ 1932.96** at `max_depth=20`, `n_estimators=200` (heavier compute).  
  - **Efficient alternative:** **RMSE ≈ 1948.52**, **train ≈ 9.52 s**, **predict ≈ 0.35 s** at `max_depth=20`, `n_estimators=50`.  
  - Excellent **latency** profile for an app with only a small RMSE trade-off vs. the heaviest RF.

- **LGBMRegressor (tuned):**  
  - Achieved the **lowest RMSE** in training (**≈ 1853.62**), benefiting from native categorical handling.  
  - **Trade-off:** noticeably slower predictions than Linear/RF; for strict real-time constraints, RF may be preferable despite a slightly higher RMSE.

**Takeaway:**  
If **lowest error** is paramount and you can tolerate higher prediction latency, choose **LightGBM**. If you need **snappy predictions** in production with near-best accuracy, deploy the **efficient Random Forest** (`depth=20, n_estimators=50`).

---

## 💡 Key Design Choices
- High-cardinality categorical features were **constrained/encoded** to avoid sparse explosions and reduce noise.
- A **repeatable CV setup** provided more reliable estimates than a single split.
- Evaluation included **end-to-end timing** so the model choice reflects **real deployment trade-offs**, not just metric scores.

---

## 🗂 Repo Structure
```
rusty-bargain-car-value-model/
├── notebooks/ <- Final cleaned Jupyter notebook
├── notebooks_archive/ <- Raw/working notebook (optional)
├── data/ <- Data source or pointer to source
├── requirements.txt <- Dependencies
├── LICENSE <- MIT (or your selection)
└── README.md <- This file
```

---

## 🚀 How to Run
1. Clone this repo:
    ```bash
    git clone https://github.com/Flamingo-Rocker/rusty-bargain-car-value-model.git
    cd rusty-bargain-car-value-model
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
3. Open the notebook in `/notebooks` to reproduce this analysis.

---

## 📦 Requirements
```
pandas==2.3.2
numpy==2.2.5
numpy-base==2.2.5            
numpydoc==1.9.0            
matplotlib==3.10.5           
matplotlib-base==3.10.5           
matplotlib-inline==0.1.6
seaborn==0.13.2
plotly==6.3.0      
scikit-learn==1.7.1 
lightgbm==4.6.0
tqdm==4.67.1        
ipython==8.30.0
m2-msys2-runtime==2.5.0.17080.65c939c
vc14_runtime==14.44.35208
vs2015_runtime==14.44.35208
```

---

## ✅ Recommendation
- Deploy: Random Forest (depth=20, n_estimators=50) for fast predictions with near-best RMSE.
- Consider: LightGBM when absolute accuracy is the priority and you can afford the latency.

---

## 🙏 Acknowledgment
Developed as part of the TripleTen Data Science Bootcamp, focusing on practical ML trade-offs (accuracy vs. latency) for real-time pricing.