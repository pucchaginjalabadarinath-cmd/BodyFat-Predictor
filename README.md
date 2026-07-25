# 📏 Body Fat Predictor

Try this project by clicking on the link: [**Body Fat Predictor**](https://bodyfat-predictor-jm8k.onrender.com/)

Public Dataset used: [**Body Fat Prediction Dataset — Kaggle**](https://www.kaggle.com/datasets/simonezappatini/body-fat-extended-dataset/data)

---

### 📖 Story Behind the Project

Every scale tells you a number. It doesn't tell you what that number is actually made of.

Back in 1996, researcher Roger Johnson published a dataset of 252 men whose body fat
percentage had been measured the accurate but deeply impractical way — underwater weighing,
special equipment, trained technicians. A year later, a companion dataset added 184 women,
measured the same way. The whole point of both studies was the same question this project
asks: **can a handful of measurements anyone can take with a tape measure and a bathroom
scale predict what would otherwise need a lab and a water tank?**

That's what this is — a regression model trained on real circumference measurements
(neck, chest, abdomen, hip, thigh, and more) that estimates body fat percentage without
any special equipment.

---

### 🔍 Project Overview

Given a set of basic body measurements (age, weight, height, and 10 circumference
measurements), the model predicts **percentage body fat** — the same idea used in
sports science and fitness assessment, just without the specialized equipment.

**Pipeline:**

1. **Cleaning** — corrected known recording errors, handled an impossible `BodyFat = 0` reading
2. **EDA** — univariate/bivariate/multivariate analysis, correlation structure, sex-split analysis
3. **Split** — train/test split
4. **Outlier Handling** — IQR-based clipping, bounds fit on the training set only (no leakage), applied only to the linear-model branch
5. **Scaling** — `StandardScaler`, fit on train only
6. **Models** — Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, XGBoost, LightGBM
7. **Hyperparameter Tuning** — `GridSearchCV` for the tree-based models
8. **Model Comparison** — trained all seven models and compared them by test R²
9. **Final Model** — selected Ridge Regression, the strongest performer on the held-out test set
10. **Deployment** — FastAPI backend + custom frontend, hosted on Render

---

### 📊 Results

| Model | Test R² | Test RMSE (% body fat) |
|---|---|---|
| **Ridge Regression (final)** | ~0.67 | ~3.6 |
| Linear Regression | ~0.66 | ~3.6 |
| Lasso | ~0.67 | ~3.6 |
| Random Forest (tuned) | ~0.58 | ~4.2 |
| XGBoost (tuned) | ~0.55 | ~4.3 |
| Decision Tree (tuned) | ~0.31 | ~5.6 |

Ridge came out on top — on a dataset this size (~435 rows) with mostly linear relationships
between circumference measurements and body fat, the simpler linear models generalize better
than the tree ensembles, even after tuning.

*(Numbers above are approximate from development runs — see `SOML_26_Project.ipynb` for
the exact final values from the training run.)*

---

### 🛠️ Tech Stack

- **Modeling:** Python, pandas, scikit-learn, XGBoost, LightGBM
- **Backend:** FastAPI, Uvicorn, joblib
- **Frontend:** HTML, CSS, vanilla JavaScript
- **Deployment:** Render

---

### 💻 Run It Locally

```bash
git clone https://github.com/<pucchaginjalabadarinath-cmd>/BodyFat-Predictor.git
cd BodyFat-Predictor

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
uvicorn main:app --reload
```

Then open `http://127.0.0.1:8000` in your browser.

---

### 📁 Repo Structure

```
bodyfat-deploy/
├── main.py                      # FastAPI backend
├── requirements.txt
├── bodyfat_model.pkl             # trained Ridge model
├── bodyfat_scaler.pkl            # fitted StandardScaler
├── bodyfat_clip_bounds.pkl       # outlier clip bounds (train-only)
├── static/
│   └── index.html                # frontend
├── SOML_26_Project.ipynb         # full training notebook (EDA → final model)
└── BodyFat.csv                   # dataset used for training
```

---

### 🎯 Try It Yourself

All you need is a tape measure and a scale — [give it a shot](https://bodyfat-predictor-jm8k.onrender.com/).

---

### 🙏 Conclusion

This project started as a personal question, not an assignment — I noticed my own weight
change between a semester at college and three months back home, and wanted an actual answer
instead of a guess. Six weeks ago I didn't know what a train-test split was; by the end of this
one I'd cleaned two merged real-world datasets, caught data leakage in my own pipeline, compared
seven models honestly instead of picking whichever looked best, and shipped my first ever
full-stack deployment — backend, frontend, and hosting, all built from scratch.

This project is part of **Summer of ML (SoM-26)**, run by the **AI/ML wing of the Coding
Club**. I'm grateful to my seniors and mentors there for the guidance, the code reviews, and
the patience — this is the first project I've built end-to-end on my own, and I wouldn't have
gotten here without the six weeks before it.
