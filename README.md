# 🌊 River Plastic Classification using Machine Learning

This project uses **Machine Learning** to classify rivers based on their plastic contribution to the ocean. It includes the complete pipeline: from data cleaning to model building, evaluation, and interpretation using SHAP.

---

## 🎯 Objective

- Classify rivers as **high** or **low** plastic contributors using environmental features.
- Build interpretable models using Logistic Regression and evaluate with proper metrics.

---

## 📂 Dataset

- Source: Public dataset on **global riverine plastic emissions into oceans**
- Format: CSV
- Loaded directly from [Google Drive](https://drive.google.com/file/d/1zIk9JOdJEu9YF7Xuv2C8f2Q8ySfG3nHd/view)

## 🧠 Machine Learning Pipeline

### 🔹 Step-by-Step:

1. **Data Loading & Cleaning**
2. **Handling missing values**
3. **Feature Engineering** – Create `plastic_contribution` column
4. **Exploratory Analysis** – Visualize top contributors
5. **Feature Scaling**
6. **Train/Test Split**
7. **Handle Class Imbalance** using SMOTE
8. **Model Training**
   - Logistic Regression + GridSearchCV
   - Random Forest
   - XGBoost
9. **Model Evaluation**
   - Accuracy, Precision, Recall
   - Confusion Matrix, ROC, PR Curve
10. **Feature Importance Analysis**
11. **SHAP Explainability**
12. **Model Comparison**
13. **Model Saving**

---

## ⚙️ Installation

Make sure you have all dependencies installed:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib shap imbalanced-learn xgboost
```
