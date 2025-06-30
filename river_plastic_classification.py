# river_plastic_classification.py

# Step 0: Install packages (for Colab, comment out if running locally)
# !pip install --quiet shap imbalanced-learn xgboost seaborn joblib

# Imports
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, classification_report,
    confusion_matrix, roc_auc_score, RocCurveDisplay, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from imblearn.over_sampling import SMOTE

# Create images directory to save plots if it doesn't exist
os.makedirs("images", exist_ok=True)

# Step 1: Load dataset
file_id = "1zIk9JOdJEu9YF7Xuv2C8f2Q8ySfG3nHd"
url = f"https://drive.google.com/uc?id={file_id}"
df = pd.read_csv(url, thousands="'")

# Step 2: Data Cleaning
df.columns = df.columns.str.strip()
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
df.replace({'': np.nan, 'NA': np.nan, 'na': np.nan, 'Na': np.nan}, inplace=True)

percentage_cols = ["P[E] [%]", "Ratio Me/MPW"]
for col in percentage_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace("%", "", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")

for col in df.columns:
    if df[col].dtype == "object" and col != "Country or Administrative area":
        df[col] = df[col].astype(str).str.replace("'", "", regex=False)
        df[col] = df[col].str.replace(",", "", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")

print("Null values before dropping NaNs:\n", df.isnull().sum())
df = df.dropna()
print(f"\nRows after dropping NaNs: {len(df)}")

# Step 3: Target creation
df["plastic_contribution"] = np.where(df["M[E] (metric tons year -1)"] > 6008, 0, 1)

# Step 4: Exploratory insight
country_contrib = df.groupby("Country or Administrative area")["M[E] (metric tons year -1)"].sum().sort_values(ascending=False)
print("\nTop 5 countries contributing plastic to oceans:")
print(country_contrib.head(5))

plt.figure(figsize=(10,5))
sns.barplot(x=country_contrib.head(5).values, y=country_contrib.head(5).index, palette="mako")
plt.xlabel("Total Plastic Emission (metric tons per year)")
plt.title("Top 5 Countries Contributing Plastic to Oceans")
plt.tight_layout()
plt.savefig("images/top_5_countries.png")
plt.close()

# Step 5: Features and target
features = df.drop(columns=["plastic_contribution", "Country or Administrative area"])
X = features
y = df["plastic_contribution"]

# Step 6: Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 7: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Step 8: Handle imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Step 9: Logistic Regression + GridSearchCV
logreg = LogisticRegression(max_iter=1000)
param_grid = {'C': [0.01, 0.1, 1, 10]}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(logreg, param_grid, cv=cv, scoring='f1', n_jobs=-1)
grid_search.fit(X_train_resampled, y_train_resampled)
best_model = grid_search.best_estimator_

# Step 10: Evaluation
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:,1]

print("\n=== Model Evaluation (Logistic Regression) ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("images/confusion_matrix.png")
plt.close()

# ROC curve
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC Score: {roc_auc:.4f}")
RocCurveDisplay.from_estimator(best_model, X_test, y_test)
plt.savefig("images/roc_curve.png")
plt.close()

# Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
plt.plot(recall, precision, marker='.')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.tight_layout()
plt.savefig("images/precision_recall.png")
plt.close()

# Step 11: Feature importance
importance = np.abs(best_model.coef_[0])
importance_df = pd.DataFrame({
    "Feature": features.columns,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 Important Features:")
print(importance_df.head(10))

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=importance_df.head(10), palette="viridis")
plt.title("Top 10 Important Features")
plt.tight_layout()
plt.savefig("images/feature_importance.png")
plt.close()

# Step 12: SHAP explainability
explainer = shap.LinearExplainer(best_model, X_train_resampled, feature_perturbation="interventional")
shap_values = explainer.shap_values(X_test)
print("\nGenerating SHAP summary plot...")
shap.summary_plot(shap_values, X_test, feature_names=features.columns, show=False)
plt.tight_layout()
plt.savefig("images/shap_summary.png")
plt.close()

# Step 13: Advanced models
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_resampled, y_train_resampled)
y_pred_rf = rf.predict(X_test)
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train_resampled, y_train_resampled)
y_pred_xgb = xgb.predict(X_test)
print("\nXGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))

# Save best model
joblib.dump(best_model, "logistic_regression_model.pkl")
print("\nModel saved as 'logistic_regression_model.pkl'")


# # river_plastic_classification.py

# # Step 0: Install packages (for Colab, comment out if running locally)
# # !pip install --quiet shap imbalanced-learn xgboost seaborn joblib

# # Imports
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from sklearn.metrics import (
#     accuracy_score, precision_score, recall_score, classification_report,
#     confusion_matrix, roc_auc_score, RocCurveDisplay, precision_recall_curve
# )
# import matplotlib.pyplot as plt
# import seaborn as sns
# import joblib
# import shap
# from imblearn.over_sampling import SMOTE

# # Step 1: Load dataset
# file_id = "1zIk9JOdJEu9YF7Xuv2C8f2Q8ySfG3nHd"
# url = f"https://drive.google.com/uc?id={file_id}"
# df = pd.read_csv(url, thousands="'")

# # Step 2: Data Cleaning
# df.columns = df.columns.str.strip()
# df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
# df.replace({'': np.nan, 'NA': np.nan, 'na': np.nan, 'Na': np.nan}, inplace=True)

# percentage_cols = ["P[E] [%]", "Ratio Me/MPW"]
# for col in percentage_cols:
#     if col in df.columns:
#         df[col] = df[col].astype(str).str.replace("%", "", regex=False)
#         df[col] = pd.to_numeric(df[col], errors="coerce")

# for col in df.columns:
#     if df[col].dtype == "object" and col != "Country or Administrative area":
#         df[col] = df[col].astype(str).str.replace("'", "", regex=False)
#         df[col] = df[col].str.replace(",", "", regex=False)
#         df[col] = pd.to_numeric(df[col], errors="coerce")

# print("Null values before dropping NaNs:\n", df.isnull().sum())
# df = df.dropna()
# print(f"\nRows after dropping NaNs: {len(df)}")

# # Step 3: Target creation
# df["plastic_contribution"] = np.where(df["M[E] (metric tons year -1)"] > 6008, 0, 1)

# # Step 4: Exploratory insight
# country_contrib = df.groupby("Country or Administrative area")["M[E] (metric tons year -1)"].sum().sort_values(ascending=False)
# print("\nTop 5 countries contributing plastic to oceans:")
# print(country_contrib.head(5))

# plt.figure(figsize=(10,5))
# sns.barplot(x=country_contrib.head(5).values, y=country_contrib.head(5).index, palette="mako")
# plt.xlabel("Total Plastic Emission (metric tons per year)")
# plt.title("Top 5 Countries Contributing Plastic to Oceans")
# plt.tight_layout()
# plt.savefig("images/top_5_countries.png")
# plt.show()

# # Step 5: Features and target
# features = df.drop(columns=["plastic_contribution", "Country or Administrative area"])
# X = features
# y = df["plastic_contribution"]

# # Step 6: Scale features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Step 7: Train/test split
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# # Step 8: Handle imbalance with SMOTE
# smote = SMOTE(random_state=42)
# X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# # Step 9: Logistic Regression + GridSearchCV
# logreg = LogisticRegression(max_iter=1000)
# param_grid = {'C': [0.01, 0.1, 1, 10]}
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# grid_search = GridSearchCV(logreg, param_grid, cv=cv, scoring='f1', n_jobs=-1)
# grid_search.fit(X_train_resampled, y_train_resampled)
# best_model = grid_search.best_estimator_

# # Step 10: Evaluation
# y_pred = best_model.predict(X_test)
# y_pred_proba = best_model.predict_proba(X_test)[:,1]

# print("\n=== Model Evaluation (Logistic Regression) ===")
# print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
# print(f"Precision: {precision_score(y_test, y_pred):.4f}")
# print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
# print("\nClassification Report:\n", classification_report(y_test, y_pred))

# # Confusion matrix
# cm = confusion_matrix(y_test, y_pred)
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
# plt.title("Confusion Matrix")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.tight_layout()
# plt.savefig("images/confusion_matrix.png")
# plt.show()

# # ROC curve
# roc_auc = roc_auc_score(y_test, y_pred_proba)
# print(f"ROC AUC Score: {roc_auc:.4f}")
# RocCurveDisplay.from_estimator(best_model, X_test, y_test)
# plt.savefig("images/roc_curve.png")
# plt.show()

# # Precision-Recall curve
# precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
# plt.plot(recall, precision, marker='.')
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.title("Precision-Recall Curve")
# plt.tight_layout()
# plt.savefig("images/precision_recall.png")
# plt.show()

# # Step 11: Feature importance
# importance = np.abs(best_model.coef_[0])
# importance_df = pd.DataFrame({
#     "Feature": features.columns,
#     "Importance": importance
# }).sort_values(by="Importance", ascending=False)

# print("\nTop 10 Important Features:")
# print(importance_df.head(10))

# plt.figure(figsize=(10, 6))
# sns.barplot(x="Importance", y="Feature", data=importance_df.head(10), palette="viridis")
# plt.title("Top 10 Important Features")
# plt.tight_layout()
# plt.savefig("images/feature_importance.png")
# plt.show()

# # Step 12: SHAP explainability
# explainer = shap.LinearExplainer(best_model, X_train_resampled, feature_perturbation="interventional")
# shap_values = explainer.shap_values(X_test)
# print("\nGenerating SHAP summary plot...")
# shap.summary_plot(shap_values, X_test, feature_names=features.columns, show=False)
# plt.tight_layout()
# plt.savefig("images/shap_summary.png")
# plt.show()

# # Step 13: Advanced models
# rf = RandomForestClassifier(random_state=42)
# rf.fit(X_train_resampled, y_train_resampled)
# y_pred_rf = rf.predict(X_test)
# print("\nRandom Forest Classification Report:")
# print(classification_report(y_test, y_pred_rf))

# xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
# xgb.fit(X_train_resampled, y_train_resampled)
# y_pred_xgb = xgb.predict(X_test)
# print("\nXGBoost Classification Report:")
# print(classification_report(y_test, y_pred_xgb))

# # Save best model
# joblib.dump(best_model, "logistic_regression_model.pkl")
# print("\nModel saved as 'logistic_regression_model.pkl'")
