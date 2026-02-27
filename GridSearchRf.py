
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score

df = pd.read_excel("d:/USveri/r2.11SAFE_1000_OTHERS_UNTOUCHED.xlsx")
df.columns = df.columns.str.strip()
target_col = "Risk_Label_5"
id_col = "Vehicle_ID"
feature_cols = [c for c in df.columns if c not in [target_col, id_col]]

X = df[feature_cols].copy()
y = df[target_col].copy()

X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.mean(numeric_only=True))

y = pd.to_numeric(y, errors="coerce")
if y.isna().any():
    raise ValueError(f"Target içinde NaN var: {y.isna().sum()} satır.")
y = y.astype(int)

# Eğitim/test 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Random Forest ile feature selection(sadece train üzerinde)
rf_fs = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"  
)

rf_fs.fit(X_train, y_train)

# Threshold: importance > 0.02 
selector = SelectFromModel(rf_fs, threshold=0.003, prefit=True)

X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)
selected_cols = X.columns[selector.get_support()]

print(f"\n RF ile seçilen önemli özellik sayısı (importance > 0.02): {len(selected_cols)}")
print("İlk 20 özellik:", list(selected_cols[:20]))

# 6) Random Forest hiperparametre optimizasyonu (seçilmiş özelliklerle)
rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)

param_distributions = {
    "n_estimators": [300, 500, 800, 1200],
    "max_depth": [None, 10, 20, 30, 40],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4, 5],
    "max_features": ["sqrt", "log2", None],
    "bootstrap": [True, False],
    "class_weight": [None, "balanced"]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf_search = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=param_distributions,
    n_iter=30,           
    scoring="f1_macro",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rf_search.fit(X_train_selected, y_train)

print("\RF Best Params:")
print(rf_search.best_params_)
print(f" Best CV F1-macro: {rf_search.best_score_:.4f}")

best_rf = rf_search.best_estimator_

#  Test performansı (RF tuned | selected features)
y_pred = best_rf.predict(X_test_selected)

print("\n=== Test Verisi Performansı (RF | Tuned | RF-Selected Features) ===")
print(classification_report(y_test, y_pred, digits=3, zero_division=0))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1-score (macro):", f1_score(y_test, y_pred, average="macro"))

# ROC-AUC 
try:
    y_proba = best_rf.predict_proba(X_test_selected)
    auc_score = roc_auc_score(y_test, y_proba, multi_class="ovr")
    print("ROC AUC (ovr):", auc_score)
except Exception as e:
    print(" ROC AUC hesaplanamadı:", e)

#  5 katlı çapraz doğrulama
X_selected_all = selector.transform(X)
cv_score = cross_val_score(best_rf, X_selected_all, y, cv=cv, scoring="f1_macro")
print(f"\n5 Katlı Çapraz Doğrulama F1 (macro) Ortalaması (RF tuned): {cv_score.mean():.4f}")

#  Seçilmiş veriyi kaydet
selected_df = pd.concat(
    [
        df[[id_col, target_col]].reset_index(drop=True),
        pd.DataFrame(X_selected_all, columns=selected_cols),
    ],
    axis=1
)
selected_df.to_excel("D:/USveri/makalerevision/US_RF_selected_features_fcm.xlsx", index=False)

# Seçilen feature adlarını ayrı kaydet
pd.Series(selected_cols).to_excel(
    "D:/USveri/makalerevision/US_RF_selected_feature_names_fcm.xlsx",
    index=False,
    header=["Selected Features"],
)

print("\ RF feature selection + RF hyperparameter optimization + evaluation tamamlandı.")
