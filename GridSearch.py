
import pandas as pd
import numpy as np

from sklearn.ensemble import ExtraTreesClassifier
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

#Extra Trees ile feature selection
et_fs = ExtraTreesClassifier(
    n_estimators=800,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
    bootstrap=False
)

et_fs.fit(X_train, y_train)

# Threshold: importance > 0.003 
selector = SelectFromModel(et_fs, threshold=0.004, prefit=True)

X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)
selected_cols = X.columns[selector.get_support()]

print(f"\ExtraTrees ile seçilen önemli özellik sayısı (importance > 0.003): {len(selected_cols)}")
print("İlk 20 özellik:", list(selected_cols[:20]))

# Extra Trees hiperparametre optimizasyonu (seçilmiş özelliklerle)
et_base = ExtraTreesClassifier(random_state=42, n_jobs=-1)

param_distributions = {
    "n_estimators": [300, 500, 800, 1200, 1600],
    "max_depth": [None, 10, 20, 30, 40, 60],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4, 5],
    "max_features": ["sqrt", "log2", None],
    "bootstrap": [False, True],
    "class_weight": [None, "balanced"],
    "criterion": ["gini", "entropy", "log_loss"],
    "max_leaf_nodes": [None, 50, 100, 200],
    "min_impurity_decrease": [0.0, 1e-4, 1e-3],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

et_search = RandomizedSearchCV(
    estimator=et_base,
    param_distributions=param_distributions,
    n_iter=30,           
    scoring="f1_macro",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

et_search.fit(X_train_selected, y_train)

print("\ExtraTrees Best Params:")
print(et_search.best_params_)
print(f"Best CV F1-macro: {et_search.best_score_:.4f}")

best_et = et_search.best_estimator_

# Test performansı
y_pred = best_et.predict(X_test_selected)

print("\n=== Test Verisi Performansı (ExtraTrees | Tuned | ET-Selected Features) ===")
print(classification_report(y_test, y_pred, digits=3, zero_division=0))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1-score (macro):", f1_score(y_test, y_pred, average="macro"))

# ROC-AUC 
try:
    y_proba = best_et.predict_proba(X_test_selected)
    auc_score = roc_auc_score(y_test, y_proba, multi_class="ovr")
    print("ROC AUC (ovr):", auc_score)
except Exception as e:
    print(" ROC AUC hesaplanamadı:", e)

# 5 katlı çapraz doğrulama
X_selected_all = selector.transform(X)
cv_score = cross_val_score(best_et, X_selected_all, y, cv=cv, scoring="f1_macro")
print(f"\ Katlı Çapraz Doğrulama F1 (macro) Ortalaması (ExtraTrees tuned): {cv_score.mean():.4f}")

# Seçilmiş veriyi kaydet
selected_df = pd.concat(
    [
        df[[id_col, target_col]].reset_index(drop=True),
        pd.DataFrame(X_selected_all, columns=selected_cols),
    ],
    axis=1
)
selected_df.to_excel("D:/USveri/makalerevision/US_ET_selected_features_fcm.xlsx", index=False)

# Seçilen feature adlarını ayrı kaydet
pd.Series(selected_cols).to_excel(
    "D:/USveri/makalerevision/US_ET_selected_feature_names_fcm.xlsx",
    index=False,
    header=["Selected Features"],
)
print("\ExtraTrees feature selection + ExtraTrees hyperparameter optimization + evaluation tamamlandı.")
