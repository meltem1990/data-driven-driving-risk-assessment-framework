
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score

df = pd.read_excel("d:/USveri/r2.11SAFE_1000_OTHERS_UNTOUCHED.xlsx")

target_col = 'Risk_Label_5'
feature_cols = [col for col in df.columns if col != target_col and col != 'Vehicle_ID']

X = df[feature_cols]
y = df[target_col]

X = X.apply(pd.to_numeric, errors='coerce')
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.mean())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# XGBoost hiperparametre ayarları 
xgb_params = dict(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.7,
    colsample_bytree=0.8,
    min_child_weight=1,
    gamma=0,
    reg_alpha=0,
    reg_lambda=1,
    
    # Diğer sabit parametreler
    objective='multi:softprob',
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1
)

#  Özellik seçimi (XGBoost ile)
model_fs = XGBClassifier(**xgb_params)
model_fs.fit(X_train, y_train)

# Özellik önemine göre seçim yap
selector = SelectFromModel(model_fs, threshold=0.004, prefit=True)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)
selected_cols = X.columns[selector.get_support()]

print(f"\Seçilen önemli özellik sayısı: **{len(selected_cols)}**")

# Nihai modeli eğit (Seçilmiş özelliklerle)
model_final = XGBClassifier(**xgb_params)
model_final.fit(X_train_selected, y_train)

# Tahmin ve metrikler
y_pred = model_final.predict(X_test_selected)
y_proba = model_final.predict_proba(X_test_selected)

print("\n=== Test Verisi Performansı (XGBoost) ===")
print(classification_report(y_test, y_pred, digits=3, zero_division=0))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1-score (macro):", f1_score(y_test, y_pred, average='macro'))

# ROC AUC (çok sınıflı)
try:

       auc_score = roc_auc_score(y_test, y_proba, multi_class='ovr')
    print("ROC AUC (ovr):", auc_score)
except:
    print("ROC AUC çok sınıflı yapı için hesaplanamadı.")

#  5 Katlı Çapraz Doğrulama (Seçilmiş özelliklerle, AYNI MODEL)
X_selected_all = selector.transform(X)
cv_model = XGBClassifier(**xgb_params)
cv_score = cross_val_score(cv_model, X_selected_all, y, cv=5, scoring='f1_macro')
print(f"\n5 Katlı Çapraz Doğrulama F1 (macro) Ortalaması: **{cv_score.mean():.4f}**")


# Seçilen veriyi kaydet (Tüm veri seti üzerinde)
selected_df = pd.concat(
    [df[['Vehicle_ID', target_col]].reset_index(drop=True),
     pd.DataFrame(X_selected_all, columns=selected_cols)],
    axis=1
)
selected_df.to_excel("D:/USveri/makalerevision/US_XGB_selected_features_fcm.xlsx", index=False)

# Seçilen özellik adlarını ayrı kaydet
pd.Series(selected_cols).to_excel(
    "D:/USveri/makalerevision/US__XGB_selected_feature_names_fcm.xlsx",
    index=False,
    header=["Selected Features"]
)

print("\XGBoost ile özellik seçimi, modelleme ve kayıt **başarıyla ve tek seferde** tamamlandı.")
