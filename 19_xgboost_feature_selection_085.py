
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

# Tüm veriyi numeriğe çevir ve eksik/sorunlu değerleri temizle
X = X.apply(pd.to_numeric, errors='coerce')
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.mean())

# Veriyi eğitim/test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Özellik seçimi: yalnızca eğitim verisine uygulanır
model_fs = XGBClassifier(eval_metric='mlogloss', random_state=42)
model_fs.fit(X_train, y_train)

# Önemli özellikleri seç
selector = SelectFromModel(model_fs, threshold=0.003, prefit=True)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)
selected_cols = X.columns[selector.get_support()]
print(f"\n Seçilen önemli özellik sayısı: {len(selected_cols)}")

# Modeli yeniden eğit ve testte tahmin yap
model_final = XGBClassifier(eval_metric='mlogloss', random_state=42)
model_final.fit(X_train_selected, y_train)
y_pred = model_final.predict(X_test_selected)
y_proba = model_final.predict_proba(X_test_selected)

# Performans metrikleri
print("\n=== Test Verisi Performansı ===")
print(classification_report(y_test, y_pred, digits=3, zero_division=0))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1-score (macro):", f1_score(y_test, y_pred, average='macro'))
87
# ROC AUC skoru
try:
    auc_score = roc_auc_score(y_test, y_proba, multi_class='ovr')
    print("ROC AUC (ovr):", auc_score)
except:
    print("ROC AUC çok sınıflı yapı için hesaplanamadı.")

# 5 katlı çapraz doğrulama (özellik seçilmiş veride)
X_selected_all = selector.transform(X)
cv_score = cross_val_score(model_final, X_selected_all, y, cv=5, scoring='f1_macro')
print(f"\5 Katlı Çapraz Doğrulama F1 (macro) Ortalaması: {cv_score.mean():.4f}")

# seçilen veriyi ve özellikleri kaydet
selected_df = pd.concat(
    [df[['Vehicle_ID', target_col]].reset_index(drop=True),
     pd.DataFrame(X_selected_all, columns=selected_cols)],
    axis=1
)
selected_df.to_excel("D:/USveri/makalerevision/ilkxgboost_selected_features_fcm.xlsx", index=False)

# Seçilen özellik adlarını ayrı dosyaya yaz 
pd.Series(selected_cols).to_excel(
    "D:/USveri/makalerevision/ilkUS_xgboost_selected_feature_names_fcm.xlsx",
    index=False,
    header=["Selected Features"]
)
print("\ Özellik seçimi, modelleme ve sonuçların kaydı başarıyla tamamlandı.")

