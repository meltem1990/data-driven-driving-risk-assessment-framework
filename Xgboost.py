import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score

df = pd.read_excel("D:/USveri/makalerevision/US_XGB_selected_features_fcm.xlsx")
df.columns = df.columns.str.strip()

target_col = "Risk_Label_5"

X = df.drop(columns=["Vehicle_ID", target_col])
y = df[target_col]

X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.mean(numeric_only=True))

y = pd.to_numeric(y, errors="coerce").astype(int)

# Eğitim/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# XGBoost optimize edilmiş parametreler
xgb_params = dict(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.7,
    colsample_bytree=0.8,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

model = XGBClassifier(**xgb_params)
model.fit(X_train, y_train)

# Tahmin
y_pred = model.predict(X_test)

# Performans
print(classification_report(y_test, y_pred, digits=3, zero_division=0))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1-macro:", f1_score(y_test, y_pred, average="macro"))
