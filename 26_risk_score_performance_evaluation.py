

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

file_path = r"D:/USveri/Risk Skor/19OZELLİKSİLİNDİ2.12US_xgboost_selected_features_fcm_with_riskscore.xlsx"
df = pd.read_excel(file_path)

target_col = "Risk_Score_norm"
drop_cols = ["Vehicle_ID"]

y = df[target_col]
feature_cols = [col for col in df.columns if col not in drop_cols + [target_col]]
X = df[feature_cols]

# numerik dönüşüm ve eksik değer temizliği
X = X.apply(pd.to_numeric, errors='coerce')
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.mean())

# Eğitim Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Modeller
models = {
    "XGBoost": XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    ),
    "RandomForest": RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),
    "ExtraTrees": ExtraTreesRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
}

#  Eğitim – Tahmin – Metrikler – Özellik Önemi
results = []  
feature_importances_dict = {}  

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2_test": r2
    })

    #  özellik önem dereceleri 
    if hasattr(model, "feature_importances_"):
        feature_importances_dict[name] = model.feature_importances_
    else:
               feature_importances_dict[name] = np.zeros(len(feature_cols))
    if name == "XGBoost":
        plt.figure(figsize=(7, 6))
        plt.scatter(y_test, y_pred, alpha=0.35)
        # y = x referans çizgisi
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)
        plt.title(f"Actual vs Predicted Risk Score ({name})")
        plt.xlabel("Actual Risk Score")
        plt.ylabel("Predicted Risk Score")
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend()
        plt.show()
# Sonuçları DataFrame'e Çevrilmesi (Metrikler ve Özellik Önemi)
# Model performans tablosu
metrics_df = pd.DataFrame(results)
# Özellik önem derecelerini tek tabloda toplama
fi_all = pd.DataFrame({"Feature": feature_cols})
for name, importances in feature_importances_dict.items():
    fi_all[f"{name}_importance"] = importances

fi_xgb = fi_all[["Feature", "XGBoost_importance"]].sort_values(
    by="XGBoost_importance", ascending=False
)
fi_rf = fi_all[["Feature", "RandomForest_importance"]].sort_values(
    by="RandomForest_importance", ascending=False
)
fi_et = fi_all[["Feature", "ExtraTrees_importance"]].sort_values(
    by="ExtraTrees_importance", ascending=False
)
output_path = r"D:/USveri/makalerevision/feature_importances_riskscore_regression.xlsx"

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    # Model performans metrikleri
    metrics_df.to_excel(writer, sheet_name="Model_Performance", index=False)
    
    # Tüm modellerin önem dereceleri tek tablodada birleştirilmesi
    fi_all_sorted.to_excel(writer, sheet_name="Feature_Import_All", index=False)
    
    # Her model için ayrı sayfada oluşturulması
    fi_xgb.to_excel(writer, sheet_name="XGBoost_Feature_Imp", index=False)
    fi_rf.to_excel(writer, sheet_name="RF_Feature_Imp", index=False)
    fi_et.to_excel(writer, sheet_name="ET_Feature_Imp", index=False)

print("Metrikler ve özellik önem dereceleri Excel dosyasına yazıldı:")
print(output_path)
