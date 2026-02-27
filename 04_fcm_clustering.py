
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from xgboost import XGBClassifier
import skfuzzy as fuzz

# FCM Fonksiyonu, data: : Kümeleme yapacağımız tüm veriyi içeren DataFrame.
def apply_fcm(data, n_clusters, feature_cols):
    scaler = StandardScaler()
    X = scaler.fit_transform(data[feature_cols])
    X_fcm = X.T  
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(X_fcm, c=n_clusters, m=2, error=0.0001, maxiter=1000)
    labels = np.argmax(u, axis=0)
    return labels, u
def plot_cpi_vs_tit(df, tit_cols, cpi_col, label_col, n_clusters):
    for tit in tit_cols:
        plt.figure()
        for label in range(n_clusters):
            subset = df[df[label_col] == label]
            plt.scatter(subset[tit], subset[cpi_col], label=f'Cluster {label}', alpha=0.8)  
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(0.01, 10)
        plt.ylim(1e-3, 1)
        plt.xticks([0.01, 0.1, 1, 10], labels=['0.01', '0.1', '1', '10'])
        plt.yticks([1e-3, 0.01, 0.1, 1], labels=['0.001', '0.01', '0.1', '1'])
        plt.xlabel(tit)
        plt.ylabel(cpi_col)
        plt.title(f'{cpi_col} vs {tit} (Clusters: {n_clusters})')
        plt.legend()
        plt.grid(True, which="both", linestyle='--', linewidth=0.5)
        plt.show()

# Sınıflandırma ve Metrik Değerlendirme
def evaluate_clustering(data, feature_cols, label_col):
    X = data[feature_cols]
    y = data[label_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    results = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred, average='weighted'),
        'AUC': roc_auc_score(y_test, y_prob, multi_class='ovr'),
        'AUPRC': average_precision_score(y_test, y_prob, average='weighted')
    }
    return results

#  Veri Okuma
df = pd.read_excel("D:\\USveri\\YENİMADR13_942_VehicleSummary.xlsx")

feature_cols = ['TIT_t1', 'TIT_t2', 'TIT_t3', 'CPI_MADR1', 'CPI_MADR2']
tit_cols = ['TIT_t1', 'TIT_t2', 'TIT_t3']
cpi_cols = ['CPI_MADR1', 'CPI_MADR2']

#  Kümeleme ve Temel Sınıflandırma
results_all = []
cluster_details = []

for n_clusters in [4, 5, 6]:
    label_col = f'FCM_Label_{n_clusters}'  
    labels, u = apply_fcm(df, n_clusters, feature_cols)
    df[label_col] = labels
     cluster_means = df.groupby(label_col)[['CPI_MADR1', 'TIT_t3']].mean()
    cluster_means['risk_score'] = cluster_means['CPI_MADR1'] + cluster_means['TIT_t3']
    sorted_clusters = cluster_means.sort_values('risk_score').index.tolist()

    cluster_to_risk_numeric = {cl: i for i, cl in enumerate(sorted_clusters)}
    df[f'Risk_Label_{n_clusters}'] = df[label_col].map(cluster_to_risk_numeric)

    risk_names = ['R0', 'R1', 'R2', 'R3', 'R4', 'R5'][:n_clusters]

    # Grafikler
    for cpi in cpi_cols:
        plot_cpi_vs_tit(df, tit_cols, cpi, label_col, n_clusters)

    # Değerlendirme metrikleri
    scores = evaluate_clustering(df, feature_cols, label_col)
    scores['Clusters'] = n_clusters
    results_all.append(scores)
    risk_counts = df[f'Risk_Label_{n_clusters}'].value_counts()
    row_dict = {'Clusters': n_clusters}
    for i in range(n_clusters):
        col_name = risk_header_mapping[i]  
        row_dict[col_name] = risk_counts.get(i, 0)  

    row_dict.update(scores) 
    cluster_details.append(row_dict)

summary_df = pd.DataFrame(cluster_details)


ordered_cols = ['Clusters'] + [risk_header_mapping[i] for i in range(6) if i in risk_header_mapping] + ['Accuracy', 'F1', 'AUC', 'AUPRC']
ordered_cols = [col for col in ordered_cols if col in summary_df.columns]
summary_df = summary_df[ordered_cols]

#  Kaydet 
print(summary_df)
summary_df.to_csv("kumeleme_ozet_4_5_6.csv", index=False)
summary_df.to_excel("D:/USveri/TTC_TIT_CPI_ALLSEGMENTS.xlsx", index=False)
# En yüksek doğruluğa sahip kümeleme sonuçlarını al
# summary_df'yiF1 değerine göre sırala
best_result = summary_df.sort_values(by='F1', ascending=False).iloc[0]
best_cluster_count = int(best_result['Clusters'])
best_label_col = f'Risk_Label_{best_cluster_count}'

# İlgili araç numarası ve etiket bilgisini al
if 'Vehicle_ID' in df.columns:
    output_df = df[['Vehicle_ID', best_label_col]].copy()
else:
    output_df = df[[best_label_col]].copy()
    output_df['Vehicle_Index'] = df.index  

output_path = f"D:/USveri/us_arac_kume_bilgileri_{best_cluster_count}_kume.xlsx"
output_df.to_excel(output_path, index=False)
print(f"\nAraç küme bilgileri dosyaya kaydedildi: {output_path}")

