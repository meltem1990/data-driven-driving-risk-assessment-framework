import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Veri setini
df = pd.read_excel(r"D:\USveri\Risk Skor\Grafik19aracrisksınıflarıduzelmisnormalize_RISK_SCORE - Kopya.xlsx")

# Her sınıf için ayrı renk
colors = {0:'#4daf4a', 1:'#377eb8', 2:'#ff7f00', 3:'#e41a1c'}

plt.figure(figsize=(10, 6))

#Tüm veri
for label in sorted(df['Risk_Label_5'].unique()):
    subset = df[df['Risk_Label_5'] == label]
    
    x_jitter = label + (np.random.rand(len(subset)) - 0.5) * 0.30
    
    plt.scatter(
        x_jitter,
        subset['Risk_Score_norm'],
        s=6,                  
        alpha=0.7,
        color=colors[label],
        label=f"R{label}"
    )

plt.xlabel("Risk Label (0–3)")
plt.ylabel("The Driving Risk Score (0–1)")
plt.title("")
plt.grid(True, linestyle='--', alpha=0.3)

plt.xticks([0, 1, 2, 3])

plt.legend(title="Classes")
plt.show()
