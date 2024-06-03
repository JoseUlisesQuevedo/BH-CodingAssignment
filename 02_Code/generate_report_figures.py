import pandas as pd
import matplotlib.pyplot as plt


results = pd.read_csv("03_Results/result_report.csv")


df_xgboost = results[results['Model'] == 'XGBoost']
df_lr = results[results['Model'] == 'Logistic Regression']
metrics = ['Precision', 'Recall', 'F1 Score', 'ROC AUC']

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Plot each metric in a separate subplot
for i, metric in enumerate(metrics):
    axes[i].plot(df_xgboost['N'], df_xgboost[metric], marker='o', label='XGBoost')
    axes[i].plot(df_lr['N'], df_lr[metric], marker='o', label='Logistic Regression')
    axes[i].set_xlabel('N')
    axes[i].set_ylabel(metric)
    axes[i].set_title(f'{metric} vs N')
    axes[i].legend()
    axes[i].grid(True)

# Adjust layout
plt.tight_layout()
plt.savefig("03_Results/metrics_vs_N.png")