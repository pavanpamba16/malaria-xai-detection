"""
Master Training, Model Export, and Visual Plot Generation Script
"""
import os
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc
import shap
import lime
import lime.lime_tabular

from src.features import extract_clinical_features
from src.models import get_base_models, build_stacking_ensemble, build_voting_ensemble
from src.evaluate import evaluate_model_performance, get_paper_baseline_benchmarks
from src.explainability import ExplainabilitySuite

# Setup directories
os.makedirs('saved_models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# 1. Load Data & Engineer Features
print("[1/6] Loading dataset and engineering domain features...")
df_raw = pd.read_csv('malaria_dataset.csv')
df_engineered = extract_clinical_features(df_raw)

# 2. Resampling & Train-Test Split (70:30 matching the paper benchmark)
print("[2/6] Preparing balanced cohorts and train/test splits...")
class_0 = df_engineered[df_engineered['severe_maleria'] == 0]
class_1 = df_engineered[df_engineered['severe_maleria'] == 1]
class_1_over = class_1.sample(len(class_0), replace=True, random_state=42)
df_balanced = pd.concat([class_1_over, class_0], axis=0).reset_index(drop=True)

X = df_balanced.drop(columns=['severe_maleria'])
y = df_balanced['severe_maleria']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

scaler = RobustScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

# 3. Model Training
print("[3/6] Training base learners and Stacking Meta-Ensemble...")
base_models = get_base_models(random_state=123)
for name, model in base_models.items():
    model.fit(X_train_scaled, y_train)

stacking_model = build_stacking_ensemble(base_models, meta_c=1.5, random_state=42)
stacking_model.fit(X_train_scaled, y_train)

voting_model = build_voting_ensemble(base_models)
voting_model.fit(X_train_scaled, y_train)

# 4. Evaluation & Benchmark Compilation
print("[4/6] Evaluating metrics against the 2025 research paper...")
benchmark_results = get_paper_baseline_benchmarks()

for name, model in base_models.items():
    res = evaluate_model_performance(f"{name} (Our Enhanced)", model, X_test_scaled, y_test)
    benchmark_results.append(res)

stack_res = evaluate_model_performance("OUR Advanced Stacking Meta-Ensemble", stacking_model, X_test_scaled, y_test)
vote_res = evaluate_model_performance("OUR Weighted Soft-Voting Ensemble", voting_model, X_test_scaled, y_test)

benchmark_results.append(stack_res)
benchmark_results.append(vote_res)

df_results = pd.DataFrame(benchmark_results).sort_values(by='Accuracy', ascending=False).reset_index(drop=True)
df_results.to_csv('plots/benchmark_comparison.csv', index=False)

print("\n" + "="*100)
print("FINAL BENCHMARK COMPARISON TABLE (VS 2025 BMC PAPER)")
print("="*100)
# Clean display string
display_df = df_results.copy()
for col in ['Accuracy', 'ROC AUC', 'MCC', 'Balanced Accuracy', 'Cohen Kappa', 'Precision', 'Recall', 'F1 Score']:
    display_df[col] = display_df[col].apply(lambda v: f"{v:.4f}")
print(display_df.to_string(index=False))

# 5. Generate & Save High-Resolution Figures
print("[5/6] Generating publication-quality charts & XAI figures...")
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Figure 1: Accuracy Comparison Bar Plot
plt.figure(figsize=(12, 7))
plot_df = df_results.sort_values(by='Accuracy', ascending=True)
colors = ['#4CAF50' if 'OUR' in m else '#2196F3' if 'Enhanced' in m else '#9E9E9E' for m in plot_df['Model']]
bars = plt.barh(plot_df['Model'], plot_df['Accuracy'], color=colors, edgecolor='black', linewidth=0.8)
plt.axvline(x=0.8195, color='red', linestyle='--', linewidth=1.5, label='Paper Best (Awe et al. 2025: 81.95%)')
for bar in bars:
    w = bar.get_width()
    plt.text(w + 0.005, bar.get_y() + bar.get_height()/2, f"{w*100:.2f}%", va='center', fontweight='bold', fontsize=10)
plt.xlim(0.50, 0.95)
plt.xlabel('Diagnosis Accuracy Score', fontsize=12, fontweight='bold')
plt.title('Accuracy Performance Benchmark: Enhanced Stacking Pipeline vs Paper Models', fontsize=14, fontweight='bold', pad=15)
plt.legend(loc='lower right', frameon=True)
plt.tight_layout()
plt.savefig('plots/model_comparison_bar.png', dpi=300)
plt.close()

# Figure 2: ROC Curves
plt.figure(figsize=(10, 8))
# Plot Stacking
y_prob_stack = stacking_model.predict_proba(X_test_scaled)[:, 1]
fpr_s, tpr_s, _ = roc_curve(y_test, y_prob_stack)
plt.plot(fpr_s, tpr_s, color='#1b5e20', lw=2.5, label=f'OUR Stacking Ensemble (AUC = {auc(fpr_s, tpr_s):.3f})')

# Plot Extra Trees & CatBoost
for m_name, col in [('Extra Trees', '#1976d2'), ('CatBoost', '#e65100'), ('Random Forest', '#7b1fa2')]:
    model = base_models[m_name]
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, lw=1.8, color=col, label=f'{m_name} (AUC = {auc(fpr, tpr):.3f})')

# Baseline reference line
plt.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Random Chance (AUC = 0.500)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12, fontweight='bold')
plt.title('Receiver Operating Characteristic (ROC) Comparison Curves', fontsize=14, fontweight='bold', pad=15)
plt.legend(loc="lower right", fontsize=11, frameon=True)
plt.tight_layout()
plt.savefig('plots/roc_auc_curves.png', dpi=300)
plt.close()

# Figure 3: Confusion Matrix
plt.figure(figsize=(7, 6))
cm = confusion_matrix(y_test, stacking_model.predict(X_test_scaled))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Non-Severe (0)', 'Severe Malaria (1)'],
            yticklabels=['Non-Severe (0)', 'Severe Malaria (1)'],
            annot_kws={"size": 14, "weight": "bold"})
plt.ylabel('Actual Clinical Ground Truth', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Diagnosis by Stacking Model', fontsize=12, fontweight='bold')
plt.title(f'Confusion Matrix: Advanced Stacking Ensemble (Accuracy: {stack_res["Accuracy"]*100:.2f}%)', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('plots/confusion_matrix.png', dpi=300)
plt.close()

# 6. SHAP & XAI Computations
print("[6/6] Computing SHAP Global & Local Interpretability values...")
cb_model = base_models['CatBoost']
shap_explainer = shap.TreeExplainer(cb_model)
shap_vals = shap_explainer(X_test_scaled)

# SHAP Beeswarm Summary Plot
plt.figure(figsize=(11, 8))
shap.summary_plot(shap_vals, X_test_scaled, max_display=15, show=False)
plt.title('Global SHAP Feature Attribution (Impact on Severe Malaria Prediction)', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('plots/shap_summary_beeswarm.png', dpi=300, bbox_inches='tight')
plt.close()

# SHAP Mean Absolute Bar Plot
plt.figure(figsize=(10, 7))
shap.plots.bar(shap_vals, max_display=15, show=False)
plt.title('Mean |SHAP Value| (Global Feature Importance Ranking)', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('plots/shap_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# Save models & metadata
joblib.dump(stacking_model, 'saved_models/malaria_stacking_model.joblib')
joblib.dump(base_models, 'saved_models/base_models.joblib')
joblib.dump(scaler, 'saved_models/scaler.joblib')
with open('saved_models/feature_columns.json', 'w') as f:
    json.dump(list(X.columns), f)

print("\nSUCCESS! All models trained, exported, and publication figures generated in /plots directory.")
