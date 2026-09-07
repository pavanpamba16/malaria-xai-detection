# 🦟 Explainable AI Malaria Severity Diagnosis & Clinical Decision Support
### **Final Year Capstone Project**
*Based on & Outperforming: Awe et al. (2025) - BMC Medical Informatics and Decision Making (25:162)*

---

## 🌟 Key Highlights
- **Accuracy Improvement**: **86.47%** vs **81.95%** in the 2025 paper (+4.52% boost).
- **Precision Gain**: **89.55%** vs **83.10%** in the 2025 paper (+6.45% boost).
- **Matthews Correlation Coefficient (MCC)**: **0.7305** vs **0.6374** in the paper.
- **Explainable AI**: SHAP (Beeswarm, Bar, Waterfall), LIME, and Counterfactual ("What-If") Clinical Reasoning.
- **Interactive Web App**: Complete Streamlit dashboard for real-time patient diagnosis and visual triage.

---

## 🚀 Quickstart Guide

### 1. Run Master Pipeline (Trains all models & generates plots)
```bash
python train_and_export.py
```

### 2. Launch Interactive Clinical Web Application
```bash
streamlit run app.py
```

### 3. Open Complete Research Notebook
```bash
jupyter notebook Malaria_XAI_Final_Year_Project.ipynb
```

---

## 📊 Benchmark Comparison vs 2025 Paper

| Model | Accuracy | ROC-AUC | MCC | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **★ OUR Stacking Meta-Ensemble** | **86.47%** | **0.8871** | **0.7305** | **89.55%** | **84.51%** | **86.96%** |
| **★ Extra Trees (Enhanced)** | 84.21% | 0.8898 | 0.6832 | 85.71% | 84.51% | 85.11% |
| **★ Random Forest (Enhanced)** | 83.46% | 0.8878 | 0.6673 | 83.56% | 85.92% | 84.72% |
| **Paper Best (Awe et al. 2025)** | **81.95%** | **0.8696** | **0.6374** | **83.10%** | **83.10%** | **83.10%** |
| **★ CatBoost (Enhanced)** | 81.20% | 0.8930 | 0.6222 | 80.26% | 85.92% | 82.99% |
| **Paper: Gradient Boosting (2025)** | 72.93% | 0.8092 | 0.4559 | 71.60% | 81.69% | 76.30% |
| **Paper: CatBoost (2025)** | 72.93% | 0.7876 | 0.4551 | 72.15% | 80.28% | 76.00% |
| **Paper: XGBoost (2025)** | 69.17% | 0.7703 | 0.3812 | 67.44% | 81.69% | 73.80% |
| **Paper: AdaBoost (2025)** | 57.89% | 0.6336 | 0.1558 | 60.87% | 59.15% | 60.00% |

---
*For detailed methodology, mathematical formulations, and clinical analysis, see [PROJECT_REPORT.md](file:///c:/Users/pamba/OneDrive/Desktop/Mani_05/PROJECT_REPORT.md).*
