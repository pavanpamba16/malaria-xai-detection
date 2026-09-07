# 🦟 Explainable AI for Enhanced Malaria Diagnosis & Severity Prediction
## **Final Year Capstone Project Report & Thesis Documentation**

---

### **1. Executive Summary**
* **Project Title**: Explainable AI for Enhanced Accuracy in Malaria Diagnosis Using Multi-Tiered Ensemble Machine Learning and Domain Feature Engineering.
* **Benchmark Study**: *Awe et al. (2025), BMC Medical Informatics and Decision Making, 25:162*.
* **Primary Achievements**:
  * **Accuracy Enhancement**: Elevated diagnostic accuracy from **81.95%** (Paper Best) to **86.47%** (**+4.52% absolute gain**).
  * **Precision & Reliability**: Boosted precision to **89.55%** (vs 83.10% in paper) and Matthews Correlation Coefficient (MCC) to **0.7305** (vs 0.6374 in paper).
  * **Domain Feature Engineering**: Formulated clinical synergy markers: *Organ Damage Score, Cerebral Malaria Index, Febrile Syndrome Score, and Age Vulnerability Interaction*.
  * **Comprehensive XAI Integration**: Implemented Global SHAP (Beeswarm, Summary Bar), Patient-Level Local SHAP (Waterfall, Force), LIME feature attribution, and Prescriptive Counterfactual ("What-If") Clinical Reasoning.
  * **Clinical Decision Support System**: Developed an interactive point-of-care diagnosis dashboard in Streamlit (`app.py`).

---

### **2. Benchmark Performance Comparison**

The table below demonstrates our system's performance evaluated under the exact 70:30 stratified split protocol compared against all published models in the 2025 BMC research paper:

| Model Architecture | Accuracy | ROC-AUC | MCC | Balanced Acc | Cohen's Kappa | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **★ OUR Stacking Meta-Ensemble** | **86.47%** | **0.8871** | **0.7305** | **86.61%** | **0.7292** | **89.55%** | **84.51%** | **86.96%** |
| **★ Extra Trees (Enhanced)** | 84.21% | 0.8898 | 0.6832 | 84.19% | 0.6831 | 85.71% | 84.51% | 85.11% |
| **★ Random Forest (Enhanced)** | 83.46% | 0.8878 | 0.6673 | 83.28% | 0.6670 | 83.56% | 85.92% | 84.72% |
| **Paper Best: Tuned RF (Awe et al. 2025)** | **81.95%** | **0.8696** | **0.6374** | **81.87%** | **0.6374** | **83.10%** | **83.10%** | **83.10%** |
| **★ CatBoost (Enhanced)** | 81.20% | 0.8930 | 0.6222 | 0.8086 | 0.6204 | 80.26% | 85.92% | 82.99% |
| **★ Weighted Soft-Voting Ensemble** | 81.20% | 0.8932 | 0.6222 | 0.8086 | 0.6204 | 80.26% | 85.92% | 82.99% |
| **Paper: Gradient Boosting (2025)** | 72.93% | 0.8092 | 0.4559 | 72.30% | 0.4505 | 71.60% | 81.69% | 76.30% |
| **Paper: CatBoost (2025)** | 72.93% | 0.7876 | 0.4551 | 72.40% | 0.4517 | 72.15% | 80.28% | 76.00% |
| **Paper: XGBoost (2025)** | 69.17% | 0.7703 | 0.3812 | 68.26% | 0.3710 | 67.44% | 81.69% | 73.80% |
| **Paper: AdaBoost (2025)** | 57.89% | 0.6336 | 0.1558 | 57.80% | 0.1557 | 60.87% | 59.15% | 60.00% |

---

### **3. Key Methodological Innovations**

#### **A. Domain-Guided Clinical Feature Engineering**
While the original paper solely passed raw binary indicators into models, clinical malaria manifestations are governed by multi-organ pathology. We formulated:
1. **Organ Damage Index**: Combining Jaundice, Coca-Cola Urine (hemoglobinuria), Hypoglycemia, and Anemia.
2. **Cerebral Malaria Score**: Combining Convulsions, Prostration, and Hyperpyrexia (>39°C).
3. **Gastrointestinal & Febrile Syndrome Indices**: Quantifying systemic inflammatory load.
4. **Age-Stratified Vulnerability**: Accounting for physiological vulnerability in children under 5 and older adults.

#### **B. Multi-Level Stacking Meta-Architecture**
Combining heterogeneous model paradigms:
- **Tree Ensembles**: Random Forest, Extra Trees
- **Gradient Boosters**: CatBoost, XGBoost, LightGBM, Gradient Boosting
- **Meta-Learner**: Calibrated L2 Logistic Regression on out-of-fold probability vectors.

#### **C. Explainable AI (XAI) Suite**
- **SHAP (SHapley Additive exPlanations)**: Calculates game-theoretic feature attributions for both global cohort trends and local patient diagnoses.
- **LIME (Local Interpretable Model-agnostic Explanations)**: Builds surrogate linear models around individual patient predictions.
- **Counterfactual "What-If" Engine**: Calculates the exact symptom resolutions needed to lower a patient's risk category.

---

### **4. Project Directory Structure**

```
Mani_05/
├── app.py                                  # Interactive Streamlit Clinical Dashboard
├── train_and_export.py                     # Master training, export & figure generation
├── Malaria_XAI_Final_Year_Project.ipynb    # Comprehensive Jupyter Notebook for thesis defense
├── malaria_dataset.csv                     # Original clinical patient dataset (337 records)
├── PROJECT_REPORT.md                       # Complete documentation & thesis chapter writeup
├── src/
│   ├── features.py                         # Clinical domain feature engineering
│   ├── models.py                           # Base models, Stacking & Voting architectures
│   ├── explainability.py                   # SHAP, LIME, and Counterfactual What-If modules
│   └── evaluate.py                         # Metric calculation & paper benchmark scoring
├── saved_models/
│   ├── malaria_stacking_model.joblib       # Trained Stacking Meta-Ensemble model
│   ├── base_models.joblib                  # Trained base estimators
│   ├── scaler.joblib                       # Fitted RobustScaler
│   └── feature_columns.json                # Feature schema definitions
└── plots/
    ├── model_comparison_bar.png            # Accuracy comparison chart vs paper
    ├── roc_auc_curves.png                  # Multi-model ROC-AUC curves
    ├── confusion_matrix.png                # Stacking ensemble confusion matrix
    ├── shap_summary_beeswarm.png           # Global SHAP beeswarm plot
    ├── shap_feature_importance.png         # Mean |SHAP| feature ranking
    └── benchmark_comparison.csv            # Tabulated metric comparisons
```

---

### **5. How to Run the Project**

1. **Retrain Models & Re-generate Publication Figures**:
   ```bash
   python train_and_export.py
   ```

2. **Launch the Interactive Web Application**:
   ```bash
   streamlit run app.py
   ```

3. **Open the Jupyter Notebook**:
   ```bash
   jupyter notebook Malaria_XAI_Final_Year_Project.ipynb
   ```
