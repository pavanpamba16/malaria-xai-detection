import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, matthews_corrcoef, 
    balanced_accuracy_score, precision_score, recall_score, cohen_kappa_score,
    confusion_matrix, classification_report
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, 
    StackingClassifier, VotingClassifier, HistGradientBoostingClassifier, AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import optuna
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 1. Load Data
df = pd.read_csv('malaria_dataset.csv')
print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# Drop sex if needed, or keep for feature engineering
# Let's create feature engineering function
def add_clinical_features(data):
    d = data.copy()
    
    # Domain Knowledge Clinical Indices
    # Severe Organ Damage / Hemoglobinuria marker
    d['organ_damage_score'] = d['jundice'] + d['cocacola_urine'] + d['hypoglycemia'] + d['Anemia']
    # Neurological / Cerebral Malaria marker
    d['cerebral_risk_score'] = d['Convulsion'] * 2.0 + d['prostraction'] * 1.5 + d['hyperpyrexia'] * 1.5
    # High Fever & Rigor combo
    d['febrile_syndrome'] = d['fever'] + d['rigor'] + d['cold'] + d['headace']
    # GI complication
    d['gi_syndrome'] = d['vomitting'] + d['diarrhea'] + d['bitter_tongue']
    # Total active symptoms
    symptom_cols = ['fever', 'cold', 'rigor', 'fatigue', 'headace', 'bitter_tongue', 
                    'vomitting', 'diarrhea', 'Convulsion', 'Anemia', 'jundice', 
                    'cocacola_urine', 'hypoglycemia', 'prostraction', 'hyperpyrexia']
    d['total_symptoms'] = d[symptom_cols].sum(axis=1)
    
    # Weighted Medical Severity Index (MSI)
    d['msi_score'] = (
        d['organ_damage_score'] * 3.0 + 
        d['cerebral_risk_score'] * 2.5 + 
        d['gi_syndrome'] * 1.5 + 
        d['febrile_syndrome'] * 1.0
    )
    
    # Age Vulnerability Interaction
    d['age_under_5'] = (d['age'] <= 5).astype(int)
    d['age_elderly'] = (d['age'] >= 55).astype(int)
    d['age_vulnerability'] = d['age_under_5'] + d['age_elderly']
    d['age_msi_interaction'] = d['age'] * d['msi_score'] / 10.0
    
    # Key non-linear pairwise clinical interactions
    d['convulsion_hyperpyrexia'] = d['Convulsion'] * d['hyperpyrexia']
    d['urine_jaundice'] = d['cocacola_urine'] * d['jundice']
    d['prostration_hypoglycemia'] = d['prostraction'] * d['hypoglycemia']
    d['anemia_prostration'] = d['Anemia'] * d['prostraction']
    
    return d

# Create the paper's balanced dataframe structure
df_feat = add_clinical_features(df)

# Drop 'sex' if desired or keep - let's check correlation
class_0 = df_feat[df_feat['severe_maleria'] == 0]
class_1 = df_feat[df_feat['severe_maleria'] == 1]
class_1_over = class_1.sample(len(class_0), replace=True, random_state=42)
df_balanced = pd.concat([class_1_over, class_0], axis=0).reset_index(drop=True)

X = df_balanced.drop(columns=['severe_maleria'])
y = df_balanced['severe_maleria']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

scaler = RobustScaler()
X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

print(f"Training shape: {X_train.shape}, Test shape: {X_test.shape}")

# Define Base Classifiers
rf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_split=2, random_state=123)
et = ExtraTreesClassifier(n_estimators=300, max_depth=14, min_samples_split=2, random_state=123)
xgb = XGBClassifier(n_estimators=250, max_depth=6, learning_rate=0.03, subsample=0.9, colsample_bytree=0.9, random_state=123, eval_metric='logloss')
lgb = LGBMClassifier(n_estimators=250, max_depth=6, learning_rate=0.03, num_leaves=31, random_state=123, verbose=-1)
cb = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.03, verbose=0, random_state=123)
gb = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.04, random_state=123)
mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, alpha=0.01, random_state=123)

models = {
    "Original Paper Best (RF)": None, # Benchmark reference (81.95%)
    "Random Forest (Enhanced)": rf,
    "Extra Trees": et,
    "XGBoost (Enhanced)": xgb,
    "LightGBM": lgb,
    "CatBoost (Enhanced)": cb,
    "Gradient Boosting": gb,
    "Multi-Layer Perceptron (MLP)": mlp
}

results = []

for name, model in models.items():
    if model is None:
        # Paper's published figures
        results.append({
            "Model": "Paper Best (Awe et al. 2025)",
            "Accuracy": 0.8195,
            "ROC AUC": 0.8696,
            "MCC": 0.6374,
            "Balanced Acc": 0.8187,
            "Cohen Kappa": 0.6374,
            "Precision": 0.8310,
            "Recall": 0.8310,
            "F1 Score": 0.8310
        })
        continue
        
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    probs = model.predict_proba(X_test_s)[:, 1] if hasattr(model, 'predict_proba') else preds
    
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, preds),
        "ROC AUC": roc_auc_score(y_test, probs),
        "MCC": matthews_corrcoef(y_test, preds),
        "Balanced Acc": balanced_accuracy_score(y_test, preds),
        "Cohen Kappa": cohen_kappa_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1 Score": f1_score(y_test, preds)
    })

# Stacking Meta-Classifier
stack_estimators = [
    ('rf', rf),
    ('et', et),
    ('xgb', xgb),
    ('lgb', lgb),
    ('cb', cb),
    ('gb', gb)
]

stack_clf = StackingClassifier(
    estimators=stack_estimators,
    final_estimator=LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42),
    cv=5,
    n_jobs=-1
)
stack_clf.fit(X_train_s, y_train)
preds_stack = stack_clf.predict(X_test_s)
probs_stack = stack_clf.predict_proba(X_test_s)[:, 1]

results.append({
    "Model": "OUR Advanced Stacking Ensemble",
    "Accuracy": accuracy_score(y_test, preds_stack),
    "ROC AUC": roc_auc_score(y_test, probs_stack),
    "MCC": matthews_corrcoef(y_test, preds_stack),
    "Balanced Acc": balanced_accuracy_score(y_test, preds_stack),
    "Cohen Kappa": cohen_kappa_score(y_test, preds_stack),
    "Precision": precision_score(y_test, preds_stack),
    "Recall": recall_score(y_test, preds_stack),
    "F1 Score": f1_score(y_test, preds_stack)
})

# Soft Voting Classifier
vote_clf = VotingClassifier(
    estimators=stack_estimators,
    voting='soft',
    weights=[1.5, 1.5, 1.2, 1.2, 1.3, 1.0]
)
vote_clf.fit(X_train_s, y_train)
preds_vote = vote_clf.predict(X_test_s)
probs_vote = vote_clf.predict_proba(X_test_s)[:, 1]

results.append({
    "Model": "OUR Weighted Soft-Voting Ensemble",
    "Accuracy": accuracy_score(y_test, preds_vote),
    "ROC AUC": roc_auc_score(y_test, probs_vote),
    "MCC": matthews_corrcoef(y_test, preds_vote),
    "Balanced Acc": balanced_accuracy_score(y_test, preds_vote),
    "Cohen Kappa": cohen_kappa_score(y_test, preds_vote),
    "Precision": precision_score(y_test, preds_vote),
    "Recall": recall_score(y_test, preds_vote),
    "F1 Score": f1_score(y_test, preds_vote)
})

res_df = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False)
print("\n" + "="*95)
print("FINAL YEAR PROJECT ACCURACY COMPARISON VS EXISTING 2025 PAPER")
print("="*95)
print(res_df.to_string(index=False))
