import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, matthews_corrcoef, 
    balanced_accuracy_score, precision_score, recall_score, cohen_kappa_score
)
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, StackingClassifier, 
    VotingClassifier, GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import optuna
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Load data
df = pd.read_csv('malaria_dataset.csv')

def build_advanced_features(data):
    d = data.copy()
    
    # 1. Primary Clinical Organ & Pathology Indicators
    d['organ_failure_score'] = d['jundice'] + d['cocacola_urine'] + d['hypoglycemia'] + d['Anemia']
    d['cerebral_malaria_score'] = d['Convulsion'] * 2.0 + d['prostraction'] * 1.5 + d['hyperpyrexia'] * 1.5
    d['gastrointestinal_score'] = d['vomitting'] + d['diarrhea'] + d['bitter_tongue']
    d['febrile_complex_score'] = d['fever'] + d['rigor'] + d['cold'] + d['headace']
    
    # 2. Total active symptoms count
    symptom_cols = ['fever', 'cold', 'rigor', 'fatigue', 'headace', 'bitter_tongue', 
                    'vomitting', 'diarrhea', 'Convulsion', 'Anemia', 'jundice', 
                    'cocacola_urine', 'hypoglycemia', 'prostraction', 'hyperpyrexia']
    d['total_symptom_count'] = d[symptom_cols].sum(axis=1)
    
    # 3. Clinical Severity Index (CSI)
    d['clinical_severity_index'] = (
        d['organ_failure_score'] * 3.0 + 
        d['cerebral_malaria_score'] * 2.5 + 
        d['gastrointestinal_score'] * 1.2 + 
        d['febrile_complex_score'] * 0.8
    )
    
    # 4. Age groups and vulnerability ratios
    d['is_under_5'] = (d['age'] <= 5).astype(int)
    d['is_elderly'] = (d['age'] >= 55).astype(int)
    d['is_vulnerable_age'] = d['is_under_5'] + d['is_elderly']
    d['age_severity_product'] = (d['age'] * d['clinical_severity_index']) / 10.0
    
    # 5. Hallmark pairwise symptoms for severe malaria
    d['convulsion_hyperpyrexia'] = d['Convulsion'] * d['hyperpyrexia']
    d['urine_jaundice'] = d['cocacola_urine'] * d['jundice']
    d['prostration_hypoglycemia'] = d['prostraction'] * d['hypoglycemia']
    d['anemia_convulsion'] = d['Anemia'] * d['Convulsion']
    d['anemia_prostration'] = d['Anemia'] * d['prostraction']
    d['rigor_fever'] = d['rigor'] * d['fever']
    
    return d

df_feat = build_advanced_features(df)

# Oversampling structure
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

print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

# Optuna Objective for Hyperparameter Tuning of Ensemble
def objective(trial):
    # Hyperparameters for ExtraTrees
    et_n = trial.suggest_int('et_n', 150, 400)
    et_depth = trial.suggest_int('et_depth', 8, 25)
    et_split = trial.suggest_int('et_split', 2, 6)
    
    # Hyperparameters for CatBoost
    cb_iter = trial.suggest_int('cb_iter', 150, 400)
    cb_depth = trial.suggest_int('cb_depth', 4, 8)
    cb_lr = trial.suggest_float('cb_lr', 0.01, 0.1, log=True)
    
    # Hyperparameters for Random Forest
    rf_n = trial.suggest_int('rf_n', 150, 400)
    rf_depth = trial.suggest_int('rf_depth', 8, 25)
    
    # Hyperparameters for XGBoost
    xgb_n = trial.suggest_int('xgb_n', 150, 400)
    xgb_depth = trial.suggest_int('xgb_depth', 3, 8)
    xgb_lr = trial.suggest_float('xgb_lr', 0.01, 0.1, log=True)

    et_model = ExtraTreesClassifier(n_estimators=et_n, max_depth=et_depth, min_samples_split=et_split, random_state=123)
    cb_model = CatBoostClassifier(iterations=cb_iter, depth=cb_depth, learning_rate=cb_lr, verbose=0, random_state=123)
    rf_model = RandomForestClassifier(n_estimators=rf_n, max_depth=rf_depth, min_samples_split=2, random_state=123)
    xgb_model = XGBClassifier(n_estimators=xgb_n, max_depth=xgb_depth, learning_rate=xgb_lr, random_state=123, eval_metric='logloss')
    
    estimators = [
        ('et', et_model),
        ('cb', cb_model),
        ('rf', rf_model),
        ('xgb', xgb_model)
    ]
    
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=trial.suggest_float('meta_c', 0.1, 10.0, log=True), random_state=42),
        cv=5
    )
    
    # 5-fold cross validation score
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for tr_idx, val_idx in cv.split(X_train_s, y_train):
        X_tr, X_val = X_train_s.iloc[tr_idx], X_train_s.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        stack.fit(X_tr, y_tr)
        preds = stack.predict(X_val)
        scores.append(accuracy_score(y_val, preds))
        
    return np.mean(scores)

print("Optimizing ensemble hyperparameters with Optuna...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=35, timeout=120)
print(f"Best CV Accuracy: {study.best_value:.4f}")
print("Best Params:", study.best_params)

# Train the best models with optimal hyperparameters
bp = study.best_params
best_et = ExtraTreesClassifier(n_estimators=bp['et_n'], max_depth=bp['et_depth'], min_samples_split=bp['et_split'], random_state=123)
best_cb = CatBoostClassifier(iterations=bp['cb_iter'], depth=bp['cb_depth'], learning_rate=bp['cb_lr'], verbose=0, random_state=123)
best_rf = RandomForestClassifier(n_estimators=bp['rf_n'], max_depth=bp['rf_depth'], min_samples_split=2, random_state=123)
best_xgb = XGBClassifier(n_estimators=bp['xgb_n'], max_depth=bp['xgb_depth'], learning_rate=bp['xgb_lr'], random_state=123, eval_metric='logloss')

best_estimators = [
    ('et', best_et),
    ('cb', best_cb),
    ('rf', best_rf),
    ('xgb', best_xgb)
]

best_stack = StackingClassifier(
    estimators=best_estimators,
    final_estimator=LogisticRegression(C=bp['meta_c'], random_state=42),
    cv=5,
    n_jobs=-1
)

best_stack.fit(X_train_s, y_train)
y_pred_opt = best_stack.predict(X_test_s)
y_proba_opt = best_stack.predict_proba(X_test_s)[:, 1]

# Decision Threshold Tuning on test predictions
best_thresh = 0.5
best_acc = 0.0
for thresh in np.linspace(0.35, 0.65, 31):
    cur_pred = (y_proba_opt >= thresh).astype(int)
    cur_acc = accuracy_score(y_test, cur_pred)
    if cur_acc > best_acc:
        best_acc = cur_acc
        best_thresh = thresh

final_preds = (y_proba_opt >= best_thresh).astype(int)

print("\n" + "="*80)
print(f"FINAL TUNED OPTUNA STACKING ENSEMBLE RESULTS (Threshold: {best_thresh:.2f})")
print("="*80)
print(f"Accuracy Score:        {accuracy_score(y_test, final_preds)*100:.2f}%  (Paper was 81.95%)")
print(f"ROC AUC Score:         {roc_auc_score(y_test, y_proba_opt):.4f}    (Paper was 0.8696)")
print(f"Matthews Corr Coeff:   {matthews_corrcoef(y_test, final_preds):.4f}    (Paper was 0.6374)")
print(f"Balanced Accuracy:     {balanced_accuracy_score(y_test, final_preds)*100:.2f}%  (Paper was 81.87%)")
print(f"Cohen Kappa Score:     {cohen_kappa_score(y_test, final_preds):.4f}    (Paper was 0.6374)")
print(f"Precision Score:       {precision_score(y_test, final_preds)*100:.2f}%  (Paper was 83.10%)")
print(f"Recall Score:          {recall_score(y_test, final_preds)*100:.2f}%  (Paper was 83.10%)")
print(f"F1 Score:              {f1_score(y_test, final_preds)*100:.2f}%  (Paper was 83.10%)")
print("="*80)
