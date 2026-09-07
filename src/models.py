"""
Machine Learning & Ensemble Architecture Module
Implements individual tree-based classifiers and advanced meta-stacking ensembles.
"""
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, 
    GradientBoostingClassifier, StackingClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def get_base_models(random_state: int = 123):
    """
    Instantiates optimized base learners configured for tabular medical symptom data.
    """
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=300, 
            max_depth=12, 
            min_samples_split=2, 
            random_state=random_state
        ),
        'Extra Trees': ExtraTreesClassifier(
            n_estimators=300, 
            max_depth=14, 
            min_samples_split=2, 
            random_state=random_state
        ),
        'CatBoost': CatBoostClassifier(
            iterations=350, 
            depth=6, 
            learning_rate=0.03, 
            verbose=0, 
            random_state=random_state
        ),
        'XGBoost': XGBClassifier(
            n_estimators=250, 
            max_depth=5, 
            learning_rate=0.03, 
            subsample=0.9, 
            colsample_bytree=0.9, 
            random_state=random_state, 
            eval_metric='logloss'
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=250, 
            max_depth=5, 
            learning_rate=0.03, 
            num_leaves=31, 
            random_state=random_state, 
            verbose=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, 
            max_depth=4, 
            learning_rate=0.04, 
            random_state=random_state
        )
    }
    return models

def build_stacking_ensemble(base_models_dict, meta_c: float = 1.0, random_state: int = 42):
    """
    Builds a multi-level stacking ensemble combining base models with a regularized meta-learner.
    """
    estimators = [(name.lower().replace(' ', '_'), model) for name, model in base_models_dict.items()]
    meta_learner = LogisticRegression(
        C=meta_c, 
        solver='lbfgs', 
        max_iter=1000, 
        random_state=random_state
    )
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1
    )
    return stacking_clf

def build_voting_ensemble(base_models_dict, weights=None):
    """
    Builds a soft-voting ensemble averaging predicted class probabilities.
    """
    estimators = [(name.lower().replace(' ', '_'), model) for name, model in base_models_dict.items()]
    if weights is None:
        weights = [1.5, 1.5, 1.3, 1.2, 1.1, 1.0]
    voting_clf = VotingClassifier(
        estimators=estimators,
        voting='soft',
        weights=weights
    )
    return voting_clf
