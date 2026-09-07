"""
Explainable AI (XAI) Suite Module
Provides Global & Local Interpretability via SHAP, LIME, and Counterfactual What-If Analysis.
"""
import shap
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ExplainabilitySuite:
    def __init__(self, model, feature_names, training_data, class_names=None):
        self.model = model
        self.feature_names = list(feature_names)
        self.training_data = np.array(training_data)
        self.class_names = class_names or ['Non-Severe Malaria', 'Severe Malaria']
        
        # Initialize LIME explainer
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.training_data,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode='classification',
            random_state=42
        )
        
        # Initialize SHAP explainer
        # If StackingClassifier, use base model or Kernel/Sampling explainer or representative tree model
        if hasattr(model, 'named_estimators_') and 'cb' in model.named_estimators_:
            self.shap_model = model.named_estimators_['cb']
        elif hasattr(model, 'estimators_') and len(model.estimators_) > 0:
            self.shap_model = model.estimators_[0]
        else:
            self.shap_model = model
            
        try:
            self.shap_explainer = shap.TreeExplainer(self.shap_model)
        except Exception:
            # Fallback for generic callable models
            self.shap_explainer = shap.Explainer(self.shap_model, self.training_data[:50])

    def compute_shap_values(self, X_sample):
        """Computes SHAP values for a sample dataset."""
        return self.shap_explainer(X_sample)

    def explain_patient_lime(self, patient_feature_vector, num_features=10):
        """
        Generates local explanation for a single patient instance using LIME.
        """
        vector = np.array(patient_feature_vector).flatten()
        predict_fn = self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.shap_model.predict_proba
        
        exp = self.lime_explainer.explain_instance(
            data_row=vector,
            predict_fn=predict_fn,
            num_features=num_features,
            top_labels=1
        )
        return exp

    def counterfactual_analysis(self, patient_df_row, symptom_cols, target_class=0, max_flips=4):
        """
        Performs counterfactual 'What-If' reasoning:
        Finds the smallest set of symptom changes needed to change the diagnosis from Severe to Non-Severe.
        """
        current_row = patient_df_row.copy()
        pred_fn = self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.shap_model.predict_proba
        initial_prob = pred_fn(current_row)[0][1]
        
        if initial_prob < 0.5:
            return {
                "initial_prediction": "Non-Severe Malaria",
                "initial_probability": float(initial_prob),
                "is_already_target": True,
                "recommendations": ["Patient is already classified as low risk / non-severe."]
            }
            
        # Identify active symptoms present in this patient
        active_symptoms = [col for col in symptom_cols if col in current_row.columns and current_row[col].values[0] == 1]
        
        # Test single symptom removals and combinations
        simulations = []
        for symptom in active_symptoms:
            test_row = current_row.copy()
            test_row[symptom] = 0
            
            # Recalculate any dependent engineered features if present
            if 'organ_failure_score' in test_row.columns:
                test_row['organ_failure_score'] = (
                    test_row.get('jundice', 0) * 1.5 + 
                    test_row.get('cocacola_urine', 0) * 2.0 + 
                    test_row.get('hypoglycemia', 0) * 1.5 + 
                    test_row.get('Anemia', 0) * 1.0
                )
            if 'cerebral_risk_score' in test_row.columns:
                test_row['cerebral_risk_score'] = (
                    test_row.get('Convulsion', 0) * 2.5 + 
                    test_row.get('prostraction', 0) * 1.8 + 
                    test_row.get('hyperpyrexia', 0) * 1.5
                )
            if 'total_symptom_count' in test_row.columns:
                test_row['total_symptom_count'] = test_row[symptom_cols].sum(axis=1)
                
            prob = pred_fn(test_row)[0][1]
            prob_reduction = initial_prob - prob
            simulations.append({
                "intervened_symptom": symptom,
                "new_probability": float(prob),
                "risk_reduction_pct": float(prob_reduction * 100),
                "reverses_severity": prob < 0.5
            })
            
        simulations.sort(key=lambda x: x['risk_reduction_pct'], reverse=True)
        
        recommendations = []
        for sim in simulations[:max_flips]:
            status = "RESOLVES SEVERITY RISK" if sim['reverses_severity'] else "Reduces Risk"
            recommendations.append(
                f"Treating/alleviating '{sim['intervened_symptom']}' drops severe malaria risk by {sim['risk_reduction_pct']:.1f}% (New Risk: {sim['new_probability']*100:.1f}%) -> {status}"
            )
            
        return {
            "initial_prediction": "Severe Malaria",
            "initial_probability": float(initial_prob),
            "is_already_target": False,
            "simulations": simulations,
            "recommendations": recommendations
        }
