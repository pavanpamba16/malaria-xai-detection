"""
Clinical Feature Engineering Module for Malaria Severity Diagnosis
Implements domain-guided biomarker indices and non-linear interactions.
"""
import pandas as pd
import numpy as np

def extract_clinical_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts medically grounded composite indices and non-linear interaction features
    from raw patient clinical symptom data.
    """
    df = df_raw.copy()
    
    # 1. Primary Organ Damage / Hemoglobinuria Marker (WHO severe malaria criteria)
    df['organ_failure_score'] = (
        df['jundice'] * 1.5 + 
        df['cocacola_urine'] * 2.0 + 
        df['hypoglycemia'] * 1.5 + 
        df['Anemia'] * 1.0
    )
    
    # 2. Cerebral Malaria / Neurological Impairment Marker
    df['cerebral_risk_score'] = (
        df['Convulsion'] * 2.5 + 
        df['prostraction'] * 1.8 + 
        df['hyperpyrexia'] * 1.5
    )
    
    # 3. Gastrointestinal Complications
    df['gastrointestinal_score'] = (
        df['vomitting'] * 1.0 + 
        df['diarrhea'] * 1.2 + 
        df['bitter_tongue'] * 0.8
    )
    
    # 4. Febrile & Systemic Syndrome
    df['febrile_syndrome_score'] = (
        df['fever'] * 1.0 + 
        df['rigor'] * 1.2 + 
        df['cold'] * 0.8 + 
        df['headace'] * 1.0 + 
        df['fatigue'] * 0.8
    )
    
    # 5. Total Active Symptom Burden
    symptom_cols = [
        'fever', 'cold', 'rigor', 'fatigue', 'headace', 'bitter_tongue', 
        'vomitting', 'diarrhea', 'Convulsion', 'Anemia', 'jundice', 
        'cocacola_urine', 'hypoglycemia', 'prostraction', 'hyperpyrexia'
    ]
    df['total_symptom_count'] = df[symptom_cols].sum(axis=1)
    
    # 6. Multi-System Clinical Severity Index (CSI)
    df['clinical_severity_index'] = (
        df['organ_failure_score'] * 3.0 + 
        df['cerebral_risk_score'] * 2.5 + 
        df['gastrointestinal_score'] * 1.2 + 
        df['febrile_syndrome_score'] * 0.8
    )
    
    # 7. Age-Stratified Physiological Vulnerability
    # Pediatric (<5 yrs) and Geriatric (>55 yrs) cohorts have significantly higher morbidity
    df['is_under_5'] = (df['age'] <= 5).astype(int)
    df['is_elderly'] = (df['age'] >= 55).astype(int)
    df['is_vulnerable_age'] = df['is_under_5'] + df['is_elderly']
    df['age_weighted_severity'] = (df['age'] * df['clinical_severity_index']) / 10.0
    
    # 8. High-Risk Multi-Symptom Interaction Indicators
    df['convulsion_and_hyperpyrexia'] = df['Convulsion'] * df['hyperpyrexia']
    df['hemoglobinuria_and_jaundice'] = df['cocacola_urine'] * df['jundice']
    df['prostration_and_hypoglycemia'] = df['prostraction'] * df['hypoglycemia']
    df['anemia_and_prostration'] = df['Anemia'] * df['prostraction']
    df['fever_and_rigor'] = df['fever'] * df['rigor']
    
    return df
