import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

def classify_shift(delta: float) -> str:
    """Helper to categorize the severity of confidence changes."""
    abs_d = abs(delta)
    if abs_d < 0.05:
        return "Low"
    elif abs_d < 0.15:
        return "Moderate"
    else:
        return "Severe"

def evaluate_fairness(model: Pipeline, cf_data: pd.DataFrame, target_class: int = 1) -> pd.DataFrame:
    """
    Audits the model by running predictions on both original and counterfactual data.
    Now includes granular confidence shift analysis.
    """
    
    # 1. Prepare Input Data
    # Drop metadata so the model sees clean features
    metadata_cols = ['_cf_type', 'ID', 'loan_approved', 'actual', 'prediction', 'probability', 'changed_attr']
    X_input = cf_data.drop(columns=[c for c in metadata_cols if c in cf_data.columns])

    # 2. Generate Predictions (NO Retraining)
    # The model predicts on the Original AND the Twin rows
    preds = model.predict(X_input)
    probs = model.predict_proba(X_input)[:, target_class]

    # 3. Structure the Results
    results = cf_data.copy()
    results['model_prediction'] = preds
    results['model_probability'] = probs

    # 4. Calculate Deviations
    # We need to map every Twin back to its Original to calculate the delta.
    # We create a lookup dictionary: {ID: Original_Prob}
    
    # Filter originals
    originals = results[results['_cf_type'] == 'Original'][['ID', 'model_probability', 'model_prediction']]
    prob_map = originals.set_index('ID')['model_probability'].to_dict()
    pred_map = originals.set_index('ID')['model_prediction'].to_dict()
    
    # Vectorized mapping to calculate delta
    # Delta = Twin_Prob - Original_Prob
    results['original_prob'] = results['ID'].map(prob_map)
    results['prob_delta'] = results['model_probability'] - results['original_prob']
    
    # Did the label flip?
    results['original_pred'] = results['ID'].map(pred_map)
    results['label_changed'] = results['model_prediction'] != results['original_pred']

    # 5. Classify Severity
    # We classify the shift based on absolute magnitude
    results['shift_severity'] = results['prob_delta'].apply(classify_shift)

    # 6. Formatting
    # Keep useful metadata for the metrics module
    output_cols = [
        'ID', '_cf_type', 'model_prediction', 'model_probability', 
        'original_prob', 'prob_delta', 'label_changed', 'shift_severity'
    ]
    
    # Add back features for context
    feature_cols = [c for c in X_input.columns if c in results.columns] 
    
    return results[output_cols + feature_cols]