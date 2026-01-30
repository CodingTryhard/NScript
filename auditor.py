import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

def evaluate_fairness(model: Pipeline, cf_data: pd.DataFrame, target_class: int = 1) -> pd.DataFrame:
    """
     Audits the model by running predictions on both original and counterfactual data.
    
    This function strictly treats the model as a 'Black Box'. It does not access coefficients
    or retrain. It simply inputs X and observes Y.

    Args:
        model: The trained scikit-learn pipeline (must have .predict() and .predict_proba()).
        cf_data: DataFrame containing the original row and its counterfactual twins 
                 (output from Module 3).
        target_class: The index of the positive class (usually 1 for 'Approved').

    Returns:
        pd.DataFrame: A detailed comparison table showing how the prediction shifts 
                      when sensitive attributes are flipped.
    """
    
    # 1. Prepare Input Data
    # We must drop metadata columns generated in Module 3 (e.g., '_cf_type')
    # so the model receives the exact shape it expects.
    # We also drop the target column if it exists in the input.
    metadata_cols = ['_cf_type', 'loan_approved'] # Add any other non-feature columns here
    X_input = cf_data.drop(columns=[c for c in metadata_cols if c in cf_data.columns])

    # 2. Generate Predictions (NO Retraining)
    # We use the frozen model object passed into the function.
    print("Running predictions on counterfactual set...")
    preds = model.predict(X_input)
    probs = model.predict_proba(X_input)[:, target_class]

    # 3. Structure the Results
    results = cf_data.copy()
    results['model_prediction'] = preds
    results['model_probability'] = probs

    # 4. Calculate Deviation
    # We extract the probability of the 'Original' instance to compare others against it.
    original_prob = results.loc[results['_cf_type'] == 'Original', 'model_probability'].values[0]
    
    # Calculate the 'Fairness Gap': How much did the probability change?
    results['prob_delta'] = results['model_probability'] - original_prob
    results['prediction_flip'] = results['prob_delta'].apply(
        lambda x: "FLIPPED" if abs(x) > 0.5 else "Stable" # Simplistic check for label change
    )
    
    # Refine prediction flip logic: Did the class label actually change?
    original_pred = results.loc[results['_cf_type'] == 'Original', 'model_prediction'].values[0]
    results['label_changed'] = results['model_prediction'] != original_pred

    # 5. Formatting for readability
    output_cols = ['_cf_type', 'model_prediction', 'model_probability', 'prob_delta', 'label_changed']
    # Add back the features for context (optional, usually good for debugging)
    feature_cols = [c for c in X_input.columns if c in results.columns] 
    
    return results[output_cols + feature_cols]