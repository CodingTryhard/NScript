import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline


def evaluate_fairness(
    model: Pipeline,
    cf_data: pd.DataFrame,
    target_class: int = 1
) -> pd.DataFrame:
    """
    Audits the model by running predictions on original and counterfactual data.
    Treats the model as a pure black box (no retraining, no coefficient access).
    """

    # --------------------------------------------------
    # 0. SAFETY: Ensure ID exists (standalone + Streamlit)
    # --------------------------------------------------
    if 'ID' not in cf_data.columns:
        cf_data = cf_data.copy()
        cf_data['ID'] = 0  # default single individual

    # --------------------------------------------------
    # 1. Prepare model input (drop metadata)
    # --------------------------------------------------
    metadata_cols = ['_cf_type', 'loan_approved', 'ID']
    X_input = cf_data.drop(
        columns=[c for c in metadata_cols if c in cf_data.columns]
    )

    # --------------------------------------------------
    # 2. Run predictions (NO retraining)
    # --------------------------------------------------
    print("Running predictions on counterfactual set...")
    preds = model.predict(X_input)
    probs = model.predict_proba(X_input)[:, target_class]

    # --------------------------------------------------
    # 3. Attach predictions to results
    # --------------------------------------------------
    results = cf_data.copy()
    results['model_prediction'] = preds
    results['model_probability'] = probs

    # --------------------------------------------------
    # 4. PER-ID counterfactual fairness logic (CRITICAL)
    # --------------------------------------------------
    results['prob_delta'] = 0.0
    results['label_changed'] = False

    for uid in results['ID'].unique():
        group = results[results['ID'] == uid]

        # find the original instance
        original = group[group['_cf_type'] == 'Original']
        if original.empty:
            continue  # defensive guard

        original = original.iloc[0]
        idx = group.index

        results.loc[idx, 'prob_delta'] = (
            group['model_probability'] - original['model_probability']
        )

        results.loc[idx, 'label_changed'] = (
            group['model_prediction'] != original['model_prediction']
        )

    # --------------------------------------------------
    # 5. Output formatting
    # --------------------------------------------------
    output_cols = [
        '_cf_type',
        'ID',
        'model_prediction',
        'model_probability',
        'prob_delta',
        'label_changed'
    ]

    feature_cols = [c for c in X_input.columns if c in results.columns]

    return results[output_cols + feature_cols]


# --------------------------------------------------
# Integration Test (Standalone Run)
# --------------------------------------------------
if __name__ == "__main__":

    # Mock biased model
    class MockModel:
        def predict(self, X):
            return np.where(X['gender'] == 'Female', 1, 0)

        def predict_proba(self, X):
            p1 = np.where(X['gender'] == 'Female', 0.85, 0.45)
            return np.column_stack((1 - p1, p1))

    mock_model = MockModel()

    # Mock counterfactual data
    mock_cf_data = pd.DataFrame({
        '_cf_type': ['Original', 'Counterfactual (gender=Male)'],
        'income': [75000, 75000],
        'gender': ['Female', 'Male'],
        'loan_approved': [1, 1]
    })

    audit_report = evaluate_fairness(mock_model, mock_cf_data)

    print("\n--- Fairness Audit Report ---")
    print(
        audit_report[
            ['_cf_type', 'model_prediction', 'model_probability', 'prob_delta', 'label_changed']
        ].to_string(index=False)
    )
