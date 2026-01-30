import shap
import pandas as pd
import numpy as np

class ExplainabilityAuditor:
    """
    Uses SHAP to attribute model decisions to specific features.
    It compares the 'Why' behind the Original decision vs. the Counterfactual decision.
    """

    def __init__(self, model_pipeline, background_data):
        """
        Args:
            model_pipeline: The trained scikit-learn pipeline (from Module 1).
            background_data: A representative sample of the training data (X_train).
                             SHAP needs this to establish a 'baseline' for comparison.
        """
        self.model = model_pipeline
        
        # We use a KernelExplainer because it is model-agnostic (works with Pipelines).
        # We pass the probability function so we see contributions to the 'Score', not just the label.
        # Summarizing background data with kmeans (k=50) speeds up calculation massively.
        self.background_summary = shap.kmeans(background_data, 50)
        self.explainer = shap.KernelExplainer(self.model.predict_proba, self.background_summary)

    def explain_flip(self, original_row: pd.Series, cf_row: pd.Series, target_class=1) -> pd.DataFrame:
        """
        Compares feature contributions for an Original vs. Counterfactual instance.

        Args:
            original_row: The data row before perturbation.
            cf_row: The data row after perturbation (e.g., gender flipped).
            target_class: Index of the positive class (1 for 'Loan Approved').

        Returns:
            pd.DataFrame: A side-by-side comparison of feature impacts (SHAP values).
        """
        
        # 1. Prepare Data
        # Combine into a single DataFrame for batch processing
        comparison_df = pd.concat([original_row.to_frame().T, cf_row.to_frame().T], ignore_index=True)
        
        # 2. Calculate SHAP Values
        # specific_shap_values will be a list of arrays (one for each class). 
        # We take the array for the target class.
        print(f"Calculating SHAP values (this may take a moment)...")
        shap_values = self.explainer.shap_values(comparison_df)
        
        # Handle SHAP output format variations (depends on version/model type)
        if isinstance(shap_values, list):
            vals = shap_values[target_class]
        else:
            vals = shap_values

        # 3. Construct Comparison Table
        explanation_df = pd.DataFrame({
            'Feature': original_row.index,
            
            # Original instance details
            'Original_Value': original_row.values,
            'Original_Impact': vals[0], # SHAP value for row 0
            
            # Counterfactual instance details
            'CF_Value': cf_row.values,
            'CF_Impact': vals[1], # SHAP value for row 1
        })

        # 4. Calculate the "Driver of Change"
        # Impact Delta = How much did this feature's contribution change?
        explanation_df['Impact_Delta'] = explanation_df['CF_Impact'] - explanation_df['Original_Impact']
        
        # 5. Ethical Interpretation Logic
        # Sort by the absolute change in impact. 
        # The top feature is the reason the prediction changed.
        explanation_df['Abs_Delta'] = explanation_df['Impact_Delta'].abs()
        explanation_df = explanation_df.sort_values(by='Abs_Delta', ascending=False).drop(columns=['Abs_Delta'])

        return explanation_df