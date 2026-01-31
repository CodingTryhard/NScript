import shap
import pandas as pd
import numpy as np


class ExplainabilityAuditor:
    """
    Uses SHAP to explain WHY a counterfactual flip occurred.
    Designed to be called ONLY after bias is detected.
    """

    def __init__(self, model_pipeline, background_data: pd.DataFrame):
        """
        model_pipeline: trained sklearn Pipeline
        background_data: raw training features (NO target, NO metadata)
        """
        self.model = model_pipeline

        # SHAP background summarization
        self.background = shap.kmeans(background_data, 50)

        # KernelExplainer for black-box pipelines
        self.explainer = shap.KernelExplainer(
            self.model.predict_proba,
            self.background
        )

    def explain_flip(
        self,
        original_row: pd.Series,
        cf_row: pd.Series,
        target_class: int = 1
    ) -> pd.DataFrame:
        """
        Explains why prediction changed between original and counterfactual.
        """

        # ---------------------------------------------
        # 1. Clean input rows (drop metadata)
        # ---------------------------------------------
        drop_cols = ['_cf_type', 'ID', 'loan_approved']
        original = original_row.drop(labels=[c for c in drop_cols if c in original_row.index])
        cf = cf_row.drop(labels=[c for c in drop_cols if c in cf_row.index])

        comparison_df = pd.concat(
            [original.to_frame().T, cf.to_frame().T],
            ignore_index=True
        )

        # ---------------------------------------------
        # 2. Compute SHAP values
        # ---------------------------------------------
        print("Calculating SHAP values (slow but precise)...")
        shap_values = self.explainer.shap_values(comparison_df)

        # Normalize SHAP output
        if isinstance(shap_values, list):
            vals = shap_values[target_class]
        elif shap_values.ndim == 3:
            vals = shap_values[target_class]
        else:
            vals = shap_values

        # ---------------------------------------------
        # 3. Build explanation table
        # ---------------------------------------------
        feature_names = comparison_df.columns

        explanation_df = pd.DataFrame({
            'Feature': feature_names,
            'Original_Value': original.values,
            'Original_Impact': vals[0],
            'CF_Value': cf.values,
            'CF_Impact': vals[1],
        })

        # ---------------------------------------------
        # 4. Delta analysis (WHY the flip happened)
        # ---------------------------------------------
        explanation_df['Impact_Delta'] = (
            explanation_df['CF_Impact'] - explanation_df['Original_Impact']
        )

        explanation_df['Abs_Delta'] = explanation_df['Impact_Delta'].abs()

        explanation_df = (
            explanation_df
            .sort_values('Abs_Delta', ascending=False)
            .drop(columns='Abs_Delta')
        )

        return explanation_df
