import pandas as pd
import numpy as np

class FairnessMetrics:
    """
    Computes quantitative fairness metrics based on counterfactual analysis.
    """

    @staticmethod
    def calculate_consistency_score(audit_df: pd.DataFrame) -> float:
        """
        Metric 1: Counterfactual Consistency Score
        Formula: (Number of Twin Predictions == Original Prediction) / Total Twins
        
        Ethical Meaning:
        Measures the stability of the model. A score of 1.0 (100%) means the model 
        is 'blind' to the sensitive attribute—it consistently gives the same decision 
        regardless of the protected trait.
        """
        # Filter out the 'Original' row to focus on the generated twins
        twins = audit_df[audit_df['_cf_type'] != 'Original']
        
        if len(twins) == 0:
            return 1.0 # Trivial consistency if no counterfactuals exist

        # Count how many twins have the same label as the original (label_changed == False)
        # Note: 'label_changed' was calculated in Module 4
        consistent_count = (~twins['label_changed']).sum()
        
        score = consistent_count / len(twins)
        return round(score, 4)

    @staticmethod
    def calculate_flip_rate(audit_df: pd.DataFrame) -> dict:
        """
        Metric 2: Flip Rate per Sensitive Attribute
        
        Ethical Meaning:
        Identifies specific vulnerabilities. A high flip rate for 'gender' vs 'race' 
        tells developers exactly where the bias is concentrated.
        """
        twins = audit_df[audit_df['_cf_type'] != 'Original'].copy()
        
        # We need to parse which attribute was changed from the string text or metadata
        # Assuming format "Counterfactual (attr=val)" from Module 3
        # A robust implementation would pass this metadata separately, 
        # but here we parse for simplicity.
        
        # Extract attribute name from string: "Counterfactual (gender=Male)" -> "gender"
        try:
            twins['changed_attr'] = twins['_cf_type'].apply(
                lambda x: x.split('(')[1].split('=')[0] if '(' in x else 'unknown'
            )
        except IndexError:
            twins['changed_attr'] = 'unknown'

        flip_rates = {}
        for attr in twins['changed_attr'].unique():
            subset = twins[twins['changed_attr'] == attr]
            flips = subset['label_changed'].sum()
            total = len(subset)
            flip_rates[attr] = round(flips / total, 4)
            
        return flip_rates

    @staticmethod
    def calculate_confidence_sensitivity(audit_df: pd.DataFrame) -> float:
        """
        Metric 3: Confidence Sensitivity (Average Absolute Deviation)
        Formula: Mean(|Prob_Original - Prob_Counterfactual|)
        
        Ethical Meaning:
        Even if the label doesn't flip (e.g., Approved -> Approved), a drop in 
        confidence (90% -> 51%) indicates 'latent bias'. The model is technically 
        fair in outcome, but treats the groups differently internally.
        """
        twins = audit_df[audit_df['_cf_type'] != 'Original']
        
        if len(twins) == 0:
            return 0.0

        # prob_delta was calculated in Module 4 (Prob_Twin - Prob_Original)
        sensitivity = twins['prob_delta'].abs().mean()
        
        return round(sensitivity, 4)

    @classmethod
    def generate_scorecard(cls, audit_df: pd.DataFrame):
        """Aggregates all metrics into a final dictionary."""
        return {
            "consistency_score": cls.calculate_consistency_score(audit_df),
            "flip_rates": cls.calculate_flip_rate(audit_df),
            "confidence_sensitivity": cls.calculate_confidence_sensitivity(audit_df)
        }