import pandas as pd
import numpy as np

class FairnessMetrics:
    """
    Computes quantitative fairness metrics and identifies drivers of bias.
    """

    @staticmethod
    def calculate_consistency_score(audit_df: pd.DataFrame) -> float:
        """
        Metric 1: Counterfactual Consistency Score
        Formula: (Number of Twin Predictions == Original Prediction) / Total Twins
        """
        # USE RAW COLUMN NAME '_cf_type'
        twins = audit_df[audit_df['_cf_type'] != 'Original']
        
        if len(twins) == 0:
            return 1.0 

        # We rely on 'label_changed' calculated in Module 4
        if 'label_changed' not in twins.columns:
             return 0.0
             
        consistent_count = (~twins['label_changed']).sum()
        score = consistent_count / len(twins)
        return round(score, 4)

    @staticmethod
    def calculate_flip_rate(audit_df: pd.DataFrame) -> dict:
        """
        Metric 2: Flip Rate per Sensitive Attribute
        """
        twins = audit_df[audit_df['_cf_type'] != 'Original'].copy()
        
        # Parse attribute name if needed
        if 'changed_attr' not in twins.columns:
             twins['changed_attr'] = twins['_cf_type'].apply(
                lambda x: x.split('(')[1].split('=')[0] if '(' in x else 'unknown'
            )

        flip_rates = {}
        # Avoid crashing on empty twins
        if twins.empty:
            return {}

        for attr in twins['changed_attr'].unique():
            subset = twins[twins['changed_attr'] == attr]
            if len(subset) > 0:
                flips = subset['label_changed'].sum()
                flip_rates[attr] = round(flips / len(subset), 4)
            else:
                flip_rates[attr] = 0.0
            
        return flip_rates

    @staticmethod
    def calculate_confidence_sensitivity(audit_df: pd.DataFrame) -> float:
        """
        Metric 3: Confidence Sensitivity
        """
        twins = audit_df[audit_df['_cf_type'] != 'Original']
        if len(twins) == 0: return 0.0
        return round(twins['prob_delta'].abs().mean(), 4)

    @staticmethod
    def rank_bias_conditions(audit_df: pd.DataFrame) -> list:
        """
        Metric 4: Bias Driver Analysis
        Identifies specific conditions (e.g. 'gender=Female') associated with negative flips.
        """
        # Ensure 'changed_attr' exists for logic
        if 'changed_attr' not in audit_df.columns:
             audit_df['changed_attr'] = audit_df['_cf_type'].apply(
                lambda x: x.split('(')[1].split('=')[0] if '(' in x else 'unknown'
            )

        # We look for where label_changed is True
        flips = audit_df[audit_df['label_changed'] == True].copy()
        
        bias_counts = {}
        
        # Iterate through unique IDs that had a flip
        # (Assuming 'ID' column was added in the backend before passing here)
        if 'ID' not in flips.columns:
            return []

        for uid in flips['ID'].unique():
            # Get the whole family (Original + Twins) for this ID
            family = audit_df[audit_df['ID'] == uid]
            
            # Safe access to original
            try:
                original = family[family['_cf_type'] == 'Original'].iloc[0]
            except IndexError:
                continue # Skip if original is missing (edge case)
            
            # Look at the twins that flipped
            flipped_twins = flips[flips['ID'] == uid]
            
            for _, twin in flipped_twins.iterrows():
                attr_name = twin['changed_attr']
                
                # Check if attribute exists in data to avoid KeyError
                if attr_name not in original or attr_name not in twin:
                    continue

                penalized_value = None
                
                # Logic: Find which value got the '0' (Reject)
                if original['model_prediction'] == 0 and twin['model_prediction'] == 1:
                    # Original was rejected. The Original's value is the "bad" one.
                    penalized_value = f"{attr_name}={original[attr_name]}"
                    
                elif original['model_prediction'] == 1 and twin['model_prediction'] == 0:
                    # Twin was rejected. The Twin's value is the "bad" one.
                    penalized_value = f"{attr_name}={twin[attr_name]}"
                
                if penalized_value:
                    bias_counts[penalized_value] = bias_counts.get(penalized_value, 0) + 1
                    
        # Sort by frequency (Highest bias first)
        sorted_bias = sorted(bias_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_bias

    @classmethod
    def generate_scorecard(cls, audit_df: pd.DataFrame):
        """Aggregates all metrics."""
        
        # Ensure helper columns exist locally if not passed
        if 'changed_attr' not in audit_df.columns:
            audit_df['changed_attr'] = audit_df['_cf_type'].apply(
                lambda x: x.split('(')[1].split('=')[0] if '(' in x else 'unknown'
            )
            
        return {
            "consistency_score": cls.calculate_consistency_score(audit_df),
            "flip_rates": cls.calculate_flip_rate(audit_df),
            "confidence_sensitivity": cls.calculate_confidence_sensitivity(audit_df),
            "bias_ranking": cls.rank_bias_conditions(audit_df)
        }