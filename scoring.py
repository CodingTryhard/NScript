import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class RiskAssessment:
    severity_score: float  # 0 to 100
    risk_level: str        # 'Low', 'Medium', 'High', 'Critical'
    breakdown: dict        # Contribution of each factor

class BiasSeverityCalculator:
    """
    Aggregates granular fairness metrics into a single 'severity' score.
    Useful for setting Go/No-Go thresholds in deployment pipelines.
    """

    def __init__(self, weights: dict = None):
        """
        Args:
            weights: Dictionary defining importance of each component.
                     Default emphasizes 'Flip Rate' (actual decision changes) 
                     over 'Sensitivity' (probability fluctuations).
        """
        self.weights = weights if weights else {
            'flip_rate': 0.50,        # Highest weight: Changing the outcome is the worst harm.
            'sensitivity': 0.30,      # Medium weight: Volatility indicates latent bias.
            'inconsistency': 0.20     # Lower weight: General instability.
        }

    def calculate_score(self, metrics_scorecard: dict) -> RiskAssessment:
        """
        Computes the Composite Bias Severity Index (0-100).
        
        Formula:
          Score = (w1 * Max_Flip_Rate) + (w2 * Conf_Sensitivity) + (w3 * (1 - Consistency))
          Scaled to 0-100 range.
        
        Args:
            metrics_scorecard: Output from Module 5 containing 'flip_rates', 
                               'confidence_sensitivity', and 'consistency_score'.
        
        Returns:
            RiskAssessment object with score and categorical label.
        """
        
        # 1. Extract Components
        # We take the MAXIMUM flip rate across all protected attributes.
        # If the model is fair for Gender but terrible for Race, the risk is still High.
        flip_rates = metrics_scorecard.get('flip_rates', {})
        max_flip_rate = max(flip_rates.values()) if flip_rates else 0.0
        
        sensitivity = metrics_scorecard.get('confidence_sensitivity', 0.0)
        
        # Invert consistency so that "High" = "Bad" (matching other metrics)
        consistency = metrics_scorecard.get('consistency_score', 1.0)
        inconsistency_rate = 1.0 - consistency

        # 2. Compute Weighted Sum
        # Note: All inputs should be in range [0, 1]
        raw_score = (
            (self.weights['flip_rate'] * max_flip_rate) +
            (self.weights['sensitivity'] * sensitivity) +
            (self.weights['inconsistency'] * inconsistency_rate)
        )
        
        # Normalize to 0-100 scale
        # The sum of weights is 1.0, so raw_score is already 0-1.
        final_score = round(raw_score * 100, 2)

        # 3. Determine Risk Category
        if final_score < 15:
            label = "Low Risk"
        elif final_score < 35:
            label = "Medium Risk"
        elif final_score < 60:
            label = "High Risk"
        else:
            label = "CRITICAL FAIL"

        return RiskAssessment(
            severity_score=final_score,
            risk_level=label,
            breakdown={
                "Max Flip Impact": round(self.weights['flip_rate'] * max_flip_rate * 100, 2),
                "Sensitivity Impact": round(self.weights['sensitivity'] * sensitivity * 100, 2),
                "Inconsistency Impact": round(self.weights['inconsistency'] * inconsistency_rate * 100, 2)
            }
        )