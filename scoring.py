import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class RiskAssessment:
    severity_score: float  # 0 to 100
    risk_level: str        # 'Low', 'Medium', 'High', 'Critical'
    breakdown: dict        # Contribution of each factor
    summary_text: str      # Executive summary

class BiasSeverityCalculator:
    """
    Aggregates granular fairness metrics into a single 'severity' score.
    UPDATED: Much stricter penalties for discriminatory flips.
    """

    def __init__(self, weights: dict = None):
        # NEW WEIGHTS: Flip Rate is now the dominant factor (70%)
        # A 10% flip rate will now cause a significant score spike.
        self.weights = weights if weights else {
            'flip_rate': 0.70,       # Increased from 0.50
            'sensitivity': 0.20,     # Reduced slightly
            'inconsistency': 0.10    # Reduced slightly
        }

    def _generate_executive_summary(self, score: float, metrics: dict, risk_label: str) -> str:
        flip_rates = metrics.get('flip_rates', {})
        # Filter for meaningful flips (> 1%)
        active_drivers = [k for k, v in flip_rates.items() if v > 0.01]
        active_drivers = sorted(active_drivers, key=lambda x: flip_rates[x], reverse=True)
        
        # Formatting
        if not active_drivers:
            drivers_text = "no specific attributes"
        elif len(active_drivers) == 1:
            drivers_text = active_drivers[0]
        else:
            drivers_text = f"{', '.join(active_drivers[:-1])} and {active_drivers[-1]}"

        # Narrative Construction
        if score < 15:
            return "The model appears stable. No significant discriminatory patterns detected."
        elif score < 40:
            return (
                f"⚠️ Bias Detected: The model decisions are influenced by {drivers_text}. "
                "While not universal, a subset of applicants receive different outcomes based purely on demographics."
            )
        else:
            return (
                f"🚨 HIGH RISK: The model shows strong bias towards {drivers_text}. "
                "Testing reveals that changing these attributes frequently reverses the hiring decision. "
                "Do not deploy without retraining."
            )

    def calculate_score(self, metrics_scorecard: dict) -> RiskAssessment:
        # 1. Extract Components
        flip_rates = metrics_scorecard.get('flip_rates', {})
        # Get the WORST flip rate (e.g., if Gender flips 15%, that's our risk ceiling)
        max_flip_rate = max(flip_rates.values()) if flip_rates else 0.0
        
        sensitivity = metrics_scorecard.get('confidence_sensitivity', 0.0)
        consistency = metrics_scorecard.get('consistency_score', 1.0)
        inconsistency_rate = 1.0 - consistency

        # 2. Compute Score
        # We apply a multiplier to flip_rate to ensure even small %s hurt the score.
        # Example: 15% flip rate * 3 multiplier = 45 points raw impact
        flip_impact = min(max_flip_rate * 3.0, 1.0) 

        raw_score = (
            (self.weights['flip_rate'] * flip_impact) +
            (self.weights['sensitivity'] * sensitivity) +
            (self.weights['inconsistency'] * inconsistency_rate)
        )
        
        # Scale to 0-100
        final_score = round(min(raw_score * 100, 100), 1)

        # 3. Determine Risk Category (Stricter Thresholds)
        if final_score < 15:
            label = "Low Risk"
        elif final_score < 40:
            label = "Medium Risk"
        else:
            label = "High Risk"

        narrative = self._generate_executive_summary(final_score, metrics_scorecard, label)

        return RiskAssessment(
            severity_score=final_score,
            risk_level=label,
            breakdown={
                "Max Flip Impact": round(self.weights['flip_rate'] * flip_impact * 100, 1),
                "Sensitivity Impact": round(self.weights['sensitivity'] * sensitivity * 100, 1)
            },
            summary_text=narrative
        )