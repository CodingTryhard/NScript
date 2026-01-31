import pandas as pd
import numpy as np
from dataclasses import dataclass, field

@dataclass
class RiskAssessment:
    severity_score: float  # 0 to 100
    risk_level: str        # 'Low', 'Medium', 'High', 'Critical'
    breakdown: dict        # Contribution of each factor
    summary_text: str      # Executive summary
    context: dict = field(default_factory=dict) # NEW: Store the user's context

class BiasSeverityCalculator:
    """
    Aggregates granular fairness metrics into a single 'severity' score.
    UPDATED: Now generates context-aware narratives.
    """

    def __init__(self, weights: dict = None):
        self.weights = weights if weights else {
            'flip_rate': 0.70,       # Dominant factor
            'sensitivity': 0.20,
            'inconsistency': 0.10
        }

    def _generate_executive_summary(self, score: float, metrics: dict, risk_label: str, context: dict) -> str:
        """
        Generates a narrative that references the specific use case.
        """
        # 1. Extract Drivers
        flip_rates = metrics.get('flip_rates', {})
        active_drivers = [k for k, v in flip_rates.items() if v > 0.01]
        active_drivers = sorted(active_drivers, key=lambda x: flip_rates[x], reverse=True)
        
        if not active_drivers:
            drivers_text = "no specific attributes"
        elif len(active_drivers) == 1:
            drivers_text = active_drivers[0]
        else:
            drivers_text = f"{', '.join(active_drivers[:-1])} and {active_drivers[-1]}"

        # 2. Extract Context
        purpose = context.get('purpose', 'General Purpose')
        demographic = context.get('demographic', 'General Population')
        
        # 3. Construct Narrative
        intro = f"**Audit Scope:** {purpose} targeting {demographic}."
        
        if score < 15:
            return (
                f"{intro}\n\n"
                f"✅ **PASS:** The model demonstrates high stability with a {risk_label} classification. "
                "Counterfactual testing reveals negligible sensitivity to protected attributes."
            )
        elif score < 40:
            return (
                f"{intro}\n\n"
                f"⚠️ **WARNING:** Bias detected linked to {drivers_text}. "
                f"While the model is generally stable, specific subgroups within the {demographic} demographic "
                "may face inconsistent outcomes. Human review is recommended."
            )
        else:
            return (
                f"{intro}\n\n"
                f"🚨 **CRITICAL RISK:** The model exhibits severe sensitivity to {drivers_text}. "
                f"For a system designed for '{purpose}', this level of instability poses significant compliance "
                "and reputational risks. Immediate mitigation is required."
            )

    def calculate_score(self, metrics_scorecard: dict, context: dict = None) -> RiskAssessment:
        """
        Computes the Composite Bias Severity Index (0-100).
        """
        if context is None: context = {}
        
        # 1. Extract Components
        flip_rates = metrics_scorecard.get('flip_rates', {})
        max_flip_rate = max(flip_rates.values()) if flip_rates else 0.0
        
        sensitivity = metrics_scorecard.get('confidence_sensitivity', 0.0)
        consistency = metrics_scorecard.get('consistency_score', 1.0)
        inconsistency_rate = 1.0 - consistency

        # 2. Compute Weighted Sum
        # Multiplier to punish high flip rates severely
        flip_impact = min(max_flip_rate * 3.0, 1.0) 

        raw_score = (
            (self.weights['flip_rate'] * flip_impact) +
            (self.weights['sensitivity'] * sensitivity) +
            (self.weights['inconsistency'] * inconsistency_rate)
        )
        
        final_score = round(min(raw_score * 100, 100), 1)

        # 3. Determine Risk Category
        if final_score < 15:
            label = "Low Risk"
        elif final_score < 40:
            label = "Medium Risk"
        else:
            label = "High Risk"

        # 4. Generate Narrative with Context
        narrative = self._generate_executive_summary(final_score, metrics_scorecard, label, context)

        return RiskAssessment(
            severity_score=final_score,
            risk_level=label,
            breakdown={
                "Max Flip Impact": round(self.weights['flip_rate'] * flip_impact * 100, 1),
                "Sensitivity Impact": round(self.weights['sensitivity'] * sensitivity * 100, 1)
            },
            summary_text=narrative,
            context=context
        )