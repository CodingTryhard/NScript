import pandas as pd
import numpy as np

class EthicalSafeguards:
    """
    Provides automated warnings regarding data suitability, 
    counterfactual plausibility, and legal interpretation.
    """
    
    @staticmethod
    def check_correlations(df: pd.DataFrame, sensitive_cols: list, threshold: float = 0.6) -> list:
        """
        Checks for high correlation between sensitive attributes and other features.
        High correlation suggests that 'flipping' one attribute while holding others 
        constant might create unrealistic data points (Proxy Bias).
        """
        warnings = []
        
        # Data preparation: Simple encoding for correlation estimation
        # (In production, use Cramer's V for categorical-categorical correlation)
        df_encoded = df.copy()
        for col in df_encoded.columns:
            if df_encoded[col].dtype == 'object':
                df_encoded[col] = df_encoded[col].astype('category').cat.codes
        
        # Calculate correlation matrix
        try:
            corr_matrix = df_encoded.corr().abs()
        except ValueError:
            return ["Could not calculate correlations due to data format issues."]
        
        for sens in sensitive_cols:
            if sens not in corr_matrix.columns:
                continue
                
            # Check correlations with other features (excluding itself)
            correlations = corr_matrix[sens].drop(labels=[sens], errors='ignore').sort_values(ascending=False)
            
            for feature, score in correlations.items():
                if score > threshold:
                    warnings.append(
                        f"⚠️ **Proxy Variable Detected:** '{sens}' is highly correlated with '{feature}' (Correlation: {score:.2f}). "
                        f"Warning: The model may use '{feature}' as a hidden proxy for '{sens}'. "
                        f"Additionally, flipping '{sens}' without adjusting '{feature}' may generate unrealistic data points."
                    )
        
        return warnings

    @staticmethod
    def get_methodology_warning() -> str:
        """
        Explains the limitations of 'Ceteris Paribus' (All else equal) testing.
        """
        return (
            "### 📉 Methodology Limitation: The 'Ceteris Paribus' Assumption\n"
            "This audit uses **Counterfactual Generation** by changing *only* the protected attribute "
            "while holding all other features constant. \n\n"
            "**Why this matters:** While effective for detecting explicit bias, this method may generate "
            "theoretically unlikely profiles (e.g., changing 'Age' from 20 to 60 without changing 'Years Employed'). "
            "Real-world bias often involves downstream effects not captured here."
        )

    @staticmethod
    def get_legal_disclaimer() -> str:
        """
        Prevents misuse of the tool for legal compliance.
        """
        return (
            "### ⚖️ Legal & Compliance Disclaimer\n"
            "**This tool provides a technical estimate of algorithmic stability, NOT a legal certification.**\n\n"
            "1. **No Legal Advice:** These results do not constitute compliance with the GDPR, ECOA, Fair Housing Act, "
            "or other non-discrimination laws.\n"
            "2. **False Negatives:** A 'Low Risk' score means the model is mathematically consistent on *this specific dataset*. "
            "It does not guarantee the model is free from historical or societal bias.\n"
            "3. **Human Review:** All algorithmic decisions affecting humans should be subject to qualitative ethical review."
        )