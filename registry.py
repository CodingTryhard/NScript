import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class FairnessMetadata:
    """
    Immutable container for fairness configuration.
    
    Attributes:
        protected_attributes: Variables legally or ethically shielded from discrimination 
                             (e.g., 'gender', 'race').
        proxy_attributes: Variables that act as stand-ins for protected attributes 
                         (e.g., 'zip_code' often proxies for 'race').
        attribute_types: Inferred data types for validation.
    """
    protected_attributes: List[str]
    proxy_attributes: List[str]
    attribute_types: Dict[str, str]

class SensitiveAttributeRegistry:
    """
    Manages the identification and validation of sensitive data features for AI auditing.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the registry with the dataset to be audited.
        
        Args:
            df (pd.DataFrame): The training or inference dataset.
        """
        self.df = df
        self.metadata: Optional[FairnessMetadata] = None

    def register_attributes(self, protected: List[str], proxies: Optional[List[str]] = None) -> FairnessMetadata:
        """
        Registers and validates sensitive attributes.

        Ethical Relevance:
            - Explicitly naming 'protected' attributes prevents 'fairness through unawareness' 
              (pretending the variable doesn't exist, which usually fails).
            - Identifying 'proxies' is crucial because models can reconstruct sensitive 
              attributes from correlations (e.g., 'zip_code' predicting 'race'), leading to 
              Indirect Discrimination.

        Args:
            protected: List of column names to protect (e.g., ['gender']).
            proxies: List of column names that correlate with protected groups (e.g., ['zip_code']).
        
        Returns:
            FairnessMetadata object containing the validated configuration.
        
        Raises:
            ValueError: If specified columns are missing from the dataframe.
        """
        if proxies is None:
            proxies = []

        # 1. Validation: Ensure all columns exist
        missing_protected = [col for col in protected if col not in self.df.columns]
        missing_proxies = [col for col in proxies if col not in self.df.columns]
        
        if missing_protected:
            raise ValueError(f"CRITICAL ERROR: Protected attributes not found in dataset: {missing_protected}")
        if missing_proxies:
            raise ValueError(f"CRITICAL ERROR: Proxy attributes not found in dataset: {missing_proxies}")

        # 2. Validation: Ensure no overlap
        overlap = set(protected).intersection(set(proxies))
        if overlap:
            raise ValueError(f"Ambiguity detected: Columns defined as both protected and proxy: {overlap}")

        # 3. Type Inference (for future causal modeling)
        # We need to know if variables are categorical or continuous for SCM generation.
        attr_types = {col: str(self.df[col].dtype) for col in protected + proxies}

        # 4. Lock in Metadata
        self.metadata = FairnessMetadata(
            protected_attributes=protected,
            proxy_attributes=proxies,
            attribute_types=attr_types
        )
        
        print(f"SUCCESS: Registered {len(protected)} protected attributes and {len(proxies)} proxies.")
        return self.metadata

    def get_audit_config(self) -> Dict:
        """Returns a dictionary representation for logging/reporting."""
        if not self.metadata:
            raise RuntimeError("Attributes have not been registered yet.")
        
        return {
            "protected": self.metadata.protected_attributes,
            "proxies": self.metadata.proxy_attributes,
            "status": "Active"
        }