import pandas as pd
import numpy as np
from itertools import product

class CounterfactualGenerator:
    """
    Generates 'Twin' data points for fairness auditing.
    
    This class implements the intervention step of causal inference.
    It asks the question: "What if this specific applicant had been different 
    only in their protected attribute, holding all else constant?"
    """

    def __init__(self, domain_mappings: dict = None):
        """
        Args:
            domain_mappings: Dictionary defining valid flips for categorical variables.
                             Example: {'gender': ['Male', 'Female'], 'race': ['White', 'Black', 'Asian']}
        """
        # Default simple binary mappings if none provided
        self.domain_mappings = domain_mappings if domain_mappings else {
            'gender': ['Male', 'Female'],
            'age_group': ['Young', 'Middle', 'Senior']
        }

    def _is_plausible(self, original_row: pd.Series, counterfactual_row: pd.Series) -> bool:
        """
        Enforces logical constraints to prevent impossible counterfactuals.
        
        Example Constraint:
        - If we flip 'pregnancy_status' to True, 'gender' must be 'Female'.
        - (For this basic implementation, we return True, but this is where 
          domain-specific logic lives).
        """
        # Example Logic:
        # if counterfactual_row['gender'] == 'Male' and original_row['is_pregnant'] == 1:
        #     return False
        return True

    def generate_counterfactuals(self, 
                                 instance: pd.Series, 
                                 sensitive_attributes: list) -> pd.DataFrame:
        """
        Generates a DataFrame of counterfactual instances.

        Fairness Logic:
        1. Identification: We isolate the specific sensitive attributes.
        2. Intervention: We iterate through all possible alternative values 
           for these attributes defined in domain_mappings.
        3. Ceteris Paribus: We intentionally DO NOT change other features 
           (like income or credit_score). This isolates the DIRECT causal effect 
           of the sensitive attribute on the model's prediction.

        Args:
            instance: A single row of data (pd.Series).
            sensitive_attributes: List of columns to perturb (e.g., ['gender']).

        Returns:
            pd.DataFrame containing the original instance AND all generated twins.
        """
        
        counterfactuals = []
        
        # Add the original instance first (Baseline)
        original_df = instance.to_frame().T
        original_df['_cf_type'] = 'Original'
        counterfactuals.append(original_df)

        for attr in sensitive_attributes:
            if attr not in self.domain_mappings:
                print(f"Warning: No mapping found for {attr}. Skipping.")
                continue

            current_val = instance[attr]
            possible_values = self.domain_mappings[attr]

            # Generate a twin for every OTHER possible value
            for val in possible_values:
                if val == current_val:
                    continue # Skip the value the user already has

                # Create the copy
                cf_instance = instance.copy()
                
                # The Intervention: Flip the bit
                cf_instance[attr] = val
                
                # Validation
                if self._is_plausible(instance, cf_instance):
                    cf_df = cf_instance.to_frame().T
                    cf_df['_cf_type'] = f'Counterfactual ({attr}={val})'
                    counterfactuals.append(cf_df)

        # Combine into a single DataFrame for batch prediction
        full_cf_df = pd.concat(counterfactuals, ignore_index=True)
        return full_cf_df

# --- Integration Test ---
if __name__ == "__main__":
    # 1. Create a mock applicant (Female, High Income)
    applicant = pd.Series({
        'income': 75000,
        'credit_score': 720,
        'years_employed': 5,
        'education': 'Bachelors',
        'gender': 'Female',      # Protected
        'loan_approved': 1       # Ground Truth (ignored for generation)
    })

    # 2. Define Mappings
    mappings = {'gender': ['Male', 'Female', 'Non-Binary']}

    # 3. Instantiate Generator
    cf_gen = CounterfactualGenerator(domain_mappings=mappings)

    # 4. Generate
    print("Generating Counterfactuals...")
    cf_results = cf_gen.generate_counterfactuals(applicant, sensitive_attributes=['gender'])

    # 5. Review
    print(cf_results[['_cf_type', 'gender', 'income', 'credit_score']])