import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of candidates
n = 200

# 1. Generate Features
data = {
    'candidate_id': range(1001, 1001 + n),
    'years_experience': np.random.randint(0, 15, n),
    'coding_test_score': np.random.normal(70, 15, n).astype(int),
    'degree': np.random.choice(['Bachelor', 'Master', 'PhD'], n, p=[0.6, 0.3, 0.1]),
    'gender': np.random.choice(['Male', 'Female'], n),
    'referral': np.random.choice([0, 1], n, p=[0.8, 0.2]) # 1 = Has referral
}

df = pd.DataFrame(data)

# Clip scores to 0-100 range
df['coding_test_score'] = df['coding_test_score'].clip(0, 100)

# 2. Define Hiring Logic (With Intentionally Injected Bias)
def hiring_logic(row):
    # Base score depends on merit
    score = (row['years_experience'] * 4) + (row['coding_test_score'] * 0.6)
    
    # Merit bonus for degree
    if row['degree'] == 'Master': score += 10
    if row['degree'] == 'PhD': score += 20
    if row['referral'] == 1: score += 15
    
    # --- INJECTED BIAS ---
    # Males get a "hidden bonus" of +15 points
    if row['gender'] == 'Male':
        score += 15
        
    # Threshold for hiring
    # This means a Female needs a higher raw score to pass the same threshold
    return 1 if score > 95 else 0

df['hired'] = df.apply(hiring_logic, axis=1)

# 3. Save to CSV
filename = "hiring_data.csv"
df.to_csv(filename, index=False)

print(f"✅ Generated {filename} with {n} rows.")
print("Bias injection: 'Male' candidates have a +15 point hidden advantage.")