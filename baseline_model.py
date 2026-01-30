import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def generate_mock_data(n_samples=1000):
    """
    Generates a synthetic dataset for Loan Approval.
    Includes potential protected attributes (e.g., 'gender') to test fairness later.
    """
    np.random.seed(42)
    data = pd.DataFrame({
        'income': np.random.normal(50000, 15000, n_samples),
        'credit_score': np.random.normal(650, 50, n_samples),
        'years_employed': np.random.randint(0, 20, n_samples),
        'education': np.random.choice(['High School', 'Bachelors', 'Masters'], n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples), # Protected Attribute
        'loan_approved': np.random.choice([0, 1], n_samples) # Target
    })
    return data

def build_model_pipeline(numeric_features, categorical_features):
    """
    Constructs a scikit-learn pipeline with preprocessing and the classifier.
    """
    # 1. Define preprocessing for numeric columns (scaling)
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # 2. Define preprocessing for categorical columns (one-hot encoding)
    # handle_unknown='ignore' ensures the model doesn't crash on unseen categories
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # 3. Combine them into a single ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 4. create the full pipeline with Logistic Regression
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(solver='liblinear', random_state=42))
    ])

    return pipeline

def train_and_predict(data, target_col, numeric_feats, cat_feats):
    """
    Splits data, trains the model, and generates predictions/probabilities.
    """
    # Split features and target
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build Pipeline
    model = build_model_pipeline(numeric_feats, cat_feats)

    # Train the model
    print("Training model...")
    model.fit(X_train, y_train)

    # Generate Predictions
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] # Probability of class 1 (Approved)

    # Calculate basic accuracy
    acc = accuracy_score(y_test, preds)
    print(f"Model trained. Accuracy on test set: {acc:.4f}")

    # Return the test set enriched with predictions for analysis
    results_df = X_test.copy()
    results_df['actual'] = y_test
    results_df['prediction'] = preds
    results_df['probability'] = probs

    return model, results_df

# --- Execution ---

if __name__ == "__main__":
    # 1. Get Data
    df = generate_mock_data()

    # 2. Define Feature Metadata
    NUMERIC_FEATURES = ['income', 'credit_score', 'years_employed']
    CATEGORICAL_FEATURES = ['education', 'gender'] # 'gender' is our protected attribute
    TARGET_COLUMN = 'loan_approved'

    # 3. Run Pipeline
    trained_model, output_df = train_and_predict(df, TARGET_COLUMN, NUMERIC_FEATURES, CATEGORICAL_FEATURES)

    # 4. Preview Output
    print("\nSample Predictions:")
    print(output_df[['gender', 'income', 'prediction', 'probability']].head())