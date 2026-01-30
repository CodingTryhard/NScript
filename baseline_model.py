import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier # Changed from LogisticRegression
from sklearn.metrics import accuracy_score

def infer_feature_types(df: pd.DataFrame, target_col: str, sensitive_cols: list):
    all_cols = [c for c in df.columns if c != target_col]
    numeric_features = df[all_cols].select_dtypes(include=['number']).columns.tolist()
    categorical_features = df[all_cols].select_dtypes(exclude=['number']).columns.tolist()
    
    # Force sensitive attributes to be categorical
    for sens in sensitive_cols:
        if sens in numeric_features:
            numeric_features.remove(sens)
        if sens not in categorical_features:
            categorical_features.append(sens)
            
    return numeric_features, categorical_features

def build_model_pipeline(numeric_features, categorical_features):
    transformers = []
    
    if numeric_features:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', numeric_transformer, numeric_features))

    if categorical_features:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        transformers.append(('cat', categorical_transformer, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers)

    # UPDATED: Using Random Forest
    # Trees capture non-linear bias (like specific threshold bonuses) much better than Regression.
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=50, 
            max_depth=10, 
            random_state=42
        ))
    ])

    return pipeline

def train_and_predict(data: pd.DataFrame, target_col: str, 
                     numeric_feats: list = None, cat_feats: list = None,
                     sensitive_cols: list = None):
    if sensitive_cols is None: sensitive_cols = []
    if target_col not in data.columns: raise ValueError(f"Target '{target_col}' not found.")
    
    if numeric_feats is None or cat_feats is None:
        numeric_feats, cat_feats = infer_feature_types(data, target_col, sensitive_cols)

    X = data.drop(columns=[target_col])
    y = data[target_col]

    if not pd.api.types.is_numeric_dtype(y):
        y = y.astype('category').cat.codes

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_model_pipeline(numeric_feats, cat_feats)
    model.fit(X_train, y_train)

    # Predict
    preds = model.predict(X_test)
    if hasattr(model.named_steps['classifier'], "predict_proba"):
        probs = model.predict_proba(X_test)[:, -1]
    else:
        probs = np.zeros(len(preds))

    results_df = X_test.copy()
    results_df['actual'] = y_test
    results_df['prediction'] = preds
    results_df['probability'] = probs

    return model, results_df