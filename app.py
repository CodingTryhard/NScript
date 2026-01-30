import streamlit as st
import pandas as pd
import numpy as np

# --- IMPORT MODULES ---
# Ensure baseline_model.py, registry.py, etc., are in the same folder
try:
    from baseline_model import train_and_predict
    from registry import SensitiveAttributeRegistry
    from generator import CounterfactualGenerator
    from auditor import evaluate_fairness
    from metrics import FairnessMetrics
    from scoring import BiasSeverityCalculator
except ImportError as e:
    st.error(f"CRITICAL ERROR: Missing Module Files. {e}")
    st.stop()

# ==========================================
# BACKEND ORCHESTRATOR
# ==========================================

class FairnessAuditorBackend:
    @staticmethod
    def run_full_audit(df: pd.DataFrame, sensitive_features: list, target_col: str):
        # 1. SETUP & TRAINING
        all_features = [c for c in df.columns if c != target_col]
        numeric_features = [c for c in df.select_dtypes(include=['number']).columns if c in all_features]
        categorical_features = [c for c in df.select_dtypes(exclude=['number']).columns if c in all_features]
        
        # Handle case where sensitive feature is numeric but needs to be treated as categorical
        for sens in sensitive_features:
            if sens not in categorical_features and sens in numeric_features:
                categorical_features.append(sens)
                numeric_features.remove(sens)

        model, _ = train_and_predict(df, target_col, numeric_features, categorical_features)

        # 2. GENERATE COUNTERFACTUALS
        sample_size = min(50, len(df))
        sample_df = df.sample(sample_size, random_state=42)
        
        # Get domains for valid flips
        domains = {col: df[col].unique().tolist() for col in sensitive_features}
        generator = CounterfactualGenerator(domain_mappings=domains) 
        
        cf_data_list = []
        for index, row in sample_df.iterrows():
            twins = generator.generate_counterfactuals(row, sensitive_features)
            twins['ID'] = index # Track ID for visualization
            cf_data_list.append(twins)
            
        full_cf_df = pd.concat(cf_data_list, ignore_index=True)

        # 3. AUDIT PREDICTIONS
        audit_results = evaluate_fairness(model, full_cf_df, target_class=1)

        # 4. METRICS & SCORING
        scorecard = FairnessMetrics.generate_scorecard(audit_results)
        risk_calc = BiasSeverityCalculator()
        risk_assessment = risk_calc.calculate_score(scorecard)

        # 5. CLEANUP & RENAMING (FIXED HERE)
        display_df = audit_results.copy()
        
        # Helper to parse what changed (if not present)
        if 'changed_attr' not in display_df.columns:
            display_df['changed_attr'] = display_df['_cf_type'].apply(
                lambda x: x.split('(')[1].split('=')[0] if '(' in x else 'None'
            )

        # RENAME COLUMNS to match UI expectations
        display_df = display_df.rename(columns={
            '_cf_type': 'Type',
            'model_prediction': 'Prediction',
            'model_probability': 'Probability'
        })

        return display_df, scorecard, risk_assessment

# ==========================================
# STREAMLIT USER INTERFACE
# ==========================================

st.set_page_config(page_title="AI Fairness Auditor", layout="wide")

st.title("🛡️ Counterfactual Fairness Auditor")
st.markdown("Stress-test your model by generating 'Twin' applicants.")

st.sidebar.header("1. Audit Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Training Data (CSV)", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success(f"Data Loaded: {len(df)} rows")
        
        all_cols = df.columns.tolist()
        target_default = len(all_cols)-1
        target_col = st.sidebar.selectbox("Target Variable (Label)", all_cols, index=target_default)
        feature_cols = [c for c in all_cols if c != target_col]
        
        sensitive_cols = st.sidebar.multiselect("Select Protected Attributes", feature_cols)

        if st.sidebar.button("🚀 RUN AUDIT", type="primary"):
            if not sensitive_cols:
                st.sidebar.error("Select at least one attribute to audit.")
            else:
                with st.spinner('Simulating Counterfactuals...'):
                    
                    audit_df, metrics, risk_obj = FairnessAuditorBackend.run_full_audit(
                        df, sensitive_cols, target_col
                    )

                    # --- DASHBOARD START ---
                    st.divider()
                    
                    # 1. SCORECARD
                    st.subheader("2. Executive Risk Assessment")
                    score = risk_obj.severity_score
                    if score < 20:
                        color = "green"
                    elif score < 50:
                        color = "orange"
                    else:
                        color = "red"
                        
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Bias Severity Index", f"{score}/100")
                    col2.metric("Consistency Score", f"{metrics['consistency_score']*100:.1f}%")
                    col3.metric("Avg Conf. Sensitivity", f"{metrics['confidence_sensitivity']:.3f}")
                    col4.markdown(f"### :{color}[{risk_obj.risk_level}]")

                    st.divider()
                    c1, c2 = st.columns([1, 2])
                    
                    # 2. FLIP RATES (Simple Table)
                    with c1:
                        st.markdown("#### 📉 Failure Rates")
                        st.caption("How often does changing this attribute change the decision?")
                        fr_df = pd.DataFrame.from_dict(metrics['flip_rates'], orient='index', columns=['Flip Rate'])
                        fr_df['Flip Rate'] = fr_df['Flip Rate'].apply(lambda x: f"{x*100:.1f}%")
                        st.table(fr_df)

                    # 3. HUMAN-READABLE COMPARISON
                    with c2:
                        st.markdown("#### 🔍 Bias Inspector")
                        st.caption("Comparing specific 'Twins' where the model was unfair.")
                        
                        # Filter only flipped cases
                        flips = audit_df[audit_df['label_changed'] == True]
                        
                        if flips.empty:
                            st.success("✅ No bias detected! The model treated all twins identically in this sample.")
                        else:
                            # Show top 3 worst offenders
                            st.warning(f"Found {len(flips)} instances of Explicit Bias.")
                            
                            unique_flip_ids = flips['ID'].unique()[:3] # Limit to 3 examples
                            
                            for uid in unique_flip_ids:
                                # Get the Original row and the Flipped Twin row
                                original_row = audit_df[(audit_df['ID'] == uid) & (audit_df['Type'] == 'Original')].iloc[0]
                                twin_rows = flips[flips['ID'] == uid]
                                
                                for _, twin_row in twin_rows.iterrows():
                                    with st.container(border=True):
                                        st.markdown(f"**Applicant #{uid} Analysis**")
                                        
                                        col_a, col_b, col_arrow = st.columns([3, 3, 1])
                                        
                                        # ORIGINAL
                                        with col_a:
                                            st.markdown(":grey[**Original Applicant**]")
                                            # Show the specific attribute value (e.g. Male)
                                            feat = twin_row['changed_attr']
                                            val = original_row[feat] if feat in original_row else "N/A"
                                            st.code(f"{feat}: {val}")
                                            
                                            # Show Outcome
                                            pred_label = "Approved" if original_row['Prediction'] == 1 else "Rejected"
                                            prob_fmt = f"{original_row['Probability']*100:.1f}%"
                                            st.metric("Outcome", pred_label, prob_fmt)

                                        # TWIN
                                        with col_b:
                                            st.markdown(f":orange[**Counterfactual Twin**]")
                                            # Show the flipped value (e.g. Female)
                                            val_twin = twin_row[feat] if feat in twin_row else "N/A"
                                            st.code(f"{feat}: {val_twin}")
                                            
                                            # Show Outcome
                                            pred_label_twin = "Approved" if twin_row['Prediction'] == 1 else "Rejected"
                                            prob_fmt_twin = f"{twin_row['Probability']*100:.1f}%"
                                            
                                            # Calculate Delta for color
                                            delta = twin_row['Probability'] - original_row['Probability']
                                            st.metric("Outcome", pred_label_twin, f"{delta*100:.1f}%", delta_color="inverse")

                                        with col_arrow:
                                            st.markdown("## ➡️")
                                            st.caption("Flipped")

    except Exception as e:
        st.error(f"Error: {e}")
        st.exception(e)
else:
    # LANDING PAGE STATE
    st.info("👋 Welcome. Please upload a CSV dataset to begin the audit.")
    
    with st.expander("Need sample data?"):
        # Generate the dummy data from Module 1 for the user to download
        # We quickly mock it here to save them import trouble if they just want a CSV
        dummy = pd.DataFrame({
            'income': np.random.normal(50000, 15000, 100),
            'credit_score': np.random.normal(650, 50, 100),
            'years_employed': np.random.randint(0, 20, 100),
            'gender': np.random.choice(['Male', 'Female'], 100),
            'loan_approved': np.random.choice([0, 1], 100)
        })
        st.download_button(
            "Download Sample CSV", 
            dummy.to_csv(index=False), 
            "sample_loan_data.csv", 
            "text/csv"
        )