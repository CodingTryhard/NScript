import streamlit as st
import pandas as pd
import numpy as np
import time

# --- IMPORT MODULES ---
try:
    from baseline_model import train_and_predict
    from registry import SensitiveAttributeRegistry
    from generator import CounterfactualGenerator
    from auditor import evaluate_fairness
    from metrics import FairnessMetrics
    from scoring import BiasSeverityCalculator
    from safeguards import EthicalSafeguards
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
        
        for sens in sensitive_features:
            if sens not in categorical_features and sens in numeric_features:
                categorical_features.append(sens)
                numeric_features.remove(sens)

        model, _ = train_and_predict(df, target_col, numeric_features, categorical_features)

        # 2. GENERATE COUNTERFACTUALS
        sample_size = min(50, len(df))
        sample_df = df.sample(sample_size, random_state=42)
        
        domains = {col: df[col].unique().tolist() for col in sensitive_features}
        generator = CounterfactualGenerator(domain_mappings=domains) 
        
        cf_data_list = []
        for index, row in sample_df.iterrows():
            twins = generator.generate_counterfactuals(row, sensitive_features)
            twins['ID'] = index 
            cf_data_list.append(twins)
            
        full_cf_df = pd.concat(cf_data_list, ignore_index=True)

        # 3. AUDIT PREDICTIONS
        audit_results = evaluate_fairness(model, full_cf_df, target_class=1)

        # 4. METRICS & SCORING
        scorecard = FairnessMetrics.generate_scorecard(audit_results)
        risk_calc = BiasSeverityCalculator()
        risk_assessment = risk_calc.calculate_score(scorecard)

        # 5. CLEANUP
        display_df = audit_results.copy()
        
        if 'changed_attr' not in display_df.columns:
            display_df['changed_attr'] = display_df['_cf_type'].apply(
                lambda x: x.split('(')[1].split('=')[0] if '(' in x else 'None'
            )

        display_df = display_df.rename(columns={
            '_cf_type': 'Type',
            'model_prediction': 'Prediction',
            'model_probability': 'Probability'
        })

        return display_df, scorecard, risk_assessment

# ==========================================
# STREAMLIT USER INTERFACE
# ==========================================

st.set_page_config(page_title="Fairness Auditor", layout="wide", page_icon="⚖️")

# --- HEADER ---
col_t1, col_t2 = st.columns([4, 1])
with col_t1:
    st.title("⚖️ Fairness Auditor")
    st.markdown("### Algorithmic Health Check")
    st.caption("Upload your data to check for hidden biases and stability issues.")

# --- SIDEBAR: CONFIG ---
with st.sidebar:
    st.header("⚙️ Audit Setup")
    uploaded_files = st.file_uploader("1. Upload Data (CSV)", type="csv", accept_multiple_files=True)
    
    run_btn = False
    
    if uploaded_files:
        st.success(f"Files Ready: {len(uploaded_files)}")
        try:
            # Config based on first file
            first_df = pd.read_csv(uploaded_files[0])
            all_cols = first_df.columns.tolist()
            
            if not all_cols:
                st.error("File appears empty.")
            else:
                st.markdown("#### 2. Select Columns")
                target_col = st.selectbox("What is the model predicting?", all_cols, index=len(all_cols)-1, help="The target label (e.g. Approved/Rejected)")
                
                feature_cols = [c for c in all_cols if c != target_col]
                sensitive_cols = st.multiselect("Which attributes should be protected?", feature_cols, help="e.g. Gender, Race, Age")
                
                st.markdown("#### 3. Options")
                enable_safeguards = st.checkbox("Check for Data Quality", value=True, help="Scans for proxy variables.")
                
                st.divider()
                run_btn = st.button("Start Fairness Audit", type="primary", use_container_width=True)
        except Exception as e:
            st.error(f"Config Error: {e}")
    else:
        st.info("Awaiting file upload...")

# --- MAIN CONTENT ---

if run_btn and uploaded_files:
    if not sensitive_cols:
        st.warning("⚠️ Please select at least one protected attribute (e.g., Gender) to audit.")
    else:
        comparison_results = []
        progress_bar = st.progress(0)
        
        for i, file in enumerate(uploaded_files):
            dataset_name = file.name
            
            # --- CARD LAYOUT FOR EACH DATASET ---
            with st.container(border=True):
                col_title, col_status = st.columns([3, 1])
                with col_title:
                    st.subheader(f"📄 Report: {dataset_name}")
                
                try:
                    file.seek(0)
                    df = pd.read_csv(file)
                    
                    # Schema Validation
                    missing = [c for c in ([target_col] + sensitive_cols) if c not in df.columns]
                    if missing:
                        st.error(f"Skipping {dataset_name}: Missing columns {missing}")
                        progress_bar.progress((i + 1) / len(uploaded_files))
                        continue

                    # Safeguards
                    proxy_warnings = []
                    if enable_safeguards:
                        with st.spinner(f"Analyzing {dataset_name} for data quality..."):
                            proxy_warnings = EthicalSafeguards.check_correlations(df, sensitive_cols)

                    # Run Backend
                    audit_df, metrics, risk_obj = FairnessAuditorBackend.run_full_audit(df, sensitive_cols, target_col)
                    
                    # --- HUMAN-READABLE RESULTS ---
                    
                    # 1. Determine Status
                    score = risk_obj.severity_score
                    consistency = metrics['consistency_score']
                    
                    if score < 20:
                        status_color = "green"
                        status_text = "Stable & Fair"
                        status_icon = "✅"
                    elif score < 50:
                        status_color = "orange"
                        status_text = "Review Needed"
                        status_icon = "⚠️"
                    else:
                        status_color = "blue"
                        status_text = "Significant Bias"
                        status_icon = "ℹ️"

                    with col_status:
                        st.markdown(f"#### :{status_color}[{status_icon} {status_text}]")

                    # 2. Main Metrics Grid
                    m1, m2, m3 = st.columns(3)
                    
                    with m1:
                        st.metric("Stability Rating", f"{consistency*100:.0f}%", help="Percentage of applicants who received the same decision regardless of sensitive attributes.")
                        st.progress(consistency)
                    
                    with m2:
                        # Inverse logic: Low bias score is good
                        health_score = 100 - score
                        st.metric("Fairness Score", f"{health_score:.0f}/100", help="Overall health score. Higher is better.")
                        st.progress(health_score / 100)
                        
                    with m3:
                        # Narrative
                        if score < 20:
                            st.caption("This model treats applicants consistently across the selected attributes.")
                        else:
                            st.caption(f"The model decision changes for {risk_obj.breakdown['Max Flip Impact']:.0f}% of applicants when protected attributes are swapped.")

                    # 3. Data Insights (Expandable)
                    if enable_safeguards and proxy_warnings:
                        with st.expander("⚠️ Data Quality Alerts", expanded=False):
                            for w in proxy_warnings:
                                st.info(w)

                    with st.expander("🔍 View Details & Charts"):
                        c_chart, c_text = st.columns([2, 1])
                        
                        with c_chart:
                            st.markdown("**Disparity by Attribute**")
                            st.caption("Which attribute triggers the most decision changes?")
                            # Clean chart data
                            chart_df = pd.DataFrame.from_dict(metrics['flip_rates'], orient='index', columns=['Change Rate'])
                            st.bar_chart(chart_df)
                            
                        with c_text:
                            st.markdown("**Executive Summary**")
                            st.write(risk_obj.summary_text)

                    # Store for comparison table
                    comparison_results.append({
                        "Dataset": dataset_name,
                        "Status": status_text,
                        "Fairness Score": f"{100-score:.0f}/100",
                        "Stability": f"{consistency*100:.1f}%"
                    })
                    
                except Exception as e:
                    st.error(f"Could not process {dataset_name}. Error: {e}")
            
            progress_bar.progress((i + 1) / len(uploaded_files))

        # --- FINAL COMPARISON ---
        if comparison_results:
            st.divider()
            st.subheader("🏆 Dataset Comparison")
            st.markdown("Quickly compare the fairness profile of your uploaded files.")
            
            comp_df = pd.DataFrame(comparison_results)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

elif not uploaded_files:
    # Empty State - Helpful Context
    st.info("👋 Welcome! Please upload a CSV file to the sidebar to start your audit.")
    
    col_help1, col_help2 = st.columns(2)
    with col_help1:
        st.markdown("""
        **What does this tool do?**
        It creates "Counterfactual Twins" for your data points—imaginary applicants who are identical in every way except for a protected attribute (like Gender).
        """)
    with col_help2:
        st.markdown("""
        **Why use it?**
        To check if your AI model is making decisions based on skills/merit, or if it's secretly relying on protected demographic traits.
        """)

    with st.expander("Need a sample file to test?"):
         dummy = pd.DataFrame({
            'income': np.random.normal(50000, 15000, 100),
            'credit_score': np.random.normal(650, 50, 100),
            'years_employed': np.random.randint(0, 20, 100),
            'gender': np.random.choice(['Male', 'Female'], 100),
            'zip_code': np.random.choice(['Urban', 'Rural'], 100),
            'loan_approved': np.random.choice([0, 1], 100)
        })
         st.download_button("Download Sample CSV", dummy.to_csv(index=False), "sample_data.csv", "text/csv")