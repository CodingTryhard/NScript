# AI Ethics Auditor
Bias, Counterfactual Fairness and Confidence Stability Assessment Tool

---

## Overview

AI Ethics Auditor is a web-based auditing system designed to evaluate machine learning decision systems for bias, proxy discrimination, counterfactual fairness violations, prediction confidence instability, and overall ethical deployment risk.

The system is built for policymakers, judges, evaluators, data scientists, and non-technical stakeholders. It prioritizes transparency, interpretability, and accountability over opaque model performance metrics.

---

## Key Features

- CSV-based auditing without requiring model retraining
- Proxy variable detection for protected attributes
- Counterfactual sensitivity analysis
- Confidence instability measurement and visualization
- Human-readable ethical assessment report
- Modular and extensible backend architecture

---

## Audit Pipeline

### Step 1 Upload Dataset
The user uploads a CSV file containing a decision target and candidate protected attributes.

### Step 2 Configure Audit
The user selects the target decision label and protected attributes for fairness evaluation.

### Step 3 Biasing Conditions and Proxy Detection
The system identifies correlations between protected attributes and other features and flags potential proxy discrimination risks. Interpretive and legal warnings are displayed.

### Step 4 Counterfactual Analysis
The system performs all-else-equal counterfactual testing to evaluate prediction sensitivity to protected attribute changes while highlighting real-world plausibility constraints.

### Step 5 Confidence Instability Analysis
Variance in prediction confidence across protected attributes is measured and visualized. High instability signals hidden bias risk.

### Step 6 Final Ethics Report
An aggregated ethical assessment is generated including bias severity score, minor and critical flags, deployment recommendation, and a plain-language summary with legal disclaimers.

---
## Technology Stack

### Backend
- Python 3.10 or higher
- Flask for web server and routing
- Pandas for dataset handling and analysis
- NumPy for numerical computation
- Scikit-learn for baseline modeling and statistical evaluation
- Jinja2 for server-side HTML templating

### Frontend
- HTML5
- Tailwind CSS via CDN for styling
- Chart.js for interactive visualizations
- Google Material Symbols for icons

### Data
- CSV-based datasets
- No database required
- Session-based state management

---
Built with a lot of suffering,  
Adwaith Shameer, Vishnu Sarang and Rahul Arun for the ISTE SC MBCET, 2026.
