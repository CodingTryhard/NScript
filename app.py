from flask import Flask, render_template, request, jsonify
import pandas as pd

from baseline_model import train_and_predict
from generator import CounterfactualGenerator
from auditor import evaluate_fairness
from metrics import FairnessMetrics
from scoring import BiasSeverityCalculator
from safeguards import EthicalSafeguards

app = Flask(
    __name__,
    template_folder="../frontend/templates",
    static_folder="../frontend/static"
)

# -------------------------------
# ROUTES (PAGES)
# -------------------------------

@app.route("/")
def landing():
    return render_template("index.html")

@app.route("/upload")
def upload():
    return render_template("upload.html")

# -------------------------------
# CORE AUDIT API
# -------------------------------

@app.route("/run_audit", methods=["POST"])
def run_audit():
    """
    Executes the full ethical audit.
    Returns JSON used by all result pages.
    """
    file = request.files["file"]
    target = request.form["target"]
    sensitive = request.form.getlist("sensitive[]")
    purpose = request.form.get("purpose", "")
    demographic = request.form.get("demographic", "")

    df = pd.read_csv(file)

    # ---------- TRAIN ----------
    all_features = [c for c in df.columns if c != target]
    numeric = df[all_features].select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in all_features if c not in numeric]

    for s in sensitive:
        if s in numeric:
            numeric.remove(s)
            categorical.append(s)

    model, _ = train_and_predict(df, target, numeric, categorical)

    # ---------- COUNTERFACTUALS ----------
    sample = df.sample(min(200, len(df)), random_state=42)
    domains = {c: df[c].unique().tolist() for c in sensitive}
    generator = CounterfactualGenerator(domains)

    cf_rows = []
    for idx, row in sample.iterrows():
        twins = generator.generate_counterfactuals(row, sensitive)
        twins["ID"] = idx
        cf_rows.append(twins)

    cf_df = pd.concat(cf_rows, ignore_index=True)

    # ---------- AUDIT ----------
    audit_results = evaluate_fairness(model, cf_df, target_class=1)
    metrics = FairnessMetrics.generate_scorecard(audit_results)
    risk = BiasSeverityCalculator().calculate_score(
        metrics,
        {"purpose": purpose, "demographic": demographic}
    )

    warnings = EthicalSafeguards.check_correlations(df, sensitive)

    return jsonify({
        "risk": {
            "severity": risk.severity_score,
            "level": risk.risk_level,
            "summary": risk.summary_text
        },
        "metrics": metrics,
        "warnings": warnings,
        "audit_results": audit_results.to_dict(orient="records")
    })


if __name__ == "__main__":
    app.run(debug=True)
