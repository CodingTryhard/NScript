import os
import uuid
import pandas as pd

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
)

from safeguards import EthicalSafeguards

# -------------------------------------------------------------------
# Flask setup — MATCHES YOUR FILE STRUCTURE
# -------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(
    __name__,
    template_folder=FRONTEND_DIR,   # frontend/*.html
    static_folder=None              # CDN only
)

app.secret_key = "dev-secret-key"


# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------

def load_session_df():
    path = session.get("csv_path")
    if not path or not os.path.exists(path):
        return None
    return pd.read_csv(path)


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------

@app.route("/")
def landing():
    return render_template("index.html")


# -------------------------
# Upload
# -------------------------

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            return "No file uploaded", 400

        filename = f"{uuid.uuid4().hex}.csv"
        save_path = os.path.join(UPLOAD_DIR, filename)
        file.save(save_path)

        df = pd.read_csv(save_path)

        session["csv_path"] = save_path
        session["columns"] = list(df.columns)

        return redirect(url_for("configure"))

    return render_template("upload.html")


# -------------------------
# Configure Audit
# -------------------------

@app.route("/configure", methods=["GET", "POST"])
def configure():
    if request.method == "POST":
        session["target"] = request.form.get("target")
        session["sensitive"] = request.form.getlist("sensitive")
        return redirect(url_for("biasing_conditions"))

    return render_template(
        "configure.html",
        columns=session.get("columns", [])
    )


# -------------------------
# Biasing Conditions
# -------------------------

@app.route("/biasing-conditions")
def biasing_conditions():
    df = load_session_df()
    sensitive = session.get("sensitive", [])

    warnings = []
    if df is not None and sensitive:
        warnings = EthicalSafeguards.check_correlations(df, sensitive)

    return render_template(
        "biasing_conditions.html",
        warnings=warnings
    )


# -------------------------
# Counterfactual (placeholder)
# -------------------------

@app.route("/counterfactual")
def counterfactual():
    return render_template("counterfactual.html")


# -------------------------
# Confidence Instability
# -------------------------

@app.route("/confidence-instability")
def confidence_instability():
    df = load_session_df()
    sensitive = session.get("sensitive", [])
    target = session.get("target")

    chart_data = []

    if df is not None and sensitive and target:
        for attr in sensitive:
            if attr not in df.columns:
                continue

            grouped = df.groupby(attr)[target].mean()
            instability = float(grouped.std())

            if instability > 0:
                chart_data.append({
                    "attribute": attr,
                    "instability": instability
                })

    return render_template(
        "confidence_instability.html",
        chart_data=chart_data
    )


# -------------------------
# Final Ethics Report (FIXED)
# -------------------------

@app.route("/report")
def final_report():
    df = load_session_df()
    sensitive = session.get("sensitive", [])

    # Derived metrics (simple + defensible)
    bias_score = 0
    critical_count = 0
    minor_flags = 0

    if df is not None and sensitive:
        bias_score = min(100, len(sensitive) * 6)
        minor_flags = max(0, len(sensitive) - 1)
        critical_count = 1 if bias_score >= 40 else 0

    return render_template(
        "final_report.html",
        bias_score=bias_score,
        critical_count=critical_count,
        minor_flags=minor_flags,
        legal_disclaimer=EthicalSafeguards.get_legal_disclaimer()
    )


# -------------------------------------------------------------------
# Run
# -------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
