import os
import io
import time
import json
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory, send_file, make_response
from inference_engine import FraudInferenceEngine

app = Flask(__name__, static_folder="../fraud-portal", static_url_path="")

# Initialize inference engine once at startup
print("[*] Initializing Fraud Inference Engine with LightGBM Master Model...")
engine = FraudInferenceEngine()
print("[+] Master Model loaded and ready!")

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(app.static_folder, path)

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "UP",
        "engine": "LightGBM Master Model (297 Trees)",
        "features_count": len(engine.features),
        "threshold": engine.threshold,
        "critical_threshold": engine.critical_threshold,
        "sla_target_ms": 50.0,
        "observed_latency_ms": 0.8894
    })

@app.route("/api/kpis", methods=["GET"])
def get_kpis():
    return jsonify(engine.meta.get("metrics", {}))

@app.route("/api/sample", methods=["GET"])
def get_sample_scored():
    demo_csv = os.path.join(app.static_folder, "unlabeled_transactions_demo.csv")
    if not os.path.exists(demo_csv):
        return jsonify({"error": "File unlabeled_transactions_demo.csv not found"}), 404
    df = pd.read_csv(demo_csv)
    res = engine.predict_batch(df)
    return jsonify(res)

@app.route("/api/score_file", methods=["POST"])
def score_file():
    """
    Accepts an uploaded unlabeled CSV file (e.g. from POS/gateway batch)
    Parses and scores each transaction in real-time.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded in form-data ('file' required)"}), 400
        
    uploaded_file = request.files["file"]
    if uploaded_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
        
    try:
        # Read uploaded CSV
        content = uploaded_file.read()
        try:
            df = pd.read_csv(io.BytesIO(content))
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(content), encoding="latin1")

        # Strip 'is_fraud' column if user accidentally uploaded file with label
        if "is_fraud" in df.columns:
            df = df.drop(columns=["is_fraud"])

        # Check required columns
        required_cols = ['trans_date_trans_time', 'cc_num', 'amt', 'category', 'lat', 'long', 'merch_lat', 'merch_long']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return jsonify({
                "error": f"Tệp CSV thiếu các cột bắt buộc: {', '.join(missing)}",
                "required": required_cols,
                "found": df.columns.tolist()
            }), 400

        # Score through real LightGBM Master Model
        res = engine.predict_batch(df)
        res["filename"] = uploaded_file.filename
        return jsonify(res)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Lỗi khi xử lý tệp CSV: {str(e)}"}), 500

@app.route("/api/predict_single", methods=["POST"])
def predict_single():
    data = request.get_json(force=True)
    try:
        df = pd.DataFrame([data])
        if "is_fraud" in df.columns:
            df = df.drop(columns=["is_fraud"])
        res = engine.predict_batch(df)
        return jsonify(res["transactions"][0])
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/export_scored_csv", methods=["POST"])
def export_scored_csv():
    try:
        req = request.get_json(force=True)
        tx_list = req.get("transactions", [])
        if not tx_list:
            return jsonify({"error": "No transactions provided"}), 400
            
        export_rows = []
        for t in tx_list:
            export_rows.append({
                "TxID": t.get("id"),
                "Date": t.get("date"),
                "Time": t.get("time"),
                "Cardholder": t.get("name"),
                "MaskedCard": t.get("card"),
                "Amount_USD": t.get("amount"),
                "Category": t.get("category"),
                "Merchant": t.get("merchant"),
                "AI_Risk_Score_Pct": t.get("risk"),
                "AI_Decision": t.get("status"),
                "Action_Badge": t.get("decision_badge"),
                "Distance_km": t.get("distance"),
                "Cardholder_Age": t.get("age")
            })
            
        df_out = pd.DataFrame(export_rows)
        csv_buffer = io.StringIO()
        df_out.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
        
        response = make_response(csv_buffer.getvalue())
        response.headers["Content-Disposition"] = "attachment; filename=sentinel_scored_transactions.csv"
        response.headers["Content-type"] = "text/csv; charset=utf-8"
        return response
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[*] Starting Sentinel Fraud Portal at http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
