
from pathlib import Path
import joblib
from flask import Flask, jsonify, request, send_from_directory
from modelo_v2 import analyze_bundle

ROOT = Path(__file__).resolve().parent
app = Flask(__name__)

bundle = joblib.load(
    ROOT / "resultados_v2" / "modelo_selecionado.joblib"
)

@app.get("/")
def home():
    return send_from_directory(ROOT / "interface_web", "index.html")

@app.get("/app.css")
def css():
    return send_from_directory(ROOT / "interface_web", "app.css")

@app.get("/app.js")
def js():
    return send_from_directory(ROOT / "interface_web", "app.js")

@app.get("/api/config")
def config():
    return jsonify({
        "version": 2,
        "model": bundle["model_name"],
        "policy": bundle["policy"],
        "scope": bundle["scope"]
    })

@app.post("/api/analyze")
def analyze():
    if not request.is_json:
        return jsonify({"error": "Envie JSON."}), 415

    data = request.get_json(silent=True)
    text = data.get("text") if isinstance(data, dict) else None

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Informe um texto."}), 400

    if len(text) > 100000:
        return jsonify({"error": "Texto muito grande."}), 413

    try:
        return jsonify(analyze_bundle(bundle, text))
    except ValueError:
        return jsonify({"error": "Informe um texto não vazio, com até 100.000 caracteres."}), 400
