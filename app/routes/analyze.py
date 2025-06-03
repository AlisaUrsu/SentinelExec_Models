from flask import Blueprint, request, jsonify
from app.services.model_service import analyze_file
from app.utils.helpers import categorize_score

analyze_bp = Blueprint('analyze', __name__)

@analyze_bp.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("file")
   
    try:
        score, raw_features = analyze_file(file)
        label, message = categorize_score(score)
        return jsonify({
            "file": file.filename,
            "score": round(score, 6),
            "label": label,
            "message": message,
            "rawFeatures": raw_features
        })
    except Exception as e:
        return jsonify({"error": "Analysis failed", "details": str(e)}), 500
