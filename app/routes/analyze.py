from flask import Blueprint, request, jsonify
from app.services.model_service import analyze_file
from app.utils.helpers import categorize_score

analyze_bp = Blueprint('analyze', __name__)

@analyze_bp.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("file")
    
    allowed_extensions = {'.exe', '.dll'}
    if not (file.filename.lower().endswith(tuple(allowed_extensions))):
        return jsonify({"error": "Invalid file type."}), 400
    
    print(f"Detected file size: {file.content_length} bytes")
    MAX_SIZE = 400 * 1024 * 1024  
    if file.content_length > MAX_SIZE:
        return jsonify({"error": "File exceeds 400MB limit."}), 400
   
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
