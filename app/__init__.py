from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app)

    from app.routes.analyze import analyze_bp
    app.register_blueprint(analyze_bp)

    return app
