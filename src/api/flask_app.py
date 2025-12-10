import logging
from pathlib import Path
from flask import Flask

from src.api.routes import setup_routes
from src.utils.logging_config import setup_logging
from src.api.admin import admin_bp

#JP
setup_logging()

templates_dir = Path(__file__).resolve().parents[1] / "templates"
app = Flask(__name__, template_folder=str(templates_dir))
app.secret_key = "secret"

setup_routes(app)
app.register_blueprint(admin_bp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
    logging.info("Aplicación finalizada")
