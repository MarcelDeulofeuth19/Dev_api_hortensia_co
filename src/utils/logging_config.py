import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import os


def setup_logging(log_filename_prefix: str = "hortensia", level: int = logging.INFO) -> None:
    """
    Configura logging con salida a consola y a archivo con rotación.

    - Crea el directorio `logs/` si no existe.
    - Archivo: logs/<prefix>.log con rotación (5 MB, 5 backups).
    - Evita duplicar handlers si ya fue inicializado.
    """
    logger = logging.getLogger()

    # Siempre asegurar nivel solicitado, aunque ya existan handlers (e.g., Uvicorn)
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # Asegurar al menos un StreamHandler visible
    has_stream = any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    if not has_stream:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logs_dir = Path("logs")
    try:
        logs_dir.mkdir(exist_ok=True)
    except Exception:
        logger.addHandler(console_handler)
        logger.info("Logging inicializado solo en consola (no se creó logs/)")
        return

    file_path = logs_dir / f"{log_filename_prefix}.log"
    has_file = any(
        isinstance(h, RotatingFileHandler) and getattr(h, 'baseFilename', None) == str(file_path)
        for h in logger.handlers
    )
    if not has_file:
        file_handler = RotatingFileHandler(
            file_path, maxBytes=5 * 1024 * 1024, backupCount=5
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.info("Logging inicializado")
