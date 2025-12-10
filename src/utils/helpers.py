import pandas as pd
import logging
import joblib
import json
import sys
from pathlib import Path

import yaml


def cargar_config(path: str | None = None):
    """Carga el YAML de configuración.

    Si no se especifica `path`, usa `src/config/config.yaml`.
    """
    if path is None:
        path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    else:
        path = Path(path)
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def cargar_modelo(modelo_path):
    """
    Carga un modelo desde un archivo (p.e., pickle) usando joblib.
    
    Args:
        modelo_path (str o Path): ruta del archivo del modelo entrenado.
    Returns:
        object: instancia del modelo cargado.
    """
    try:
        modelo = joblib.load(modelo_path)
        logging.info("Modelo cargado exitosamente desde %s", modelo_path)
        return modelo
    except FileNotFoundError:
        logging.error("No se pudo encontrar el archivo del modelo en la ruta especificada: %s", modelo_path)
        sys.exit(1)
    except Exception as e:
        logging.error("Error al cargar el modelo: %s", e)
        sys.exit(1)


def cargar_json(ruta):
    """
    Carga los datos de un archivo JSON desde la carpeta 'data' dentro de src.
    La ruta puede ser solo el nombre del archivo o un path relativo a 'data/'.
    """
    try:
        base_dir = Path(__file__).resolve().parents[1]  # apunta a /src
        data_dir = base_dir / "data"

        # Si te pasan "departamentos_colombia.json" o "data/departamentos_colombia.json"
        p = Path(ruta)
        if p.parent.name == "data":
            json_path = base_dir / p
        else:
            json_path = data_dir / p.name

        if not json_path.exists():
            raise FileNotFoundError(json_path)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        logging.info("Datos cargados exitosamente desde %s", json_path)
        return data

    except FileNotFoundError as e:
        logging.error("No se pudo encontrar el archivo JSON: %s", e)
        sys.exit(1)
    except Exception as e:
        logging.error("Error al cargar el archivo JSON %s: %s", ruta, e)
        sys.exit(1)
