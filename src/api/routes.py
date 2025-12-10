from flask import jsonify, request, redirect, url_for, Response
from src.services.hortensia_contraofertas_matrix import MotorPrediccionContraofertas as MotorMatrix
from src.services.hortensia_CF_matrix_BACK import MotorPrediccionHrespaldo
from src.services.hortensia_CF_matrix_NCL import MotorPrediccionContraofertas as MotorMatrix_NCL

import logging
import os
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass
from functools import wraps
from src.config.config import (
    VERSION_CONTRAOFERTAS,
    VERSION_NCLF,
    VERSION_HRESPALDO,
    VERSION_FPD,
    VERSION_FPD_HRESPALDO,
    VERSION_NCLF_FPD,
    MODELO_PATH,
    SCALER_PATH_H,
    MODELO_FPD_PATH,
    SCALER_PATH_FPD,
    MODELO_NCLF_PATH,
    SCALER_NCLF_PATH,
    MODELO_FPD_NCLF_PATH,
    MODELO_BACK_PATH,
    SCALER_BACK_PATH,
    MODELO_FPD_BACK_PATH
)
from src.utils.responses import error_response
from src.core.decision import decide_and_predict, ValidationError


USERNAME = os.getenv("ADMIN_USERNAME", "admin")
PASSWORD = os.getenv("ADMIN_PASSWORD", "")
if not PASSWORD:
    logging.warning("ADMIN_PASSWORD no definido en .env; usando valor vacío")

class MotorRechazoZF:
    VERSION_HRESPALDO = VERSION_HRESPALDO
    VERSION_FPD_HRESPALDO = VERSION_FPD_HRESPALDO

    def predecir(self, datos, grupo_retailer=None):
        cliente = datos.get("cliente", {})
        return {
            "dni_cliente_consultado": cliente.get("dni_cliente"),
            "Respuesta": "Rechazado, ZF",
            "Razon_H": "ZF",
            "CodigoHortensia": "4",
            "Motor": self.VERSION_HRESPALDO,
            "Motor_FPD_Version": self.VERSION_FPD_HRESPALDO,
            "grupo_retailer": grupo_retailer or datos.get("grupo_retailer"),
            "departamento": cliente.get("departamento_tienda"),
        }

motor_zf = MotorRechazoZF()

def authenticate():
    """
    Responde con un 401 y el encabezado WWW-Authenticate para solicitar credenciales.
    """
    return Response(
        "Acceso no autorizado. Se requiere autenticación.\n",
        401,
        {"WWW-Authenticate": 'Basic realm="Login Required"'},
    )


def requires_auth(f):
    """
    Decorador que protege las rutas con autenticación básica.
    Verifica tanto el nombre de usuario como la contraseña.
    """

    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or auth.username != USERNAME or auth.password != PASSWORD:
            return authenticate()
        return f(*args, **kwargs)

    return decorated

motor_regular = MotorMatrix(MODELO_PATH, SCALER_PATH_H, MODELO_FPD_PATH, SCALER_PATH_FPD)
motor_nclf = MotorMatrix_NCL(MODELO_NCLF_PATH, SCALER_NCLF_PATH, MODELO_FPD_NCLF_PATH)
motor_backup = MotorPrediccionHrespaldo(MODELO_BACK_PATH, SCALER_BACK_PATH, MODELO_FPD_BACK_PATH)

@requires_auth
def index():
    """
    Ruta de inicio protegida con autenticación básica.
    """

    return ()
    
def predict_nocreditlife_cf_new(datos_cliente, grupo_retailer):
    """
    Realiza una predicción para clientes sin vida crediticia.
    """
    try:
        logging.info("Características de entrada (No Credit Life): %s", datos_cliente)
        resultado_prediccion = motor_nclf.predecir(
            datos_cliente, grupo_retailer
        )
        logging.info(
            "Resultado de la predicción (No Credit Life): %s", resultado_prediccion
        )
        return jsonify({"resultado": resultado_prediccion}), 200

    except KeyError as ke:
        logging.error("Clave faltante en el JSON: %s", ke)
        return  error_response(VERSION_NCLF,VERSION_NCLF_FPD,dni_cliente_consultado=None)
    except Exception as e:
        logging.error("Error durante la predicción: %s", str(e))
        return jsonify(error_response(VERSION_NCLF,VERSION_NCLF_FPD,dni_cliente_consultado=None))

#JP
def predict_contraofertas_new(datos_cliente, grupo_retailer):
    """
    Realiza una prediccion para contraofertas.
    """
    try:
        resultado_prediccion = motor_regular.predecir(
            datos_cliente, grupo_retailer
        )
        logging.info(
            "Resultado de la prediccion (Contraofertas): %s", resultado_prediccion
        )
        return jsonify({"resultado": resultado_prediccion}), 200

    except KeyError as ke:
        logging.error("Clave faltante en el JSON: %s", ke)
        return error_response(VERSION_CONTRAOFERTAS,VERSION_FPD,dni_cliente_consultado=None)
    except Exception as e:
        logging.error("Error durante la prediccion: %s", str(e))
        return jsonify(error_response(VERSION_CONTRAOFERTAS,VERSION_FPD,dni_cliente_consultado=None))

def hortensia_respaldo(datos_cliente, grupo_retailer ):

    try:
        resultado_prediccion = motor_backup.predecir(datos_cliente, grupo_retailer)
        logging.info("Resultado de la prediccion (Respaldo): %s", resultado_prediccion)
        return jsonify({"resultado": resultado_prediccion}), 200

    except KeyError as ke:
        logging.error("Clave faltante en el JSON: %s", ke)
        return jsonify({"error": f"Clave faltante en el JSON: {str(ke)}"}), 400
    except Exception as e:
        logging.error("Error durante la prediccion: %s", str(e))
        return jsonify(error_response(VERSION_HRESPALDO,VERSION_FPD_HRESPALDO,dni_cliente_consultado=None))


def predict_contraoferta():
    """
    Realiza una predicción centralizando la lógica en app_core.decision.
    """
    try:
        datos_cliente = request.get_json()
        logging.info("Datos recibidos del cliente: %s", datos_cliente)

        motores = {
            "contra": motor_regular,
            "backup": motor_backup,
            "NCL": motor_nclf,
            "ZF": motor_zf
        }

        resultado, status = decide_and_predict(datos_cliente, motores)
        if status == 200:
            return jsonify({"resultado": resultado}), 200
        return jsonify(resultado), status

    except ValidationError as ve:
        logging.error("Validación fallida: %s", ve)
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logging.error("Error durante la predicción: %s", str(e))
        return jsonify(error_response(VERSION_CONTRAOFERTAS, VERSION_FPD, dni_cliente_consultado=None))


def setup_routes(app):
    @app.route("/")
    @requires_auth
    def index_route():
        return redirect(url_for("admin.admin_menu"))

    @app.route("/predict_contraoferta", methods=["POST"])
    def predict_contraoferta_route():
        return predict_contraoferta()
