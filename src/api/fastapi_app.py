from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import logging

from src.services.hortensia_contraofertas_matrix import MotorPrediccionContraofertas as MotorMatrix
from src.services.hortensia_CF_matrix_BACK import MotorPrediccionHrespaldo
from src.services.hortensia_CF_matrix_NCL import MotorPrediccionContraofertas as MotorMatrix_NCL
from src.utils.logging_config import setup_logging
from src.core.decision import decide_and_predict, ValidationError
from src.config.config import (
    VERSION_HRESPALDO,
    VERSION_FPD_HRESPALDO,
    MODELO_PATH,
    SCALER_PATH_H,
    MODELO_FPD_PATH,
    SCALER_PATH_FPD,
    MODELO_NCLF_PATH,
    SCALER_NCLF_PATH,
    MODELO_FPD_NCLF_PATH,
    MODELO_BACK_PATH,
    SCALER_BACK_PATH,
    MODELO_FPD_BACK_PATH,
)

setup_logging()
app = FastAPI()

# Instancias de motores
motor_regular = MotorMatrix(MODELO_PATH, SCALER_PATH_H, MODELO_FPD_PATH, SCALER_PATH_FPD)
motor_nclf = MotorMatrix_NCL(MODELO_NCLF_PATH, SCALER_NCLF_PATH, MODELO_FPD_NCLF_PATH)
motor_backup = MotorPrediccionHrespaldo(MODELO_BACK_PATH, SCALER_BACK_PATH, MODELO_FPD_BACK_PATH)

class MotorRechazoZF:
    VERSION_HRESPALDO = VERSION_HRESPALDO
    VERSION_FPD_HRESPALDO = VERSION_FPD_HRESPALDO

    def predecir(self, datos, grupo_retailer=None):
        cliente = datos.get("cliente", {})
        return {
            "resultado": {
                "dni_cliente_consultado": cliente.get("dni_cliente"),
                "Respuesta": "Rechazado",
                "Razon_H": "ZF",
                "CodigoHortensia": "4",
                "Motor": self.VERSION_HRESPALDO,
                "Motor_FPD_Version": self.VERSION_FPD_HRESPALDO,
                "grupo_retailer": grupo_retailer or datos.get("grupo_retailer"),
                "departamento": cliente.get("departamento_tienda"),
            }
        }
    
motor_zf = MotorRechazoZF()

@app.post("/predecir/")
async def predecir(request: Request):
    try:
        datos = await request.json()

        motores = {
            "contra": motor_regular,
            "backup": motor_backup,
            "NCL": motor_nclf,
            "ZF": motor_zf
        }

        resultado, status = decide_and_predict(datos, motores)
        return JSONResponse(content=resultado, status_code=status)
    except ValidationError as ve:
        logging.warning("Validación fallida: %s", ve)
        return JSONResponse(content={"error": str(ve)}, status_code=400)
    except Exception as e:
        logging.error(f"Error al procesar la solicitud: {e}")
        return JSONResponse(content={"error": "Error interno"}, status_code=500)
