import logging
import pandas as pd
import numpy as np
#from xml_procces import procesar_xml, rutas_descripciones
from src.services.extraccion_API import procesar_informe
from src.models.feature_utils import normalize_and_select
from src.services.preprocess import (
    calcular_variables_cliente,
    preprocesscomportamiento,
    calcular_tendencia
)
from src.utils.helpers import cargar_modelo
from src.models.predict_utils import (
    extraer_datos_cliente_campos,
    procesar_xml_experian,
    assign_nested_bins
)
#cargar_json
from src.config.config import(
VERSION_CONTRAOFERTAS,
VERSION_FPD,
EDGES_CFG,
REJECT_THRESHOLDS,
CREDIT_CONDITIONS,
CUOTAS
)
from src.utils.responses import build_rejection_response, build_approval_response, error_response

class MotorPrediccionContraofertas:
    """Clase que maneja la lógica de predicción y generación de contraofertas."""

    def __init__(self, MODELO_PATH, SCALER_PATHS_H, MODELO_PATH_FPD, SCALER_PATHS_FPD):
        self.modelo_h = cargar_modelo(MODELO_PATH)
        self.min_max_scaler_H = cargar_modelo(SCALER_PATHS_H)
        self.modelo_fpd = cargar_modelo(MODELO_PATH_FPD)
        self.min_max_scaler_FPD = cargar_modelo(SCALER_PATHS_FPD)

    def normalizar_y_seleccionar_features(self, df):
        df["puntaje_quanto"] = df["quanto"]
        return normalize_and_select(
            df,
            model_h=self.modelo_h,
            scaler_h=self.min_max_scaler_H,
            model_fpd=self.modelo_fpd,
            scaler_fpd=self.min_max_scaler_FPD,
        )
#JP
    def predecir(self, datos_cliente_json, grupo_retailer):
        """Realiza el proceso completo de predicción y genera la respuesta correspondiente."""
        try:
            datos_cliente, cliente_experian_xml, p3, score_experian, dni_cliente_consultado, tid = extraer_datos_cliente_campos(datos_cliente_json, ["p6", "score_experian"])
            dni = datos_cliente.get("dni_cliente")
            if score_experian == 3:
                mensaje = "Rechazado, Cliente reportado como fallecido"
                codigo = "0"
                return build_rejection_response(
                    VERSION_CONTRAOFERTAS,
                    VERSION_FPD,
                    dni_cliente_consultado=dni_cliente_consultado,
                    mensaje=mensaje,
                    codigo=codigo
                )
            logging.info("Iniciando el proceso de extracción de experian.")
            
            datos_cliente = procesar_xml_experian(datos_cliente, cliente_experian_xml, procesar_informe)
            logging.info("Iniciando un nuevo proceso de predicción.")
            datos_cliente = calcular_variables_cliente(datos_cliente)
            datos_cliente = preprocesscomportamiento(datos_cliente)
            datos_cliente = calcular_tendencia(datos_cliente)
            logging.info("Variables adicionales calculadas exitosamente.")
            proba_pagar, proba_fpd = self.generar_probabilidad(datos_cliente)
            logging.info("Probabilidades generadas exitosamente.")
            segmentoH, segmentoFPD = assign_nested_bins(proba_h= proba_pagar, proba_fpd = proba_fpd, edges_cfg = EDGES_CFG[grupo_retailer])
            logging.info(f"Segmentos de cliente identificados: {segmentoH} {segmentoFPD}.")
            contraofertas, mensaje, razones_rechazo = self.generar_contraofertas(p3, proba_pagar, proba_fpd , grupo_retailer, segmentoH, segmentoFPD, dni)

            if mensaje == 'Aprobado':
                return build_approval_response(
                    VERSION_CONTRAOFERTAS,
                    VERSION_FPD,
                    dni_cliente_consultado=dni_cliente_consultado,
                    proba_h=proba_pagar,
                    proba_fpd=proba_fpd,
                    contraofertas=contraofertas,
                    codigo="2",
                )
            elif mensaje == 'Rechazado':
                return build_rejection_response(
                    VERSION_CONTRAOFERTAS,
                    VERSION_FPD,
                    dni_cliente_consultado=dni_cliente_consultado,
                    mensaje=razones_rechazo,
                    codigo="0",
                    proba_h=proba_pagar,
                    proba_fpd=proba_fpd,
                    contraofertas=contraofertas,
                    only_group_in_contraofertas=True,
                )
        except KeyError as ke:
            logging.exception("Error en la clave")
            return error_response(VERSION_CONTRAOFERTAS, VERSION_FPD, dni_cliente_consultado)
        except Exception as e:
            logging.exception("Error inesperado")
            return error_response(VERSION_CONTRAOFERTAS, VERSION_FPD, dni_cliente_consultado)

    def generar_probabilidad(self, cliente):
        """Genera las probabilidades de pago y FPD a partir de los datos del cliente."""
        logging.info("Generando probabilidad de pago...")
        df = cliente.copy()
        df["ident_genero"] = df['genero_exp']
        df['Edad'] = df['edad_cliente']
        df_H, featuresH, df_FPD, featuresFPD = self.normalizar_y_seleccionar_features(df)
        #logging.info("Features normalizados y seleccionado: %s", df_H.to_dict(orient="records")[0])
        logging.info("Calculando probablidad")
        proba_pagar = self.modelo_h.predict_proba(df_H[featuresH])[:, 1]
        proba_fpd = self.modelo_fpd.predict_proba(df_FPD[featuresFPD])[:, 1]
        logging.info("Probabilidad de pago: %s", proba_pagar)
        logging.info("Probabilidad FPD: %s", proba_fpd)
        return proba_pagar, proba_fpd

    def generar_contraofertas(self,p3 ,proba_pagar, proba_fpd, grupo_retailer, segmentoH, segmentoFPD, dni):
        logging.info("Generando contraofertas...")

        
        # Trazas para diagnosticar tipos/valores
        h = str(segmentoH)[-1] if segmentoH is not None else '1'
        razones_rechazo = []
        if proba_pagar < EDGES_CFG[grupo_retailer]['edges_h'][0]:
            mensaje = "Rechazado"
            contraofertas = {"grupo_cliente": f"{grupo_retailer}-H0_F0"}
            razones_rechazo.append("H")
        if proba_fpd >= EDGES_CFG[grupo_retailer]['edges_f_by_h'][h][3]:
            mensaje = "Rechazado"
            contraofertas = {"grupo_cliente": f"{grupo_retailer}-H0_F0"}
            razones_rechazo.append("HFPD")
        rechazado = False

        for x in range(REJECT_THRESHOLDS['diagonal']["corte_H"], REJECT_THRESHOLDS['diagonal']["corte_H2"]):
            x = x /100
            y = REJECT_THRESHOLDS['diagonal']["m"] * x + REJECT_THRESHOLDS['diagonal']["b"]
            if proba_pagar >= x and proba_fpd >= y:
                rechazado = True
                break
        if rechazado:
            mensaje = "Rechazado"
            contraofertas = {"grupo_cliente": f"{grupo_retailer}-H0_F0"}
            razones_rechazo = ["RE"]

        elif not razones_rechazo and p3:
        # Solo se aplica si no hubo ninguna otra razón antes
            mensaje = "Rechazado"
            contraofertas = {"grupo_cliente": f"{grupo_retailer}-{segmentoH}_{segmentoFPD}"}
            razones_rechazo = ["P3"]

        # Champion challenger
        def es_par(c):
            return c.isdigit() and int(c) % 2 == 0

        Champion, grupo_retailer2 = "Estandar", grupo_retailer
 
        condicion_A = (grupo_retailer == "A" and segmentoH == "H03" and segmentoFPD == "F01")

        if (condicion_A) and es_par(dni):
            Champion, grupo_retailer2 =  "Retador", grupo_retailer2 + "-2"

        if not razones_rechazo:
            mensaje = "Aprobado"
            contraofertas = {
                "List_price": CREDIT_CONDITIONS[Champion][grupo_retailer][f"{segmentoH}_{segmentoFPD}"]['list_price'],
                "Monto_max": CREDIT_CONDITIONS[Champion][grupo_retailer][f"{segmentoH}_{segmentoFPD}"]['financial_amount'],
                "Opciones_finan": CREDIT_CONDITIONS[Champion][grupo_retailer][f"{segmentoH}_{segmentoFPD}"]['percentaje'],
                "lapso_cuotas": CUOTAS,
                "valor_accesorios": CREDIT_CONDITIONS[Champion][grupo_retailer][f"{segmentoH}_{segmentoFPD}"]['accesories_amount'],
                "grupo_cliente": f"{grupo_retailer2}-{segmentoH}_{segmentoFPD}",
            }
        logging.info("Contraofertas: %s", contraofertas)
        return contraofertas, mensaje, "+".join(razones_rechazo)