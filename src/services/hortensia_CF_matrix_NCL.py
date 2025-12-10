import logging
import lightgbm as lgb
import numpy as np
import pandas as pd
from src.services.extraccion_API_NCL import procesar_informe
from src.services.preprocess_NCL import (
    calcular_variables_cliente,
    preprocesscomportamiento,
    calcular_tendencia
)
from src.models.feature_utils import normalize_and_select
from src.utils.helpers import cargar_modelo, cargar_json
from src.models.predict_utils import (
    extraer_datos_cliente_campos,
    procesar_xml_experian,
    assign_nested_bins
)
from src.utils.responses import build_rejection_response, build_approval_response, error_response
#cargar_json
from src.config.config import(
VERSION_NCLF,
VERSION_NCLF_FPD,
EDGES_CFG_NCL,
REJECT_THRESHOLDS_NCL,
CREDIT_CONDITIONS_NCL,
CUOTAS
)

class MotorPrediccionContraofertas:
    """Clase que maneja la lógica de predicción y generación de contraofertas."""

    def __init__(self, MODELO_PATH, SCALER_PATH_H, MODELO_PATH_FPD):
        self.modelo_h = cargar_modelo(MODELO_PATH)
        self.modelo_fpd = cargar_modelo(MODELO_PATH_FPD)
        self.min_max_scaler = cargar_modelo(SCALER_PATH_H)
        self.departamentos_colombia = cargar_json("departamentos_colombia.json")
        self.tipo_trabajo_mapping = cargar_json("tipo_trabajo_mapping.json")
        self.genero_mapping = cargar_json("genero_mapping.json")

    def normalizar_y_seleccionar_features(self, df):
        """
        Normaliza las columnas indicadas en FEATURES_MINMAX utilizando el MinMaxScaler
        y retorna un DataFrame final, junto con las columnas finales a utilizar en el modelo.

        Args:
            df (pd.DataFrame): DataFrame con las variables del cliente.

        Returns:
            Tuple[pd.DataFrame, List[str]]:
                - DataFrame con las variables normalizadas.
                - Lista con las variables finales a utilizar en el modelo.
        """
        df["puntaje_quanto"] = df["quanto"]        
        return normalize_and_select(
            df,
            model_h=self.modelo_h,
            scaler_h=self.min_max_scaler,
            model_fpd=self.modelo_fpd,
            scaler_fpd=self.min_max_scaler,
        )

    def predecir(self, datos_cliente_json, grupo_retailer):
        """
        Procesa el JSON con los datos del cliente, calcula variables, normaliza, predice probabilidades
        y genera contraofertas o razones de rechazo. Retorna un diccionario con el resultado final.

        Args:
            datos_cliente_json (dict): JSON con datos del cliente.

        Returns:
            dict: Respuesta con información de aprobación o rechazo, puntuaciones y contraofertas.
        """
        
        try:
            datos_cliente, cliente_experian_xml, p3, score_experian, dni_cliente_consultado, tid = extraer_datos_cliente_campos(datos_cliente_json, ["p6", "score_experian"])

            if score_experian == 3:
                mensaje = "Cliente reportado como fallecido"
                codigo = "0"
                return build_rejection_response(
                    VERSION_NCLF,
                    VERSION_NCLF_FPD,
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
            proba_pagar, proba_fpd = self.generar_probabilidad(datos_cliente, tid)
            logging.info("Probabilidades generadas exitosamente.")
            segmentoH, segmentoFPD = assign_nested_bins(proba_h= proba_pagar, proba_fpd = proba_fpd, edges_cfg = EDGES_CFG_NCL[grupo_retailer])
            logging.info(f"Segmentos de cliente identificados: {segmentoH} {segmentoFPD}.")
            contraofertas, mensaje, razones_rechazo = self.generar_contraofertas(p3, proba_pagar, proba_fpd , grupo_retailer, segmentoH, segmentoFPD)

            # Lógica de respuesta final
            if mensaje == 'Aprobado':
                return build_approval_response(
                    VERSION_NCLF,
                    VERSION_NCLF_FPD,
                    dni_cliente_consultado=dni_cliente_consultado,
                    proba_h=proba_pagar,
                    proba_fpd=proba_fpd,
                    contraofertas=contraofertas,
                    codigo="2",
                )
            elif mensaje == 'Rechazado':
                return build_rejection_response(
                    VERSION_NCLF,
                    VERSION_NCLF_FPD,
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
            return error_response(VERSION_NCLF, VERSION_NCLF_FPD, dni_cliente_consultado)
        except Exception as e:
            logging.exception("Error inesperado")
            return error_response(VERSION_NCLF, VERSION_NCLF_FPD, dni_cliente_consultado)

    def generar_probabilidad(self, cliente, tid: int):
        """
        Genera la probabilidad de aprobación Hrespaldo y la probabilidad FPD para un cliente.

        Args:
            cliente (pd.DataFrame): DataFrame con variables ya calculadas del cliente.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Probabilidades Hrespaldo y FPD.
        """
        logging.info("Generando probabilidad de pago...")
        df = cliente.copy()
        df["ident_genero"] = df['genero_exp']
        df['Edad'] = df['edad_cliente']
        df_H, featuresH, df_FPD, featuresFPD  = self.normalizar_y_seleccionar_features(df)
        logging.info("Features normalizados y seleccionados H: %s", df_H.to_dict(orient="records")[0])
        logging.info("Calculando probablidad H")
        
        proba_pagar = self.modelo_h.predict_proba(df_H[featuresH])[:, 1]
        if tid == 6:
            proba_pagar = max(proba_pagar - 0.07, 0.0)
        logging.info("Features normalizados y seleccionados FPD: %s", df_H.to_dict(orient="records")[0])
        logging.info("Calculando probablidad FPD")
        
        proba_fpd = self.modelo_fpd.predict_proba(df_FPD[featuresFPD])[:, 1]
        logging.info("Probabilidad de pago NCL: %s", proba_pagar)
        logging.info("Probabilidad FPD NCL: %s", proba_fpd)

        return proba_pagar, proba_fpd

    def generar_contraofertas(self,p3 ,proba_pagar, proba_fpd, grupo_retailer, segmentoH, segmentoFPD):
        logging.info("Generando contraofertas...")
        # Trazas para diagnosticar tipos/valores
        h = str(segmentoH)[-1] if segmentoH is not None else '1'
        razones_rechazo = []
        if proba_pagar < EDGES_CFG_NCL[grupo_retailer]['edges_h'][0]:
            mensaje = "Rechazado"
            contraofertas = {"grupo_cliente": f"{grupo_retailer}-H0_F0"}
            razones_rechazo.append("H")
        if proba_fpd >= EDGES_CFG_NCL[grupo_retailer]['edges_f_by_h'][h][3]:
            mensaje = "Rechazado"
            contraofertas = {"grupo_cliente": f"{grupo_retailer}-H0_F0"}
            razones_rechazo.append("HFPD")
        if proba_fpd >= (
            (REJECT_THRESHOLDS_NCL['diagonal']["m"] * proba_pagar)
            + REJECT_THRESHOLDS_NCL['diagonal']["b"]
        ):
            mensaje = "Rechazado"
            contraofertas = {"grupo_cliente": f"{grupo_retailer}-H0_F0"}
            razones_rechazo = ["RE"]
            
        if not razones_rechazo:
            mensaje = "Aprobado"
            contraofertas = {
                "List_price": CREDIT_CONDITIONS_NCL[grupo_retailer][f"{segmentoH}_{segmentoFPD}"]['list_price'],
                "Monto_max": CREDIT_CONDITIONS_NCL[grupo_retailer][f"{segmentoH}_{segmentoFPD}"]['financial_amount'],
                "Opciones_finan": CREDIT_CONDITIONS_NCL[grupo_retailer][f"{segmentoH}_{segmentoFPD}"]['percentaje'],
                "lapso_cuotas": CUOTAS,
                "valor_accesorios": CREDIT_CONDITIONS_NCL[grupo_retailer][f"{segmentoH}_{segmentoFPD}"]['accesories_amount'],
                "grupo_cliente": f"{grupo_retailer}-{segmentoH}_{segmentoFPD}",
            }
        
        logging.info("Contraofertas: %s", contraofertas)
        return contraofertas, mensaje, "+".join(razones_rechazo)