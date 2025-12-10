import logging
import numpy as np
import pandas as pd
from pathlib import Path
from src.utils.helpers import cargar_modelo, cargar_json
from src.models.feature_utils import normalize_and_select 
from src.models.predict_utils import extraer_datos_iniciales, assign_nested_bins
from src.utils.responses import build_rejection_response, build_approval_response, error_response
#cargar_json
from src.config.config import(
VERSION_HRESPALDO,
VERSION_FPD_HRESPALDO,
EDGES_CFG_BACKUP,
CREDIT_CONDITIONS_BACKUP,
Var_pct_IPC_3,
Var_TRM_1,
IBR_Var,
Var_TPM_1,
Var_TD3,
PREPROC_COMMON_ERR,
CUOTAS
)
BASE_DIR = Path(__file__).resolve().parents[1]  # sube desde /services hasta /src


class MotorPrediccionHrespaldo:
    """
    La clase MotorPrediccionHrespaldo ofrece métodos para predecir la probabilidad
    de aprobación de un crédito (Hrespaldo) y la probabilidad de FPD (First Payment Default),
    además de generar contraofertas basadas en estas probabilidades.

    Este motor:
    - Carga modelos de predicción y transformaciones.
    - Normaliza y selecciona variables del cliente.
    - Produce una respuesta final, ya sea aprobación, rechazo o contraofertas.
    """

    def __init__(self, modelo_path: str, min_max_scaler_path: str, modelo_fpd_path: str):
        """
        Inicializa el motor de predicción, cargando modelos, mapeos y escaladores necesarios.

        Args:
            modelo_path (str): Ruta al modelo principal de predicción Hrespaldo.
            min_max_scaler_path (str): Ruta al objeto MinMaxScaler para normalización.
        """
        # Carga de modelos
        self.modelo_h = cargar_modelo(modelo_path)
        self.modelo_fpd = cargar_modelo(modelo_fpd_path)
        self.min_max_scaler = cargar_modelo(min_max_scaler_path)

        # Cargar mapeos desde archivos JSON
        self.departamentos_colombia = cargar_json("departamentos_colombia.json")
        self.tipo_trabajo_mapping = cargar_json("tipo_trabajo_mapping.json")
        self.genero_mapping = cargar_json("genero_mapping.json")


        # Gestor de configuración (ampliable para futuras mejoras)

    def preprocesar(self, df:pd.DataFrame) -> pd.DataFrame:
        """Preprocesa los datos del cliente

        Args:
            df (pd.DataFrame): dataframe con los datos del cliente

        Returns:
            pd.DataFrame: dataframe preprocesado
        """
        common_err = PREPROC_COMMON_ERR
        df["constitucion_department_retailer"] = df["constitucion_department_retailer"].replace(
            common_err
        )
        df["region_ret"] = df["constitucion_department_retailer"].map(self.departamentos_colombia)
        df['Var_pct_IPC_3'] = Var_pct_IPC_3
        df['Var_TRM_1'] = Var_TRM_1 
        df['IBR_Var'] = IBR_Var
        df['Var_TPM_1'] = Var_TPM_1
        df['Var_TD3'] = Var_TD3

        return df


    def calcular_variables_cliente(self, datos_cliente: dict) -> dict:
        """
        Calcula y asigna nuevas variables categóricas y numéricas al cliente en base a mapeos
        y datos geográficos, estado civil, tipo de trabajo y género.

        Args:
            datos_cliente (dict): Diccionario con datos originales del cliente.

        Returns:
            dict: Diccionario del cliente actualizado con las nuevas variables.
        """
        logging.info("Calculando variables del cliente...")
        
        datos_cliente["region_res"] = self.departamentos_colombia.get(
            datos_cliente.get("departamento_actual"), 6
        )
        datos_cliente["region_nac"] = self.departamentos_colombia.get(
            datos_cliente.get("dpto_nac"), None
        )
        datos_cliente["tipo_trabajo"] = self.tipo_trabajo_mapping.get(
            datos_cliente.get("tipo_trabajo"), None
        )
        datos_cliente["genero"] = self.genero_mapping.get(
            datos_cliente.get("genero"), None
        )
        
        return datos_cliente


#JP
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
            logging.info("Datos del cliente recibidos en el JSON: %s", datos_cliente_json)

            datos_cliente, tid = extraer_datos_iniciales(datos_cliente_json, ["score_experian"])
            dni_cliente_consultado = datos_cliente.get("dni_cliente", "")
            # retailer_id = datos_cliente.get("retailer", "")

            logging.info("Iniciando proceso de predicción para el cliente %s", dni_cliente_consultado)

            # Cálculo de variables adicionales del cliente
            datos_cliente = self.calcular_variables_cliente(datos_cliente)
            datos_cliente_df = pd.DataFrame([datos_cliente])
            datos_cliente_df = self.preprocesar(datos_cliente_df)
            # con el loogin imprimimos las columnas finales
            logging.info("Variables finales: %s", datos_cliente_df)
            datos_cliente_df = datos_cliente_df.replace({None: np.nan})
            logging.info("Variables adicionales calculadas exitosamente.")
            proba_pagar, proba_fpd = self.generar_probabilidad(datos_cliente_df, tid)
            logging.info("Probabilidades generadas exitosamente.")
            segmentoH, segmentoFPD = assign_nested_bins(proba_h= proba_pagar, proba_fpd = proba_fpd, edges_cfg = EDGES_CFG_BACKUP)
            logging.info(f"Segmentos de cliente identificados: {segmentoH} {segmentoFPD}.")
            contraofertas, mensaje, razones_rechazo = self.generar_contraofertas(proba_pagar, proba_fpd , grupo_retailer, segmentoH, segmentoFPD)

            # Lógica de respuesta final
            if mensaje == 'Aprobado':
                return build_approval_response(
                    VERSION_HRESPALDO,
                    VERSION_FPD_HRESPALDO,
                    dni_cliente_consultado=dni_cliente_consultado,
                    proba_h=proba_pagar,
                    proba_fpd=proba_fpd,
                    contraofertas=contraofertas,
                    codigo="2",
                )
            elif mensaje == 'Rechazado':
                return build_rejection_response(
                    VERSION_HRESPALDO,
                    VERSION_FPD_HRESPALDO,
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
            return error_response(VERSION_HRESPALDO, VERSION_FPD_HRESPALDO, dni_cliente_consultado)
        except Exception as e:
            logging.exception("Error inesperado")
            return error_response(VERSION_HRESPALDO, VERSION_FPD_HRESPALDO, dni_cliente_consultado)


    def generar_probabilidad(self, cliente, tid):
        """
        Genera la probabilidad de aprobación Hrespaldo y la probabilidad FPD para un cliente.

        Args:
            cliente (pd.DataFrame): DataFrame con variables ya calculadas del cliente.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Probabilidades Hrespaldo y FPD.
        """
        # Renombrar variables para el modelo
        cliente["genero_cliente"] = cliente["genero"]
        cliente["numero_hijos_cliente"] = cliente["numero_hijos"]
        cliente["tiene_tarjeta_credito"] = cliente["tarjeta_credito"]
        cliente["tipo_trabajo_cliente"] = cliente["tipo_trabajo"]
        cliente["Edad"] = cliente["edad_al_contratar"]

        df_H, featuresH, df_FPD, featuresFPD  = normalize_and_select(
            cliente,
            model_h=self.modelo_h,
            scaler_h=self.min_max_scaler,
            model_fpd=self.modelo_fpd,
            scaler_fpd=self.min_max_scaler,
        )
        logging.info("Datos del cliente normalizados y seleccionados exitosamente.")
        proba_hrespaldo = self.modelo_h.predict_proba(df_H[featuresH])[:, 1]
        proba_hrespaldo_fpd = self.modelo_fpd.predict_proba(df_FPD[featuresFPD])[:, 1]
        if tid == 6:
            proba_hrespaldo = max(proba_hrespaldo - 0.05, 0.0)
        return proba_hrespaldo, proba_hrespaldo_fpd

    def generar_contraofertas(self,proba_pagar, proba_fpd, grupo_retailer, segmentoH, segmentoFPD):
        logging.info("Generando contraofertas...")
        # Trazas para diagnosticar tipos/valores
        h = str(segmentoH)[-1] if segmentoH is not None else '1'
        razones_rechazo = []
        if proba_pagar < EDGES_CFG_BACKUP['edges_h'][0]:
            mensaje = "Rechazado"
            contraofertas = {"grupo_cliente": f"{grupo_retailer}-H0_F0"}
            razones_rechazo.append("H")
        if proba_fpd >= EDGES_CFG_BACKUP['edges_f_by_h'][h][3]:
            mensaje = "Rechazado"
            contraofertas = {"grupo_cliente": f"{grupo_retailer}-H0_F0"}
            razones_rechazo.append("HFPD")

        if not razones_rechazo:
            mensaje = "Aprobado"
            contraofertas = {
                "List_price": CREDIT_CONDITIONS_BACKUP[f"{segmentoH}_{segmentoFPD}"]['list_price'],
                "Monto_max": CREDIT_CONDITIONS_BACKUP[f"{segmentoH}_{segmentoFPD}"]['financial_amount'],
                "Opciones_finan": CREDIT_CONDITIONS_BACKUP[f"{segmentoH}_{segmentoFPD}"]['percentaje'],
                "lapso_cuotas": CUOTAS,
                "valor_accesorios": CREDIT_CONDITIONS_BACKUP[f"{segmentoH}_{segmentoFPD}"]['accesories_amount'],
                "grupo_cliente": f"{grupo_retailer}-{segmentoH}_{segmentoFPD}",
            }
        
        logging.info("Contraofertas: %s", contraofertas)
        return contraofertas, mensaje, "+".join(razones_rechazo)