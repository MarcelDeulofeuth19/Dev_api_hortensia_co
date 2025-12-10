"""Preprocesamiento y features (No Credit Life)."""
import pandas as pd
import numpy as np
import logging
import json
from sklearn.linear_model import LinearRegression

from src.utils.helpers import cargar_json
from src.utils.helpers import cargar_modelo, cargar_json, cargar_config
from src.config.config import (
    PREPROC_COMMON_ERR,
    PREPROC_COMP_MAP,
    PREPROC_COMP_COLUMNS_NCLF,
    PREPROC_TENDENCIA_COLS_NCLF,
    PREPROC_NULOS_NCLF,
    PREPROC_IMPUTAR_NCLF,
    Var_pct_IPC_3,
    Var_TRM_1,
    IBR_Var,
    Var_TPM_1,
    Var_TD3
)

common_err = PREPROC_COMMON_ERR

# Cargar los mapeos desde archivos JSON
departamentos_colombia = cargar_json(
    "data/departamentos_colombia.json"
)
tipo_trabajo_mapping = cargar_json("data/tipo_trabajo_mapping.json")
genero_mapping = cargar_json("data/genero_mapping.json")


def calcular_variables_cliente(datos_cliente):
    logging.info("[preprocess_NCL] calcular_variables_cliente: start | type=%s", type(datos_cliente).__name__)
    if isinstance(datos_cliente, pd.DataFrame):
        if datos_cliente.empty:
            raise ValueError("datos_cliente DataFrame vacío")
        datos_dict = datos_cliente.iloc[0].to_dict()
    elif isinstance(datos_cliente, pd.Series):
        datos_dict = datos_cliente.to_dict()
    elif isinstance(datos_cliente, dict):
        datos_dict = dict(datos_cliente)
    else:
        raise TypeError(f"Tipo no soportado para datos_cliente: {type(datos_cliente)}")

    tipo_trabajo_val = datos_dict.get("tipo_trabajo")
    genero_val = datos_dict.get("genero")
    if isinstance(tipo_trabajo_val, pd.Series):
        tipo_trabajo_val = tipo_trabajo_val.iloc[0]
    if isinstance(genero_val, pd.Series):
        genero_val = genero_val.iloc[0]

    datos_dict["tipo_trabajo"] = tipo_trabajo_mapping.get(tipo_trabajo_val, None)
    datos_dict["genero_cliente"] = genero_mapping.get(genero_val, None)

    datos_dict["edad_cliente"] = datos_dict.pop("edad_al_contratar", 0)
    datos_dict["tipo_trabajo_cliente"] = datos_dict.pop("tipo_trabajo", 0)
    datos_dict["puntaje_p6"] = datos_dict.get("p6", 0)
    
    logging.info("[preprocess_NCL] calcular_variables_cliente: socio-demográfico OK")

    datos_cliente_df = pd.DataFrame([datos_dict])
    
    if 'departamento_exp' in datos_cliente_df.columns:
        datos_cliente_df.loc[datos_cliente_df['departamento_exp'].isnull(), 'departamento_exp'] = datos_cliente_df['dpto_nac']
        datos_cliente_df.loc[datos_cliente_df['ciudad_exp']=='BOGOTA D.C.', 'departamento_exp'] = 'bogota'
        datos_cliente_df.departamento_exp = (
            datos_cliente_df.departamento_exp.str.lower().str.strip()
        )
        datos_cliente_df["departamento_exp"] = datos_cliente_df["departamento_exp"].replace(common_err)
        datos_cliente_df["region_exp"] = datos_cliente_df["departamento_exp"].map(departamentos_colombia)
    else:
        datos_cliente_df["region_exp"] = -1

    datos_cliente_df["dpto_nac"] = datos_cliente_df["dpto_nac"].replace(common_err)
    datos_cliente_df["region_nac"] = datos_cliente_df["dpto_nac"].map(departamentos_colombia)

    datos_cliente_df["departamento_actual"] = datos_cliente_df["departamento_actual"].replace(common_err)
    datos_cliente_df["region_res"] = datos_cliente_df["departamento_actual"].map(departamentos_colombia)

    datos_cliente_df["constitucion_department_retailer"] = datos_cliente_df["constitucion_department_retailer"].replace(common_err)
    datos_cliente_df["region_ret"] = datos_cliente_df["constitucion_department_retailer"].map(departamentos_colombia)
    
    datos_cliente_df['Var_pct_IPC_3'] = Var_pct_IPC_3
    datos_cliente_df['Var_TRM_1'] = Var_TRM_1
    datos_cliente_df['IBR_Var'] = IBR_Var
    datos_cliente_df['Var_TPM_1'] = Var_TPM_1
    datos_cliente_df['Var_TD3'] = Var_TD3 
    
    logging.info(f"[preprocess][macros] Var_pct_IPC_3 {Var_pct_IPC_3}")
    logging.info(f"[preprocess][macros] Var_TRM_1 {Var_TRM_1}")
    logging.info(f"[preprocess][macros] IBR_Var {IBR_Var}")
    logging.info(f"[preprocess][macros] Var_TPM_1 {Var_TPM_1}")
    logging.info(f"[preprocess][macros] Var_TD3 {Var_TD3}")
    
    logging.info("[preprocess_NCL] calcular_variables_cliente: end | cols=%d", len(datos_cliente_df.columns))

    return datos_cliente_df

def preprocesscomportamiento(df):
    logging.info("Preprocesando comportamiento...")
    df = pd.DataFrame(df)
    comportamiento_dicc = PREPROC_COMP_MAP

    # Lista de columnas que queremos procesar.
    columnas_comportamiento = PREPROC_COMP_COLUMNS_NCLF
    
    for col in columnas_comportamiento:
        # Si la columna no existe, la creamos y asignamos valor 1.
        if col not in df.columns:
            df[col] = 1
        else:
            df[col] = df[col].map(comportamiento_dicc)
            df[col] = df[col].fillna(1)
    
    logging.info("Comportamiento preprocesado exitosamente...")
    
    if all(f'trimestre_{i}_moraMaxima' in df.columns for i in [1, 2, 3]):
        for i in [1, 2, 3]:  
            col = f'trimestre_{i}_moraMaxima'
            df[col] = df[col].astype(str).str.replace('M ', '').astype(float)

    return df

def calcular_tendencia(df):
    logging.info("Calculando tendencia...")
    
    def calcular_pendiente(row):
        y = row.values
        mask = ~np.isnan(y)
        X = np.arange(len(y))[mask].reshape(-1, 1)
        model = LinearRegression().fit(X, y)
        return model.coef_[0]
    

    tri_cols = ['trimestre_1_saldo', 'trimestre_2_saldo', 'trimestre_3_saldo']

    for col in tri_cols:
        if col not in df.columns:
            df[col] = 0
    df["saldos_tri_tendencia"] = df[tri_cols].apply(lambda row: calcular_pendiente(row), axis=1)

    columnas = PREPROC_TENDENCIA_COLS_NCLF

    X = np.arange(1, 13).reshape(-1, 1) #Modificar API
    for col in columnas:
        varcols = [f"{col}_{i}" for i in range(1, 13)]
        
        for vcol in varcols:
            if vcol not in df.columns:
                df[vcol] = 0
        
        df[varcols] = df[varcols].fillna(0)
        df[varcols] = df[varcols].astype(float)

        logging.info(f"Procesando columnas: {varcols}")
        
        df[col + "_mean"]      = df[varcols].mean(axis=1)
        df[col + "_std"]       = df[varcols].std(axis=1)
        df[col + "_max"]       = df[varcols].max(axis=1)
        df[col + "_min"]       = df[varcols].min(axis=1)
        df[col + "_tendencia"] = df[varcols].apply(lambda row: calcular_pendiente(row), axis=1) #Modificar API
        df[col + "_range"]     = df[col + "_max"] - df[col + "_min"]
        df[col + "cambio_1al12"] = df[varcols[-1]] - df[varcols[0]]
        
        df[col + "_mes_saldo_max"] = df[varcols].idxmax(axis=1).str.split('_', expand=True)[1]
        df[col + "_mes_saldo_min"] = df[varcols].idxmin(axis=1).str.split('_', expand=True)[1]

        df[col + "_mes_saldo_max"] = df[col + "_mes_saldo_max"].astype(float)
        df[col + "_mes_saldo_min"] = df[col + "_mes_saldo_min"].astype(float)
        
        #Modificar API
        for i in range(1, len(varcols)):
            col_anterior = varcols[i - 1]
            col_actual = varcols[i]
            col_nueva = f'var_pct_{varcols[i]}'
            df[col_nueva] = ((df[col_actual] - df[col_anterior]) / df[col_anterior])
        #Modificar API
        for i in range(0, len(varcols), 3):
            col_nueva = f'mean_m_{varcols[i]}'
            df[col_nueva] = df[varcols[i: i+3]].mean(axis=1)
        #mean_m_numCreditos30_1


    X = np.arange(1, 4).reshape(-1, 1)

    logging.info(f"Tendencia para {col} calculada exitosamente.")

    logging.info(f"Verificando columnas")
    lista_nulos = PREPROC_NULOS_NCLF
    
    for col in lista_nulos:
        if col not in df.columns:
            df[col] = np.nan  

    lista_imputar =   PREPROC_IMPUTAR_NCLF


    for col in lista_imputar:
        if col not in df.columns:
            df[col] = 0
  
    df['agr_prinp_antiguedadDesde'] = pd.to_datetime(df['agr_prinp_antiguedadDesde'])
    df['agr_prinp_antiguedadDesde'] = (df['fechaConsulta'] - df['agr_prinp_antiguedadDesde']).dt.days
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df
