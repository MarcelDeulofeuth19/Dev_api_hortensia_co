import yaml
from pathlib import Path

# Ruta del archivo de configuración al lado de este módulo
CONFIG_FILE = Path(__file__).resolve().parent / "config.yaml"

with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

# 3. Exponer las variables tal como antes, para mantener compatibilidad.

# Versiones Hortensia
VERSION_CONTRAOFERTAS = CONFIG["versions_hortensia"]["hortensia_contraofertas"]
VERSION_HRESPALDO = CONFIG["versions_hortensia"]["hrespaldo"]
VERSION_NCLF = CONFIG["versions_hortensia"]["hortensia_NCLF"]

# Versiones FPD
VERSION_FPD = CONFIG["versions_fpd"]["fpd_hortensia"]
VERSION_FPD_HRESPALDO = CONFIG["versions_fpd"]["fpd_hrespaldo"]
VERSION_NCLF_FPD = CONFIG["versions_fpd"]["fpd_hortensia_NCLF"]

# Rutas de modelos y escaladores
MODEL_PATHS = CONFIG["model_paths"]
MODEL_PATHS_FPD = CONFIG["model_paths_fpd"]
SCALER_PATHS = CONFIG["scaler_paths"]

MODELO_PATH = MODEL_PATHS['hortensia']
MODELO_FPD_PATH = MODEL_PATHS_FPD['hortensia_fpd']
SCALER_PATH_H = SCALER_PATHS['min_max_scaler_experian_H']
SCALER_PATH_FPD = SCALER_PATHS['min_max_scaler_experian_FPD']

# Contra ofertas  NCL

MODELO_NCLF_PATH = MODEL_PATHS["hortensia_NCLF"]
MODELO_FPD_NCLF_PATH = MODEL_PATHS_FPD["hortensia_NCLF_FPD"]
SCALER_NCLF_PATH = SCALER_PATHS["min_max_scaler_NCL"]

# Modelos respaldo
MODELO_BACK_PATH = MODEL_PATHS["respaldo"]
MODELO_FPD_BACK_PATH = MODEL_PATHS_FPD["respaldo_fpd"]
SCALER_BACK_PATH = SCALER_PATHS["min_max_scaler_hrespaldo"]

# Alias para compatibilidad retroactiva
MODEL_PATH = MODELO_PATH
MODEL_PATH_FPD = MODELO_FPD_PATH
SCALER_PATH = SCALER_PATHS
MODEL_PATH_NCLF = MODELO_NCLF_PATH
MODEL_PATH_NCLF_FPD = MODELO_FPD_NCLF_PATH
MIN_MAX_SCALER_PATH_NCLF = SCALER_NCLF_PATH

# Edges config para matriz
EDGES_CFG = CONFIG.get("edges_cfg", {})
REJECT_THRESHOLDS = CONFIG['reject_thresholds']
############ Parámetro para escoger entre la lista de descuento y la lista sin descuento ###########################################
CREDIT_CONDITIONS = CONFIG['credit_conditions_sin_descuento']
#CREDIT_CONDITIONS = CONFIG['credit_conditions_con_descuento']

# Edges config para matriz NCL
EDGES_CFG_NCL = CONFIG.get("edges_cfg_NCL", {})
REJECT_THRESHOLDS_NCL = CONFIG['reject_thresholds_NCL']
CREDIT_CONDITIONS_NCL = CONFIG['credit_conditions_NCL']

# Edges config para matriz NCL
EDGES_CFG_BACKUP = CONFIG.get("edges_cfg_BACKUP", {})
CREDIT_CONDITIONS_BACKUP = CONFIG['credit_conditions_BACKUP']

CUOTAS = CONFIG["lapso_cuotas"]

# Indicadores macro_económicos
Var_pct_IPC_3 = CONFIG['Macroeconomicas']['Var_pct_IPC_3']
Var_TRM_1 = CONFIG['Macroeconomicas']['Var_TRM_1']
IBR_Var = CONFIG['Macroeconomicas']['IBR_Var']
Var_TPM_1 = CONFIG['Macroeconomicas']['Var_TPM_1']
Var_TD3 = CONFIG['Macroeconomicas']['Var_TD3']

# Reglas de negocio
PPT_BLACKLIST_DEPARTMENTS = set(
    CONFIG.get('business_rules', {}).get('ppt_blacklist_departments', [])
)

# Preprocesamiento
PREPROC_COMMON_ERR = CONFIG.get('preprocessing', {}).get('common_err_map', {})
PREPROC_COMP_MAP = CONFIG.get('preprocessing', {}).get('comportamiento_map', {})
PREPROC_COMP_COLUMNS_NCLF = CONFIG.get('preprocessing', {}).get('comportamiento_columns_nclf', [])
PREPROC_TENDENCIA_COLS_REG = CONFIG.get('preprocessing', {}).get('tendencia_cols_regular', [])
PREPROC_TENDENCIA_COLS_NCLF = CONFIG.get('preprocessing', {}).get('tendencia_cols_nclf', [])
PREPROC_NULOS_REGULAR = CONFIG.get('preprocessing', {}).get('nulos_regular', [])
PREPROC_IMPUTAR_REGULAR = CONFIG.get('preprocessing', {}).get('imputar_regular', [])
PREPROC_NULOS_NCLF = CONFIG.get('preprocessing', {}).get('nulos_nclf', [])
PREPROC_IMPUTAR_NCLF = CONFIG.get('preprocessing', {}).get('imputar_nclf', [])
