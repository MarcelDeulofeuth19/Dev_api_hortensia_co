import logging
import numpy as np
import pandas as pd
from typing import Optional, Tuple


def normalize_and_select(
    df: pd.DataFrame,
    model_h,
    scaler_h,
    model_fpd,
    scaler_fpd,
) -> Tuple[pd.DataFrame, list, pd.DataFrame, Optional[list]]:
    """
    Normaliza features para modelos H y FPD y retorna dataframes y listas de features.
    """
    logging.info("Normalizando y seleccionando features (util)")

    # H
    features_minmax_h = scaler_h.feature_names_in_
    logging.info("[features] scaler_H features_in full: %s", features_minmax_h)

    df_minmax_h = df[features_minmax_h].copy().astype(float)
    df_minmax_h.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_h = df.copy()
    df_h[features_minmax_h] = scaler_h.transform(df_minmax_h)
    features_h = model_h.feature_name_
    logging.info("[features] model_H features_in full: %s", features_h)

    # FPD
    features_minmax_fpd = scaler_fpd.feature_names_in_
    logging.info("[features] scaler_FPD features_in full: %s", features_minmax_fpd) 
    df_minmax_fpd = df[features_minmax_fpd].copy().astype(float)
    df_minmax_fpd.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_fpd = df.copy()
    df_fpd[features_minmax_fpd] = scaler_fpd.transform(df_minmax_fpd)
    features_fpd = model_fpd.feature_name_
    logging.info("[features] model_FPD features_in full: %s", features_fpd)

    return df_h, features_h, df_fpd, features_fpd