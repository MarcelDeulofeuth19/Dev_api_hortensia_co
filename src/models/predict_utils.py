import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Callable


def validar_datos_cliente(datos_cliente: Dict[str, Any], required_keys: List[str]) -> None:
    for key in required_keys:
        if key not in datos_cliente:
            raise KeyError(f"Clave faltante en los datos del cliente: {key}")


def extraer_datos_cliente_campos(
    datos_cliente_json: Dict[str, Any],
    required_initial_keys: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], Optional[str], int, int, int]:
    """
    Extrae y valida los datos iniciales (incluye p3, score_experian y dni_cliente).
    Retorna: (datos_cliente, cliente_experian_xml, p3, score_experian, dni)
    """
    datos_cliente = datos_cliente_json.get("cliente", {})
    cliente_experian_xml = datos_cliente_json.get("experianXML", None)
    if not datos_cliente:
        raise ValueError("Datos del cliente no encontrados en el JSON")

    required = required_initial_keys or ["p6", "score_experian"]
    validar_datos_cliente(datos_cliente, required)
    tid = int(datos_cliente.get("tipo_documento", 0))
    p3 = int(datos_cliente.get("p3", 0))
    score_experian = int(datos_cliente.get("score_experian", 0))
    dni_cliente_consultado = int(datos_cliente.get("dni_cliente", 0))
    return datos_cliente, cliente_experian_xml, p3, score_experian, dni_cliente_consultado, tid


def extraer_datos_iniciales(
    datos_cliente_json: Dict[str, Any],
    required_initial_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Extrae y valida los datos del cliente (sin p3/score/dni)."""
    datos_cliente = datos_cliente_json.get("cliente", {})
    if not datos_cliente:
        raise ValueError("Datos del cliente no encontrados en el JSON")

    required = required_initial_keys or ["score_experian"]
    validar_datos_cliente(datos_cliente, required)
    tid = int(datos_cliente.get("tipo_documento", 0))
    return datos_cliente, tid


def procesar_xml_experian(
    datos_cliente: Dict[str, Any],
    cliente_experian_xml: Optional[str],
    procesar_informe_func: Callable[[str], pd.DataFrame],
) -> Dict[str, Any]:
    """Fusiona los campos del XML Experian (si existe) con los datos del cliente."""
    if cliente_experian_xml:
        datos_experian_df = procesar_informe_func(cliente_experian_xml)
        if datos_experian_df is not None and not datos_experian_df.empty:
            logging.info(
                "Valores obtenidos del XML de Experian: %s",
                datos_experian_df.to_dict(orient="records")[0],
            )
            datos_cliente_df = pd.DataFrame([datos_cliente])
            datos_cliente_df = pd.concat([datos_cliente_df, datos_experian_df], axis=1)
            #datos_cliente = datos_cliente_df.to_dict(orient="records")[0]
        else:
            logging.info("No se pudo procesar el XML de Experian")
    else:
        logging.info("No se recibió XML de Experian")
    return datos_cliente_df

def assign_nested_bins(
        proba_h,
        proba_fpd,
        edges_cfg: dict | None = None,
        h_prefix: str = "H",
        f_prefix: str = "F",
        pad: int = 2,
    ) -> tuple:
        """
        Asigna y devuelve los grupos para una sola pareja de valores (H, FPD).

        Entradas:
        - proba_h: probabilidad de H (float o array-like)
        - proba_fpd: probabilidad de FPD (float o array-like)
        - edges_cfg: dict con la forma:
            {
              'edges_h': [e0, e1, ..., ek],
              'edges_f_by_h': {
                   '1': [f0, f1, ..., fm],
                   '2': [...],
                   ...
              }
            }
        Salida:
        - (grupoH, grupoFPD) por ejemplo ("H01", "F02"). Si no se proveen bordes
          válidos, devuelve (None, None).
        """

        # Normalizar a float (tomar el primer elemento si es array-like)
        try:
            proba_h = float(np.atleast_1d(proba_h)[0])
        except Exception:
            logging.debug("assign_nested_bins: no se pudo convertir proba_h a float")
            return (None, None)

        try:
            proba_fpd = float(np.atleast_1d(proba_fpd)[0])
        except Exception:
            logging.debug("assign_nested_bins: no se pudo convertir proba_fpd a float")
            return (None, None)

        if not edges_cfg or "edges_h" not in edges_cfg or "edges_f_by_h" not in edges_cfg:
            logging.debug("assign_nested_bins: edges_cfg ausente o incompleto")
            return (None, None)

        edges_h = np.asarray(edges_cfg.get("edges_h", []), dtype=float)
        edges_f_by_h = edges_cfg.get("edges_f_by_h", {})

        if edges_h.size < 2:
            logging.debug("assign_nested_bins: edges_h insuficiente")
            return (None, None)

        # Encontrar índice del grupo H siguiendo una lógica similar a pandas.cut(include_lowest=True, right=True)
        # Bins: [e0, e1], (e1, e2], ..., (e_{k-1}, e_k]
        h_index = None
        k = edges_h.size - 1
        for i in range(1, k + 1):
            low = edges_h[i - 1]
            high = edges_h[i]
            if (i == 1 and (proba_h >= low) and (proba_h <= high)) or (
                i > 1 and (proba_h > low) and (proba_h <= high)
            ):
                h_index = i
                break

        if h_index is None:
            logging.debug("assign_nested_bins: proba_h fuera de los bordes")
            return (None, None)

        grupoH = f"{h_prefix}{h_index:0{pad}d}"

        # Buscar bordes FPD para el grupo H encontrado
        edges_f = edges_f_by_h.get(str(h_index))
        if not edges_f:
            logging.debug("assign_nested_bins: sin edges_f para H=%s", h_index)
            return (grupoH, None)

        edges_f = np.asarray(edges_f, dtype=float)
        if edges_f.size < 2:
            logging.debug("assign_nested_bins: edges_f insuficiente para H=%s", h_index)
            return (grupoH, None)

        f_index = None
        m = edges_f.size - 1
        for j in range(1, m + 1):
            low = edges_f[j - 1]
            high = edges_f[j]
            if (j == 1 and (proba_fpd >= low) and (proba_fpd <= high)) or (
                j > 1 and (proba_fpd > low) and (proba_fpd <= high)
            ):
                f_index = j
                break

        if f_index is None:
            logging.debug("assign_nested_bins: proba_fpd fuera de los bordes para H=%s", h_index)
            return (grupoH, None)

        grupoFPD = f"{f_prefix}{f_index:0{pad}d}"
        return grupoH, grupoFPD