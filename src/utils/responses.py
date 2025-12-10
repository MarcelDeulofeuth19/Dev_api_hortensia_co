def _as_version(value, default_label):
    """Permite pasar string o dict de versiones; devuelve string consistente."""
    if isinstance(value, dict):
        # Intentar primeras claves conocidas
        for key in ("hortensia_contraofertas", "fpd_hortensia", default_label):
            if key in value:
                return value[key]
        # Fallback: primera clave
        return next(iter(value.values())) if value else ""
    return str(value) if value is not None else ""


def error_response(VERSION_H, VERSION_FPD, dni_cliente_consultado=None):
    """Arma la respuesta cuando ocurre un error interno en la aplicación."""
    return {
        "dni_cliente_consultado": f"{dni_cliente_consultado}" if dni_cliente_consultado else "",
        "Motor": VERSION_H,
        "Motor_FPD_Version": VERSION_FPD,
        "CodigoHortensia": 99,
        "Razon_H": "Error Interno",
        "Respuesta": "Error Interno",
        "Mensaje": "Error Interno."
    }


def _as_percent(v):
    try:
        return f"{round(float(v) * 100, 2)}"
    except Exception:
        return ""


def build_rejection_response(
    version_main,
    version_fpd,
    dni_cliente_consultado=None,
    mensaje="Rechazado",
    codigo="0",
    proba_h=None,
    proba_fpd=None,
    contraofertas=None,
    only_group_in_contraofertas=False,
):
    """
    Construye respuesta de rechazo estandarizada para cualquier motor.
    """
    resp = {
        "dni_cliente_consultado": f"{dni_cliente_consultado}" if dni_cliente_consultado else "",
        "Motor": _as_version(version_main, "hortensia_contraofertas"),
        "Motor_FPD_Version": _as_version(version_fpd, "fpd_hortensia"),
        "Respuesta": "Rechazado",
        "Razon_H": mensaje,
        "CodigoHortensia": codigo,
    }
    if codigo == "0":
        resp["Respuesta"] = f"Rechazado, {mensaje}"
    if proba_fpd is not None:
        resp["Puntuacion_HFPD"] = _as_percent(proba_fpd)
    if proba_h is not None:
        resp["Puntuacion_H"] = _as_percent(proba_h)
    if contraofertas:
        if only_group_in_contraofertas:
            resp["Contraofertas"] = {k: v for k, v in contraofertas.items() if k == "grupo_cliente"}
        else:
            resp["Contraofertas"] = contraofertas
    return resp


def build_approval_response(
    version_main,
    version_fpd,
    dni_cliente_consultado=None,
    proba_h=None,
    proba_fpd=None,
    contraofertas=None,
    codigo="2",
):
    """
    Construye respuesta de aprobación estandarizada para cualquier motor.
    """
    resp = {
        "dni_cliente_consultado": f"{dni_cliente_consultado}" if dni_cliente_consultado else "",
        "Motor": _as_version(version_main, "hortensia_contraofertas"),
        "Motor_FPD_Version": _as_version(version_fpd, "fpd_hortensia"),
        "Respuesta": "Aprobado",
        "Razon_H": "Aprobado",
        "CodigoHortensia": codigo,
        "Puntuacion_H": _as_percent(proba_h) if proba_h is not None else "",
        "Puntuacion_HFPD": _as_percent(proba_fpd) if proba_fpd is not None else "",
    }
    if contraofertas:
        resp["Contraofertas"] = contraofertas
    return resp
