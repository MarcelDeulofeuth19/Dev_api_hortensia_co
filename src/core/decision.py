import logging
from typing import Any, Dict, Tuple
from src.config.config import PPT_BLACKLIST_DEPARTMENTS
class ValidationError(Exception):
    pass


def validate_payload(datos: Dict[str, Any]) -> None:
    if not isinstance(datos, dict):
        raise ValidationError("El cuerpo debe ser un objeto JSON")

    cliente = datos.get("cliente")
    if not isinstance(cliente, dict):
        raise ValidationError("Falta el objeto 'cliente'")

    required_cliente = ["tipo_documento", "score_experian"]
    missing = [k for k in required_cliente if k not in cliente]
    if missing:
        raise ValidationError(f"Faltan claves en 'cliente': {', '.join(missing)}")


def decide_and_predict(
    datos: Dict[str, Any],
    motores: Dict[str, Any],
) -> Tuple[Dict[str, Any], int]:
    validate_payload(datos)

    cliente = datos["cliente"]
    tipo_documento = int(cliente.get("tipo_documento"))
    tipo_modelo = int(cliente.get("score_experian", 30))
    grupo_retailer = datos.get("grupo_tienda") or datos.get("grupo_retailer")
    departamento_tienda = cliente.get("departamento_tienda")
    dni = int(cliente.get("dni_cliente"))
    id_retailer = cliente.get("retailer")

    logging.info(
        "Solicitud | dept=%s tipo_doc=%s score_experian=%s grupo=%s retailer=%s",
        departamento_tienda,
        tipo_documento,
        tipo_modelo,
        grupo_retailer,
        id_retailer,
    )
    
    if tipo_documento in (1, 4):
        if tipo_modelo == 30:
            motor = motores["backup"]
        elif tipo_modelo in (0, 1, 2, 3, 4):
            motor = motores["NCL"]
        else:
            motor = motores["contra"]
            
    elif tipo_documento == 6:
        if departamento_tienda and str(departamento_tienda).lower().replace(" ", "") in PPT_BLACKLIST_DEPARTMENTS:
            motor = motores["ZF"]
            resultado = motor.predecir(datos, grupo_retailer)
            return (resultado, 200) 
        if tipo_modelo == 30:
            motor = motores["backup"]
        elif tipo_modelo in (0, 1, 2, 3, 4):
            motor = motores["NCL"]
        else:
            motor = motores["contra"]

    try:
        resultado = motor.predecir(datos, grupo_retailer)
        return (resultado, 200)
    except Exception:
        logging.exception("Fallo en motor de predicción")
        return ({"error": "Error interno"}, 500)

