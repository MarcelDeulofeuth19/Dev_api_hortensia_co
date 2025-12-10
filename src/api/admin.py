from flask import Blueprint, render_template, request, redirect, url_for, flash
import yaml
from src.config.config import CONFIG, CONFIG_FILE
from .routes import requires_auth  # Reutiliza el decorador definido en rutas

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

@admin_bp.route('/indicadores', methods=["GET", "POST"])
@requires_auth
def update_indicadores():
    if request.method == "POST":
        try:
            # Capturamos y convertimos los valores del formulario a float.
            nueva_IPC_3 = float(request.form.get("IPC_3"))
            nueva_TRM_1 = float(request.form.get("TRM_1"))
            nueva_IBR_6_1 = float(request.form.get("IBR_6_1"))
            nueva_TPM_1 = float(request.form.get("TPM_1"))
            nueva_TD_NACIONAL_3 = float(request.form.get("TD_NACIONAL_3"))
            
            # Se actualiza el diccionario de configuración en memoria.
            CONFIG["Macroeconomicas"] = {
                "IPC_3": nueva_IPC_3,
                "TRM_1": nueva_TRM_1,
                "IBR_6_1": nueva_IBR_6_1,
                "TPM_1": nueva_TPM_1,
                "TD_NACIONAL_3": nueva_TD_NACIONAL_3,
            }
            
            # Se sobrescribe el archivo YAML para persistir los cambios.
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                yaml.safe_dump(CONFIG, f)
            
            flash("Los indicadores macroeconómicos han sido actualizados correctamente.", "success")
        except Exception as e:
            flash(f"Error al actualizar: {str(e)}", "danger")
        return redirect(url_for("admin.update_indicadores"))
    
    # En el método GET, se obtienen los valores actuales de la sección "Macroeconomicas".
    indicadores = CONFIG.get("Macroeconomicas", {})
    return render_template("update_indicadores.html", indicadores=indicadores)


@admin_bp.route('/menu')
@requires_auth
def admin_menu():
    return render_template("admin_menu.html")


@admin_bp.route('/umbrales', methods=["GET", "POST"])
@requires_auth
def update_umbrales():
    if request.method == "POST":
        try:
            new_umbrales = {}
            # Se iteran las claves conocidas (G, F, E, D, C, B, A)
            for key in ['G', 'F', 'E', 'D', 'C', 'B', 'A']:
                lower = float(request.form.get(f"{key}_lower"))
                upper = float(request.form.get(f"{key}_upper"))
                new_umbrales[key] = [lower, upper]
            
            # Se actualiza la sección 'credit_umbrales' del bloque 'contraofertas'
            CONFIG["contraofertas"]["credit_umbrales"] = new_umbrales
            
            # Se sobrescribe el archivo YAML para persistir los cambios.
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                yaml.safe_dump(CONFIG, f)
            
            flash("Los umbrales han sido actualizados correctamente.", "success")
        except Exception as e:
            flash(f"Error al actualizar los umbrales: {str(e)}", "danger")
        return redirect(url_for("admin.update_umbrales"))
    
    # En GET, se recupera la configuración actual de los umbrales.
    current_umbrales = CONFIG.get("contraofertas", {}).get("credit_umbrales", {})
    return render_template("update_umbrales.html", umbrales=current_umbrales)
