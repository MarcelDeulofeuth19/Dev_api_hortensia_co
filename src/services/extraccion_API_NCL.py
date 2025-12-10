import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import re
import logging
from datetime import datetime
from collections import defaultdict, OrderedDict
from sklearn.linear_model import LinearRegression
from pandas.api.types import is_numeric_dtype

# Logging centralizado desde logging_config.setup_logging()
def variables_income(df):
    cuota = [col for col in df.columns if col.startswith('Sector') and col.endswith('cuota')]
    df['promedio_cuota'] = df.loc[:, cuota].sum(axis=1) / df.loc[:, cuota].notna().sum(axis=1)
    cartera_Cuota = []
    cartera_totalCuotas = []
    cartera_cuotasCanceladas = []
    for col in df.columns:
        if 'Sector' in col:
            if col.endswith('cuota'):
                cartera_Cuota.append(col)
            elif 'totalCuotas' in col:
                cartera_totalCuotas.append(col)
            elif 'cuotasCanceladas' in col:
                cartera_cuotasCanceladas.append(col)

    # Conversión de tipos
    cols_to_convert = cartera_Cuota + cartera_totalCuotas + cartera_cuotasCanceladas
    for col in cols_to_convert:
        if col in df.columns and not is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Cálculo de amortización general (con control de división por cero)
    denom_amort = df[cartera_totalCuotas].sum(axis=1) * df[cartera_Cuota].sum(axis=1)
    denom_amort = denom_amort.replace(0, np.nan)

    
    df["amortizacion_cartera"] = (
        (df[cartera_cuotasCanceladas].sum(axis=1) * df[cartera_Cuota].sum(axis=1)) / denom_amort
    )
    # Sector3: cartera real
    cartera_real_cuotasCanceladas = [col for col in df.columns if 'Sector3' in col and col.endswith('cuotasCanceladas')]
    cartera_real_totalCuotas = [col for col in df.columns if 'Sector3' in col and 'totalCuotas' in col]

    cuotas_canceladas_real = df[cartera_real_cuotasCanceladas].replace([0, -1], np.nan)
    total_cuotas_real = df[cartera_real_totalCuotas].replace([0, -1], np.nan)

    df["ratio_cartera_real"] = cuotas_canceladas_real.sum(axis=1) / total_cuotas_real.sum(axis=1)

    # Sector4: telcos
    cartera_telcos_cuotas = [col for col in df.columns if 'Sector4' in col and col.endswith('cuota')]
    cuotas_canceladas_telcos = df[cartera_telcos_cuotas].where(lambda x: (x != 0))

    df["ratio_cartera_telcos"] = cuotas_canceladas_telcos.sum(axis=1) / cuotas_canceladas_telcos.count(axis=1)
    # Limpiar infs
    df['ratio_cartera_real'] = df['ratio_cartera_real'].replace([np.inf, -np.inf], np.nan)
    df['ratio_cartera_telcos'] = np.abs(df['ratio_cartera_telcos'])

    return df

def procesar_cuotas_y_amortizacion(df):
    totales_cantidad = [col for col in df.columns if 'total_cantidad' in col ]
    activas = [col for col in df.columns if 'Activa_' in col and 'cantidad' in col ]
    aldia =  [col for col in df.columns if '_Al dia_' in col and 'cantidad' in col]
    buenas = activas + aldia 

    df['portafolio_total_ahorros'] = np.where(
        df.get('total_tipo_1', 0) == 'AHO',
        df.get('total_cantidad_1', 0),
        0
    )

    df['portafolio_positivas_ahorros'] = np.where(
        df.get('total_tipo_1', 0) == 'AHO',
        df.get('AHO_Activa_cantidad_1', 0),
        0
    )
    df['portafolio_totales_diferentes'] = df[totales_cantidad].apply(lambda x: x.notna() & (x != ''), axis=1).sum(axis=1)
    df['portafolio_aldia_diferentes'] = df[buenas].apply(lambda x: x.notna() & (x != ''), axis=1).sum(axis=1)
    df['portafolio_totales'] = df[totales_cantidad].apply(pd.to_numeric, errors='coerce').sum(axis=1)
    df['portafolio_aldia'] = df[buenas].apply(pd.to_numeric, errors='coerce').sum(axis=1)
    df['portafolio_mora'] = df['portafolio_totales'] - df['portafolio_aldia']

    df['portafolio_positivas_ahorros'] = pd.to_numeric(df['portafolio_positivas_ahorros'], errors='coerce')
    df['portafolio_total_ahorros'] = pd.to_numeric(df['portafolio_total_ahorros'], errors='coerce')
    df['portafolio_num_can_n'] = (df['portafolio_mora'] - (df['portafolio_total_ahorros'] - df['portafolio_positivas_ahorros']) )/ (df['portafolio_totales'] - df['portafolio_total_ahorros'])
    df['portafolio_num_can_p'] = (df['portafolio_aldia'] + df['portafolio_positivas_ahorros'] )/ (df['portafolio_totales'] - df['portafolio_total_ahorros'])

    df['portafolio_num_can_p'] = df['portafolio_num_can_p'].replace([np.inf, -np.inf], np.nan)
    df['portafolio_num_can_n'] = df['portafolio_num_can_n'].replace([np.inf, -np.inf], np.nan)

    # Consultas Nit
    consultas_nit = df.filter(regex='nitSuscriptor').columns
    df[consultas_nit] = df[consultas_nit].apply(pd.to_numeric, errors='coerce')
    entidades_set = {
        901344787.0, 901258467.0, 901308756.0, 901279350.0, 901310399.0
    }
    df['Consultas_competencia_72h'] = df[consultas_nit].isin(entidades_set).sum(axis=1)

    cartera_estados = [col for col in df.columns if 'Cartera' in col and col.endswith('codigo')]
    valores_buenas = [1.0, 3.0]
    valores_activas = [1.0, 2.0]
    cuenta_validas = df[cartera_estados].notna().sum(axis=1)
    df['buenas_carteras'] = df[cartera_estados].isin(valores_buenas).sum(axis=1)
    df['activos'] = df[cartera_estados].isin(valores_activas).sum(axis=1)
    df['carteras_activas_pp'] = df['activos'] / cuenta_validas
    df['carteras_buenas_pp'] = df['buenas_carteras'] / cuenta_validas

    # Trimestres
    trim1 = [col for col in df.columns if 'trim_1_saldoMora' in col]
    trim2 = [col for col in df.columns if 'trim_2_saldoMora' in col]
    trim3 = [col for col in df.columns if 'trim_3_saldoMora' in col]

    t1 = df[trim1].astype('float')
    t2 = df[trim2].astype('float')
    t3 = df[trim3].astype('float')

    df['Telcos_trim1'] = t1.sum(axis=1)
    df['Telcos_trim2'] = t2.sum(axis=1)
    df['Telcos_trim3'] = t3.sum(axis=1)

    df['variacion_mora_telcos'] = (
        (df['Telcos_trim2'] - df['Telcos_trim1']) +
        (df['Telcos_trim3'] - df['Telcos_trim2'])
    ) / 2

    #Saldos productos
    df['productos_saldo_total'] = df['tdc_saldo_actual'] + df['cartera_saldo_actual']
    df['productos_mora_total'] = df['tdc_saldo_mora'] + df['cartera_saldo_mora']

    # Fechas
    cols_fechas = ['tdc_fechaApertura_reciente', 'cartera_fechaApertura_reciente', 'ahorros_fechaApertura_reciente']
    df[cols_fechas] = df[cols_fechas].apply(pd.to_datetime, errors='coerce')
    df['fechaConsulta'] = pd.to_datetime(df['fechaConsulta'], errors='coerce')
    
    df['fechaApertura_max'] = df[cols_fechas].max(axis=1)
    
    df['Meses_apertura'] = (
        (df['fechaConsulta'] - df['fechaApertura_max']).dt.days // 30
    ).astype('Int64')
    
    # Mora por sector
    sector_mora_COM = [col for col in df.columns if 'COM_trim' in col]
    sector_mora_CTC = [col for col in df.columns if 'CTC_trim' in col]
    sector_mora_CDC = [col for col in df.columns if 'CDC_trim' in col]

    for sector in [sector_mora_COM, sector_mora_CTC, sector_mora_CDC]:
        df[sector] = df[sector].apply(pd.to_numeric, errors='coerce')


    df['Telcos_mora_trimestre_COM'] = df[sector_mora_COM].sum(axis=1) / df[sector_mora_COM].count(axis=1)
    df['Telcos_mora_trimestre_CTC'] = df[sector_mora_CTC].sum(axis=1) / df[sector_mora_CTC].count(axis=1)
    df['Telcos_mora_trimestre_CDC'] = df[sector_mora_CDC].sum(axis=1) / df[sector_mora_CDC].count(axis=1)
    telcos = [col for col in df.columns if 'Telcos_mora_trimestre' in col]
    df['Telcos_mora_trimestre'] = df[telcos].sum(axis=1) / df[telcos].count(axis=1)

    # Ratios finales
    df['agr_prinp_creditoVigentes'] = pd.to_numeric(df['agr_prinp_creditoVigentes'], errors='coerce')
    df['agr_saldos_saldoTotalEnMora'] = pd.to_numeric(df['agr_saldos_saldoTotalEnMora'], errors='coerce')
    df['saldo_prom_mora_prod'] = df['agr_saldos_saldoTotalEnMora'] / df['agr_prinp_creditoVigentes']

    df['agr_saldos_cuotaMensual'] = pd.to_numeric(df['agr_saldos_cuotaMensual'], errors='coerce')
    df['agr_saldos_saldoTotal'] = pd.to_numeric(df['agr_saldos_saldoTotal'], errors='coerce')
    df['ratio_cuota_saldo'] = df['agr_saldos_cuotaMensual'] / df['agr_saldos_saldoTotal']
    df['Dias_ultimo_producto'] = (df['fechaConsulta'] - df['cartera_fechaApertura_reciente']).dt.days
    return df

def procesar_consultas_experian(df):
    # 1. Conversión de fechas
    df['periodo_consulta'] = df['fechaConsulta'].astype(str).str[:10]
    
    df['fechaConsulta'] = pd.to_datetime(df['fechaConsulta'], errors='coerce')
    df['periodo_consulta'] = pd.to_datetime(df['periodo_consulta'], errors='coerce')
    df['cartera_fechaApertura_reciente'] = pd.to_datetime(df['cartera_fechaApertura_reciente'], errors='coerce')
    
    # 2. Identificación de columnas
    consultas_SFI = [col for col in df.columns if col.startswith('Consulta_SFI') and col.endswith('cantidad')]
    consultas_fecha = [col for col in df.columns if col.startswith('Consulta_') and col.endswith('_fecha')]
    
    df[consultas_fecha] = df[consultas_fecha].apply(pd.to_datetime, errors='coerce')
    df['fechaConsulta'] = pd.to_datetime(df['fechaConsulta'], errors='coerce')
    df['max_fecha_consulta'] = df[consultas_fecha].max(axis=1)
    df['Dias_ultimaconsul'] = (df['fechaConsulta'] - df['max_fecha_consulta']).dt.days

    # 5. Agregados

    df.loc[:, 'Consultas_entidad'] = df[consultas_fecha].notna().sum(axis=1)
    df.loc[:, 'Consultas_SFI'] = df[consultas_SFI].sum(axis=1)
    df.loc[:, 'Consultas_ult_mes'] = (
        df[consultas_fecha].gt(df['periodo_consulta'] - pd.DateOffset(months=1), axis=0)
    ).sum(axis=1)

    return df

def procesar_informe(xml_string):
    try:
        if xml_string.startswith("<?xml"):
            xml_string = xml_string.split("?>", 1)[1]
        root = ET.fromstring(xml_string)
    except ET.XMLSyntaxError as e:
        logging.error("No se pudo parsear el XML: %s", e)
        return None

    informe = root.find(".//Informe")
    
    if informe is None:
        logging.error("No se encontró la etiqueta 'Informe' en el XML")
        return None
    
    datos = OrderedDict()
    
    # métodos frecuentes
    get_attr = lambda el, k, d='': el.get(k, d) if el is not None else d
    
    # 1. Metadatos básicos
    datos['fechaConsulta'] = informe.get('fechaConsulta', '')
    datos['identificacionDigitada'] = informe.get('identificacionDigitada', '')
    
    # 2. Sección NaturalNacional
    natural_nacional = informe.find('NaturalNacional')
    identificacion = natural_nacional.find('Identificacion') if natural_nacional is not None else None
    for key in ['ciudad', 'departamento', 'genero']:
        datos[f'{key}_exp'] = get_attr(identificacion, key)
    
    # 3. Score
    score = informe.find('Score')
    datos['puntaje_experian'] = get_attr(score, 'puntaje')
        # Extraer códigos de razones 
    if score is not None:
        razones = score.findall('Razon')
        codigos_razon = [razon.get('codigo') for razon in razones]
        # Puedes guardar solo los primeros 2 o hasta N si necesitas
        datos['razon_codigo_1'] = codigos_razon[0] if len(codigos_razon) > 0 else None 
        datos['razon_codigo_2'] = codigos_razon[1] if len(codigos_razon) > 1 else None
    else:
        datos['razon_codigo_1'] = None
        datos['razon_codigo_2'] = None
    
    # 4. Cuentas Ahorro
    cuentas_ahorro = informe.findall('CuentaAhorro')
    fechas_apertura1 = [ahorro.get('fechaApertura') for ahorro in cuentas_ahorro
                        if ahorro.get('fechaApertura', '').strip()]
    datos['ahorros_fechaApertura_reciente'] = max(fechas_apertura1) if fechas_apertura1 else ''
    datos['Ctas_pCliente'] = len(cuentas_ahorro) 
    datos['Cuentas_sector1'] = sum(1 for cuenta in cuentas_ahorro if cuenta.attrib.get('sector') == '1')

    # 5. Cuentas Cartera
    contador_por_sector = defaultdict(int)
    fechas_apertura2 = []
    carteras = informe.findall('CuentaCartera')

    for i, cartera in enumerate(carteras, 1):
        sector = cartera.get('sector', '')
        
        if fecha_str := cartera.get('fechaApertura'):
            fecha_str = fecha_str.strip()
            if len(fecha_str) == 10 and fecha_str.count('-') == 2:
                try:
                    fecha_dt = datetime.strptime(fecha_str, '%Y-%m-%d')
                    fechas_apertura2.append(fecha_dt)
                except ValueError:
                    pass

        if valores := cartera.find('Valores'):
            for tipo in valores.findall('Valor') or []:
                contador_por_sector[sector] += 1
                idx = contador_por_sector[sector]
                datos.update({
                    f'Sector{sector}_{idx}_cuota': tipo.get('cuota', ''),
                    f'Sector{sector}_{idx}_totalCuotas': tipo.get('totalCuotas', ''),
                    f'Sector{sector}_{idx}_cuotasCanceladas': tipo.get('cuotasCanceladas', '')
                })

        estados = cartera.find('Estados')
        if estados is not None:
            estado_cuentas = estados.findall('EstadoCuenta')
            for j, estado in enumerate(estado_cuentas, 1): 
                for key in ['codigo']:
                    datos[f'Cartera_{i}_{j}_{key}'] = estado.attrib.get(key, '')

    datos['cartera_fechaApertura_reciente'] = max(fechas_apertura2) if fechas_apertura2 else ''
    
    datos['CC_TipoContrato_1'] = sum(
        1 for cartera in carteras
        if (caract := cartera.find('Caracteristicas')) is not None and caract.attrib.get('tipoContrato') == '1'
    )

    datos['cc_tpobl_2_con'] = sum(
        1 for cartera in carteras
        if (caract := cartera.find('Caracteristicas')) is not None and caract.attrib.get('tipoObligacion') == '2'
    )

    # 6. Tarjeta de Crédito
    tdc = informe.findall('TarjetaCredito')
    fechas_apertura3 = [tc.get('fechaApertura') for tc in tdc
                        if tc.get('fechaApertura', '').strip()]

    datos['tdc_fechaApertura_reciente'] = max(fechas_apertura3) if fechas_apertura3 else ''

    # 7. Consultas en Experian
    consultas = informe.findall('Consulta')
    consultas_por_tipo = defaultdict(list)
    for consulta in consultas:
        consultas_por_tipo[consulta.get('tipoCuenta', '')].append(consulta)
    
    for tipo in sorted(consultas_por_tipo):
        for i, consulta in enumerate(consultas_por_tipo[tipo], 1):
            datos.update({
                f'Consulta_{tipo}_{i}_cantidad': consulta.get('cantidad', ''),
                f'Consulta_{tipo}_{i}_nitSuscriptor': consulta.get('nitSuscriptor', ''),
                f'Consulta_{tipo}_{i}_fecha': consulta.get('fecha', '')
            })
    # 8. ProductosValores
    datos['quanto'] = get_attr(informe.find('productosValores'), 'valor1')
    datos['quanto_pct'] = get_attr(informe.find('productosValores'), 'valor1smlv')

    # 9. infoAgregada
    info = informe.find('InfoAgregada')
    res = info.find('Resumen') if info is not None else None

    # Principales
    principales = res.find('Principales') if res is not None else None
    for key in ['cuentasCerradasAHOCCB', 'creditosCerrados', 'antiguedadDesde', 'creditoVigentes','consultadasUlt6meses']:
        datos[f'agr_prinp_{key}'] = get_attr(principales, key)

    # Saldos
    saldo = res.find('Saldos') if res is not None else None
    for key in ['cuotaMensual', 'saldoTotal', 'saldoTotalEnMora', 'saldoCreditoMasAlto']:
        datos[f'agr_saldos_{key}'] = get_attr(saldo, key)

    if saldo is not None:
        for i, mes in enumerate(saldo.findall('Mes') or [], 1):
            datos.update({
                f'saldoTotalMora_{i}': mes.get('saldoTotalMora', ''),
                f'saldoTotal_{i}': mes.get('saldoTotal', '')
            })

    # Comportamiento
    comp = res.find('Comportamiento') if res is not None else None
    if comp is not None:
        for i, mes in enumerate(comp.findall('Mes') or [], 1):
            datos.update({
                f'comportamiento_{i}': mes.get('comportamiento', ''),
                f'cantidad_{i}': mes.get('cantidad', '')
            })

    # Portafolio
    if info is not None and (por := info.find('ComposicionPortafolio')) is not None:
        for i, tipo in enumerate(por.findall('TipoCuenta'), 1):
            if tipo is not None and hasattr(tipo, 'attrib'):
                for key in ['tipo', 'cantidad']:
                    datos[f'total_{key}_{i}'] = tipo.attrib.get(key, '')

                for j, est in enumerate(tipo.findall('Estado'), 1):
                    if est is not None and hasattr(est, 'attrib'):
                        codigo_valor = est.attrib.get('codigo', '')
                        if codigo_valor in {'Al dia', 'Activa'}:
                            for key in ['cantidad']:
                                datos[f'{tipo.attrib.get("tipo", "tipo")}_{codigo_valor}_{key}_{j}'] = est.attrib.get(key, '')

    # Evolución Deuda
    # evo = info.find('EvolucionDeuda') if info is not None else None
    # if evo is not None:
    #     if ap := evo.find('AnalisisPromedio'):
    #         for key in ['cuota', 'porcentajeUso', 'totalCerradas', 'totalAbiertas', 'saldo']:
    #             datos[f'agr_analisisPromedio_{key}'] = ap.get(key, '')
    evo = info.find('EvolucionDeuda') if info is not None else None
    if evo is not None:
        # Buscar el nodo AnalisisPromedio robustamente
        for ap in evo.iter('AnalisisPromedio'):
            for key in ['cuota', 'porcentajeUso', 'totalCerradas', 'totalAbiertas', 'saldo']:
                datos[f'agr_analisisPromedio_{key}'] = ap.get(key, '')
            break  # Solo queremos el primero
        
        # Buscar nodos Trimestre robustamente
        for i, trimestre in enumerate(evo.iter('Trimestre'), 1):
            datos.update({
                f'trimestre_{i}_cuota': trimestre.get('cuota', 'missing'),
                f'trimestre_{i}_cupoTotal': trimestre.get('cupoTotal', 'missing'),
                f'trimestre_{i}_moraMaxima': trimestre.get('moraMaxima', 'missing'),
                f'trimestre_{i}_saldo': trimestre.get('saldo', 'missing')
            })


    # 10. InfoAgregadaMicrocredito

    micro = informe.find('InfoAgregadaMicrocredito')
    if micro is not None:
        resumen_path = micro.find('Resumen')
        
        # 1. Extraer datos de CreditosCerrados
        if resumen_path is not None:
            perfil_general_path = resumen_path.find('PerfilGeneral')
            if perfil_general_path is not None:
                creditos_cerrados_path = perfil_general_path.find('CreditosCerrados')
                if creditos_cerrados_path is not None:
                    datos['Cc_sectorTelcos'] = creditos_cerrados_path.get('sectorTelcos', '')
                    datos['Cc_totalComoPrincipal'] = creditos_cerrados_path.get('totalComoPrincipal', '')

        # 2. Extraer datos de EvolucionDeuda
        if (evd := micro.find('EvolucionDeuda')) is not None:
            for sector in evd.findall('EvolucionDeudaSector'):
                if sector.get('codSector') == "4":
                    for tipo_cuenta in sector.findall('EvolucionDeudaTipoCuenta'):
                        tipo = tipo_cuenta.get('tipoCuenta', '')
                        for i, trimestre in enumerate(tipo_cuenta.findall('EvolucionDeudaValorTrimestre')[:3], 1):
                            datos[f'{tipo}_trim_{i}_saldoMora'] = trimestre.get('saldoMora', '')

        # VectorSaldosYMoras 
        if resumen_path is not None: 
            vector_saldos = resumen_path.find('VectorSaldosYMoras')
            if vector_saldos is not None:
                saldos_moras_list = vector_saldos.findall('SaldosYMoras')
                if saldos_moras_list:
                    for i, saldos_moras in enumerate(saldos_moras_list, 1):
                        datos.update({
                            f'saldoDeudaTotalMora_{i}': saldos_moras.get('saldoDeudaTotalMora', ''),
                            f'saldoDeudaTotal_{i}': saldos_moras.get('saldoDeudaTotal', ''),
                            f'numCreditosMayorIgual60_{i}': saldos_moras.get('numCreditosMayorIgual60', ''),
                            f'totalCuentasMora_{i}': saldos_moras.get('totalCuentasMora', '')
                        })
    
    # 11–14. Sumas de saldos
    def sumar_valores(xpath, atributo):
        return sum(
            float(valor.get(atributo, 0))
            for valor in informe.findall(xpath)
            if valor.get(atributo, '').replace('.', '', 1).isdigit()
        )
    
    datos.update({
        'cartera_saldo_actual': sumar_valores(".//CuentaCartera/Valores/Valor", 'saldoActual'),
        'cartera_saldo_mora': sumar_valores(".//CuentaCartera/Valores/Valor", 'saldoMora'),
        'tdc_saldo_actual': sumar_valores(".//TarjetaCredito/Valores/Valor", 'saldoActual'),
        'tdc_saldo_mora': sumar_valores(".//TarjetaCredito/Valores/Valor", 'saldoMora')
    })
    
    if datos:
        df = pd.DataFrame([datos])
        
        cols_cats = [
            'fechaConsulta', 'ciudad_exp', 'departamento_exp', 
            'ahorros_fechaApertura_reciente', 'cartera_fechaApertura_reciente', 'tdc_fechaApertura_reciente',
            'agr_prinp_antiguedadDesde', 'comportamiento_1', 'comportamiento_2', 
            'comportamiento_3', 'comportamiento_4', 'comportamiento_5', 'comportamiento_6', 
            'comportamiento_7', 'comportamiento_8', 
            'comportamiento_9', 'comportamiento_10', 'comportamiento_11', 'comportamiento_12', 
            'comportamiento_13', 'comportamiento_14', 'comportamiento_15', 'comportamiento_16', 
            'comportamiento_17', 'comportamiento_18', 'comportamiento_19', 'comportamiento_20', 
            'comportamiento_21', 'comportamiento_22', 'comportamiento_23', 'comportamiento_24', 
            'total_tipo_1', 'total_tipo_2', 'total_tipo_3', 
            'trimestre_1_moraMaxima', 'trimestre_2_moraMaxima', 'trimestre_3_moraMaxima']
        for col in df.loc[:,(df.columns.str.endswith('_fecha'))&(df.columns.str.startswith('Consulta_'))].columns:
            cols_cats.append(col)
        
        cols_to_convert = df.columns.difference(cols_cats)
        df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors='coerce')
        
        df = variables_income(df.copy())
        
        cartera_Cuota = []
        cartera_totalCuotas = []
        cartera_cuotasCanceladas = []
        
        for col in df.columns:
            if 'Sector' in col:
                if col.endswith('cuota'):
                    cartera_Cuota.append(col)
                elif 'totalCuotas' in col:
                    cartera_totalCuotas.append(col)
                elif 'cuotasCanceladas' in col:
                    cartera_cuotasCanceladas.append(col)
        cols_to_convert = cartera_Cuota + cartera_totalCuotas + cartera_cuotasCanceladas
            # 7. Eliminar columnas temporales
        columnas_a_eliminar = [col for col in cols_to_convert if col in df.columns]
        df.drop(columns=columnas_a_eliminar, inplace=True, errors='ignore')

        df = procesar_cuotas_y_amortizacion(df.copy())
            # Eliminar columnas de 'Cartera'
        df.drop(columns=[col for col in df.columns if 'Cartera' in col and col.endswith('codigo')], inplace=True)
        
        df['quanto'] = pd.to_numeric(df["quanto"], errors="coerce")
        df['agr_saldos_cuotaMensual'] = pd.to_numeric(df["agr_saldos_cuotaMensual"], errors="coerce")

        df["capacidad_endeudamiento"] = df["quanto"] - (df["quanto"] * 0.4) - df["agr_saldos_cuotaMensual"]
        df['ratio_endeudamiento'] = np.nan
        df.loc[df["quanto"] > 0, 'ratio_endeudamiento'] = df["capacidad_endeudamiento"] / df["quanto"]
        #datos = pd.DataFrame([datos])
        
        if consultas:
            df = procesar_consultas_experian(df.copy())
        return df

    return []
