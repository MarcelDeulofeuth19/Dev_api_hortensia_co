"""Extraccion y procesamiento de XML Experian (Regular)."""
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
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
# Logging centralizado desde logging_config.setup_logging()

def variables_income(df):
    logging.info("[extract] variables_income: start | cols=%d", len(df.columns))
    df = df.copy()
    
    df['Avance_global_cartera'] = np.nan
    df.loc[
        (df['cartera_valorInicial_activa'] != 0) & (~df['cartera_valorInicial_activa'].isnull()),
        'Avance_global_cartera'
    ] = (
        df['cartera_saldo_actual'] / df['cartera_valorInicial_activa']
    )
    
    cuota = [col for col in df.columns if col.startswith('Sector') and col.endswith('cuota')]
    logging.info("[extract] variables_income: found %d 'cuota' cols", len(cuota))
    df['promedio_cuota'] = df.loc[:, cuota].sum(axis=1) / df.loc[:, cuota].notna().sum(axis=1)
    df['promedio_cuota'] = df['promedio_cuota'].fillna(0)
    
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
    logging.info(
        "[extract] variables_income: cartera sets | cuota=%d totalCuotas=%d canceladas=%d",
        len(cartera_Cuota), len(cartera_totalCuotas), len(cartera_cuotasCanceladas),
    )
    for col in cols_to_convert:
        if col in df.columns and not is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Cálculo de amortización general (con control de división por cero)
    denom_amort = df[cartera_totalCuotas].sum(axis=1) * df[cartera_Cuota].sum(axis=1)
    denom_amort = denom_amort.replace(0, np.nan)
    df["amortizacion_cartera"] = (
        (df[cartera_cuotasCanceladas].sum(axis=1) * df[cartera_Cuota].sum(axis=1)) / denom_amort
    )
    logging.info("[extract] variables_income: computed amortizacion_cartera")

    # Sector3: cartera real
    cartera_real_cuotasCanceladas = [col for col in df.columns if 'Sector3' in col and col.endswith('cuotasCanceladas')]
    cartera_real_totalCuotas = [col for col in df.columns if 'Sector3' in col and 'totalCuotas' in col]

    cuotas_canceladas_real = df[cartera_real_cuotasCanceladas].replace([0, -1], np.nan)
    total_cuotas_real = df[cartera_real_totalCuotas].replace([0, -1], np.nan)

    df["ratio_cartera_real"] = cuotas_canceladas_real.sum(axis=1) / total_cuotas_real.sum(axis=1)
    logging.info("[extract] variables_income: computed ratio_cartera_real")

    # Sector4: telcos  ##Cambio
    cartera_telcos_cuotas = [col for col in df.columns if 'Sector4' in col and col.endswith('cuota')]
    cuotas_canceladas_telcos = df[cartera_telcos_cuotas].where(lambda x: (x != 0))

    df["ratio_cartera_telcos"] = cuotas_canceladas_telcos.sum(axis=1) / cuotas_canceladas_telcos.count(axis=1)
    df['ratio_cartera_real'] = df['ratio_cartera_real'].replace([np.inf, -np.inf], np.nan)
    df['ratio_cartera_telcos'] = np.abs(df['ratio_cartera_telcos'])
    logging.info("[extract] variables_income: computed ratio_cartera_telcos")
    
    
        # AGREGAR A LA API
    df['ratio_creditos_Neg'] = np.where(
        (df['agr_prinp_creditoVigentes'] == 0) | (df['agr_prinp_creditoVigentes'].isna()),
        0,  
        df['agr_prinp_creditosActualesNegativos'] / df['agr_prinp_creditoVigentes']
    )
    
    suma_mora = df['saldoTotalMora_1'] + df['saldoTotalMora_2'] + df['saldoTotalMora_3']
    suma_saldo = df['saldoTotal_1'] + df['saldoTotal_2'] + df['saldoTotal_3']

    df['ratio_mora_saldo_3m'] = np.where(
        suma_saldo == 0,
        0,
        suma_mora / suma_saldo
    )
    logging.info("[extract] variables_income: computed ratio_mora_saldo_3m")
    
    relacion_t = np.where(df['trimestre_1_saldo'] == 0, 0, df['trimestre_1_cuota'] / df['trimestre_1_saldo'])
    relacion_t_1 = np.where(df['trimestre_2_saldo'] == 0, 0, df['trimestre_2_cuota'] / df['trimestre_2_saldo'])

    df['ratio_cuota_saldo_6m'] = np.where(
        relacion_t_1 == 0,
        0,
        ((relacion_t - relacion_t_1) / relacion_t_1)
    )

        # AGREGAR A LA API
    cols = ['trimestre_1_porcentajeUso', 'trimestre_2_porcentajeUso', 'trimestre_3_porcentajeUso']
    if all(col in df.columns for col in cols):
        x = np.arange(1, len(cols)+1)
        Y = df[cols].values

        N = len(x)
        sum_x = np.sum(x)
        sum_x2 = np.sum(x**2)
        sum_y = np.sum(Y, axis=1)
        sum_xy = np.sum(Y * x, axis=1)

        num = N * sum_xy - sum_x * sum_y
        denom = N * sum_x2 - sum_x**2
        df['VarPctUso'] = num / denom
        logging.info("[extract] variables_income: computed VarPctUso")
    else:
        df['VarPctUso'] = 0
        logging.info("[extract] variables_income: VarPctUso defaulted to 0 (missing cols)")
        
    if 'fecha_max_vencimiento' in df.columns: # modificar API
        df['Periodos_max_vencimiento'] = (
            (pd.to_datetime(df['fecha_max_vencimiento']).dt.year - pd.to_datetime(df['fechaConsulta']).dt.year) * 12 +
            (pd.to_datetime(df['fecha_max_vencimiento']).dt.month - pd.to_datetime(df['fechaConsulta']).dt.month)
        )
        logging.info("[extract] variables_income: computed Periodos_max_vencimiento")

    def calcular_variacion1_2(row):
        x, y = row['trimestre_1_porcentajeUso'], row['trimestre_2_porcentajeUso']
        if x == 0 and y == 0:
            return 0
        elif x == 0:
            return np.nan
        else:
            return (y - x) / x

    def calcular_variacion1_3(row):
        x, y = row['trimestre_1_porcentajeUso'], row['trimestre_3_porcentajeUso']
        if x == 0 and y == 0:
            return 0
        elif x == 0:
            return np.nan
        else:
            return (y - x) / x

    df['porcentajeUso_1al2'] = df.apply(calcular_variacion1_2, axis=1)
    df['porcentajeUso_1al3'] = df.apply(calcular_variacion1_3, axis=1)
    logging.info("[extract] variables_income: computed porcentajeUso deltas")
    
    return df

def procesar_cuotas_y_amortizacion(df):
    logging.info("[extract] procesar_cuotas_y_amortizacion: start")
    totales_cantidad = [col for col in df.columns if 'total_cantidad' in col ]
    activas = [col for col in df.columns if 'Activa_' in col and 'cantidad' in col ]
    aldia =  [col for col in df.columns if '_Al dia_' in col and 'cantidad' in col]
    buenas = activas + aldia 
    logging.info(
        "[extract] cuotas: totales=%d activas=%d aldia=%d",
        len(totales_cantidad), len(activas), len(aldia),
    )

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
    logging.debug("portafolio_total_ahorros: %s", df['portafolio_total_ahorros'].to_list() if hasattr(df['portafolio_total_ahorros'], 'to_list') else df['portafolio_total_ahorros'])
    df['portafolio_positivas_ahorros'] = pd.to_numeric(df['portafolio_positivas_ahorros'], errors='coerce')
    df['portafolio_total_ahorros'] = pd.to_numeric(df['portafolio_total_ahorros'], errors='coerce')
    
        # AGREGAR A LA API
    df['portafolio_num_can_n'] = df['portafolio_mora'] / (df['portafolio_totales'] - df['portafolio_total_ahorros'])
    df['portafolio_num_can_p'] = (df['portafolio_aldia'] - df['portafolio_positivas_ahorros'] ) / (df['portafolio_totales'] - df['portafolio_total_ahorros'])
    logging.info("[extract] cuotas: computed portafolio ratios")
# CAMBIO RESTA NUM CAN P
    df['portafolio_num_can_p'] = df['portafolio_num_can_p'].replace([np.inf, -np.inf], np.nan)
    df['portafolio_num_can_n'] = df['portafolio_num_can_n'].replace([np.inf, -np.inf], np.nan)
    
    #df['portafolio_num_can_p'] = df['portafolio_num_can_p'].fillna(0)
    logging.info("[extract] procesar_cuotas_y_amortizacion: end")
    df['portafolio_num_can_n'] = df['portafolio_num_can_n'].fillna(0)


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

    # Productos saldos
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

    return df

def procesar_consultas_experian(df):
    logging.info("[extract] procesar_consultas_experian: start")
   
    df = df.copy()

    # 1. Conversión de fechas
    df.loc[:, 'periodo_consulta'] = df['fechaConsulta'].astype(str).str[:10]
    df.loc[:, 'fechaConsulta'] = pd.to_datetime(df['fechaConsulta'], errors='coerce')
    df.loc[:, 'periodo_consulta'] = pd.to_datetime(df['periodo_consulta'], errors='coerce')
    df.loc[:, 'cartera_fechaApertura_reciente'] = pd.to_datetime(df['cartera_fechaApertura_reciente'], errors='coerce')

    consultas_SFI = [col for col in df.columns if col.startswith('Consulta_SFI') and col.endswith('cantidad')]
    consultas_fecha = [col for col in df.columns if col.startswith('Consulta_') and col.endswith('_fecha')]

    df[consultas_fecha] = df[consultas_fecha].apply(pd.to_datetime, errors='coerce')
    df['fechaConsulta'] = pd.to_datetime(df['fechaConsulta'], errors='coerce')
    df['max_fecha_consulta'] = df[consultas_fecha].max(axis=1)
    df['Dias_ultimaconsul'] = (df['fechaConsulta'] - df['max_fecha_consulta']).dt.days
    
    df.loc[:, 'Dias_ultimo_producto'] = (df['fechaConsulta'] - df['cartera_fechaApertura_reciente']).dt.days

    # 5. Agregados

    df.loc[:, 'Consultas_entidad'] = df[consultas_fecha].notna().sum(axis=1)
    df.loc[:, 'Consultas_SFI'] = df[consultas_SFI].sum(axis=1)
    df.loc[:, 'Consultas_ult_mes'] = (
        df[consultas_fecha].gt(df['periodo_consulta'] - pd.DateOffset(months=1), axis=0)
    ).sum(axis=1)


    logging.info("[extract] procesar_consultas_experian: end")
    return df

def procesar_informe(xml_string):
    logging.info("[extract] procesar_informe: start | xml_len=%d", len(xml_string) if isinstance(xml_string, str) else -1)
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
    
    # 4. Cuentas Ahorro
    cuentas_ahorro = informe.findall('CuentaAhorro')
    fechas_apertura1 = [ahorro.get('fechaApertura') for ahorro in cuentas_ahorro
                        if ahorro.get('fechaApertura', '').strip()]
    datos['ahorros_fechaApertura_reciente'] = max(fechas_apertura1) if fechas_apertura1 else ''
    datos['Ctas_pCliente'] = len(cuentas_ahorro) 
    logging.info("[extract] ahorro: cuentas=%d", len(cuentas_ahorro))
    datos['Cuentas_sector1'] = sum(1 for cuenta in cuentas_ahorro if cuenta.attrib.get('sector') == '1')

    # 5. Cuentas Cartera
    contador_por_sector = defaultdict(int)
    fechas_apertura2 = []
    carteras = informe.findall('CuentaCartera')
    
    fechas_vencimiento = [
        cartera.get('fechaVencimiento')
        for cartera in carteras
        if cartera.get('fechaVencimiento', '').strip()
    ]
    datos['fecha_max_vencimiento'] = max(fechas_vencimiento) if fechas_vencimiento else ''

    logging.info("[extract] cartera: cuentas=%d", len(carteras))
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
    logging.info("[extract] tdc: tarjetas=%d", len(tdc))

    # 7. Consultas en Experian
    consultas = informe.findall('Consulta')
    consultas_por_tipo = defaultdict(list)
    for consulta in consultas:
        consultas_por_tipo[consulta.get('tipoCuenta', '')].append(consulta)

    logging.info("[extract] consultas: total=%d tipos=%d", len(consultas), len(consultas_por_tipo))
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
    logging.info("[extract] productosValores: quanto=%s quanto_pct=%s", datos['quanto'], datos['quanto_pct'])

    # 9. infoAgregada
    info = informe.find('InfoAgregada')
    res = info.find('Resumen') if info is not None else None

    # Principales
    principales = res.find('Principales') if res is not None else None     # AGREGAR A LA API
    for key in ['creditoVigentes','creditosCerrados', 'creditosActualesNegativos', 'histNegUlt12Meses', 'cuentasAbiertasAHOCCB','cuentasCerradasAHOCCB','consultadasUlt6meses','desacuerdosALaFecha','antiguedadDesde', 'reclamosVigentes']: # Modificar API
        datos[f'agr_prinp_{key}'] = get_attr(principales, key)

    # Saldos
    saldo = res.find('Saldos') if res is not None else None
    for key in ['saldoTotalEnMora', 'saldoM30', 'saldoM60', 'saldoM90', 'cuotaMensual', 'saldoCreditoMasAlto', 'saldoTotal']: # Modificar API
        datos[f'agr_saldos_{key}'] = get_attr(saldo, key)

    if saldo is not None:
        for i, mes in enumerate(saldo.findall('Mes') or [], 1):
            datos.update({
                f'saldoTotalMora_{i}': mes.get('saldoTotalMora', ''),
                f'saldoTotal_{i}': mes.get('saldoTotal', '')
            })
        logging.info("[extract] saldos: meses=%d", i if 'i' in locals() else 0)

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

    evo = info.find('EvolucionDeuda') if info is not None else None
    if evo is not None:
        for ap in evo.iter('AnalisisPromedio'):
            for key in ['cuota', 'porcentajeUso', 'totalCerradas', 'totalAbiertas', 'saldo']:
                datos[f'agr_analisisPromedio_{key}'] = ap.get(key, '')
            break
        for i, trimestre in enumerate(evo.iter('Trimestre'), 1):
            datos.update({
                f'trimestre_{i}_cuota': trimestre.get('cuota', 'missing'),
                f'trimestre_{i}_cupoTotal': trimestre.get('cupoTotal', 'missing'),
                f'trimestre_{i}_moraMaxima': trimestre.get('moraMaxima', 'missing'),
                f'trimestre_{i}_saldo': trimestre.get('saldo', 'missing'),                # AGREGAR A LA API
                f'trimestre_{i}_porcentajeUso': trimestre.get('porcentajeUso', 'missing')
                # AGREGAR A LA API
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
                            f'totalCuentasMora_{i}': saldos_moras.get('totalCuentasMora', ''),
                            f'numCreditos30_{i}': saldos_moras.get('numCreditos30', ''),
                        })

    # 11–14. Sumas de saldos
    def sumar_valores(xpath, atributo):
        return sum(
            float(valor.get(atributo, 0))
            for valor in informe.findall(xpath)
            if valor.get(atributo, '').replace('.', '', 1).isdigit()
        )

    def sumar_valor_inicial_activa(xpath):
        return sum(
            float(valor.get('valorInicial', 0))
            for valor in informe.findall(xpath)
            if valor.get('saldoActual', '').replace('.', '', 1).isdigit() and 
            float(valor.get('saldoActual', 0)) not in [0, -1]
        )

    def sumar_cupo_total_estado_tarjetas(xpath_tarjeta, codigos_validos, ):
        suma = 0.0
        for tarjeta in informe.findall(xpath_tarjeta):
            estados = tarjeta.find('Estados')
            codigo_estado = None
            if estados is not None:
                for estado in estados.findall('EstadoCuenta'):
                    codigo_estado = estado.get('codigo')

            if codigo_estado in codigos_validos:
                valores = tarjeta.find('Valores')
                if valores is not None:
                    for valor in valores.findall('Valor'):
                        v = valor.get('cupoTotal', '')
                        if v.replace('.', '', 1).isdigit():
                            suma += float(v)
        return suma
    
    datos.update({
        'cartera_saldo_actual': sumar_valores(".//CuentaCartera/Valores/Valor", 'saldoActual'),
        'cartera_saldo_mora': sumar_valores(".//CuentaCartera/Valores/Valor", 'saldoMora'),
        'tdc_saldo_actual': sumar_valores(".//TarjetaCredito/Valores/Valor", 'saldoActual'),
        'tdc_saldo_mora': sumar_valores(".//TarjetaCredito/Valores/Valor", 'saldoMora'),
        'cartera_valorInicial_activa': sumar_valor_inicial_activa(".//CuentaCartera/Valores/Valor"),
        'tdc_cupototal_activo': sumar_cupo_total_estado_tarjetas(".//TarjetaCredito",['01', '13', '14', '15', '16'])
    })
    logging.info(
        "[extract] sumas: cartera_saldo=%.2f tdc_saldo=%.2f cupo_activo=%.2f",
        datos.get('cartera_saldo_actual', 0) or 0,
        datos.get('tdc_saldo_actual', 0) or 0,
        datos.get('tdc_cupototal_activo', 0) or 0,
    )
    
    if datos:
        logging.info("[extract] dataframe: construyendo con %d claves", len(datos))
        df = pd.DataFrame([datos])
        
        cols_cats = [
        'fechaConsulta',  'fecha_max_vencimiento',
        'ciudad_exp','departamento_exp','agr_prinp_antiguedadDesde',
        'ahorros_fechaApertura_reciente','cartera_fechaApertura_reciente','tdc_fechaApertura_reciente',
        'comportamiento_1', 'comportamiento_2', 'comportamiento_3', 'comportamiento_4',
        'comportamiento_5', 'comportamiento_6', 'comportamiento_7', 'comportamiento_8',
        'comportamiento_9', 'comportamiento_10', 'comportamiento_11', 'comportamiento_12',
        'comportamiento_13', 'comportamiento_14', 'comportamiento_15', 'comportamiento_16',
        'comportamiento_17', 'comportamiento_18', 'comportamiento_19', 'comportamiento_20',
        'comportamiento_21', 'comportamiento_22', 'comportamiento_23', 'comportamiento_24',
        'trimestre_1_moraMaxima','trimestre_2_moraMaxima','trimestre_3_moraMaxima',
        'total_tipo_1','total_tipo_2','total_tipo_3']
        
        for col in df.loc[:,(df.columns.str.endswith('_fecha'))&(df.columns.str.startswith('Consulta_'))].columns:
            cols_cats.append(col)
        
        cols_to_convert = df.columns.difference(cols_cats)
        df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors='coerce')
        
        logging.info("[extract] variables_income: calling")
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

        logging.info("[extract] cuotas_y_amortizacion: calling")
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
            logging.info("[extract] consultas_experian: calling")
            df = procesar_consultas_experian(df.copy())
        logging.info("[extract] procesar_informe: end | final_cols=%d", len(df.columns))
        return df

    return []
