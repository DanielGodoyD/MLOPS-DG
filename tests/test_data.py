"""tests/test_data.py — Tests del generador de datos."""
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, "src")
#from generate_data import generate  # noqa: E402

DATA_PATH = Path("data/Resultado_renovacion_prestamo.csv")

FEATURES = ['MES', 'CLIENTE', 'Plazo_Renovado',  'Nro_Entidades',
       'Dif_Entidades', 'Meses_oferta', 'EDAD', 'Flag_LimProv',
       'Uso_Linea_LOG', 'Uso_TrimLinea_LOG', 'Saldo_Consumo_LOG',
       'SUELDO_ESTIMADO_LOG', 'ANTIGUEDAD_MES_LOG', 'Linea_Renovado_LOG',
       'Ahorro_LOG', 'Prestamo_vigente_LOG', 'Promed_6Mdeuda_LOG',
       'Deuda_Cubierta%_LOG', 'REGION_CALLAO', 'REGION_CENTRO',
       'REGION_LIMA BALNEARIO', 'REGION_LIMA CENTRO', 'REGION_LIMA ESTE',
       'REGION_LIMA MODERNA', 'REGION_LIMA NORTE', 'REGION_LIMA PROVINCIA',
       'REGION_LIMA SUR', 'REGION_NORTE', 'REGION_OESTE', 'REGION_ORIENTE',
       'REGION_SIERRA CENTRAL', 'REGION_SUR', 'SEXO_F', 'SEXO_M',
       'EST_CIVIL_C', 'EST_CIVIL_D', 'EST_CIVIL_S', 'EST_CIVIL_U',
       'EST_CIVIL_V', 'EST_CIVIL_X', 'EST_CIVIL_Y', 'Cluster']

TARGET = "FLAG_VENTA"

def test_generate_has_correct_columns():
    """El dataset debe tener todas las columnas esperadas."""
    df = pd.read_csv(DATA_PATH)
    for col in FEATURES + [TARGET]:
        assert col in df.columns, f"Columna faltante: {col}"


def test_generate_target_is_binary():
    """El target FLAG_VENTA debe ser binario (0 o 1)."""
    df = pd.read_csv(DATA_PATH)
    assert set(df[TARGET].unique()).issubset({0, 1})


def test_generate_class_imbalance():
    """La tasa de clase 1 debe estar entre 5% y 20%."""
    df = pd.read_csv(DATA_PATH)
    rate = df[TARGET].mean()
    assert 0.05 <= rate <= 0.20, f"Tasa de clase 1 fuera de rango: {rate:.2%}"


def test_generate_no_nulls():
    """El dataset no debe tener valores nulos."""
    df = pd.read_csv(DATA_PATH)
    assert df.isnull().sum().sum() == 0


def test_generate_n_rows():
    """El dataset debe tener más de n filas."""
    df = pd.read_csv(DATA_PATH)
    assert len(df) >= 400

