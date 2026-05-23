"""tests/test_data.py — Tests del generador de datos."""
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, "src")
from train_pipeline import FEATURES, TARGET # noqa: E402

# Ruta absoluta al archivo de datos (para que lo lea el makefile sin importar el directorio de trabajo)
DATA_PATH = Path(__file__).parent.parent / "data" / "Resultado_renovacion_prestamo.csv"


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
    assert 0.03 <= rate <= 0.20, f"Tasa de clase 1 fuera de rango: {rate:.2%}"


def test_generate_no_nulls():
    """El dataset no debe tener valores nulos."""
    df = pd.read_csv(DATA_PATH)
    assert df.isnull().sum().sum() == 0


def test_generate_n_rows():
    """El dataset debe tener más de n filas."""
    df = pd.read_csv(DATA_PATH)
    assert len(df) >= 400

