"""tests/test_model.py — Tests del modelo serializado."""
import os
import pickle
import sys
import pandas as pd
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, "src")
from train_pipeline import train # noqa: E402

# Ruta absoluta al archivo de datos (no depende del directorio de trabajo actual)
DATA_PATH = Path(__file__).parent.parent / "data" / "Resultado_renovacion_prestamo.csv"

@pytest.fixture(scope="module")
def modelo_entrenado(tmp_path_factory):
    """Fixture: entrena el modelo una vez para todos los tests del módulo."""
    tmp = tmp_path_factory.mktemp("workdir")
    os.chdir(tmp)
    df = pd.read_csv(DATA_PATH)
    train(df)
    with open(tmp / "artifacts" / "modelo.pkl", "rb") as f:
        return pickle.load(f)


def test_model_has_predict(modelo_entrenado):
    """El modelo debe tener el método predict."""
    assert hasattr(modelo_entrenado, "predict")


def test_model_has_predict_proba(modelo_entrenado):
    """El modelo debe tener el método predict_proba."""
    assert hasattr(modelo_entrenado, "predict_proba")


def test_model_predicts_binary(modelo_entrenado):
    """Las predicciones deben ser binarias (0 o 1)."""
    
    # Definimos un conjunto de datos ficticios
    X = pd.DataFrame({'Plazo_Renovado':[12],
                            'Nro_Entidades':[3],
                            'Dif_Entidades':[0],
                            'Meses_oferta':[6],
                            'EDAD':[34],
                            'Flag_LimProv':[0],
                            'Uso_Linea_LOG':[0.440676],
                            'Uso_TrimLinea_LOG':[0.440676],
                            'Saldo_Consumo_LOG':[8.88],
                            'SUELDO_ESTIMADO_LOG':[8.134],
                            'ANTIGUEDAD_MES_LOG':[5],
                            'Linea_Renovado_LOG':[1500],
                            'Ahorro_LOG':[2093],
                            'Prestamo_vigente_LOG':[18.512],
                            'Promed_6Mdeuda_LOG':[8.051],
                            'Deuda_Cubierta%_LOG':[0.67],
                            'REGION_CALLAO':[0],
                            'REGION_CENTRO':[1],
                            'REGION_LIMA BALNEARIO':[0],
                            'REGION_LIMA CENTRO':[0],
                            'REGION_LIMA ESTE':[0],
                            'REGION_LIMA MODERNA':[0],
                            'REGION_LIMA NORTE':[0],
                            'REGION_LIMA PROVINCIA':[0],
                            'REGION_LIMA SUR':[0],
                            'REGION_NORTE':[0],
                            'REGION_OESTE':[0],
                            'REGION_ORIENTE':[0],
                            'REGION_SIERRA CENTRAL':[0],
                            'REGION_SUR':[0],
                            'SEXO_F':[0],
                            'SEXO_M':[1],
                            'EST_CIVIL_C':[1],
                            'EST_CIVIL_D':[0],
                            'EST_CIVIL_S':[0],
                            'EST_CIVIL_U':[0],
                            'EST_CIVIL_V':[0],
                            'EST_CIVIL_X':[0],
                            'EST_CIVIL_Y':[0],
                            'Cluster':[1]})

    y_pred = modelo_entrenado.predict(X)
    assert set(y_pred).issubset({0, 1})


def test_model_predict_proba_shape(modelo_entrenado):
    """predict_proba debe retornar shape (n, 2) con valores en [0, 1]."""
    
    X = pd.DataFrame({'Plazo_Renovado':[12],
                            'Nro_Entidades':[3],
                            'Dif_Entidades':[0],
                            'Meses_oferta':[6],
                            'EDAD':[34],
                            'Flag_LimProv':[0],
                            'Uso_Linea_LOG':[0.440676],
                            'Uso_TrimLinea_LOG':[0.440676],
                            'Saldo_Consumo_LOG':[8.88],
                            'SUELDO_ESTIMADO_LOG':[8.134],
                            'ANTIGUEDAD_MES_LOG':[5],
                            'Linea_Renovado_LOG':[1500],
                            'Ahorro_LOG':[2093],
                            'Prestamo_vigente_LOG':[18.512],
                            'Promed_6Mdeuda_LOG':[8.051],
                            'Deuda_Cubierta%_LOG':[0.67],
                            'REGION_CALLAO':[0],
                            'REGION_CENTRO':[1],
                            'REGION_LIMA BALNEARIO':[0],
                            'REGION_LIMA CENTRO':[0],
                            'REGION_LIMA ESTE':[0],
                            'REGION_LIMA MODERNA':[0],
                            'REGION_LIMA NORTE':[0],
                            'REGION_LIMA PROVINCIA':[0],
                            'REGION_LIMA SUR':[0],
                            'REGION_NORTE':[0],
                            'REGION_OESTE':[0],
                            'REGION_ORIENTE':[0],
                            'REGION_SIERRA CENTRAL':[0],
                            'REGION_SUR':[0],
                            'SEXO_F':[0],
                            'SEXO_M':[1],
                            'EST_CIVIL_C':[1],
                            'EST_CIVIL_D':[0],
                            'EST_CIVIL_S':[0],
                            'EST_CIVIL_U':[0],
                            'EST_CIVIL_V':[0],
                            'EST_CIVIL_X':[0],
                            'EST_CIVIL_Y':[0],
                            'Cluster':[1]})

    proba = modelo_entrenado.predict_proba(X)
    assert proba.shape == (1, 2)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_model_predict_proba_sums_to_one(modelo_entrenado):
    """Las probabilidades por fila deben sumar 1."""
    
    X = pd.DataFrame({'Plazo_Renovado':[12],
                            'Nro_Entidades':[3],
                            'Dif_Entidades':[0],
                            'Meses_oferta':[6],
                            'EDAD':[34],
                            'Flag_LimProv':[0],
                            'Uso_Linea_LOG':[0.440676],
                            'Uso_TrimLinea_LOG':[0.440676],
                            'Saldo_Consumo_LOG':[8.88],
                            'SUELDO_ESTIMADO_LOG':[8.134],
                            'ANTIGUEDAD_MES_LOG':[5],
                            'Linea_Renovado_LOG':[1500],
                            'Ahorro_LOG':[2093],
                            'Prestamo_vigente_LOG':[18.512],
                            'Promed_6Mdeuda_LOG':[8.051],
                            'Deuda_Cubierta%_LOG':[0.67],
                            'REGION_CALLAO':[0],
                            'REGION_CENTRO':[1],
                            'REGION_LIMA BALNEARIO':[0],
                            'REGION_LIMA CENTRO':[0],
                            'REGION_LIMA ESTE':[0],
                            'REGION_LIMA MODERNA':[0],
                            'REGION_LIMA NORTE':[0],
                            'REGION_LIMA PROVINCIA':[0],
                            'REGION_LIMA SUR':[0],
                            'REGION_NORTE':[0],
                            'REGION_OESTE':[0],
                            'REGION_ORIENTE':[0],
                            'REGION_SIERRA CENTRAL':[0],
                            'REGION_SUR':[0],
                            'SEXO_F':[0],
                            'SEXO_M':[1],
                            'EST_CIVIL_C':[1],
                            'EST_CIVIL_D':[0],
                            'EST_CIVIL_S':[0],
                            'EST_CIVIL_U':[0],
                            'EST_CIVIL_V':[0],
                            'EST_CIVIL_X':[0],
                            'EST_CIVIL_Y':[0],
                            'Cluster':[1]})

    proba = modelo_entrenado.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_model_is_random_forest(modelo_entrenado):
    """El modelo debe ser un RandomForestClassifier."""
    from sklearn.ensemble import RandomForestClassifier 
    assert isinstance(modelo_entrenado, RandomForestClassifier)
