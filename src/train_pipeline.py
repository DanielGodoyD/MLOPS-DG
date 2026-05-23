"""train_pipeline.py — Entrena el modelo y guarda artefactos."""
import json
import pickle
import logging
from pathlib import Path

import numpy as np  # noqa: F401
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import make_scorer, recall_score, precision_score, f1_score, roc_auc_score, accuracy_score


# from generate_data import generate, DATA_PATH

DATA_PATH = Path("data/Resultado_renovacion_prestamo.csv")

ARTIFACTS = Path("artifacts")
MODEL_PATH = ARTIFACTS / "modelo.pkl"
METRICS_PATH = ARTIFACTS / "metrics.json"

FEATURES = ['Plazo_Renovado',  'Nro_Entidades',
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
RANDOM_STATE = 123
TEST_SIZE = 0.30
RECALL_MIN = 0.60  # quality gate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | TRAIN | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_data() -> pd.DataFrame:
    """Carga o genera el dataset."""
    if not DATA_PATH.exists():
        log.info("No hay data")
    return pd.read_csv(DATA_PATH)


def train(df: pd.DataFrame) -> dict:
    """Entrena con GridSearchCV + SMOTE y retorna métricas."""

    X, y = df[FEATURES], df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Primero, volvemos a crear df_train a partir de X_train y y_train
    df_train = pd.concat([X_train, y_train], axis=1)

    # Luego, obtenemos los conteos de las clases
    count_class_0, count_class_1 = df_train[TARGET].value_counts()

    # Dividimos en datasets de clase 0 y clase 1
    df_class_0 = df_train[df_train[TARGET] == 0]
    df_class_1 = df_train[df_train[TARGET] == 1]

    # Realizamos el undersampling para la clase mayoritaria
    df_class_0_under = df_class_0.sample(count_class_1, random_state=42)

    # Concatenamos para crear el conjunto de entrenamiento undersampleado
    df_train_under = pd.concat([df_class_0_under, df_class_1], axis=0)

    # Obtener X_train e y_train undersampleados
    X_tr_s = df_train_under.drop(TARGET, axis=1)
    y_tr_s = df_train_under[TARGET]

    # Definimos las métricas que nos interesan para la clase positiva (FLAG_VENTA = 1)
    # Usamos make_scorer para poder pasarlos a GridSearchCV
    scoring = {
        'roc_auc': make_scorer(roc_auc_score),
        'recall_pos': make_scorer(recall_score, pos_label=1),
        'precision_pos': make_scorer(precision_score, pos_label=1),
        'f1_pos': make_scorer(f1_score, pos_label=1)
    }

    # Creamos el modelo base de RandomForestClassifier
    rf_model = RandomForestClassifier(random_state=42)

    # Definimos los hiperparámetros a buscar
    param_grid = {
        'n_estimators': [50, 100, 150],  # Número de árboles en el bosque
        'max_depth': [None, 10, 20],   # Profundidad máxima del árbol
        'min_samples_leaf': [1, 2, 4], # Número mínimo de muestras requeridas para estar en un nodo hoja
        'class_weight': ['balanced']   # Para manejar el desbalance de clases
    }

    # Inicializamos GridSearchCV
    # refit='f1_pos' significa que el modelo final se entrenará con la combinación de parámetros que maximice el F1-Score de la clase positiva.
    gs = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        scoring=scoring,
        refit='f1_pos', # Refit con la métrica f1_pos para optimizar el balance entre recall y precision de la clase 1
        cv=3, # Número de folds para la validación cruzada
        verbose=2, # Nivel de verbosidad
        n_jobs=-1  # Usa todos los núcleos disponibles
    )

    gs.fit(X_tr_s, y_tr_s)

    y_pred = gs.best_estimator_.predict(X_test)

    metricas = {
        "f1": round(f1_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "params": gs.best_params_,
        "recall_minimo": RECALL_MIN,
    }

    log.info(
        "F1=%.4f | Recall=%.4f | Acc=%.4f",
        metricas["f1"], metricas["recall"], metricas["accuracy"],
    )

    ARTIFACTS.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(gs.best_estimator_, f)
    with open(METRICS_PATH, "w") as f:
        json.dump(metricas, f, indent=2)

    log.info("Artefactos guardados en %s", ARTIFACTS)
    return metricas


if __name__ == "__main__":
    df = load_data()
    log.info("Dataset: %d filas", len(df))
    metricas = train(df)
    print(json.dumps(metricas, indent=2))
