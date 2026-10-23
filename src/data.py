"""Leitura e preparação da base PEJOTA, como no notebook.

As 20 variáveis são todas as colunas exceto COD_CLI e STATUS. UF é codificada
como inteiro (MG = 0, RJ = 1, SP = 2), reproduzindo o LabelEncoder do notebook.
A imputação e a padronização ficam dentro dos Pipelines (models.py).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_PATH, SEED, TEST_SIZE

COLS_COM_BRANCOS = ["IFIN1", "IFIN2", "IFIN4", "IFIN5", "IFIN7", "IFIN8"]
CODIGO_UF = {"MG": 0, "RJ": 1, "SP": 2}


def load_pejota() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_excel(DATA_PATH, sheet_name="PEJOTA")
    df.columns = df.columns.str.strip()
    df[COLS_COM_BRANCOS] = (df[COLS_COM_BRANCOS]
                            .replace(r"^\s*$", np.nan, regex=True)
                            .apply(pd.to_numeric, errors="coerce"))
    df["UF"] = df["UF"].astype(str).str.strip().map(CODIGO_UF)
    y = (df["STATUS"] == "MAU").astype(int).rename("y")
    X = df.drop(columns=["COD_CLI", "STATUS"])
    return X, y


@dataclass
class Split:
    X_tr: pd.DataFrame
    X_te: pd.DataFrame
    y_tr: pd.Series
    y_te: pd.Series


def make_split(X, y, test_size: float = TEST_SIZE, seed: int = SEED) -> Split:
    return Split(*train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y))
