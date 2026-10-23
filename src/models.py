"""Os três modelos da Questão 17 e o KNN da Questão 18, todos como Pipeline.

Imputação e padronização ficam dentro do Pipeline: em cada fold da validação
cruzada, são ajustadas só com os dados de treino daquele fold.
"""
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import SEED


def logistica() -> Pipeline:
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                     ("model", LogisticRegression(max_iter=1000, random_state=SEED))])


def arvore(max_depth: int = 4) -> Pipeline:
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("model", DecisionTreeClassifier(max_depth=max_depth, random_state=SEED))])


def knn(k: int = 5, imputar: bool = True) -> Pipeline:
    passos = [("imp", SimpleImputer(strategy="median"))] if imputar else []
    passos += [("sc", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=k))]
    return Pipeline(passos)


def modelos_q17() -> dict:
    return {"Regressão logística": logistica(),
            "Árvore de decisão (prof. 4)": arvore(4),
            "KNN (k = 5)": knn(5)}
