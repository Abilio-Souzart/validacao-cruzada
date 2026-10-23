"""Questão 18: escolha de k no KNN por validação cruzada (Iris, do próprio scikit-learn)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold, cross_val_score

from .config import KS_IRIS, N_FOLDS, SEED
from .models import knn
from .utils import header, save_figure, save_table


def run(n_rep: int = 50) -> dict:
    header("QUESTÃO 18 - KNN na Iris: escolha de k por CV")
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target

    # (a) Exatamente como no enunciado: 5 folds, semente 42
    cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    scores = {k: cross_val_score(knn(k, imputar=False), X, y, cv=cv) for k in KS_IRIS}
    tab = pd.DataFrame({"acurácia média": {k: s.mean() for k, s in scores.items()},
                        "desvio padrão": {k: s.std() for k, s in scores.items()},
                        "flores erradas (de 150)": {k: int(round(150 * (1 - s.mean())))
                                                     for k, s in scores.items()}})
    tab.index.name = "k"
    melhor = int(tab["acurácia média"].idxmax())
    save_table(tab.round(4), "q18a_cv_por_k")
    print("(a) 5-fold CV estratificada (semente 42):")
    print(tab.round(4).to_string())
    print(f"    Melhor k: {melhor}")

    # Complemento: o mesmo procedimento com 50 sorteios de folds
    rep = pd.DataFrame({k: [cross_val_score(knn(k, imputar=False), X, y,
                                            cv=StratifiedKFold(N_FOLDS, shuffle=True, random_state=s)).mean()
                            for s in range(n_rep)] for k in KS_IRIS})
    melhores = rep.idxmax(axis=1).value_counts().reindex(KS_IRIS, fill_value=0)
    resumo = pd.DataFrame({"média em 50 sorteios": rep.mean(), "desvio entre sorteios": rep.std(),
                           "vezes como melhor k": melhores})
    resumo.index.name = "k"
    save_table(resumo.round(4), "q18_cv_repetida")
    print(f"\nComplemento: {n_rep} sorteios diferentes de folds:")
    print(resumo.round(4).to_string())

    # (b) Gráfico
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [1.4, 1]})
    m, s = tab["acurácia média"], tab["desvio padrão"]
    axes[0].plot(KS_IRIS, m, "o-", color="steelblue", lw=2, label="Semente 42 (enunciado)")
    axes[0].fill_between(KS_IRIS, m - s, m + s, alpha=0.2, color="steelblue", label="± 1 desvio entre folds")
    axes[0].plot(KS_IRIS, resumo["média em 50 sorteios"], "s--", color="#FB8C00",
                 label="Média de 50 sorteios de folds")
    axes[0].axvline(melhor, color="tomato", ls=":", label=f"Melhor k (semente 42) = {melhor}")
    axes[0].set_xticks(KS_IRIS)
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Acurácia de CV")
    axes[0].set_ylim(0.85, 1.0)
    axes[0].set_title("(b) Acurácia média ± 1 desvio padrão por k")
    axes[0].legend(fontsize=8, loc="lower right")
    axes[1].bar([str(k) for k in KS_IRIS], melhores.values, color="#90A4AE")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Nº de sorteios")
    axes[1].set_title(f"Qual k vence em cada um dos {n_rep} sorteios de folds")
    fig.tight_layout()
    save_figure(fig, "q18b_knn_iris")
    return {"tab": tab, "melhor": melhor, "resumo": resumo}
