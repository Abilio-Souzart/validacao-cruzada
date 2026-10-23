"""Item 17(e): vazamento de dados quando o pré-processamento fica fora do Pipeline.

1. Na base PEJOTA: imputação e padronização fora x dentro do Pipeline.
2. Em dados sintéticos sem nenhum sinal: seleção de variáveis fora x dentro.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import N_FOLDS, SEED
from .models import logistica
from .utils import header, save_figure, save_table


def pejota_fora_x_dentro(split, n_rep: int = 20) -> pd.DataFrame:
    linhas = []
    for seed in range(n_rep):
        cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
        dentro = cross_val_score(logistica(), split.X_tr, split.y_tr, cv=cv).mean()
        X_prep = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(split.X_tr))
        fora = cross_val_score(LogisticRegression(max_iter=1000, random_state=SEED),
                               X_prep, split.y_tr, cv=cv).mean()
        linhas.append({"seed": seed, "dentro do Pipeline": dentro, "fora do Pipeline": fora})
    return pd.DataFrame(linhas).set_index("seed")


def ruido_selecao(n=200, p=2000, k=20, n_rep=20) -> pd.DataFrame:
    """Rótulos aleatórios e variáveis de puro ruído: a acurácia verdadeira é 50%."""
    rng = np.random.default_rng(SEED)
    linhas = []
    for rep in range(n_rep):
        X = rng.standard_normal((n, p))
        y = rng.integers(0, 2, n)
        cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=rep)
        # ERRADO: escolhe as k variáveis mais associadas a y usando todos os dados
        X_sel = SelectKBest(f_classif, k=k).fit_transform(X, y)
        fora = cross_val_score(LogisticRegression(max_iter=1000), X_sel, y, cv=cv).mean()
        # CERTO: a seleção é refeita em cada fold, só com os dados de treino do fold
        pipe = Pipeline([("sel", SelectKBest(f_classif, k=k)),
                         ("model", LogisticRegression(max_iter=1000))])
        dentro = cross_val_score(pipe, X, y, cv=cv).mean()
        linhas.append({"rep": rep, "seleção fora do Pipeline": fora,
                       "seleção dentro do Pipeline": dentro})
    return pd.DataFrame(linhas).set_index("rep")


def plot(pej: pd.DataFrame, ruido: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, df, titulo, ylim in [
        (axes[0], pej, "PEJOTA: imputação e padronização", (0.66, 0.71)),
        (axes[1], ruido, "Ruído puro: seleção de 20 entre 2.000 variáveis", (0.3, 1.0)),
    ]:
        cores = ["#F44336" if "fora" in c else "#4CAF50" for c in df.columns]
        ax.bar(range(len(df.columns)), df.mean(), yerr=df.std(), color=cores, width=0.5, capsize=6)
        ax.set_xticks(range(len(df.columns)), [c.replace(" do", "\ndo") for c in df.columns])
        for i, v in enumerate(df.mean()):
            ax.text(i, v + df.std().iloc[i] + 0.005, f"{v:.1%}", ha="center", fontweight="bold")
        ax.set_ylim(*ylim)
        ax.set_ylabel("Acurácia média de CV")
        ax.set_title(titulo)
    axes[1].axhline(0.5, color="k", ls="--", lw=1, label="Acurácia verdadeira (50%)")
    axes[1].legend(loc="upper right")
    fig.suptitle("(e) Pré-processamento fora × dentro do Pipeline")
    fig.tight_layout()
    save_figure(fig, "q17e_vazamento")


def run(q17: dict) -> dict:
    header("ITEM 17(e) - Vazamento de dados")
    pej = pejota_fora_x_dentro(q17["split"])
    ruido = ruido_selecao()
    save_table(pej.round(5), "q17e_pejota_fora_dentro")
    save_table(ruido.round(4), "q17e_ruido_selecao")
    dif = pej["fora do Pipeline"] - pej["dentro do Pipeline"]
    print(f"PEJOTA (20 sorteios de folds): dentro = {pej['dentro do Pipeline'].mean():.4f} | "
          f"fora = {pej['fora do Pipeline'].mean():.4f} | diferença média = {100 * dif.mean():+.2f} p.p. "
          f"(máx |dif| = {100 * dif.abs().max():.2f})")
    print(f"Ruído puro (20 repetições): seleção fora = {ruido.iloc[:, 0].mean():.3f} | "
          f"dentro = {ruido.iloc[:, 1].mean():.3f}")
    plot(pej, ruido)
    return {"pejota": pej, "ruido": ruido}
