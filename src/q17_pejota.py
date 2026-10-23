"""Questão 17: validação cruzada na base PEJOTA, itens (a) a (c)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from .config import N_FOLDS, SEED
from .data import load_pejota, make_split
from .models import logistica, modelos_q17
from .utils import header, save_figure, save_table


def erro_padrao_acuracia(acc: float, n: int) -> float:
    """Erro padrão de uma proporção: quanto a acurácia oscilaria entre conjuntos de teste de tamanho n."""
    return float(np.sqrt(acc * (1 - acc) / n))


def plot_folds(scores: dict) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    nomes = list(scores)
    ax.boxplot([scores[n] for n in nomes], tick_labels=[n.replace(" (", "\n(") for n in nomes],
               widths=0.5, patch_artist=True, boxprops=dict(facecolor="#BBDEFB"),
               medianprops=dict(color="navy", lw=2))
    for i, n in enumerate(nomes, start=1):
        ax.scatter(np.full(len(scores[n]), i) + np.linspace(-0.08, 0.08, len(scores[n])),
                   scores[n], color="k", s=18, zorder=3)
    ax.axhline(0.5, color="grey", ls=":", lw=1, label="Baseline (50%)")
    ax.set_ylabel("Acurácia no fold de validação")
    ax.set_ylim(0.45, 0.8)
    ax.set_title("(b) Acurácia por fold: CV estratificada de 5 folds nos 80% de treino")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_figure(fig, "q17b_acuracia_por_fold")


def run() -> dict:
    header("QUESTÃO 17 - Validação cruzada na base PEJOTA")
    X, y = load_pejota()
    split = make_split(X, y)
    print(f"20 variáveis | treino: {len(split.y_tr)} | teste definitivo: {len(split.y_te)}")

    # (a)
    m = logistica().fit(split.X_tr, split.y_tr)
    acc_tr = accuracy_score(split.y_tr, m.predict(split.X_tr))
    acc_te = accuracy_score(split.y_te, m.predict(split.X_te))
    ep = erro_padrao_acuracia(acc_te, len(split.y_te))
    print(f"\n(a) Acurácia no treino = {acc_tr:.4f} | no teste = {acc_te:.4f} "
          f"(erro padrão do teste ≈ {ep:.3f}, IC 95% ≈ {acc_te - 1.96 * ep:.3f} a {acc_te + 1.96 * ep:.3f})")

    # (b)
    cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    scores = {n: cross_val_score(p, split.X_tr, split.y_tr, cv=cv, scoring="accuracy")
              for n, p in modelos_q17().items()}
    tab = pd.DataFrame({
        "acurácia média": {n: s.mean() for n, s in scores.items()},
        "desvio padrão": {n: s.std() for n, s in scores.items()},
        "mín": {n: s.min() for n, s in scores.items()},
        "máx": {n: s.max() for n, s in scores.items()},
    })
    save_table(tab.round(4), "q17b_cv_modelos")
    save_table(pd.DataFrame(scores, index=[f"fold {i}" for i in range(1, N_FOLDS + 1)]).round(4),
               "q17b_cv_por_fold")
    print("\n(b) CV estratificada de 5 folds (80% de treino):")
    print(tab.round(4).to_string())
    plot_folds(scores)

    # Diferença fold a fold (os modelos usam os mesmos folds)
    lr, dt = scores["Regressão logística"], scores["Árvore de decisão (prof. 4)"]
    print(f"    Logística − árvore, fold a fold: {np.round(lr - dt, 3).tolist()}")

    # (c)
    melhor = tab["acurácia média"].idxmax()
    final = modelos_q17()[melhor].fit(split.X_tr, split.y_tr)
    acc_final = accuracy_score(split.y_te, final.predict(split.X_te))
    print(f"\n(c) Melhor pelo CV: {melhor} ({tab.loc[melhor, 'acurácia média']:.4f})")
    print(f"    Acurácia no teste definitivo: {acc_final:.4f} | diferença CV − teste = "
          f"{tab.loc[melhor, 'acurácia média'] - acc_final:+.4f} "
          f"(≈ {(acc_final - tab.loc[melhor, 'acurácia média']) / ep:.1f} erros padrão do teste)")

    return {"X": X, "y": y, "split": split, "acc_tr": acc_tr, "acc_te": acc_te,
            "tab": tab, "scores": scores, "melhor": melhor, "acc_final": acc_final}
