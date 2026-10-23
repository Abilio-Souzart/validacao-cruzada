"""Análises complementares à Questão 17.

1. Quanto as estimativas variam (itens a e c): repetindo a divisão treino/teste
   e a divisão em folds.
2. O viés de escolher o modelo pelo teste (item d), medido por simulação.
3. Validação cruzada aninhada: a estimativa honesta de todo o processo de seleção.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from .config import N_FOLDS, N_REPETICOES, N_REPETICOES_CV, N_REPETICOES_SELECAO, SEED
from .models import arvore, knn, logistica, modelos_q17
from .utils import header, save_figure, save_table


def candidatos() -> dict:
    """Conjunto amplo de modelos para as simulações de seleção (21 candidatos)."""
    c = {"logística": logistica()}
    c.update({f"árvore prof. {d}": arvore(d) for d in range(1, 11)})
    c.update({f"KNN k = {k}": knn(k) for k in range(1, 40, 4)})
    return c


# ---------------------------------------------------------------------------
def variabilidade_divisao(X, y, n_rep=N_REPETICOES) -> pd.DataFrame:
    linhas = []
    for seed in range(n_rep):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
        m = logistica().fit(X_tr, y_tr)
        linhas.append({"seed": seed, "acc_treino": m.score(X_tr, y_tr), "acc_teste": m.score(X_te, y_te)})
    return pd.DataFrame(linhas).set_index("seed")


def variabilidade_folds(split, n_rep=N_REPETICOES_CV) -> pd.DataFrame:
    linhas = []
    for seed in range(n_rep):
        cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
        linhas.append({"seed": seed, **{n: cross_val_score(p, split.X_tr, split.y_tr, cv=cv).mean()
                                         for n, p in modelos_q17().items()}})
    return pd.DataFrame(linhas).set_index("seed")


def plot_variabilidade(div: pd.DataFrame, acc_tr_42: float, acc_te_42: float,
                       folds: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    gap = 100 * (div["acc_treino"] - div["acc_teste"])
    axes[0].hist(gap, bins=30, color="#90A4AE")
    axes[0].axvline(gap.mean(), color="k", lw=2, label=f"média = {gap.mean():+.1f} p.p.")
    axes[0].axvline(100 * (acc_tr_42 - acc_te_42), color="#F44336", ls="--", lw=2,
                    label="divisão do notebook (semente 42)")
    axes[0].set_xlabel("Acurácia no treino − acurácia no teste (p.p.)")
    axes[0].set_ylabel("Nº de divisões")
    axes[0].set_title(f"(a) Regressão logística em {len(div)} divisões 80/20")
    axes[0].legend(fontsize=9)
    for col in folds:
        axes[1].hist(100 * folds[col], bins=15, alpha=0.6, label=col)
    axes[1].set_xlabel("Acurácia média de CV (%)")
    axes[1].set_title(f"(b) CV de 5 folds com {len(folds)} sorteios diferentes de folds")
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    save_figure(fig, "comp_variabilidade")


# ---------------------------------------------------------------------------
def _uma_repeticao(X, y, rep: int) -> dict:
    X_a, X_novo, y_a, y_novo = train_test_split(X, y, test_size=0.2, random_state=1000 + rep, stratify=y)
    X_tr, X_te, y_tr, y_te = train_test_split(X_a, y_a, test_size=0.25, random_state=2000 + rep, stratify=y_a)
    cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=rep)
    res = {}
    for nome, p in candidatos().items():
        cv_acc = cross_val_score(clone(p), X_tr, y_tr, cv=cv).mean()
        p.fit(X_tr, y_tr)
        res[nome] = (p.score(X_te, y_te), cv_acc, p.score(X_novo, y_novo))
    r = pd.DataFrame(res, index=["teste", "cv", "novo"]).T
    pelo_teste, pelo_cv = r["teste"].idxmax(), r["cv"].idxmax()
    return {
        "rep": rep,
        "escolhido_pelo_teste": pelo_teste,
        "acc_teste_prometida": r.loc[pelo_teste, "teste"],
        "acc_novo_pelo_teste": r.loc[pelo_teste, "novo"],
        "escolhido_pelo_cv": pelo_cv,
        "acc_cv_prometida": r.loc[pelo_cv, "cv"],
        "acc_teste_pelo_cv": r.loc[pelo_cv, "teste"],
        "acc_novo_pelo_cv": r.loc[pelo_cv, "novo"],
    }


def simular_selecao(X, y, n_rep: int = N_REPETICOES_SELECAO) -> pd.DataFrame:
    """Em cada repetição: treino 60% / teste 20% / dados novos 20%.

    Escolhe o melhor de 21 modelos (a) pelo teste e (b) por CV no treino, e
    compara a acurácia que cada critério "promete" com a obtida em dados novos.
    As repetições rodam em paralelo.
    """
    from joblib import Parallel, delayed
    linhas = Parallel(n_jobs=-1)(delayed(_uma_repeticao)(X, y, rep) for rep in range(n_rep))
    return pd.DataFrame(linhas).set_index("rep")


def plot_selecao(sim: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    d1 = 100 * (sim["acc_teste_prometida"] - sim["acc_novo_pelo_teste"])
    d2 = 100 * (sim["acc_teste_pelo_cv"] - sim["acc_novo_pelo_cv"])
    bins = np.linspace(min(d1.min(), d2.min()), max(d1.max(), d2.max()), 30)
    ax.hist(d2, bins=bins, alpha=0.6, color="#4CAF50",
            label=f"Escolhido por CV, avaliado no teste (média {d2.mean():+.1f} p.p.)")
    ax.hist(d1, bins=bins, alpha=0.6, color="#F44336",
            label=f"Escolhido pelo teste, avaliado no teste (média {d1.mean():+.1f} p.p.)")
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("Acurácia estimada − acurácia em dados novos (p.p.)")
    ax.set_ylabel("Nº de repetições")
    ax.set_title(f"(d) Escolher entre 21 modelos pelo teste infla a estimativa ({len(sim)} repetições)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save_figure(fig, "comp_vies_selecao_no_teste")


# ---------------------------------------------------------------------------
def cv_aninhada(split) -> dict:
    """CV interna escolhe o modelo entre os 21 candidatos; CV externa avalia o processo."""
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                     ("model", LogisticRegression(max_iter=1000, random_state=SEED))])
    grade = [
        {"model": [LogisticRegression(max_iter=1000, random_state=SEED)]},
            {"model": [DecisionTreeClassifier(random_state=SEED)], "model__max_depth": list(range(1, 11))},
        {"model": [KNeighborsClassifier()], "model__n_neighbors": list(range(1, 40, 4))},
    ]
    interna = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    externa = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED + 1)
    busca = GridSearchCV(pipe, grade, cv=interna, scoring="accuracy", n_jobs=-1)

    busca.fit(split.X_tr, split.y_tr)
    melhor_cv = busca.best_score_
    escolhido = busca.best_params_
    aninhada = cross_val_score(busca, split.X_tr, split.y_tr, cv=externa, scoring="accuracy")
    teste = busca.score(split.X_te, split.y_te)
    return {"melhor CV (não aninhada)": melhor_cv, "CV aninhada": aninhada.mean(),
            "CV aninhada (desvio)": aninhada.std(), "teste definitivo": teste,
            "modelo escolhido": str({k: (v.__class__.__name__ if k == "model" else v)
                                     for k, v in escolhido.items()})}


# ---------------------------------------------------------------------------
def run(q17: dict) -> dict:
    header("ANÁLISES COMPLEMENTARES - Questão 17")
    X, y, split = q17["X"], q17["y"], q17["split"]

    div = variabilidade_divisao(X, y)
    save_table(div.round(4), "comp_divisoes_treino_teste")
    pct = (div["acc_teste"] < q17["acc_te"]).mean()
    print(f"1a. {len(div)} divisões 80/20 (logística): treino {div['acc_treino'].mean():.4f} | "
          f"teste {div['acc_teste'].mean():.4f} ± {div['acc_teste'].std():.4f} | "
          f"treino > teste em {(div['acc_treino'] > div['acc_teste']).mean():.0%} das divisões")
    print(f"    A divisão do notebook (teste = {q17['acc_te']:.4f}) é melhor que {pct:.0%} das divisões.")

    folds = variabilidade_folds(split)
    save_table(folds.round(4), "comp_cv_sementes")
    print(f"1b. CV de 5 folds com {len(folds)} sorteios de folds:")
    print(folds.agg(["mean", "std", "min", "max"]).round(4).to_string())
    print(f"    Logística > árvore em {(folds.iloc[:, 0] > folds.iloc[:, 1]).mean():.0%} dos sorteios")
    plot_variabilidade(div, q17["acc_tr"], q17["acc_te"], folds)

    sim = simular_selecao(X, y)
    save_table(sim, "comp_vies_selecao")
    print(f"\n2. Viés de seleção ({len(sim)} repetições, 21 modelos):")
    print(f"   Pelo teste: estimativa {sim['acc_teste_prometida'].mean():.4f} | em dados novos "
          f"{sim['acc_novo_pelo_teste'].mean():.4f} | otimismo "
          f"{100 * (sim['acc_teste_prometida'] - sim['acc_novo_pelo_teste']).mean():+.2f} p.p.")
    print(f"   Por CV:     teste {sim['acc_teste_pelo_cv'].mean():.4f} | em dados novos "
          f"{sim['acc_novo_pelo_cv'].mean():.4f} | diferença "
          f"{100 * (sim['acc_teste_pelo_cv'] - sim['acc_novo_pelo_cv']).mean():+.2f} p.p.")
    print(f"   CV prometia {sim['acc_cv_prometida'].mean():.4f} para o modelo escolhido por ela")
    print("   Modelos escolhidos pelo teste (mais frequentes):",
          sim["escolhido_pelo_teste"].value_counts().head(5).to_dict())
    print("   Modelos escolhidos por CV (mais frequentes):",
          sim["escolhido_pelo_cv"].value_counts().head(5).to_dict())
    plot_selecao(sim)

    an = cv_aninhada(split)
    save_table(pd.Series(an).rename("valor").to_frame(), "comp_cv_aninhada")
    print("\n3. CV aninhada (seleção entre 21 modelos dentro de cada fold externo):")
    for k, v in an.items():
        print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")
    return {"div": div, "folds": folds, "sim": sim, "aninhada": an}
