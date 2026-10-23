"""Testes do pré-processamento e de propriedades usadas na análise."""
import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.data import load_pejota, make_split
from src.models import logistica
from src.q17_vazamento import ruido_selecao


@pytest.fixture(scope="module")
def dados():
    return load_pejota()


def test_variaveis(dados):
    X, y = dados
    assert X.shape == (1500, 20)
    assert not {"COD_CLI", "STATUS"} & set(X.columns)
    assert set(X["UF"].unique()) == {0, 1, 2}
    assert y.mean() == pytest.approx(0.5)


def test_split_estratificado(dados):
    s = make_split(*dados)
    assert len(s.y_te) == 300 and s.y_te.mean() == pytest.approx(0.5)


def test_scaler_ajustado_so_no_treino_do_fold(dados):
    """Dentro do Pipeline, a média aprendida pelo scaler é a do treino, não a da base inteira."""
    s = make_split(*dados)
    pipe = logistica().fit(s.X_tr, s.y_tr)
    imputado = pipe.named_steps["imp"].transform(s.X_tr)
    assert np.allclose(pipe.named_steps["sc"].mean_, imputado.mean(axis=0))


def test_reproduz_notebook(dados):
    s = make_split(*dados)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    assert cross_val_score(logistica(), s.X_tr, s.y_tr, cv=cv).mean() == pytest.approx(0.6883, abs=1e-4)


def test_vazamento_na_selecao():
    r = ruido_selecao(n_rep=3)
    assert r["seleção fora do Pipeline"].mean() > 0.65
    assert abs(r["seleção dentro do Pipeline"].mean() - 0.5) < 0.1
