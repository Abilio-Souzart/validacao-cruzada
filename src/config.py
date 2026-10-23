"""Caminhos e parâmetros globais."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "base_pejota.xls"
FIG_DIR = ROOT / "results" / "figures"
TABLE_DIR = ROOT / "results" / "tables"

SEED = 42
TEST_SIZE = 0.20          # teste definitivo da Questão 17
N_FOLDS = 5
KS_IRIS = [1, 3, 5, 7, 9, 11, 13, 15]

# Análises complementares
N_REPETICOES = 200        # divisões aleatórias nas simulações
N_REPETICOES_CV = 50      # repetições de CV com sementes diferentes
N_REPETICOES_SELECAO = 100  # repetições da simulação de viés de seleção


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
