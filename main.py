"""Executa a análise completa de validação cruzada.

Uso:
    python main.py                    # tudo (cerca de 4 minutos em 1 núcleo)
    python main.py --sem-complementos # só os itens do enunciado e o 17(e)
"""
import argparse
import warnings

import matplotlib

matplotlib.use("Agg")

from src import q17_complementos, q17_pejota, q17_vazamento, q18_iris  # noqa: E402
from src.config import FIG_DIR, TABLE_DIR, ensure_dirs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sem-complementos", action="store_true",
                        help="não roda as simulações da Questão 17 (as mais demoradas)")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)
    ensure_dirs()

    q17 = q17_pejota.run()
    q17_vazamento.run(q17)
    if not args.sem_complementos:
        q17_complementos.run(q17)
    q18_iris.run()
    print(f"\nTabelas: {TABLE_DIR}\nFiguras: {FIG_DIR}")


if __name__ == "__main__":
    main()
