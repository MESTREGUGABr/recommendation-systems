"""Terminal navegável: menu passo a passo para rodar as implementações sem
precisar decorar caminhos e comandos.

Passo 1 — gerar a base de dados
Passo 2 — escolher o algoritmo (user-based, item-based, content-based ou
          knowledge-based); ao entrar, já roda o exemplo do slide:
          - KNN colaborativo: as 3 perguntas (1. Similaridade
            2. Quantos vizinhos 3. Estimar a nota)
          - Content-based: TF-IDF dos itens, cosseno com o perfil e ranking
          - Knowledge-based: o exemplo das câmeras do capítulo (consulta,
            QuickXPlain, diagnósticos, reparos, utilidade e críticas)
Passo 3 — treinar com a base real e recomendar, ou avaliar o modelo

Uso (a partir de qualquer lugar, com Python 3.10+ no PATH):

    py implementations/menu.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).parent
USERBASED = ROOT / "userbased-nearestneighbor"
ITEMBASED = ROOT / "itembased-nearestneighbor"
CONTENTBASED = ROOT / "content-based_recommendation"
KNOWLEDGEBASED = ROOT / "knowledge-based_recommendation"

# Módulos que cada pasta de implementação define com o mesmo nome; precisam
# ser removidos do cache antes de importar os de outra pasta, senão o
# Python reaproveita o que já tinha importado da pasta anterior.
_SHARED_MODULE_NAMES = ("present", "example_slide", "recommender", "evaluate")


def load_present(folder: Path) -> ModuleType:
    """Importa o present.py da pasta do algoritmo escolhido, sem misturar
    com o módulo de mesmo nome da outra pasta."""
    folder_str = str(folder)
    for name in _SHARED_MODULE_NAMES:
        sys.modules.pop(name, None)
    sys.path.insert(0, folder_str)
    try:
        import present
    finally:
        sys.path.remove(folder_str)
    return present


def run(*args: str) -> None:
    print("\n" + "-" * 60)
    subprocess.run(args)
    print("-" * 60)
    input("\nPressione Enter para continuar...")


def ask(label: str) -> str:
    return input(f"  {label} (Enter para usar o padrão): ").strip()


def read_choice(prompt: str, valid: set[str]) -> str:
    while True:
        choice = input(prompt).strip()
        if choice in valid:
            return choice
        print("Opção inválida.")


# ---------------------------------------------------------------- Passo 1 --

def step1_generate_dataset() -> None:
    print("\n=== Passo 1 — Base de dados ===")
    print("  1) Gerar/regerar a base (grava em userbased-nearestneighbor/data/,")
    print("     itembased-nearestneighbor/data/ e content-based_recommendation/data/)")
    print("  2) Pular (já tenho uma base gerada)")
    choice = read_choice("\nEscolha uma opção: ", {"1", "2"})
    if choice == "1":
        seed = ask("seed")
        args = [sys.executable, str(ROOT / "generate_dataset.py")]
        if seed:
            args += ["--seed", seed]
        run(*args)


# ---------------------------------------------------------------- Passo 2 --

ALGORITHMS = {
    "1": ("User-based (similaridade de Pearson)", USERBASED),
    "2": ("Item-based (cosseno ajustado)", ITEMBASED),
    "3": ("Content-based (TF-IDF + cosseno)", CONTENTBASED),
    "4": ("Knowledge-based (restrições e casos)", KNOWLEDGEBASED),
}


def step2_choose_algorithm() -> tuple[str, Path] | None:
    print("\n=== Passo 2 — Escolher o algoritmo ===")
    for key, (label, _folder) in ALGORITHMS.items():
        print(f"  {key}) {label}")
    print("  0) Sair")
    choice = read_choice("\nEscolha uma opção: ", {"0", *ALGORITHMS})
    if choice == "0":
        return None
    label, folder = ALGORITHMS[choice]
    return label, folder


# ---------------------------------------------------------------- Passo 3 --

def evaluate_userbased() -> None:
    print("Padrões: test-ratio=0.2, seed=42, min-train=3, k=2, min-common=3.")
    test_ratio = ask("test-ratio")
    seed = ask("seed")
    min_train = ask("min-train")
    k = ask("k")
    min_common = ask("min-common")
    args = [sys.executable, str(USERBASED / "evaluate.py")]
    for flag, value in (
        ("--test-ratio", test_ratio),
        ("--seed", seed),
        ("--min-train", min_train),
        ("--k", k),
        ("--min-common", min_common),
    ):
        if value:
            args += [flag, value]
    run(*args)


def evaluate_contentbased() -> None:
    print("Padrões: test-ratio=0.2, seed=42, min-train=3, k=3, top-n=5.")
    test_ratio = ask("test-ratio")
    seed = ask("seed")
    min_train = ask("min-train")
    k = ask("k")
    top_n = ask("top-n")
    args = [sys.executable, str(CONTENTBASED / "evaluate.py")]
    for flag, value in (
        ("--test-ratio", test_ratio),
        ("--seed", seed),
        ("--min-train", min_train),
        ("--k", k),
        ("--top-n", top_n),
    ):
        if value:
            args += [flag, value]
    run(*args)


def step3_run_algorithm(label: str, folder: Path) -> None:
    # Entrar no algoritmo já roda o exemplo do slide.
    present = load_present(folder)
    print("\n" + "-" * 60)
    present.step_test()
    print("-" * 60)
    input("\nPressione Enter para continuar...")

    # O item-based ainda não tem avaliação própria, e o knowledge-based não usa
    # notas (não há o que esconder para testar).
    evaluators = {USERBASED: evaluate_userbased, CONTENTBASED: evaluate_contentbased}
    evaluate = evaluators.get(folder)
    while True:
        print(f"\n=== Passo 3 — {label} ===")
        if folder == KNOWLEDGEBASED:
            print("  1) Informar requisitos e receber recomendações (catálogo de câmeras)")
        else:
            print("  1) Treinar com a base real e recomendar")
        if evaluate:
            if folder == CONTENTBASED:
                print("  2) Avaliar o modelo (MAE/RMSE do KNN e Precision/Recall do perfil)")
            else:
                print("  2) Avaliar o modelo (MAE/RMSE sobre notas escondidas de teste)")
        print("  3) Voltar (escolher outro algoritmo)")
        print("  0) Sair")

        valid = {"1", "3", "0"} | ({"2"} if evaluate else set())
        choice = read_choice("\nEscolha uma opção: ", valid)

        if choice == "0":
            raise SystemExit
        if choice == "3":
            return
        if choice == "1":
            present = load_present(folder)
            print("\n" + "-" * 60)
            model = present.step_train()
            present.step_recommend(model)
            print("-" * 60)
            input("\nPressione Enter para continuar...")
        elif choice == "2" and evaluate:
            evaluate()


def main() -> None:
    print("=== Sistemas de Recomendação — terminal navegável ===")
    try:
        step1_generate_dataset()
        while True:
            chosen = step2_choose_algorithm()
            if chosen is None:
                break
            step3_run_algorithm(*chosen)
    except (SystemExit, KeyboardInterrupt, EOFError):
        pass
    print("\nAté mais.")


if __name__ == "__main__":
    main()
