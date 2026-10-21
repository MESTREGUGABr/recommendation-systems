"""Apresentação guiada: teste (matriz da Alice) -> treino (base real) -> recomendação.

Pensado para rodar ao vivo na aula: primeiro reproduz a matriz da Alice
respondendo as 3 perguntas de todo KNN (similaridade, quantos vizinhos,
nota estimada), depois treina com a base real em data/ratings.csv e gera
recomendações de verdade.
"""

from __future__ import annotations

from pathlib import Path

from example_slide import RATINGS
from recommender import ItemBasedKNN, load_ratings

DATA_PATH = Path(__file__).parent / "data" / "ratings.csv"


def pause(msg: str = "Pressione Enter para continuar...") -> None:
    input(f"\n{msg}")


def section(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def step_test() -> None:
    section("Passo 1 — Teste: matriz da Alice (mesma do Slide 2)")
    model = ItemBasedKNN(RATINGS, k=2, min_common=2)
    prediction = model.predict("Alice", "Item5")
    assert prediction is not None

    print("\n1) Similaridade — cosseno ajustado entre Item5 e cada item que Alice")
    print("   avaliou, sobre os usuários que avaliaram os dois itens:")
    for item_id in ("Item1", "Item2", "Item3", "Item4"):
        print(f"     Item5 x {item_id}: {model.similarity('Item5', item_id):+.4f}")

    print(f"\n2) Quantos vizinhos — k={model.k}, só entram os de similaridade positiva:")
    for neighbor in prediction.neighbors:
        print(f"     {neighbor.item_id}  (similaridade={neighbor.similarity:.4f}, "
              f"nota de Alice={neighbor.user_rating:g})")

    print(f"\n3) Nota estimada — média ponderada das notas de Alice nos itens vizinhos:")
    print(f"     Alice em Item5 ≈ {prediction.score:.4f}")


def step_train() -> ItemBasedKNN:
    section("Passo 2 — Treino: carregar a base real (data/ratings.csv)")
    ratings = load_ratings(DATA_PATH)
    model = ItemBasedKNN(ratings, k=2, min_common=2)
    n_ratings = sum(len(items) for items in ratings.values())
    print(f"\nUsuários carregados: {len(ratings)}")
    print(f"Itens distintos: {len(model.items)}")
    print(f"Avaliações carregadas: {n_ratings}")
    return model


def step_recommend(model: ItemBasedKNN) -> None:
    section("Passo 3 — Recomendação (base real)")
    chosen = input("\nPara qual usuário recomendar? (Enter = primeiro da base): ").strip()
    user_id = chosen or sorted(model.ratings)[0]
    if user_id not in model.ratings:
        print(f"Usuário {user_id!r} não encontrado na base.")
        return

    print(f"\n2) Quantos vizinhos — k={model.k}, só entram os de similaridade "
          f"positiva (com pelo menos {model.min_common} usuários em comum)")

    recommendations = model.recommend(user_id, n=5)
    print(f"\n1) Similaridade e 3) Nota estimada — Top-{len(recommendations)} "
          f"recomendações para {user_id}:")
    if not recommendations:
        print("     Nenhuma recomendação disponível (sem vizinhos positivos para nenhum item).")
        return
    for prediction in recommendations:
        print(f"\n     {prediction.item_id} — nota estimada {prediction.score:.2f}")
        print(f"       Similaridade com os itens que {user_id} já avaliou:")
        for neighbor in prediction.neighbors:
            print(f"         {neighbor.item_id}: similaridade={neighbor.similarity:.4f}, "
                  f"nota de {user_id}={neighbor.user_rating:g}")


def main() -> None:
    step_test()
    pause("Pressione Enter para treinar com a base real...")
    model = step_train()
    pause("Pressione Enter para gerar recomendações...")
    step_recommend(model)


if __name__ == "__main__":
    main()
