"""Apresentação guiada: teste (Slide 2) -> treino (base real) -> recomendação.

Pensado para rodar ao vivo na aula: primeiro reproduz o exemplo da Alice
respondendo as 3 perguntas de todo KNN (similaridade, quantos vizinhos,
nota estimada), depois treina com a base real em data/ratings.csv e gera
recomendações de verdade.
"""

from __future__ import annotations

from pathlib import Path

from example_slide import RATINGS
from recommender import UserBasedKNN, load_ratings

DATA_PATH = Path(__file__).parent / "data" / "ratings.csv"


def pause(msg: str = "Pressione Enter para continuar...") -> None:
    input(f"\n{msg}")


def section(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def step_test() -> None:
    section("Passo 1 — Teste: exemplo do Slide 2 (matriz da Alice)")
    model = UserBasedKNN(RATINGS, k=2)
    prediction = model.predict("Alice", "Item5")
    assert prediction is not None

    print("\n1) Similaridade — correlação de Pearson entre Alice e cada usuário,")
    print("   calculada só sobre os itens que os dois avaliaram em comum:")
    for user_id in ("User1", "User2", "User3", "User4"):
        print(f"     Alice x {user_id}: {model.similarity('Alice', user_id):+.4f}")

    print(f"\n2) Quantos vizinhos — k={model.k}, só entram os de similaridade positiva:")
    for neighbor in prediction.neighbors:
        print(f"     {neighbor.user_id}  (similaridade={neighbor.similarity:.4f}, "
              f"nota em Item5={neighbor.item_rating:g}, média={neighbor.user_mean:.2f})")

    print(f"\n3) Nota estimada — média de Alice + desvio ponderado dos vizinhos:")
    print(f"     Alice em Item5 ≈ {prediction.score:.4f}")


def step_train() -> UserBasedKNN:
    section("Passo 2 — Treino: carregar a base real (data/ratings.csv)")
    ratings = load_ratings(DATA_PATH)
    model = UserBasedKNN(ratings, k=2)
    n_ratings = sum(len(items) for items in ratings.values())
    print(f"\nUsuários carregados: {len(ratings)}")
    print(f"Itens distintos: {len(model.items)}")
    print(f"Avaliações carregadas: {n_ratings}")
    return model


def step_recommend(model: UserBasedKNN) -> None:
    section("Passo 3 — Recomendação (base real)")
    chosen = input("\nPara qual usuário recomendar? (Enter = primeiro da base): ").strip()
    user_id = chosen or sorted(model.ratings)[0]
    if user_id not in model.ratings:
        print(f"Usuário {user_id!r} não encontrado na base.")
        return

    print(f"\n1) Similaridade — usuários mais parecidos com {user_id} (Pearson, "
          f"top 5):")
    similarities = sorted(
        ((other, model.similarity(user_id, other))
         for other in model.ratings if other != user_id),
        key=lambda pair: -pair[1],
    )
    for other_id, sim in similarities[:5]:
        print(f"     {user_id} x {other_id}: {sim:+.4f}")

    print(f"\n2) Quantos vizinhos — k={model.k}, só entram os de similaridade "
          f"positiva (com pelo menos {model.min_common} itens em comum)")

    recommendations = model.recommend(user_id, n=5)
    print(f"\n3) Nota estimada — Top-{len(recommendations)} recomendações para {user_id}:")
    if not recommendations:
        print("     Nenhuma recomendação disponível (sem vizinhos positivos para nenhum item).")
        return
    for prediction in recommendations:
        print(f"\n     {prediction.item_id} — nota estimada {prediction.score:.2f}")
        print("       Similaridade dos vizinhos usados nessa previsão:")
        for neighbor in prediction.neighbors:
            print(f"         {neighbor.user_id}: similaridade={neighbor.similarity:.4f}, "
                  f"nota={neighbor.item_rating:g}, média={neighbor.user_mean:.2f}")


def main() -> None:
    step_test()
    pause("Pressione Enter para treinar com a base real...")
    model = step_train()
    pause("Pressione Enter para gerar recomendações...")
    step_recommend(model)


if __name__ == "__main__":
    main()
