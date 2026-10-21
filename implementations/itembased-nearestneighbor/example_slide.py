"""Reaproveita a matriz da Alice (mesma do Slide 2) para o método baseado em item."""

from recommender import ItemBasedKNN


RATINGS = {
    "Alice": {"Item1": 5, "Item2": 3, "Item3": 4, "Item4": 4},
    "User1": {"Item1": 3, "Item2": 1, "Item3": 2, "Item4": 3, "Item5": 3},
    "User2": {"Item1": 4, "Item2": 3, "Item3": 4, "Item4": 3, "Item5": 5},
    "User3": {"Item1": 3, "Item2": 3, "Item3": 1, "Item4": 5, "Item5": 4},
    "User4": {"Item1": 1, "Item2": 5, "Item3": 5, "Item4": 2, "Item5": 1},
}
ITEMS = ["Item1", "Item2", "Item3", "Item4", "Item5"]


def print_matrix(ratings: dict, estimate: tuple[str, str, float] | None = None) -> None:
    """Imprime a matriz usuário x item x nota, igual à do slide.

    `estimate` é (usuário, item, nota) e preenche a célula que está em
    branco na matriz original com o valor que o algoritmo previu.
    """
    est_user, est_item, est_score = estimate if estimate else (None, None, None)
    print("          " + "".join(f"{item:>8}" for item in ITEMS))
    for user, user_ratings in ratings.items():
        cells = []
        for item in ITEMS:
            if user == est_user and item == est_item:
                cells.append(f"{est_score:>7.2f}*")
            elif item in user_ratings:
                cells.append(f"{user_ratings[item]:>8g}")
            else:
                cells.append(f"{'?':>8}")
        print(f"{user:<10}" + "".join(cells))
    if estimate:
        print("(* nota estimada pelo algoritmo — em branco na matriz original)")


if __name__ == "__main__":
    model = ItemBasedKNN(RATINGS, k=2, min_common=2)

    print("Matriz original (? = ainda não avaliado):")
    print_matrix(RATINGS)

    print("\nSimilaridade de cada item com Item5 (cosseno ajustado):")
    for item_id in ("Item1", "Item2", "Item3", "Item4"):
        print(f"  {item_id}: {model.similarity(item_id, 'Item5'):.4f}")

    prediction = model.predict("Alice", "Item5")
    if prediction is not None:
        print(f"\nPrevisão de Alice para Item5: {prediction.score:.4f}")
        for neighbor in prediction.neighbors:
            print(f"  {neighbor.item_id}: nota de Alice={neighbor.user_rating:g}, "
                  f"similaridade={neighbor.similarity:.4f}")

        print("\nMatriz com a nota estimada:")
        print_matrix(RATINGS, estimate=("Alice", "Item5", prediction.score))
