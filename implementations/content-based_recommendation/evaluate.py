"""Avalia a filtragem baseada em conteúdo com divisão treino/teste reproduzível.

Esconde uma parte das notas de cada usuário (como no user-based) e mede:

- ContentBasedKNN (vizinhos mais próximos): MAE, RMSE e cobertura sobre as
  notas escondidas — o quanto a nota estimada se aproxima da real.
- RocchioProfile (perfil): Precision@N e Recall@N — dos N itens recomendados
  a cada usuário, quantos eram itens que ele de fato curtiu (nota ≥ limiar)
  e estavam escondidos no teste.

Os vetores TF-IDF dependem só do conteúdo dos itens (não das notas), então
usar a coleção inteira de itens não vaza notas de teste.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from recommender import (ContentBasedKNN, Ratings, RocchioProfile, TfIdf,
                         load_items, load_ratings)

DATA_DIR = Path(__file__).parent / "data"


def split_ratings(ratings: Ratings, test_ratio: float = 0.2, seed: int = 42,
                  min_train: int = 3) -> tuple[Ratings, list[tuple[str, str, float]]]:
    """Move uma amostra reproduzível de notas para o teste.

    Cada usuário mantém pelo menos `min_train` notas no treino. As notas de
    teste somem do treino, logo não participam de nenhum cálculo do modelo.
    (Mesma lógica do evaluate.py do user-based.)
    """
    if not 0 <= test_ratio < 1:
        raise ValueError("test_ratio deve estar entre 0 (inclusive) e 1 (exclusive).")
    rng = random.Random(seed)
    train: Ratings = {user: dict(items) for user, items in ratings.items()}
    test: list[tuple[str, str, float]] = []
    for user in sorted(train):
        items = train[user]
        item_ids = sorted(items)
        rng.shuffle(item_ids)
        max_removable = max(0, len(item_ids) - min_train)
        n_test = min(max_removable, round(len(item_ids) * test_ratio))
        for item_id in item_ids[:n_test]:
            test.append((user, item_id, items.pop(item_id)))
    return train, test


def evaluate_knn(train: Ratings, test: list[tuple[str, str, float]],
                 vectors: dict, k: int = 3) -> dict[str, float]:
    """MAE, RMSE e cobertura das notas estimadas pelos vizinhos mais próximos."""
    model = ContentBasedKNN(train, vectors, k=k)
    errors: list[float] = []
    for user_id, item_id, true_rating in test:
        prediction = model.predict(user_id, item_id)
        if prediction is not None:
            errors.append(prediction.score - true_rating)
    mae = sum(abs(e) for e in errors) / len(errors) if errors else math.nan
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors)) if errors else math.nan
    coverage = len(errors) / len(test) if test else math.nan
    return {"mae": mae, "rmse": rmse, "coverage": coverage,
            "evaluated": len(errors), "total_test": len(test)}


def evaluate_rocchio(train: Ratings, test: list[tuple[str, str, float]],
                     vectors: dict, n: int = 5,
                     like_threshold: float = 4.0) -> dict[str, float]:
    """Precision@N e Recall@N sobre os itens curtidos que ficaram escondidos."""
    model = RocchioProfile(train, vectors, like_threshold=like_threshold)
    hidden_liked: dict[str, set[str]] = {}
    for user_id, item_id, rating in test:
        if rating >= like_threshold:
            hidden_liked.setdefault(user_id, set()).add(item_id)
    precisions: list[float] = []
    recalls: list[float] = []
    for user_id, liked in hidden_liked.items():
        recommended = {rec.item_id for rec in model.recommend(user_id, n)}
        hits = len(recommended & liked)
        precisions.append(hits / n)
        recalls.append(hits / len(liked))
    count = len(precisions)
    return {
        "precision": sum(precisions) / count if count else math.nan,
        "recall": sum(recalls) / count if count else math.nan,
        "users": count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-train", type=int, default=3)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    ratings = load_ratings(args.data / "ratings.csv")
    items = load_items(args.data / "items.csv")
    vectors = TfIdf({item_id: item.tokens for item_id, item in items.items()}).vectors
    train, test = split_ratings(ratings, args.test_ratio, args.seed, args.min_train)

    knn = evaluate_knn(train, test, vectors, args.k)
    print(f"Usuários: {len(ratings)}  Itens: {len(items)}  Notas de teste: {knn['total_test']}")
    print(f"\nVizinhos mais próximos (k={args.k}) — nota estimada vs. nota real")
    print(f"  Cobertura: {knn['coverage']:.1%} ({knn['evaluated']} previstas)")
    print(f"  MAE:  {knn['mae']:.4f}")
    print(f"  RMSE: {knn['rmse']:.4f}")

    rocchio = evaluate_rocchio(train, test, vectors, args.top_n)
    print(f"\nPerfil de Rocchio — Top-{args.top_n} contra os itens curtidos escondidos "
          f"({rocchio['users']} usuários)")
    print(f"  Precision@{args.top_n}: {rocchio['precision']:.4f}")
    print(f"  Recall@{args.top_n}:    {rocchio['recall']:.4f}")


if __name__ == "__main__":
    main()
