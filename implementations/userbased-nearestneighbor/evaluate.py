"""Avalia o UserBasedKNN com uma divisão treino/teste reproduzível (fase 2)."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from recommender import Ratings, UserBasedKNN, load_ratings

DEFAULT_PATH = Path(__file__).parent / "data" / "ratings.csv"


def split_ratings(ratings: Ratings, test_ratio: float = 0.2, seed: int = 42,
                   min_train: int = 3) -> tuple[Ratings, list[tuple[str, str, float]]]:
    """Move uma amostra reproduzível de notas para o teste.

    Cada usuário mantém pelo menos `min_train` notas no treino, para que a
    similaridade continue calculável. As notas de teste somem do treino, logo
    não participam da similaridade nem das médias usadas na previsão.
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


def evaluate(train: Ratings, test: list[tuple[str, str, float]], k: int = 2,
             min_common: int = 3) -> dict[str, float]:
    """Treina sobre `train` e mede erro apenas sobre as notas escondidas em `test`."""
    model = UserBasedKNN(train, k=k, min_common=min_common)
    errors: list[float] = []
    for user_id, item_id, true_rating in test:
        prediction = model.predict(user_id, item_id)
        if prediction is not None:
            errors.append(prediction.score - true_rating)
    mae = sum(abs(e) for e in errors) / len(errors) if errors else math.nan
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors)) if errors else math.nan
    coverage = len(errors) / len(test) if test else math.nan
    return {
        "mae": mae,
        "rmse": rmse,
        "coverage": coverage,
        "evaluated": len(errors),
        "total_test": len(test),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-train", type=int, default=3)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--min-common", type=int, default=3)
    args = parser.parse_args()

    ratings = load_ratings(args.data)
    train, test = split_ratings(ratings, args.test_ratio, args.seed, args.min_train)
    metrics = evaluate(train, test, args.k, args.min_common)

    print(f"Usuários: {len(ratings)}  Notas de teste: {metrics['total_test']}")
    print(f"Cobertura: {metrics['coverage']:.1%} ({metrics['evaluated']} previstas)")
    print(f"MAE:  {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")


if __name__ == "__main__":
    main()
