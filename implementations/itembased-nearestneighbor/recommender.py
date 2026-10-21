"""Filtragem colaborativa baseada em itens, com notas explícitas de 1 a 5."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


Ratings = dict[str, dict[str, float]]


def load_ratings(path: str | Path) -> Ratings:
    """Lê user_id,item_id,rating; pares ausentes ficam ausentes da matriz."""
    ratings: Ratings = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ["user_id", "item_id", "rating"]:
            raise ValueError("O CSV deve ter as colunas user_id,item_id,rating.")
        for line_number, row in enumerate(reader, start=2):
            user_id = row["user_id"].strip()
            item_id = row["item_id"].strip()
            if not user_id or not item_id:
                raise ValueError(f"Linha {line_number}: IDs não podem estar vazios.")
            try:
                rating = float(row["rating"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Linha {line_number}: nota inválida.") from exc
            if not math.isfinite(rating) or not 1 <= rating <= 5:
                raise ValueError(f"Linha {line_number}: nota fora da escala de 1 a 5.")
            user_ratings = ratings.setdefault(user_id, {})
            if item_id in user_ratings:
                raise ValueError(f"Linha {line_number}: par usuário-item repetido.")
            user_ratings[item_id] = rating
    if not ratings:
        raise ValueError("O CSV não contém avaliações.")
    return ratings


def _items_to_users(ratings: Mapping[str, Mapping[str, float]]) -> dict[str, dict[str, float]]:
    """Transpõe user->item->nota para item->user->nota."""
    by_item: dict[str, dict[str, float]] = {}
    for user_id, items in ratings.items():
        for item_id, rating in items.items():
            by_item.setdefault(item_id, {})[user_id] = rating
    return by_item


def adjusted_cosine(item_a: Mapping[str, float], item_b: Mapping[str, float],
                     user_means: Mapping[str, float], min_common: int = 2) -> float:
    """Cosseno ajustado (Sarwar et al.) sobre usuários que avaliaram os dois itens."""
    if min_common < 2:
        raise ValueError("min_common deve ser pelo menos 2.")
    common = item_a.keys() & item_b.keys()
    if len(common) < min_common:
        return 0.0
    numerator = sum((item_a[u] - user_means[u]) * (item_b[u] - user_means[u]) for u in common)
    a_squared = sum((item_a[u] - user_means[u]) ** 2 for u in common)
    b_squared = sum((item_b[u] - user_means[u]) ** 2 for u in common)
    if a_squared == 0 or b_squared == 0:
        return 0.0
    return numerator / math.sqrt(a_squared * b_squared)


@dataclass(frozen=True)
class Neighbor:
    item_id: str
    similarity: float
    user_rating: float


@dataclass(frozen=True)
class Prediction:
    item_id: str
    score: float
    neighbors: tuple[Neighbor, ...]


class ItemBasedKNN:
    """Consulta a matriz de avaliações diretamente; não há ajuste de parâmetros."""

    def __init__(self, ratings: Mapping[str, Mapping[str, float]], k: int = 2,
                 min_common: int = 2) -> None:
        if k < 1:
            raise ValueError("k deve ser pelo menos 1.")
        if min_common < 2:
            raise ValueError("min_common deve ser pelo menos 2.")
        self.ratings = {user: dict(items) for user, items in ratings.items()}
        if not self.ratings or any(not items for items in self.ratings.values()):
            raise ValueError("Cada usuário precisa ter pelo menos uma avaliação.")
        self.k = k
        self.min_common = min_common
        self.user_means = {
            user: sum(items.values()) / len(items)
            for user, items in self.ratings.items()
        }
        self.by_item = _items_to_users(self.ratings)
        self.items = set(self.by_item.keys())
        self._similarity_cache: dict[tuple[str, str], float] = {}

    def similarity(self, item_a: str, item_b: str) -> float:
        key = (item_a, item_b) if item_a <= item_b else (item_b, item_a)
        if key not in self._similarity_cache:
            self._similarity_cache[key] = adjusted_cosine(
                self.by_item[item_a], self.by_item[item_b], self.user_means, self.min_common,
            )
        return self._similarity_cache[key]

    def predict(self, user_id: str, item_id: str) -> Prediction | None:
        """Prevê item não avaliado; None indica ausência de vizinhos válidos."""
        user_ratings = self.ratings[user_id]
        if item_id in user_ratings:
            raise ValueError("O usuário já avaliou este item.")
        if item_id not in self.items:
            raise ValueError("Item desconhecido.")
        neighbors = []
        for other_item, rating in user_ratings.items():
            similarity = self.similarity(item_id, other_item)
            if similarity > 0:
                neighbors.append(Neighbor(other_item, similarity, rating))
        neighbors.sort(key=lambda neighbor: (-neighbor.similarity, neighbor.item_id))
        selected = tuple(neighbors[:self.k])
        if not selected:
            return None
        weighted_sum = sum(neighbor.similarity * neighbor.user_rating for neighbor in selected)
        weight_total = sum(neighbor.similarity for neighbor in selected)
        score = max(1.0, min(5.0, weighted_sum / weight_total))
        return Prediction(item_id, score, selected)

    def recommend(self, user_id: str, n: int = 5) -> list[Prediction]:
        """Devolve os n itens não avaliados com maior nota prevista."""
        if n < 1:
            raise ValueError("n deve ser pelo menos 1.")
        seen = self.ratings[user_id]
        predictions = [self.predict(user_id, item) for item in self.items - seen.keys()]
        available = [prediction for prediction in predictions if prediction is not None]
        return sorted(available, key=lambda prediction: (-prediction.score, prediction.item_id))[:n]
