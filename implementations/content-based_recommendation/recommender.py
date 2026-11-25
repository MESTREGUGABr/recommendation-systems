"""Filtragem baseada em conteúdo (Slide 3): TF-IDF + similaridade do cosseno.

Três peças, na ordem do slide:

1. **Representação do item** — cada item vira um vetor TF-IDF calculado a
   partir do seu texto (gênero + palavras-chave):

       TF(i, j)     = contagem do termo i no item j
       IDF(i)       = log(N / n(i))      (N itens; n(i) itens que contêm i)
       TF-IDF(i, j) = TF(i, j) * IDF(i)

2. **Similaridade** — cosseno do ângulo entre dois vetores:

       sim(a, b) = a·b / (|a| * |b|)

3. **Recomendação** — duas formas, ambas do slide:

   - *Vizinhos mais próximos* (`ContentBasedKNN`): a nota de um item novo é
     estimada a partir dos k itens mais parecidos que o usuário já avaliou.
   - *Perfil de Rocchio* (`RocchioProfile`): o perfil do usuário é um vetor
     (a média dos itens que ele curtiu, menos a dos que não curtiu) e os
     itens ainda não vistos são ordenados pelo cosseno com esse perfil.
"""

from __future__ import annotations

import csv
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


Ratings = dict[str, dict[str, float]]
Vector = dict[str, float]

LOG_BASE = 10  # o slide não fixa a base do log; a mesma do exemplo tfidf_toy.py

# Remover stop words é uma das melhorias do vetorial citadas no slide
# ("a", "the", "on", ...). Lista curta, só pra não deixar palavras vazias de
# sentido pesarem no vetor.
STOPWORDS = frozenset({
    "a", "o", "as", "os", "de", "do", "da", "dos", "das", "e", "em", "um", "uma",
    "para", "com", "no", "na", "por", "and", "the", "of", "in", "on", "into",
})


# ----------------------------------------------------------------- leitura --

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


def tokenize(text: str) -> list[str]:
    """Minúsculas, só letras (com acento), sem stop words nem termos de 1 letra."""
    words = re.findall(r"[^\W\d_]+", text.lower())
    return [word for word in words if len(word) > 1 and word not in STOPWORDS]


@dataclass(frozen=True)
class Item:
    item_id: str
    title: str
    genre: str
    description: str

    @property
    def tokens(self) -> list[str]:
        """O 'conteúdo' do item: gênero + palavras-chave."""
        return tokenize(f"{self.genre} {self.description}")


def load_items(path: str | Path) -> dict[str, Item]:
    """Lê item_id,title,genre,description (gerado por generate_dataset.py)."""
    items: dict[str, Item] = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ["item_id", "title", "genre", "description"]:
            raise ValueError(
                "O items.csv deve ter as colunas item_id,title,genre,description. "
                "Rode `py generate_dataset.py` para gerar a base com o conteúdo dos itens."
            )
        for line_number, row in enumerate(reader, start=2):
            item_id = row["item_id"].strip()
            if not item_id:
                raise ValueError(f"Linha {line_number}: item_id vazio.")
            if item_id in items:
                raise ValueError(f"Linha {line_number}: item repetido.")
            items[item_id] = Item(item_id, row["title"].strip(), row["genre"].strip(),
                                  row["description"].strip())
    if not items:
        raise ValueError("O CSV não contém itens.")
    return items


# ------------------------------------------------------------------ TF-IDF --

class TfIdf:
    """Calcula o vetor TF-IDF de cada documento de uma coleção.

    `documents` mapeia o id do documento à lista de termos dele. Termos que
    aparecem em todos os documentos têm IDF = 0 e, portanto, peso zero (é o
    efeito de "reduzir o peso de termos que aparecem em todos os documentos").
    """

    def __init__(self, documents: Mapping[str, Sequence[str]],
                 log_base: float = LOG_BASE) -> None:
        if not documents:
            raise ValueError("A coleção precisa ter pelo menos um documento.")
        self.n_docs = len(documents)
        self.log_base = log_base
        self.df: Counter[str] = Counter()          # n(i)
        for terms in documents.values():
            self.df.update(set(terms))
        self.idf: dict[str, float] = {
            term: math.log(self.n_docs / count, log_base)
            for term, count in self.df.items()
        }
        self.vectors: dict[str, Vector] = {
            doc_id: self.vectorize(terms) for doc_id, terms in documents.items()
        }

    def vectorize(self, terms: Sequence[str]) -> Vector:
        """TF-IDF de um texto qualquer (ex.: um perfil) com o IDF desta coleção.

        Termos que a coleção nunca viu (n(i) = 0) ficam de fora.
        """
        counts = Counter(terms)
        vector = {term: count * self.idf[term]
                  for term, count in counts.items() if term in self.idf}
        return {term: weight for term, weight in vector.items() if weight != 0}


def cosine(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    """sim(a, b) = a·b / (|a| * |b|); 0 se algum dos vetores for nulo."""
    if len(a) > len(b):
        a, b = b, a
    dot = sum(weight * b[term] for term, weight in a.items() if term in b)
    norm_a = math.sqrt(sum(w * w for w in a.values()))
    norm_b = math.sqrt(sum(w * w for w in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def top_terms(vector: Mapping[str, float], n: int = 5) -> list[tuple[str, float]]:
    """Os n termos de maior peso de um vetor (para explicar um perfil)."""
    return sorted(vector.items(), key=lambda pair: (-pair[1], pair[0]))[:n]


# ------------------------------------------------ vizinhos mais próximos ----

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


class ContentBasedKNN:
    """Estima a nota de um item novo pelos k itens mais parecidos já avaliados.

    "Simple method: nearest neighbors" do slide — dado o conjunto de itens que
    o usuário já avaliou, acha os k mais parecidos (cosseno) com o item novo e
    usa as notas que o usuário deu a eles, ponderadas pela similaridade:

        nota(u, i) = Σ sim(i, v) * r(u, v) / Σ sim(i, v)     (v nos k vizinhos)
    """

    def __init__(self, ratings: Mapping[str, Mapping[str, float]],
                 vectors: Mapping[str, Vector], k: int = 3) -> None:
        if k < 1:
            raise ValueError("k deve ser pelo menos 1.")
        self.ratings = {user: dict(items) for user, items in ratings.items()}
        self.vectors = dict(vectors)
        self.k = k
        self.items = set(self.vectors)

    def similarity(self, item_a: str, item_b: str) -> float:
        return cosine(self.vectors[item_a], self.vectors[item_b])

    def predict(self, user_id: str, item_id: str) -> Prediction | None:
        """Prevê item não avaliado; None indica ausência de vizinhos parecidos."""
        if item_id not in self.vectors:
            raise ValueError("Item sem conteúdo cadastrado.")
        user_ratings = self.ratings[user_id]
        if item_id in user_ratings:
            raise ValueError("O usuário já avaliou este item.")
        neighbors = []
        for rated_id, rating in user_ratings.items():
            if rated_id not in self.vectors:
                continue
            similarity = self.similarity(item_id, rated_id)
            if similarity > 0:
                neighbors.append(Neighbor(rated_id, similarity, rating))
        neighbors.sort(key=lambda n: (-n.similarity, n.item_id))
        selected = tuple(neighbors[:self.k])
        if not selected:
            return None
        score = sum(n.similarity * n.user_rating for n in selected) / sum(
            n.similarity for n in selected)
        return Prediction(item_id, max(1.0, min(5.0, score)), selected)

    def recommend(self, user_id: str, n: int = 5) -> list[Prediction]:
        """Devolve os n itens não avaliados com maior nota prevista."""
        if n < 1:
            raise ValueError("n deve ser pelo menos 1.")
        seen = self.ratings[user_id]
        predictions = [self.predict(user_id, item) for item in self.items - seen.keys()]
        available = [p for p in predictions if p is not None]
        return sorted(available, key=lambda p: (-p.score, p.item_id))[:n]


# ------------------------------------------------------------------ Rocchio --

@dataclass(frozen=True)
class Recommendation:
    item_id: str
    score: float  # cosseno entre o perfil do usuário e o item


class RocchioProfile:
    """Perfil do usuário pelo método de Rocchio (Slide 3).

        Q(i+1) = α * Q(i) + β * média(D+) − γ * média(D−)

    D+ são os itens curtidos (nota ≥ `like_threshold`) e D− os não curtidos
    (nota ≤ `dislike_threshold`). Sem consulta inicial (Q = 0), só sobram os
    dois centroides. Como diz o slide, "muitas vezes só o feedback positivo é
    usado": por padrão β = 1 e γ = 0.
    """

    def __init__(self, ratings: Mapping[str, Mapping[str, float]],
                 vectors: Mapping[str, Vector], alpha: float = 1.0, beta: float = 1.0,
                 gamma: float = 0.0, like_threshold: float = 4.0,
                 dislike_threshold: float = 2.0) -> None:
        if like_threshold <= dislike_threshold:
            raise ValueError("like_threshold deve ser maior que dislike_threshold.")
        self.ratings = {user: dict(items) for user, items in ratings.items()}
        self.vectors = dict(vectors)
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.like_threshold = like_threshold
        self.dislike_threshold = dislike_threshold
        self.items = set(self.vectors)

    def liked(self, user_id: str) -> list[str]:
        """D+ — itens que o usuário curtiu."""
        return sorted(item for item, rating in self.ratings[user_id].items()
                      if rating >= self.like_threshold and item in self.vectors)

    def disliked(self, user_id: str) -> list[str]:
        """D− — itens que o usuário não curtiu."""
        return sorted(item for item, rating in self.ratings[user_id].items()
                      if rating <= self.dislike_threshold and item in self.vectors)

    def _centroid(self, item_ids: Sequence[str]) -> Vector:
        total: Vector = {}
        for item_id in item_ids:
            for term, weight in self.vectors[item_id].items():
                total[term] = total.get(term, 0.0) + weight
        return {term: weight / len(item_ids) for term, weight in total.items()}

    def profile(self, user_id: str, query: Mapping[str, float] | None = None) -> Vector:
        """Vetor de perfil Q(i+1); `query` é a consulta inicial Q(i), se houver."""
        result: Vector = {term: self.alpha * w for term, w in (query or {}).items()}
        positive = self._centroid(self.liked(user_id)) if self.liked(user_id) else {}
        negative = self._centroid(self.disliked(user_id)) if self.disliked(user_id) else {}
        for term, weight in positive.items():
            result[term] = result.get(term, 0.0) + self.beta * weight
        for term, weight in negative.items():
            result[term] = result.get(term, 0.0) - self.gamma * weight
        return {term: weight for term, weight in result.items() if weight != 0}

    def recommend(self, user_id: str, n: int = 5) -> list[Recommendation]:
        """Itens não avaliados, do mais para o menos parecido com o perfil."""
        if n < 1:
            raise ValueError("n deve ser pelo menos 1.")
        profile = self.profile(user_id)
        seen = self.ratings[user_id]
        scored = [Recommendation(item, cosine(profile, self.vectors[item]))
                  for item in self.items - seen.keys()]
        scored = [rec for rec in scored if rec.score > 0]
        return sorted(scored, key=lambda rec: (-rec.score, rec.item_id))[:n]
