"""Gera a base dummy (ratings.csv + items.csv) direto nas pastas de implementação.

Substitui a geração manual por LLM: a mesma lógica, com uma semente fixa,
pode ser reexecutada por qualquer pessoa do grupo. A quantidade de usuários,
de itens e de avaliações por usuário é sorteada a cada execução (dentro de
faixas razoáveis), então o tamanho da base muda de uma semente para outra —
só as notas em si continuam seguindo o padrão por grupo de preferência, para
os vizinhos mais próximos fazerem sentido.

Características buscadas (as mesmas pedidas originalmente ao DeepSeek, agora
codificadas aqui):

- Formato longo: user_id,item_id,rating (sem pares repetidos). Pares
  ausentes representam "ainda não avaliado" — nunca 0.
- Usuários, itens e total de avaliações sorteados por execução (ver
  `USERS_RANGE`, `ITEMS_RANGE`, `RATINGS_PER_USER_RANGE`); notas inteiras de
  1 a 5.
- Cada usuário com pelo menos `MIN_RATINGS_PER_USER` avaliações; cada item
  avaliado por vários usuários.
- 3 grupos de preferência (gêneros) reconhecíveis, com alguma mistura entre
  grupos e pequenas variações de nota — o suficiente para os vizinhos mais
  próximos fazerem sentido sem o resultado ser artificialmente perfeito.
- Casos didáticos propositais: um par de usuários muito similares (mesmo
  grupo, poucas notas destoantes), um par de gostos opostos (grupos opostos,
  notas quase invertidas) e usuários com poucas avaliações.

Uso (a partir de `implementations/`):
    python generate_dataset.py
    python generate_dataset.py --seed 7
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

# Faixas sorteadas a cada execução — isso é o que varia o "tamanho" da base.
USERS_RANGE = (22, 38)
ITEMS_RANGE = (14, 26)
RATINGS_PER_USER_RANGE = (8, 20)
MIN_RATINGS_PER_USER = 8
RATING_SCALE = (1, 5)
DEFAULT_SEED = 42

# Pastas que recebem uma cópia idêntica da base gerada, para os dois métodos
# serem comparáveis sobre os mesmos dados.
TARGET_DIRS = ("userbased-nearestneighbor", "itembased-nearestneighbor")

# Pool de itens com gênero predominante; o gerador sorteia quantos (e quais)
# entram na base de cada execução. Usuários de um grupo tendem a gostar mais
# dos itens do "seu" gênero.
ITEM_POOL: list[tuple[str, str]] = [
    ("Cidade em Chamas", "acao"),
    ("Rota de Fuga", "acao"),
    ("O Último Turno", "acao"),
    ("Código Vermelho", "acao"),
    ("Perseguição Noturna", "acao"),
    ("Linha de Frente", "acao"),
    ("Alvo em Movimento", "acao"),
    ("Zona de Impacto", "acao"),
    ("Contagem Regressiva", "acao"),
    ("Fronteira Hostil", "acao"),
    ("Cartas de Outono", "drama"),
    ("Café na Praça", "drama"),
    ("O Silêncio entre Nós", "drama"),
    ("Dias de Verão", "drama"),
    ("A Casa Amarela", "drama"),
    ("Reencontro em Lisboa", "drama"),
    ("Noite de Estreia", "drama"),
    ("Retrato de Família", "drama"),
    ("O Peso da Ausência", "drama"),
    ("Entre Duas Cidades", "drama"),
    ("Órbita Zero", "ficcao"),
    ("A Última Fronteira", "ficcao"),
    ("Máquinas de Vidro", "ficcao"),
    ("O Reino das Marés", "ficcao"),
    ("Estação Aurora", "ficcao"),
    ("A Biblioteca Infinita", "ficcao"),
    ("Viagem ao Desconhecido", "ficcao"),
    ("Sinais de Outro Mundo", "ficcao"),
    ("O Último Algoritmo", "ficcao"),
    ("Memórias Sintéticas", "ficcao"),
]
GENRES = ("acao", "drama", "ficcao")

# Afinidade média (1 a 5) de um usuário de cada grupo por item de cada gênero.
# A diagonal é alta (gosta do próprio gênero); fora da diagonal há alguma
# mistura, nunca rejeição total, para não deixar o resultado perfeito demais.
AFFINITY: dict[str, dict[str, float]] = {
    "acao":   {"acao": 4.4, "drama": 2.3, "ficcao": 3.0},
    "drama":  {"acao": 2.2, "drama": 4.5, "ficcao": 2.6},
    "ficcao": {"acao": 3.1, "drama": 2.4, "ficcao": 4.3},
}


def _clamp_rating(value: float) -> int:
    low, high = RATING_SCALE
    return max(low, min(high, round(value)))


def _pick_items(rng: random.Random) -> list[tuple[str, str, str]]:
    """Sorteia quantos e quais itens do pool entram nesta execução."""
    n_items = rng.randint(*ITEMS_RANGE)
    chosen = rng.sample(ITEM_POOL, n_items)
    return [(f"I{i:02d}", title, genre) for i, (title, genre) in enumerate(chosen, start=1)]


def _assign_user_groups(rng: random.Random, n_users: int) -> list[str]:
    """Distribui os usuários pelos 3 grupos, o mais equilibrado possível."""
    groups = [GENRES[i % len(GENRES)] for i in range(n_users)]
    rng.shuffle(groups)
    return groups


def generate_ratings(seed: int = DEFAULT_SEED) -> tuple[list[tuple[str, str, int]], list[tuple[str, str, str]]]:
    rng = random.Random(seed)
    items = _pick_items(rng)
    min_raters = max(3, min(len(items) // 3, rng.randint(4, 6)))

    n_users = rng.randint(*USERS_RANGE)
    user_ids = [f"U{i:02d}" for i in range(1, n_users + 1)]
    user_groups = dict(zip(user_ids, _assign_user_groups(rng, n_users)))

    rows: list[tuple[str, str, int]] = []
    seen: set[tuple[str, str]] = set()

    def add_rating(user_id: str, item_id: str, score: float) -> None:
        key = (user_id, item_id)
        if key in seen:
            return
        seen.add(key)
        rows.append((user_id, item_id, _clamp_rating(score)))

    # U01/U02 (par similar) e U03/U04 (par oposto) recebem um tratamento à
    # parte: para a correlação de Pearson ficar visível, os dois precisam
    # compartilhar o mesmo conjunto de itens, não só uma amostra em comum.
    scripted_users = set(user_ids[:4]) if n_users >= 4 else set()
    max_ratings = min(RATINGS_PER_USER_RANGE[1], len(items))
    min_ratings = min(MIN_RATINGS_PER_USER, max_ratings)

    for user_id in user_ids:
        if user_id in scripted_users:
            continue
        group = user_groups[user_id]
        n_ratings = rng.randint(min_ratings, max_ratings)
        rated_items = rng.sample(items, n_ratings)
        for item_id, _title, genre in rated_items:
            base = AFFINITY[group][genre]
            noise = rng.gauss(0, 0.6)
            add_rating(user_id, item_id, base + noise)

    if scripted_users:
        _make_similar_pair(rng, user_ids, items, user_groups, add_rating, min_ratings, max_ratings)
        _make_opposite_pair(rng, user_ids, items, user_groups, add_rating, min_ratings, max_ratings)
    _ensure_item_coverage(rng, user_ids, items, add_rating, min_raters)

    rng.shuffle(rows)
    return rows, items


def _make_similar_pair(rng, user_ids, items, user_groups, add_rating, min_ratings, max_ratings) -> None:
    """U01 e U02: mesmo grupo, mesmos itens, notas quase idênticas."""
    u1, u2 = user_ids[0], user_ids[1]
    group = user_groups[u1]
    user_groups[u2] = group
    n_ratings = rng.randint(min_ratings, max_ratings)
    shared_items = rng.sample(items, n_ratings)
    for item_id, _title, genre in shared_items:
        base = AFFINITY[group][genre] + rng.gauss(0, 0.4)
        add_rating(u1, item_id, base)
        add_rating(u2, item_id, base + rng.gauss(0, 0.3))


def _make_opposite_pair(rng, user_ids, items, user_groups, add_rating, min_ratings, max_ratings) -> None:
    """U03 e U04: mesmos itens, notas sistematicamente invertidas (6 - nota)."""
    u3, u4 = user_ids[2], user_ids[3]
    group = user_groups[u3]
    n_ratings = rng.randint(min_ratings, max_ratings)
    shared_items = rng.sample(items, n_ratings)
    for item_id, _title, genre in shared_items:
        base = AFFINITY[group][genre] + rng.gauss(0, 0.4)
        add_rating(u3, item_id, base)
        add_rating(u4, item_id, 6 - base + rng.gauss(0, 0.3))


def _ensure_item_coverage(rng, user_ids, items, add_rating, min_raters: int) -> None:
    """Garante que todo item tenha avaliações de pelo menos `min_raters` usuários."""
    n_raters = min(min_raters, len(user_ids))
    for item_id, _title, _genre in items:
        raters = rng.sample(user_ids, n_raters)
        for user_id in raters:
            base = 3.0 + rng.uniform(-1.0, 1.0)
            add_rating(user_id, item_id, base)


def write_dataset(rows: list[tuple[str, str, int]], items: list[tuple[str, str, str]],
                   out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ratings_path = out_dir / "ratings.csv"
    items_path = out_dir / "items.csv"

    with ratings_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["user_id", "item_id", "rating"])
        writer.writerows(rows)

    with items_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item_id", "title"])
        for item_id, title, _genre in items:
            writer.writerow([item_id, title])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    rows, items = generate_ratings(args.seed)

    base_dir = Path(__file__).parent
    for target in TARGET_DIRS:
        write_dataset(rows, items, base_dir / target / "data")

    users = {user for user, _item, _rating in rows}
    rated_items = {item for _user, item, _rating in rows}
    print(f"Gerado: {len(rows)} avaliações, {len(users)} usuários, {len(rated_items)} itens")
    print("Escrito em: " + ", ".join(f"{target}/data/" for target in TARGET_DIRS))


if __name__ == "__main__":
    main()
