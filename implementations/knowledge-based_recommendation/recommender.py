"""Recomendação baseada em conhecimento (Slide 4), sobre um catálogo de câmeras.

O usuário não tem histórico de notas: ele diz o que precisa (requisitos) e o
sistema usa conhecimento sobre os itens para achar o que serve. Duas famílias:

**Baseada em restrições (constraint-based)**
  - `select` — consulta conjuntiva σ[REQ](P): itens que cumprem *todos* os
    requisitos.
  - `quickxplain` — Algoritmo 4.1: um conjunto de conflito mínimo, quando
    nenhum item cumpre todos os requisitos.
  - `all_conflict_sets` / `diagnoses` — todos os conjuntos de conflito e os
    diagnósticos (conjuntos mínimos de requisitos a abrir mão).
  - `repairs` — reparos π[atributos(d)] σ[REQ − d](P): que valores os
    requisitos de um diagnóstico teriam de assumir para existir solução.
  - `utility_ranking` — teoria da utilidade multiatributo: cada item recebe
    notas de "qualidade" e "economia" e o cliente pondera as duas.

**Baseada em casos (case-based)**
  - `CaseBasedRetrieval.rank` — similaridade
    `Σ w_r * sim(p, r) / Σ w_r`, com "mais é melhor" (MIB), "menos é
    melhor" (LIB) ou valor-alvo.
  - `unit_critique` / `compound_critiques` / `apply_critique` — navegação por
    críticas ("mais barata", "mais barata e com mais mpix"), com críticas
    compostas descobertas por Apriori (Algoritmo 4.4).
"""

from __future__ import annotations

import csv
import operator
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable, Mapping, Sequence

Product = dict[str, object]

NUMERIC = ("price", "mpix", "opt_zoom", "lcd_size")
BOOLEAN = ("movies", "sound", "waterproof")
ATTRIBUTES = NUMERIC + BOOLEAN
LABELS = {
    "price": "preço (€)", "mpix": "mpix", "opt_zoom": "zoom óptico",
    "lcd_size": "tela LCD", "movies": "filmes", "sound": "som",
    "waterproof": "à prova d'água",
}


def format_value(attribute: str, value: object) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if attribute == "opt_zoom":
        return f"{value:g}×"
    return f"{value:g}"


# ----------------------------------------------------------------- catálogo --

def load_catalog(path: str | Path) -> list[Product]:
    """Lê id,price,mpix,opt_zoom,lcd_size,movies,sound,waterproof."""
    catalog: list[Product] = []
    with Path(path).open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ["id", *ATTRIBUTES]:
            raise ValueError("O CSV deve ter as colunas id," + ",".join(ATTRIBUTES) + ".")
        for line_number, row in enumerate(reader, start=2):
            product: Product = {"id": row["id"].strip()}
            for attribute in NUMERIC:
                try:
                    product[attribute] = float(row[attribute])
                except ValueError as exc:
                    raise ValueError(f"Linha {line_number}: {attribute} inválido.") from exc
            for attribute in BOOLEAN:
                text = row[attribute].strip().lower()
                if text not in ("yes", "no"):
                    raise ValueError(f"Linha {line_number}: {attribute} deve ser yes ou no.")
                product[attribute] = text == "yes"
            catalog.append(product)
    if not catalog:
        raise ValueError("O catálogo está vazio.")
    return catalog


# ------------------------------------------------- restrições: σ[REQ](P) ----

_OPS: dict[str, Callable[[object, object], bool]] = {
    "<=": operator.le, ">=": operator.ge, "<": operator.lt, ">": operator.gt,
    "=": operator.eq,
}


@dataclass(frozen=True)
class Requirement:
    """Um requisito do usuário, ex.: r1: price <= 150."""
    rid: str
    attribute: str
    op: str
    value: float | bool

    def __post_init__(self) -> None:
        if self.attribute not in ATTRIBUTES:
            raise ValueError(f"Atributo desconhecido: {self.attribute!r}.")
        if self.op not in _OPS:
            raise ValueError(f"Operador desconhecido: {self.op!r}.")

    def holds(self, product: Mapping[str, object]) -> bool:
        return _OPS[self.op](product[self.attribute], self.value)

    def __str__(self) -> str:
        return f"{self.rid}: {self.attribute} {self.op} {format_value(self.attribute, self.value)}"


def select(catalog: Sequence[Product], requirements: Sequence[Requirement]) -> list[Product]:
    """Consulta conjuntiva σ[REQ](P): itens que satisfazem todos os requisitos."""
    return [p for p in catalog if all(r.holds(p) for r in requirements)]


def quickxplain(catalog: Sequence[Product],
                requirements: Sequence[Requirement]) -> list[Requirement]:
    """Algoritmo 4.1 — um conjunto de conflito mínimo de `requirements`.

    Devolve [] se os requisitos já são consistentes (existe item que cumpre
    todos) ou se não há requisitos. Divide os requisitos ao meio e usa o
    resultado de uma metade como conhecimento de fundo para a outra, então
    acha o conflito com poucas consultas ao catálogo.

    Observação: no slide a primeira linha está impressa como
    "σ[REQ](P) = ∅ or REQ = ∅ then return ∅"; o glifo "≠" se perdeu. A
    versão correta (Junker, 2004), usada aqui, retorna vazio quando há
    solução, ou seja, quando σ[REQ](P) ≠ ∅.
    """
    reqs = list(requirements)
    if not reqs or select(catalog, reqs):
        return []
    return _qx(catalog, [], [], reqs)


def _qx(catalog: Sequence[Product], background: list[Requirement],
        delta: list[Requirement], reqs: list[Requirement]) -> list[Requirement]:
    # Se Δ ≠ ∅ e o conhecimento de fundo B já é inconsistente, Δ não contribui.
    if delta and not select(catalog, background):
        return []
    if len(reqs) == 1:
        return list(reqs)
    k = len(reqs) // 2
    reqs1, reqs2 = reqs[:k], reqs[k:]
    delta2 = _qx(catalog, background + reqs1, reqs1, reqs2)
    delta1 = _qx(catalog, background + delta2, delta2, reqs1)
    return delta1 + delta2


def all_conflict_sets(catalog: Sequence[Product],
                      requirements: Sequence[Requirement]) -> list[tuple[Requirement, ...]]:
    """Todos os conjuntos de conflito mínimos (força bruta; poucos requisitos)."""
    reqs = list(requirements)
    conflicts: list[tuple[Requirement, ...]] = []
    for size in range(1, len(reqs) + 1):
        for subset in combinations(reqs, size):
            if any(set(found) <= set(subset) for found in conflicts):
                continue  # contém um conflito menor: não é mínimo
            if not select(catalog, subset):
                conflicts.append(subset)
    return conflicts


def diagnoses(conflicts: Sequence[Sequence[Requirement]],
              requirements: Sequence[Requirement]) -> list[tuple[Requirement, ...]]:
    """Diagnósticos mínimos: conjuntos mínimos que "acertam" todo conflito.

    Abrir mão de todos os requisitos de um diagnóstico remove todos os
    conflitos (é um hitting set mínimo dos conjuntos de conflito).
    """
    reqs = list(requirements)
    found: list[tuple[Requirement, ...]] = []
    for size in range(1, len(reqs) + 1):
        for subset in combinations(reqs, size):
            if any(set(d) <= set(subset) for d in found):
                continue
            if all(set(subset) & set(conflict) for conflict in conflicts):
                found.append(subset)
    return found


def repairs(catalog: Sequence[Product], requirements: Sequence[Requirement],
            diagnosis: Sequence[Requirement]) -> list[dict[str, object]]:
    """π[atributos(d)] σ[REQ − d](P): valores que o diagnóstico precisaria assumir."""
    rest = [r for r in requirements if r not in diagnosis]
    attributes = list(dict.fromkeys(r.attribute for r in diagnosis))
    projected: list[dict[str, object]] = []
    for product in select(catalog, rest):
        row = {attribute: product[attribute] for attribute in attributes}
        if row not in projected:
            projected.append(row)
    return projected


# ------------------------------------------- utilidade multiatributo (MAUT) --

DIMENSIONS = ("quality", "economy")

# Para cada atributo: lista de (condição sobre o valor, nota_qualidade,
# nota_economia) — a tabela do slide 17.
UTILITY_TABLE: dict[str, list[tuple[Callable[[object], bool], int, int]]] = {
    "price":      [(lambda v: v <= 250, 5, 10), (lambda v: v > 250, 10, 5)],
    "mpix":       [(lambda v: v <= 8, 4, 10), (lambda v: v > 8, 10, 6)],
    "opt_zoom":   [(lambda v: v <= 9, 6, 9), (lambda v: v > 9, 10, 6)],
    "lcd_size":   [(lambda v: v <= 2.7, 6, 10), (lambda v: v > 2.7, 9, 5)],
    "movies":     [(lambda v: v is True, 10, 7), (lambda v: v is False, 3, 10)],
    "sound":      [(lambda v: v is True, 10, 8), (lambda v: v is False, 7, 10)],
    "waterproof": [(lambda v: v is True, 10, 6), (lambda v: v is False, 8, 10)],
}


def dimension_scores(product: Mapping[str, object]) -> dict[str, list[int]]:
    """Notas de cada dimensão por atributo, na ordem de ATTRIBUTES."""
    scores: dict[str, list[int]] = {"quality": [], "economy": []}
    for attribute in ATTRIBUTES:
        for condition, quality, economy in UTILITY_TABLE[attribute]:
            if condition(product[attribute]):
                scores["quality"].append(quality)
                scores["economy"].append(economy)
                break
    return scores


@dataclass(frozen=True)
class RankedItem:
    product: Product
    quality: int
    economy: int
    utility: float


def utility_ranking(products: Sequence[Product],
                    weights: Mapping[str, float]) -> list[RankedItem]:
    """Ordena por utilidade = Σ interesse(dimensão) * nota(dimensão).

    `weights` é o interesse do cliente, ex.: {"quality": 0.8, "economy": 0.2}.
    Empates saem em ordem de id.
    """
    ranked = []
    for product in products:
        scores = dimension_scores(product)
        quality, economy = sum(scores["quality"]), sum(scores["economy"])
        utility = weights["quality"] * quality + weights["economy"] * economy
        ranked.append(RankedItem(product, quality, economy, utility))
    return sorted(ranked, key=lambda item: (-item.utility, str(item.product["id"])))


# ------------------------------------------------- baseada em casos (CBR) ----

@dataclass(frozen=True)
class CaseRequirement:
    """Preferência de um atributo: mode = mib | lib | target | equal."""
    attribute: str
    mode: str
    weight: float = 1.0
    value: float | bool | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("mib", "lib", "target", "equal"):
            raise ValueError(f"Modo desconhecido: {self.mode!r}.")
        if self.mode in ("target", "equal") and self.value is None:
            raise ValueError("Os modos target e equal precisam de um valor.")
        if self.weight <= 0:
            raise ValueError("O peso deve ser positivo.")


class CaseBasedRetrieval:
    """Recupera os itens mais parecidos com os requisitos do cliente.

        similarity(p, REQ) = Σ w_r * sim(p, r) / Σ w_r

    Similaridade local de um atributo numérico, com min/max do catálogo:
      MIB (mais é melhor):  (φ − min) / (max − min)
      LIB (menos é melhor): (max − φ) / (max − min)
      target (valor-alvo):  1 − |φ − alvo| / (max − min)
      equal (atributo yes/no): 1 se igual ao desejado, senão 0
    """

    def __init__(self, catalog: Sequence[Product]) -> None:
        self.catalog = list(catalog)
        self.bounds = {
            attribute: (min(p[attribute] for p in self.catalog),
                        max(p[attribute] for p in self.catalog))
            for attribute in NUMERIC
        }

    def local_similarity(self, product: Mapping[str, object], req: CaseRequirement) -> float:
        value = product[req.attribute]
        if req.mode == "equal":
            return 1.0 if value == req.value else 0.0
        low, high = self.bounds[req.attribute]
        span = high - low
        if span == 0:
            return 1.0
        if req.mode == "mib":
            return (value - low) / span
        if req.mode == "lib":
            return (high - value) / span
        return max(0.0, 1.0 - abs(value - req.value) / span)

    def similarity(self, product: Mapping[str, object],
                   requirements: Sequence[CaseRequirement]) -> float:
        total = sum(r.weight for r in requirements)
        return sum(r.weight * self.local_similarity(product, r) for r in requirements) / total

    def rank(self, requirements: Sequence[CaseRequirement],
             candidates: Sequence[Product] | None = None) -> list[tuple[Product, float]]:
        pool = self.catalog if candidates is None else candidates
        scored = [(p, self.similarity(p, requirements)) for p in pool]
        return sorted(scored, key=lambda pair: (-pair[1], str(pair[0]["id"])))

    def item_similarity(self, a: Mapping[str, object], b: Mapping[str, object]) -> float:
        """Similaridade entre dois itens (todos os atributos, mesmo peso)."""
        reqs = [CaseRequirement(attr, "target", 1.0, b[attr]) for attr in NUMERIC]
        reqs += [CaseRequirement(attr, "equal", 1.0, b[attr]) for attr in BOOLEAN]
        return self.similarity(a, reqs)


# ---------------------------------------------------------------- críticas --

Critique = tuple[tuple[str, str], ...]  # ((atributo, "<" | ">"), ...)

CRITIQUE_WORDS = {"<": "menor", ">": "maior"}


def describe_critique(critique: Critique) -> str:
    return " e ".join(f"{LABELS[attr]} {CRITIQUE_WORDS[sign]}" for attr, sign in critique)


def critique_pattern(entry: Mapping[str, object], candidate: Mapping[str, object]) -> frozenset:
    """Como o candidato difere do item de entrada, nos atributos numéricos."""
    pattern = set()
    for attribute in NUMERIC:
        if candidate[attribute] < entry[attribute]:
            pattern.add((attribute, "<"))
        elif candidate[attribute] > entry[attribute]:
            pattern.add((attribute, ">"))
    return frozenset(pattern)


def satisfies_critique(entry: Mapping[str, object], candidate: Mapping[str, object],
                       critique: Critique) -> bool:
    return critique_pattern(entry, candidate) >= set(critique)


def apply_critique(retrieval: CaseBasedRetrieval, entry: Product, candidates: Sequence[Product],
                   critique: Critique) -> Product | None:
    """O item mais parecido com `entry` entre os que atendem à crítica."""
    pool = [c for c in candidates
            if c["id"] != entry["id"] and satisfies_critique(entry, c, critique)]
    if not pool:
        return None
    return max(pool, key=lambda c: (retrieval.item_similarity(c, entry), -_id_number(c)))


def _id_number(product: Mapping[str, object]) -> int:
    digits = "".join(ch for ch in str(product["id"]) if ch.isdigit())
    return int(digits) if digits else 0


def unit_critique(retrieval: CaseBasedRetrieval, entry: Product, candidates: Sequence[Product],
                  attribute: str, sign: str) -> Product | None:
    """Crítica de um atributo só (ex.: "mais barata" = price, "<")."""
    return apply_critique(retrieval, entry, candidates, ((attribute, sign),))


def compound_critiques(entry: Product, candidates: Sequence[Product], k: int = 3,
                       min_support: float = 0.3) -> list[tuple[Critique, float]]:
    """CompoundCritiques(r, CI, k, σmin) do Algoritmo 4.4.

    CritiquePatterns: padrão de diferença de cada candidato em relação ao
    item atual. Apriori: padrões que ocorrem em pelo menos `min_support` dos
    candidatos, com 2 ou mais condições. SelectCritiques: as `k` de menor
    suporte (as mais discriminantes, como em Reilly et al.; o slide não fixa
    o critério).
    """
    pool = [c for c in candidates if c["id"] != entry["id"]]
    if not pool:
        return []
    patterns = [critique_pattern(entry, c) for c in pool]
    n = len(patterns)

    def support(itemset: frozenset) -> float:
        return sum(1 for pattern in patterns if itemset <= pattern) / n

    level = {frozenset([item]) for pattern in patterns for item in pattern}
    level = {itemset for itemset in level if support(itemset) >= min_support}
    frequent: dict[frozenset, float] = {}
    while level:
        for itemset in level:
            frequent[itemset] = support(itemset)
        size = len(next(iter(level))) + 1
        candidates_next = {a | b for a in level for b in level if len(a | b) == size}
        level = {c for c in candidates_next if support(c) >= min_support}

    # Só os conjuntos "fechados": descarta um conjunto quando existe um maior
    # com o mesmo suporte (ele diria a mesma coisa com menos condições).
    compound = [(itemset, sup) for itemset, sup in frequent.items()
                if len(itemset) >= 2
                and not any(itemset < other and sup == other_sup
                            for other, other_sup in frequent.items())]
    compound.sort(key=lambda pair: (pair[1], sorted(pair[0])))
    return [(tuple(sorted(itemset)), sup) for itemset, sup in compound[:k]]
