"""Reproduz o exemplo do capítulo (Slide 4): catálogo de câmeras P1 a P8.

Cada bloco confere o resultado calculado com o valor impresso no slide
(`verify`) — se algum número divergir, o script para com um erro.

Execute `python example_slide.py` para ver o passo a passo completo.
"""

from __future__ import annotations

from pathlib import Path

from recommender import (ATTRIBUTES, CaseBasedRetrieval, CaseRequirement, Requirement,
                         all_conflict_sets, apply_critique, compound_critiques,
                         describe_critique, diagnoses, dimension_scores, format_value,
                         load_catalog, quickxplain, repairs, select, unit_critique,
                         utility_ranking)

CATALOG = load_catalog(Path(__file__).parent / "data" / "catalog.csv")

# Slide 9: σ[mpix≥10, price<300](P) = {p4, p7}
QUERY = [Requirement("c1", "mpix", ">=", 10.0), Requirement("c2", "price", "<", 300.0)]

# Slide 15: REQ = {r1: price ≤ 150, r2: opt-zoom = 5x, r3: sound = yes, r4: waterproof = yes}
R1 = Requirement("r1", "price", "<=", 150.0)
R2 = Requirement("r2", "opt_zoom", "=", 5.0)
R3 = Requirement("r3", "sound", "=", True)
R4 = Requirement("r4", "waterproof", "=", True)
REQ = [R1, R2, R3, R4]

# Slide 18: interesse de cada cliente em qualidade x economia
CUSTOMERS = {"cu1": {"quality": 0.8, "economy": 0.2},
             "cu2": {"quality": 0.4, "economy": 0.6}}

# Valores impressos no slide 18: (soma qualidade, soma economia, utilidade cu1, cu2)
SLIDE_UTILITY = {
    "P1": (41, 65, 45.8, 55.4), "P2": (49, 64, 52.0, 58.0),
    "P3": (53, 61, 54.6, 57.8), "P4": (58, 55, 57.4, 56.2),
    "P5": (53, 60, 54.4, 57.2), "P6": (58, 55, 57.4, 56.2),
    "P7": (63, 50, 60.4, 55.2), "P8": (69, 43, 63.8, 53.4),
}

# O slide imprime para P5 as notas de LCD (3.0") como 10 (qualidade) e 6 (economia),
# mas a própria tabela do slide 17 (LCD > 2.7) dá 9 e 5 — e é o que o slide usa em
# P6, P7 e P8, que também têm LCD 3.0". Com a tabela, P5 fica (52, 59, 53.4, 56.2).
SLIDE_TYPOS = {"P5": (52, 59, 53.4, 56.2)}

# Slide 16: reparos de cada diagnóstico
SLIDE_REPAIRS = {
    ("r1", "r2"): [{"price": 278, "opt_zoom": 10}],
    ("r1", "r4"): [{"price": 182, "waterproof": False}],
    ("r2", "r3"): [{"opt_zoom": 4, "sound": False}],
}


def ids(products) -> list[str]:
    return [p["id"] for p in products]


def rids(requirements) -> tuple[str, ...]:
    return tuple(r.rid for r in requirements)


def verify() -> None:
    """Confere cada resultado com o slide; levanta AssertionError se divergir."""
    assert ids(select(CATALOG, QUERY)) == ["P4", "P7"]

    conflicts = all_conflict_sets(CATALOG, REQ)
    assert {rids(c) for c in conflicts} == {("r1", "r2"), ("r2", "r4"), ("r1", "r3")}

    diags = diagnoses(conflicts, REQ)
    assert {rids(d) for d in diags} == set(SLIDE_REPAIRS)

    for diagnosis in diags:
        assert repairs(CATALOG, REQ, diagnosis) == SLIDE_REPAIRS[rids(diagnosis)]

    for name, weights in CUSTOMERS.items():
        column = {"cu1": 2, "cu2": 3}[name]
        for item in utility_ranking(CATALOG, weights):
            expected = SLIDE_TYPOS.get(item.product["id"], SLIDE_UTILITY[item.product["id"]])
            assert (item.quality, item.economy) == expected[:2]
            assert abs(item.utility - expected[column]) < 0.05


def main() -> None:
    print("Catálogo (slide 7):")
    print("  id  " + "".join(f"{a:>12}" for a in ATTRIBUTES))
    for p in CATALOG:
        print(f"  {p['id']:<4}" + "".join(f"{format_value(a, p[a]):>12}" for a in ATTRIBUTES))

    print("\n1) Consulta conjuntiva (slide 9):  σ[mpix≥10, price<300](P)")
    print("   =", ids(select(CATALOG, QUERY)))

    print("\n2) QuickXPlain (slide 14-15) sobre REQ =")
    for r in REQ:
        print("     ", r)
    print("   σ[REQ](P) =", ids(select(CATALOG, REQ)) or "∅  (nenhum item serve)")
    print("   um conjunto de conflito mínimo:", [r.rid for r in quickxplain(CATALOG, REQ)])

    conflicts = all_conflict_sets(CATALOG, REQ)
    print("\n3) Todos os conflitos e diagnósticos (slide 13):")
    for i, conflict in enumerate(conflicts, start=1):
        print(f"     CS{i} = {{{', '.join(r.rid for r in conflict)}}}")
    diags = diagnoses(conflicts, REQ)
    for i, diagnosis in enumerate(diags, start=1):
        print(f"     d{i} = {{{', '.join(r.rid for r in diagnosis)}}}")

    print("\n4) Reparos π[atributos(d)] σ[REQ − d](P) (slide 16):")
    for i, diagnosis in enumerate(diags, start=1):
        for repair in repairs(CATALOG, REQ, diagnosis):
            text = ", ".join(f"{a}={format_value(a, v)}" for a, v in repair.items())
            print(f"     d{i}: {text}")

    print("\n5) Ranking por utilidade (slide 17-18):")
    print("   P   Σqualidade  Σeconomia   cu1 [pos]    cu2 [pos]")
    ranks = {name: {item.product['id']: (pos, item.utility)
                    for pos, item in enumerate(utility_ranking(CATALOG, w), start=1)}
             for name, w in CUSTOMERS.items()}
    for p in CATALOG:
        s = dimension_scores(p)
        pid = p["id"]
        print(f"   {pid}  {sum(s['quality']):>9}  {sum(s['economy']):>9}   "
              f"{ranks['cu1'][pid][1]:>5.1f} [{ranks['cu1'][pid][0]}]   "
              f"{ranks['cu2'][pid][1]:>5.1f} [{ranks['cu2'][pid][0]}]")
    print("   Divergências em relação ao slide 18:")
    print("   - P5: o slide imprime Σ(5,4,6,10,10,10,8)=53 e Σ(10,10,9,6,7,8,10)=60, mas pela")
    print("     tabela do slide 17 (LCD 3.0 > 2.7 -> 9 e 5, como em P6, P7 e P8) dá 52 e 59.")
    print("   - P4 e P6 empatam; o slide as ordena de forma diferente em cu1 e cu2,")
    print("     aqui o desempate é pelo id.")

    print("\n6) Baseada em casos (slide 19) — exemplo ilustrativo, não é do slide:")
    retrieval = CaseBasedRetrieval(CATALOG)
    wishes = [CaseRequirement("price", "lib", 0.5),
              CaseRequirement("mpix", "mib", 0.3),
              CaseRequirement("opt_zoom", "mib", 0.2)]
    print("   preço menor (peso 0,5), mpix maior (0,3), zoom maior (0,2):")
    ranking = retrieval.rank(wishes)
    for product, score in ranking[:4]:
        print(f"     {product['id']}  similaridade={score:.4f}")

    entry = ranking[0][0]
    print(f"\n7) Críticas (slide 20-22) a partir do item de entrada {entry['id']}:")
    cheaper = unit_critique(retrieval, entry, CATALOG, "price", "<")
    print("   crítica unitária 'mais barata':",
          cheaper["id"] if cheaper else "nenhum item atende")
    for critique, support in compound_critiques(entry, CATALOG, k=3, min_support=0.2):
        chosen = apply_critique(retrieval, entry, CATALOG, critique)
        print(f"   crítica composta [{describe_critique(critique)}] "
              f"(suporte {support:.0%}) -> {chosen['id'] if chosen else '-'}")

    verify()
    print("\nOK: tudo confere com o slide (exceto a divergência de P5 apontada no item 5).")


if __name__ == "__main__":
    main()
