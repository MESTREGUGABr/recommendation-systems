"""Apresentação guiada: teste (Slide 4) -> treino (catálogo) -> recomendação.

Pensado para rodar ao vivo na aula: primeiro reproduz o exemplo do capítulo
(catálogo de câmeras P1 a P8) passo a passo, depois carrega o catálogo de
data/ e conduz um diálogo: o usuário informa requisitos e o sistema devolve
os itens que servem (ranqueados por utilidade) ou, se nenhum serve, explica o
conflito e propõe diagnósticos e reparos. No fim, é possível navegar por
críticas ("mais barata", "mais mpix"...).
"""

from __future__ import annotations

from pathlib import Path

import example_slide
from recommender import (ATTRIBUTES, CaseBasedRetrieval, Product, Requirement, all_conflict_sets,
                         apply_critique, compound_critiques, describe_critique, diagnoses,
                         format_value, load_catalog, quickxplain, repairs, select,
                         utility_ranking)

DATA_PATH = Path(__file__).parent / "data" / "catalog.csv"


def pause(msg: str = "Pressione Enter para continuar...") -> None:
    input(f"\n{msg}")


def section(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def step_test() -> None:
    section("Passo 1 — Teste: exemplo do capítulo (Slide 4, câmeras P1 a P8)")
    example_slide.main()


def step_train() -> list[Product]:
    section("Passo 2 — Treino: carregar o catálogo (data/catalog.csv)")
    catalog = load_catalog(DATA_PATH)
    print(f"\nItens no catálogo: {len(catalog)}")
    print("Base de conhecimento: o próprio catálogo + a tabela de utilidade do slide 17.")
    print("(Não há notas de usuários: o sistema usa o que o usuário pede, não o histórico.)")
    return catalog


# ------------------------------------------------------------ requisitos --

# Valores padrão do diálogo: um conjunto mínimo e razoável de recursos para uma
# câmera, usado quando o usuário só aperta Enter. Assim ninguém precisa conhecer
# o catálogo para conseguir uma sugestão. Digite "-" para ignorar um requisito.
DEFAULTS: dict[str, float | bool | None] = {
    "price": 250.0,       # preço máximo (€)
    "mpix": 8.0,          # megapixels mínimos
    "opt_zoom": 4.0,      # zoom óptico mínimo (×)
    "lcd_size": 2.5,      # tela LCD mínima (pol)
    "movies": True,       # grava filmes
    "sound": True,        # grava som
    "waterproof": None,   # tanto faz (None = sem requisito por padrão)
}


def _default_text(default: float | bool | None, yes_no: bool) -> str:
    if default is None:
        return "tanto faz"
    if yes_no:
        return "s" if default else "n"
    return f"{default:g}"


def _ask_number(label: str, default: float | None) -> float | None:
    hint = f"padrão {_default_text(default, False)}, - ignora"
    while True:
        text = input(f"  {label} [{hint}]: ").strip().replace(",", ".")
        if not text:
            return default
        if text == "-":
            return None
        try:
            return float(text)
        except ValueError:
            print("    Digite um número (ou - para ignorar).")


def _ask_yes_no(label: str, default: bool | None) -> bool | None:
    hint = f"padrão {_default_text(default, True)}, - ignora"
    while True:
        text = input(f"  {label} [s/n; {hint}]: ").strip().lower()
        if not text:
            return default
        if text == "-":
            return None
        if text in ("s", "sim", "y", "yes"):
            return True
        if text in ("n", "nao", "não", "no"):
            return False
        print("    Responda s, n ou - para ignorar.")


def ask_requirements() -> list[Requirement]:
    print("\nInforme os requisitos da câmera (Enter aceita o valor padrão):")
    wishes = [
        ("price", "<=", _ask_number("preço máximo (€)", DEFAULTS["price"])),
        ("mpix", ">=", _ask_number("megapixels mínimos", DEFAULTS["mpix"])),
        ("opt_zoom", ">=", _ask_number("zoom óptico mínimo (×)", DEFAULTS["opt_zoom"])),
        ("lcd_size", ">=", _ask_number("tela LCD mínima (pol)", DEFAULTS["lcd_size"])),
        ("movies", "=", _ask_yes_no("grava filmes", DEFAULTS["movies"])),
        ("sound", "=", _ask_yes_no("grava som", DEFAULTS["sound"])),
        ("waterproof", "=", _ask_yes_no("à prova d'água", DEFAULTS["waterproof"])),
    ]
    requirements = []
    for attribute, op, value in wishes:
        if value is not None:
            requirements.append(Requirement(f"r{len(requirements) + 1}", attribute, op, value))
    return requirements


def ask_quality_weight() -> float:
    text = input("\nQuanto você valoriza qualidade vs. economia? "
                 "Qualidade em % (Enter = 50): ").strip().replace(",", ".")
    try:
        return min(100.0, max(0.0, float(text))) / 100 if text else 0.5
    except ValueError:
        return 0.5


# -------------------------------------------------------------- saída ----

def show_product(product: Product) -> str:
    return "  ".join(f"{a}={format_value(a, product[a])}" for a in ATTRIBUTES)


def explain_failure(catalog: list[Product], requirements: list[Requirement]) -> None:
    print("\nNenhuma câmera cumpre todos os requisitos (σ[REQ](P) = ∅).")
    conflict = quickxplain(catalog, requirements)
    print("\nQuickXPlain — um conjunto de conflito mínimo:")
    print("   " + ", ".join(str(r) for r in conflict))

    conflicts = all_conflict_sets(catalog, requirements)
    print("\nTodos os conflitos mínimos:")
    for conflict in conflicts:
        print("   {" + ", ".join(r.rid for r in conflict) + "}")

    diags = diagnoses(conflicts, requirements)
    print("\nDiagnósticos (requisitos a relaxar para existir solução) e reparos:")
    for i, diagnosis in enumerate(diags, start=1):
        names = ", ".join(r.rid for r in diagnosis)
        options = repairs(catalog, requirements, diagnosis)
        text = " | ".join(
            ", ".join(f"{a}={format_value(a, v)}" for a, v in option.items())
            for option in options)
        print(f"   d{i} = {{{names}}}  ->  {text}")


def _build_options(retrieval: CaseBasedRetrieval, entry: Product,
                   pool: list[Product]) -> list[tuple[str, tuple, Product, float]]:
    """Críticas que levam a algum item: (rótulo, crítica, item resultante, similaridade).

    Só entram críticas que de fato têm um item restante para mostrar, então o
    usuário nunca escolhe uma opção que não leva a lugar nenhum.
    """
    candidates: list[tuple[str, tuple]] = [
        ("mais barata", (("price", "<"),)),
        ("mais megapixels", (("mpix", ">"),)),
        ("mais zoom óptico", (("opt_zoom", ">"),)),
    ]
    for critique, support in compound_critiques(entry, pool, k=3, min_support=0.2):
        candidates.append((f"{describe_critique(critique)} (suporte {support:.0%})", critique))
    options = []
    for label, critique in candidates:
        chosen = apply_critique(retrieval, entry, pool, critique)
        if chosen is not None:
            options.append((label, critique, chosen, retrieval.item_similarity(chosen, entry)))
    return options


def show_final(retrieval: CaseBasedRetrieval, matches: list[Product], path: list, weights: dict) -> None:
    """Fecha a navegação: mostra a recomendação final, o trajeto e as similaridades."""
    final = path[-1][0]
    section(f"Recomendação final: {final['id']}")
    item = utility_ranking([final], weights)[0]
    print(f"\n  {show_product(final)}")
    print(f"  utilidade={item.utility:.1f}  (Σqualidade={item.quality}, Σeconomia={item.economy})")

    if len(path) > 1:
        print("\nTrajeto da navegação (similaridade de cada passo com o item anterior):")
        print(f"  {path[0][0]['id']}  (item inicial, 1º do ranking por utilidade)")
        for product, label, similarity in path[1:]:
            print(f"  -> {product['id']}  pela crítica '{label}'  (similaridade {similarity:.4f})")
        print(f"\nSimilaridade da recomendação final com o item inicial: "
              f"{retrieval.item_similarity(final, path[0][0]):.4f}")

    others = sorted(((retrieval.item_similarity(c, final), c) for c in matches
                     if c["id"] != final["id"]), key=lambda pair: (-pair[0], str(pair[1]["id"])))
    if others:
        print(f"\nOutras câmeras que cumprem os requisitos, por similaridade com {final['id']}:")
        for similarity, product in others[:3]:
            print(f"  {product['id']}  similaridade={similarity:.4f}  {show_product(product)}")


def critique_loop(catalog: list[Product], candidates: list[Product], entry: Product,
                  weights: dict) -> None:
    """Navegação por críticas: a cada passo mostra para onde cada crítica leva."""
    retrieval = CaseBasedRetrieval(catalog)
    pool = [c for c in candidates if c["id"] != entry["id"]]  # CI <- CI - r
    path: list[tuple[Product, str | None, float | None]] = [(entry, None, None)]

    section("Navegação por críticas")
    print("A cada passo, escolha uma crítica ao item atual; o sistema mostra o item")
    print("mais parecido que atende à crítica. Digite 0 (ou Enter) para aceitar o item")
    print("atual e ver a recomendação final.")

    while True:
        options = _build_options(retrieval, entry, pool)
        print(f"\nItem atual: {entry['id']}  {show_product(entry)}")
        if not options:
            print("Não há mais câmeras diferentes que atendam a uma crítica: fim da navegação.")
            break
        print("Críticas possíveis (-> item que ela traz, com a similaridade ao item atual):")
        for number, (label, _critique, chosen, similarity) in enumerate(options, start=1):
            print(f"  {number}) {label}  ->  {chosen['id']}  (similaridade {similarity:.4f})")
        print(f"  0) Aceitar {entry['id']} e ver a recomendação final")
        choice = input("Escolha: ").strip() or "0"
        if choice == "0":
            break
        if not choice.isdigit() or not 1 <= int(choice) <= len(options):
            print("Opção inválida.")
            continue
        label, _critique, chosen, similarity = options[int(choice) - 1]
        path.append((chosen, label, similarity))
        pool = [c for c in pool if c["id"] != chosen["id"]]
        entry = chosen

    show_final(retrieval, candidates, path, weights)


def step_recommend(catalog: list[Product]) -> None:
    section("Passo 3 — Recomendação (diálogo com requisitos)")
    requirements = ask_requirements()
    if not requirements:
        print("\nTodos os requisitos foram ignorados: nada para filtrar.")
        return

    print("\nRequisitos:")
    for requirement in requirements:
        print("   ", requirement)

    matches = select(catalog, requirements)
    if not matches:
        explain_failure(catalog, requirements)
        return

    quality = ask_quality_weight()
    weights = {"quality": quality, "economy": 1 - quality}
    ranked = utility_ranking(matches, weights)
    print(f"\n{len(matches)} câmera(s) cumprem todos os requisitos, "
          f"ranqueadas por utilidade (qualidade {quality:.0%}, economia {1 - quality:.0%}):")
    for position, item in enumerate(ranked, start=1):
        print(f"  {position}. {item.product['id']}  utilidade={item.utility:.1f}  "
              f"(Σqualidade={item.quality}, Σeconomia={item.economy})")
        print(f"       {show_product(item.product)}")

    if len(ranked) > 1:
        print("\nNavegação por críticas: parte do 1º item e permite pedir, por exemplo, "
              "'mais barata'\nou 'mais megapixels', mostrando para qual câmera cada crítica leva.")
        answer = input("Quer navegar por críticas? [s/N]: ").strip().lower()
        if answer in ("s", "sim", "y", "yes"):
            critique_loop(catalog, matches, ranked[0].product, weights)


def main() -> None:
    step_test()
    pause("Pressione Enter para carregar o catálogo...")
    catalog = step_train()
    pause("Pressione Enter para informar requisitos...")
    step_recommend(catalog)


if __name__ == "__main__":
    main()
