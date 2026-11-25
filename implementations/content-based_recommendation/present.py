"""Apresentação guiada: teste (Slide 3) -> treino (base real) -> recomendação.

Pensado para rodar ao vivo na aula: primeiro reproduz o exemplo do slide
(livros + perfil do usuário, TF-IDF e cosseno), depois monta os vetores dos
itens da base real em data/ e gera recomendações de verdade, pelos dois
métodos do slide: perfil de Rocchio e vizinhos mais próximos.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import example_slide
from recommender import (ContentBasedKNN, Item, Ratings, RocchioProfile, TfIdf,
                         load_items, load_ratings, top_terms)

DATA_DIR = Path(__file__).parent / "data"
K = 3


@dataclass
class Trained:
    ratings: Ratings
    items: dict[str, Item]
    tfidf: TfIdf
    knn: ContentBasedKNN
    rocchio: RocchioProfile


def pause(msg: str = "Pressione Enter para continuar...") -> None:
    input(f"\n{msg}")


def section(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def step_test() -> None:
    section("Passo 1 — Teste: exemplo do Slide 3 (livros e perfil do usuário)")
    model, profile, ranking = example_slide.build()

    print("\n1) Representação — cada livro vira um vetor TF-IDF:")
    print("   TF = contagem do termo; IDF = log(N / n(i)); TF-IDF = TF * IDF")
    print(f"   Coleção com N={model.n_docs} livros; termos do perfil e seus IDF:")
    for term in sorted(profile):
        print(f"     {term:<12} n(i)={model.df[term]}  IDF={model.idf[term]:.3f}")

    print("\n2) Similaridade — cosseno entre o perfil do usuário e cada livro:")
    for title, score in ranking:
        print(f"     perfil x {title:<22} {score:.4f}")

    print(f"\n3) Recomendação — o livro mais parecido com o perfil é:")
    print(f"     {ranking[0][0]}  (cosseno {ranking[0][1]:.4f})")


def step_train() -> Trained:
    section("Passo 2 — Treino: carregar a base real (data/) e montar o TF-IDF")
    ratings = load_ratings(DATA_DIR / "ratings.csv")
    items = load_items(DATA_DIR / "items.csv")
    tfidf = TfIdf({item_id: item.tokens for item_id, item in items.items()})
    knn = ContentBasedKNN(ratings, tfidf.vectors, k=K)
    rocchio = RocchioProfile(ratings, tfidf.vectors)
    n_ratings = sum(len(rated) for rated in ratings.values())
    print(f"\nUsuários carregados: {len(ratings)}")
    print(f"Itens com conteúdo: {len(items)}  (vocabulário: {len(tfidf.idf)} termos)")
    print(f"Avaliações carregadas: {n_ratings}")
    return Trained(ratings, items, tfidf, knn, rocchio)


def _label(model: Trained, item_id: str) -> str:
    item = model.items[item_id]
    return f"{item_id} {item.title} [{item.genre}]"


def step_recommend(model: Trained) -> None:
    section("Passo 3 — Recomendação (base real)")
    chosen = input("\nPara qual usuário recomendar? (Enter = primeiro da base): ").strip()
    user_id = chosen or sorted(model.ratings)[0]
    if user_id not in model.ratings:
        print(f"Usuário {user_id!r} não encontrado na base.")
        return

    liked = model.rocchio.liked(user_id)
    print(f"\n1) Perfil de {user_id} (Rocchio, só feedback positivo) — "
          f"{len(liked)} itens curtidos (nota >= {model.rocchio.like_threshold:g}):")
    for item_id in liked:
        print(f"     {_label(model, item_id)} (nota {model.ratings[user_id][item_id]:g})")
    profile = model.rocchio.profile(user_id)
    if not profile:
        print("     Sem itens curtidos: não dá pra montar um perfil.")
    else:
        terms = ", ".join(f"{term} ({weight:.2f})" for term, weight in top_terms(profile, 6))
        print(f"     Termos de maior peso no perfil: {terms}")

    rocchio_recs = model.rocchio.recommend(user_id, n=5)
    print(f"\n2) Similaridade — Top-{len(rocchio_recs)} itens novos mais próximos do perfil "
          f"(cosseno):")
    if not rocchio_recs:
        print("     Nenhum item novo com similaridade positiva com o perfil.")
    for rec in rocchio_recs:
        print(f"     {_label(model, rec.item_id)}  cosseno={rec.score:.4f}")

    knn_recs = model.knn.recommend(user_id, n=5)
    print(f"\n3) Nota estimada — Top-{len(knn_recs)} pelos k={model.knn.k} itens mais "
          f"parecidos que {user_id} já avaliou:")
    if not knn_recs:
        print("     Nenhuma recomendação disponível (sem itens parecidos já avaliados).")
    for prediction in knn_recs:
        print(f"\n     {_label(model, prediction.item_id)} — nota estimada "
              f"{prediction.score:.2f}")
        for neighbor in prediction.neighbors:
            print(f"       parecido com {_label(model, neighbor.item_id)}: "
                  f"similaridade={neighbor.similarity:.4f}, nota de {user_id}="
                  f"{neighbor.user_rating:g}")


def main() -> None:
    step_test()
    pause("Pressione Enter para treinar com a base real...")
    model = step_train()
    pause("Pressione Enter para gerar recomendações...")
    step_recommend(model)


if __name__ == "__main__":
    main()
