"""Reproduz o exemplo de representação de itens e perfil do Slide 3 (página 4).

Três livros descritos por gênero, autor, tipo e palavras-chave, e um perfil de
usuário na mesma estrutura. Cada item e o perfil viram vetores TF-IDF (com o
IDF calculado sobre os 3 livros) e os livros são ordenados pelo cosseno com o
perfil. O preço (campo numérico) fica de fora — a similaridade aqui é só
sobre o texto.
"""

from __future__ import annotations

from recommender import TfIdf, cosine, tokenize

BOOKS = {
    "The Night of the Gun": {
        "genre": "Memoir", "author": "David Carr", "type": "Paperback",
        "keywords": "Press and journalism, drug addiction, personal memoirs, New York",
    },
    "The Lace Reader": {
        "genre": "Fiction, Mystery", "author": "Brunonia Barry", "type": "Hardcover",
        "keywords": "American contemporary fiction, detective, historical",
    },
    "Into the Fire": {
        "genre": "Romance, Suspense", "author": "Suzanne Brockmann", "type": "Hardcover",
        "keywords": "American fiction, murder, neo-Nazism",
    },
}

USER_PROFILE = {
    "genre": "Fiction", "author": "Brunonia, Barry, Ken Follett", "type": "Paperback",
    "keywords": "Detective, murder, New York",
}


def text_of(record: dict[str, str]) -> str:
    return " ".join(record.values())


def build() -> tuple[TfIdf, dict[str, float], list[tuple[str, float]]]:
    """Devolve o modelo TF-IDF, o vetor do perfil e o ranking (livro, cosseno)."""
    model = TfIdf({title: tokenize(text_of(book)) for title, book in BOOKS.items()})
    profile = model.vectorize(tokenize(text_of(USER_PROFILE)))
    ranking = sorted(
        ((title, cosine(profile, vector)) for title, vector in model.vectors.items()),
        key=lambda pair: -pair[1],
    )
    return model, profile, ranking


if __name__ == "__main__":
    model, profile, ranking = build()

    print(f"Coleção: {model.n_docs} livros, {len(model.idf)} termos distintos\n")
    print("IDF = log(N / n(i)) dos termos do perfil que existem nos livros:")
    for term in sorted(profile):
        print(f"  {term:<12} n(i)={model.df[term]}  IDF={model.idf[term]:.3f}")

    print("\nCosseno entre o perfil do usuário e cada livro:")
    for title, score in ranking:
        print(f"  {title:<22} {score:.4f}")
