# Content-Based Recommendation

Filtragem baseada em **conteúdo** (Slide 3): em vez de comparar usuários ou
notas entre si, descreve-se cada item pelo seu texto (gênero + palavras-chave),
converte-se esse texto em vetor **TF-IDF** e recomenda-se o que for parecido,
pelo **cosseno**, com o que o usuário já curtiu — "mostre mais do mesmo que eu
gostei".

## Como o código funciona

### `recommender.py`

- **`load_ratings(path)`** — idêntica à dos outros métodos: lê
  `user_id,item_id,rating`, sem usar 0 para "não avaliado".

- **`load_items(path)`** — lê `item_id,title,genre,description`, o `items.csv`
  que o `generate_dataset.py` grava nesta pasta. Cada `Item` expõe `.tokens`
  (gênero + palavras-chave, sem stop words).

- **`TfIdf(documents)`** — as fórmulas do slide, sobre os `N` itens:
  ```
  TF(i, j)     = contagem do termo i no item j
  IDF(i)       = log(N / n(i))          (log base 10, como em tfidf_toy.py)
  TF-IDF(i, j) = TF(i, j) * IDF(i)
  ```
  Termos presentes em todos os itens ficam com IDF 0 (peso zero).
  `.vectorize(termos)` calcula o vetor de um texto novo (ex.: um perfil) com o
  IDF da coleção.

- **`cosine(a, b)`** — `a·b / (|a| * |b|)` sobre vetores esparsos; 0 se
  algum vetor for nulo.

- **`ContentBasedKNN(ratings, vectors, k)`** — "Simple method: nearest
  neighbors" do slide. Para um item novo, pega os `k` itens mais parecidos
  (cosseno) **que o usuário já avaliou** e estima
  `nota = Σ sim·nota / Σ sim`. Sem nenhum item parecido, devolve `None`.

- **`RocchioProfile(ratings, vectors, alpha, beta, gamma)`** — perfil do
  usuário pela fórmula de Rocchio do slide,
  `Q = α·Q₀ + β·média(D+) − γ·média(D−)`, onde D+ são os itens curtidos
  (nota ≥ 4) e D− os não curtidos (nota ≤ 2). Por padrão só feedback positivo
  (β = 1, γ = 0), como o slide sugere. `.recommend()` ordena os itens novos
  pelo cosseno com o perfil.

### `evaluate.py`

Esconde uma parte das notas de cada usuário (mesma divisão do user-based) e
mede, sobre o que foi escondido:

- **KNN**: MAE, RMSE e cobertura da nota estimada.
- **Rocchio**: Precision@N e Recall@N — dos N itens recomendados, quantos eram
  itens curtidos que estavam escondidos.

## Passo a passo da implementação

1. **Descrever os itens** como texto (gênero + palavras-chave) — `Item.tokens`.
2. **Pesar os termos** com TF-IDF — `TfIdf`.
3. **Comparar vetores** com o cosseno — `cosine()`.
4. **Estimar notas** pelos itens parecidos já avaliados — `ContentBasedKNN`.
5. **Montar o perfil** do usuário e ordenar os itens novos — `RocchioProfile`.
6. **Avaliar** escondendo notas reais — `evaluate.py`.

## Executar

Na pasta `implementations/content-based_recommendation` (ou use
`py ../menu.py` para o terminal navegável):

```powershell
py example_slide.py
py evaluate.py
py present.py
```

`evaluate.py` aceita `--test-ratio`, `--seed`, `--min-train`, `--k` e
`--top-n`. Antes, gere a base com `py ../generate_dataset.py` — o
`data/items.csv` desta pasta tem as colunas extras `genre` e `description`.

## Exemplo do Slide 3

`example_slide.py` reproduz o exemplo de itens e perfil da página 4 do slide
(três livros e um perfil de usuário), calcula o TF-IDF e ordena os livros pelo
cosseno com o perfil: *The Lace Reader* (0,4692), *The Night of the Gun*
(0,3242) e *Into the Fire* (0,1563).

## Limitações (também citadas no slide)

- Só vê o que está no texto: sem palavras-chave boas, a recomendação piora.
- **Superespecialização**: tende a sugerir "mais do mesmo" (a Precision@N baixa
  reflete em parte isso).
- **Ramp-up**: precisa de algumas notas do usuário para montar o perfil.
