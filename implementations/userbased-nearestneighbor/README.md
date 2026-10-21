# User-Based Nearest Neighbors

Filtragem colaborativa baseada em **usuário**: para prever a nota que um
usuário daria a um item que ainda não avaliou, procura-se outros usuários
com padrão de notas parecido (os "vizinhos") e usa-se as notas deles nesse
item, ponderadas pela semelhança.

## Como o código funciona

### `recommender.py`

- **`load_ratings(path)`** — lê o CSV `user_id,item_id,rating` e devolve
  `dict[user_id][item_id] = nota`. Valida as colunas, a faixa de 1 a 5 e que
  não haja par usuário-item repetido. Um par ausente do dicionário significa
  "ainda não avaliado" — nunca é representado como nota 0.

- **`pearson(a, b, min_common)`** — correlação de Pearson entre dois
  usuários, calculada **só sobre os itens que os dois avaliaram em comum**
  (as médias usadas na fórmula também são só desses itens comuns, não a
  média geral de cada usuário). Se tiverem menos de `min_common` itens em
  comum, ou se algum dos dois não tiver variação nas notas desses itens
  (denominador zero), devolve `0.0` — "não dá pra saber se são parecidos".

- **`Neighbor` / `Prediction`** — dataclasses só pra carregar o resultado de
  forma estruturada: quem contribuiu pra previsão, com que similaridade e
  qual nota deu, pra dar pra explicar o resultado (e testar) sem só devolver
  um número solto.

- **`UserBasedKNN.__init__(ratings, k, min_common)`** — guarda uma cópia das
  notas (não o dicionário original, pra não ser afetado se quem chamou
  mudar depois), calcula a **média de cada usuário sobre todas as notas que
  ele deu** (isso é diferente da média usada no Pearson, que é só dos itens
  em comum) e guarda o conjunto de todos os itens avaliados por alguém.

- **`.similarity(a, b)`** — chama `pearson` com o `min_common` do modelo.

- **`.predict(user, item)`** — o coração do algoritmo:
  1. erro se o usuário já avaliou esse item (não faz sentido prever o que
     já se sabe);
  2. para cada outro usuário que avaliou esse item, calcula a similaridade
     com o usuário alvo;
  3. **descarta similaridade ≤ 0** — vizinho de gosto neutro/oposto não
     ajuda a prever, só adiciona ruído;
  4. ordena os que sobraram pela similaridade e pega os `k` melhores;
  5. se não sobrou nenhum, devolve `None` (sem vizinhos válidos, sem
     previsão — isso é o que a cobertura mede em `evaluate.py`);
  6. senão, estima a nota como a **média do usuário + o desvio ponderado
     dos vizinhos em relação à própria média deles**:
     ```
     nota = média(user) + Σ sim(user,v)·(nota(v,item) − média(v)) / Σ sim(user,v)
     ```
     (usa o desvio, não a nota bruta do vizinho, porque usuários têm
     escalas diferentes — um pode dar notas em geral mais altas que outro);
  7. limita o resultado entre 1 e 5.

- **`.recommend(user, n)`** — chama `.predict` para todo item que o usuário
  ainda não avaliou, descarta os `None`, ordena pela nota prevista
  (decrescente) e devolve os `n` melhores.

### `evaluate.py` (fase 2 — avaliação)

- **`split_ratings(ratings, test_ratio, seed, min_train)`** — separa uma
  fração das notas de **cada usuário** para teste, sorteando com uma
  semente fixa (reprodutível) e garantindo que sobrem pelo menos
  `min_train` notas no treino daquele usuário (senão não dá pra calcular
  similaridade). As notas de teste somem do dicionário de treino — não
  podem vazar pro cálculo de similaridade nem de médias.

- **`evaluate(train, test, k, min_common)`** — treina um `UserBasedKNN` só
  com `train`, prevê cada `(usuário, item)` de `test` e compara com a nota
  real escondida. Devolve **MAE** (erro médio absoluto), **RMSE** (penaliza
  mais os erros grandes) e **cobertura** (fração das notas de teste que
  tiveram previsão — as que caíram em `None` por falta de vizinho contam
  contra a cobertura, não entram no erro).

## Passo a passo da implementação

Essa é a ordem lógica pra construir (ou explicar) o algoritmo do zero:

1. **Representar os dados** como matriz usuário × item × nota, sem usar 0
   para "não avaliado" — é isso que `load_ratings` faz ao ler o CSV.
2. **Medir a semelhança entre dois usuários** olhando só pro que os dois
   avaliaram em comum — é a Pearson de `pearson()`.
3. **Escolher os vizinhos**: de todos que avaliaram o item alvo, manter só
   quem tem similaridade positiva e pegar os `k` mais parecidos — dentro de
   `.predict()`, antes de calcular a nota.
4. **Estimar a nota** como a média do usuário ajustada pelo quanto cada
   vizinho se desviou da própria média naquele item, ponderado pela
   similaridade — a fórmula no fim de `.predict()`.
5. **Gerar recomendações** repetindo o passo 4 para todo item não avaliado
   e ordenando pela nota prevista — `.recommend()`.
6. **Avaliar o modelo** escondendo notas reais, prevendo-as com o resto da
   base como treino, e medindo o erro — `evaluate.py`.

## Executar

Na pasta `implementations/userbased-nearestneighbor` (veja o
[README de `implementations/`](../README.md) para o setup do ambiente, ou
use `py ../menu.py` para o terminal navegável):

```powershell
py example_slide.py
py evaluate.py
py present.py
```

`evaluate.py` aceita `--test-ratio`, `--seed`, `--min-train`, `--k` e
`--min-common` para experimentar outras divisões; por padrão usa 20% das
notas de cada usuário como teste (semente fixa `42`, mínimo de 3 notas no
treino). Sobre a base dummy atual (seed `42`, 35 usuários, 24 itens, 600
avaliações), a cobertura é de 100% com MAE ≈ 0,76 e RMSE ≈ 0,97 — esses
números mudam a cada `generate_dataset.py --seed` diferente.

## Resultado do exemplo do Slide 2

Reproduzindo a matriz da Alice: as similaridades calculadas são `0,85`
(User1), `0,71` (User2), `0,00` (User3) e `-0,79` (User4) — a página 9 do
slide parece trocar os valores de User2 e User3. Com `k=2`, os vizinhos
positivos são User1 e User2, e a previsão de Alice para o Item5 é `4,872`.

## Interface

`load_ratings("data/ratings.csv")` devolve `dict[user_id][item_id] = nota`
(veja [`implementations/generate_dataset.py`](../generate_dataset.py) para
como essa base é gerada).
`UserBasedKNN(ratings, k=2, min_common=3)` recebe **somente notas de treino**.
`predict(user_id, item_id)` devolve `Prediction` ou `None` se não houver vizinhos
válidos. `recommend(user_id, n=5)` devolve previsões ordenadas para itens ainda
não vistos. Uma avaliação escondida para teste deve ser removida do dicionário
antes de criar o modelo; nunca pode participar da similaridade nem das médias.

## Divisão do trabalho

**Fase 1 — algoritmo (`recommender.py`, `example_slide.py`):** concluída.

**Fase 2 — avaliação (`evaluate.py`):** concluída (MAE, RMSE, cobertura).
NDCG@N fica como próximo passo, caso a apresentação precise avaliar também
a ordenação do Top-N, não só o erro de nota a nota.
