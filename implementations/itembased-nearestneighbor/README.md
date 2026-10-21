# Item-Based Nearest Neighbors

Filtragem colaborativa baseada em **item**: em vez de comparar usuários entre
si (como em [`userbased-nearestneighbor`](../userbased-nearestneighbor)),
compara-se cada par de itens pelo padrão de notas que os mesmos usuários
deram a eles, e a nota prevista para um item novo é a média ponderada das
notas que o próprio usuário já deu aos itens mais parecidos.

> Não há ainda um slide oficial da disciplina dedicado a este método (o Slide 2
> cobre apenas a versão baseada em usuário). A similaridade usada aqui é o
> **cosseno ajustado** (Sarwar et al., 2001), a formulação clássica para
> filtragem baseada em item descrita em Jannach et al., *Recommender Systems:
> An Introduction* (bibliografia da disciplina). Ajustar caso a professora
> apresente uma formulação diferente em sala.

## Como o código funciona

### `recommender.py`

- **`load_ratings(path)`** — idêntica à do `userbased-nearestneighbor`: lê
  `user_id,item_id,rating` e devolve `dict[user_id][item_id] = nota`, sem
  usar 0 para "não avaliado".

- **`_items_to_users(ratings)`** — transpõe a matriz: em vez de
  `usuário → item → nota`, monta `item → usuário → nota`. É essa visão por
  item que o resto do módulo usa, porque a similaridade aqui é entre itens.

- **`adjusted_cosine(item_a, item_b, user_means, min_common)`** — cosseno
  ajustado entre dois itens, sobre os **usuários que avaliaram os dois**
  (pelo menos `min_common`). "Ajustado" porque subtrai de cada nota a média
  **do usuário que a deu** (não a média do item) antes de calcular o
  cosseno — isso remove o viés de usuários que, em geral, dão notas mais
  altas ou mais baixas que a média, igual ao que o Pearson faz no método
  baseado em usuário.

- **`Neighbor` / `Prediction`** — mesma ideia do método baseado em usuário:
  dataclasses pra guardar quem contribuiu pra previsão e com que peso.

- **`ItemBasedKNN.__init__(ratings, k, min_common)`** — guarda a matriz de
  notas, a média de cada usuário (precisa dela pro cosseno ajustado), a
  visão transposta (`_items_to_users`) e um cache de similaridade (calcular
  cosseno ajustado toda vez que dois itens se repetem em previsões
  diferentes seria redundante, já que a matriz de treino não muda).

- **`.similarity(item_a, item_b)`** — consulta o cache; se não tiver,
  calcula `adjusted_cosine` e guarda (a chave do cache é ordenada, porque
  `sim(i,j) == sim(j,i)`).

- **`.predict(user, item)`** — mesma lógica estrutural do método baseado em
  usuário, só que comparando itens:
  1. erro se o usuário já avaliou esse item;
  2. para cada item que o usuário **já avaliou**, calcula a similaridade
     dele com o item alvo;
  3. descarta similaridade ≤ 0;
  4. ordena pela similaridade e pega os `k` melhores;
  5. sem vizinhos válidos → `None`;
  6. senão, a nota prevista é a **média ponderada das notas que o próprio
     usuário deu** aos itens vizinhos (sem ajuste por média — aqui quem
     avalia é sempre o mesmo usuário nos dois lados, então não há o viés de
     escalas diferentes que existia entre usuários):
     ```
     nota = Σ sim(item, j)·nota(user, j) / Σ sim(item, j)
     ```
  7. limita o resultado entre 1 e 5.

- **`.recommend(user, n)`** — igual ao método baseado em usuário: chama
  `.predict` para todo item não avaliado, descarta `None`, ordena e
  devolve os `n` melhores.

## Passo a passo da implementação

1. **Representar os dados** do mesmo jeito (matriz esparsa, sem 0 para
   "não avaliado") — `load_ratings`.
2. **Olhar a matriz por item**, não por usuário — `_items_to_users`, porque
   agora quem é comparado são os itens.
3. **Medir a semelhança entre dois itens** pelos usuários que avaliaram os
   dois, ajustando pela média de cada usuário — `adjusted_cosine`.
4. **Escolher os vizinhos**: dos itens que o usuário já avaliou, manter só
   os de similaridade positiva com o item alvo e pegar os `k` mais
   parecidos — dentro de `.predict()`.
5. **Estimar a nota** como a média ponderada das notas que o usuário já deu
   a esses itens vizinhos — fórmula no fim de `.predict()`.
6. **Gerar recomendações** repetindo o passo 5 para todo item não avaliado
   — `.recommend()`.

Falta o passo de **avaliação** (treino/teste, MAE/RMSE/cobertura) — ver
"Próximos passos" abaixo; o `userbased-nearestneighbor/evaluate.py` serve de
modelo pra isso.

## Executar

Na pasta `implementations/itembased-nearestneighbor` (veja o
[README de `implementations/`](../README.md) para o setup do ambiente, ou
use `py ../menu.py` para o terminal navegável):

```powershell
py example_slide.py
py present.py
```

(ou `python` no lugar de `py`, dependendo da instalação.)

## Resultado do exemplo (matriz da Alice)

Reaproveitando a matriz da Alice do Slide 2, as similaridades de cada item com
o Item5 são `0,80` (Item1), `-0,91` (Item2), `-0,76` (Item3) e `0,43` (Item4).
Com `k=2`, os vizinhos positivos são Item1 e Item4, e a previsão de Alice para
o Item5 é `4,650` — próxima da previsão `4,872` obtida pelo método baseado em
usuário, útil para comparar os dois métodos na apresentação.

## Interface

`load_ratings("data/ratings.csv")` devolve `dict[user_id][item_id] = nota`.
`ItemBasedKNN(ratings, k=2, min_common=2)` recebe **somente notas de treino**.
`predict(user_id, item_id)` devolve `Prediction` ou `None` se não houver
vizinhos válidos. `recommend(user_id, n=5)` devolve previsões ordenadas para
itens ainda não vistos. A mesma base dummy de `userbased-nearestneighbor` é
usada aqui — veja [`implementations/generate_dataset.py`](../generate_dataset.py),
que escreve uma cópia idêntica em ambas as pastas para os dois métodos serem
comparáveis sobre os mesmos dados.

## Próximos passos

- `evaluate.py` com o mesmo split treino/teste do método baseado em usuário,
  para comparar MAE/RMSE/cobertura entre os dois algoritmos lado a lado.
