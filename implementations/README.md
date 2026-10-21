# Implementações

Implementações em Python puro (sem dependências externas) dos algoritmos de
filtragem colaborativa vistos na disciplina, usando a mesma base dummy de
avaliações (gerada por [`generate_dataset.py`](generate_dataset.py)) para
permitir comparar os métodos entre si.

| Pasta | Método | Status |
|---|---|---|
| [`userbased-nearestneighbor`](userbased-nearestneighbor) | Filtragem colaborativa baseada em **usuário** (similaridade de Pearson) | Algoritmo + avaliação (MAE/RMSE/cobertura) |
| [`itembased-nearestneighbor`](itembased-nearestneighbor) | Filtragem colaborativa baseada em **item** (cosseno ajustado) | Algoritmo + exemplo; avaliação pendente |

## Terminal navegável

Em vez de decorar os comandos abaixo, dá pra rodar tudo por um menu
interativo (biblioteca padrão, sem dependências):

```powershell
cd implementations
py menu.py
```

Ele guia em 3 passos:

1. **Base de dados** — gerar/regerar ou pular.
2. **Escolher o algoritmo** (user-based ou item-based) — ao entrar, já roda
   o exemplo do slide respondendo as 3 perguntas de todo KNN: **1.
   similaridade**, **2. quantos vizinhos**, **3. nota estimada**.
3. **O que fazer com ele** — treinar com a base real e recomendar pra um
   usuário, ou (só no user-based) avaliar o modelo (MAE/RMSE sobre notas
   escondidas de teste).

Os parâmetros são pedidos só quando fazem sentido (ex.: pra qual usuário
recomendar, ou os parâmetros da avaliação), com um padrão sensato se você só
apertar Enter. Sai com `0` em qualquer passo.

## Base dummy

`generate_dataset.py` escreve `data/ratings.csv` (`user_id,item_id,rating`) e
`data/items.csv` (`item_id,title`) **direto dentro de cada pasta de
implementação** (`userbased-nearestneighbor/data/` e
`itembased-nearestneighbor/data/`), com uma cópia idêntica nas duas — nada
manual, nada pedido a um LLM.

O tamanho da base é sorteado a cada execução, não fixo: entre 22 e 38
usuários, entre 14 e 26 itens (de um pool de 30, em 3 gêneros), e entre 8 e
20 avaliações por usuário — então o total de avaliações varia de uma semente
para outra. Todo usuário tem pelo menos 8 avaliações e todo item é avaliado
por vários usuários; pares não aparecem repetidos e nota ausente nunca é 0
(significa "ainda não avaliado"). O gerador também cria de propósito um par
de usuários muito parecidos (`U01`/`U02`), um par de gostos opostos
(`U03`/`U04`, notas quase invertidas) — úteis para mostrar na apresentação
como cada caso se comporta.

A aleatoriedade usa uma semente (`--seed`, padrão `42`), então a mesma
semente sempre reproduz exatamente a mesma base; mudar a semente sorteia um
tamanho e um conteúdo novos:

```powershell
cd implementations
py generate_dataset.py            # semente 42 (padrão)
py generate_dataset.py --seed 7   # outro tamanho/base, igualmente reproduzível
```

## Ambiente de execução

Nenhuma dependência externa é necessária — tudo usa a biblioteca padrão do
Python (`csv`, `math`, `random`, `argparse`). Não precisa de venv nem de
`pip install` nada.

**Requisito:** Python 3.10 ou mais recente.

```powershell
py --version   # confirma que o Python 3.10+ está instalado
cd implementations
py menu.py
```
