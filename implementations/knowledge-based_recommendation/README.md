# Knowledge-Based Recommendation

Recomendação **baseada em conhecimento** (Slide 4): em vez de notas de
usuários (colaborativa) ou do histórico de itens curtidos (conteúdo), o
usuário diz o que precisa e o sistema usa conhecimento sobre os itens para
achar o que serve. Útil quando há poucas avaliações, quando o tempo importa
(uma câmera de cinco anos atrás) ou quando o cliente quer definir requisitos
explícitos ("a cor do carro deve ser preta").

O catálogo é o **exemplo do capítulo**: oito câmeras (P1 a P8) em
[`data/catalog.csv`](data/catalog.csv).

## Como o código funciona

### Baseada em restrições (`recommender.py`)

- **`select(catalog, requisitos)`** — consulta conjuntiva σ[REQ](P): os itens
  que cumprem *todos* os requisitos. Ex.: `σ[mpix≥10, price<300](P) = {P4, P7}`.

- **`quickxplain(catalog, requisitos)`** — Algoritmo 4.1. Quando
  `σ[REQ](P) = ∅`, devolve um conjunto de conflito **mínimo** (um grupo de
  requisitos que não dá certo junto, sem nenhum requisito sobrando),
  dividindo os requisitos ao meio. Nota: o slide imprime a 1ª linha como
  "σ[REQ](P) = ∅ … return ∅" (o `≠` se perdeu); implementamos a versão
  correta, que devolve vazio quando *existe* solução.

- **`all_conflict_sets` / `diagnoses`** — todos os conflitos mínimos e, a
  partir deles, os **diagnósticos**: conjuntos mínimos de requisitos a
  relaxar que desfazem todos os conflitos. No exemplo:
  `CS = {r1,r2}, {r2,r4}, {r1,r3}` → `d1={r1,r2}, d2={r1,r4}, d3={r2,r3}`.

- **`repairs`** — π[atributos(d)] σ[REQ − d](P): consulta o catálogo com os
  requisitos que sobram e mostra os valores que os requisitos do diagnóstico
  precisariam ter (ex.: `d2 → {price=182, waterproof=no}`).

- **`utility_ranking`** — utilidade multiatributo (slide 17-18): cada atributo
  dá uma nota de **qualidade** e outra de **economia** (tabela
  `UTILITY_TABLE`); o cliente pondera as duas (cu1: 80%/20%, cu2: 40%/60%) e
  os itens são ordenados pela soma ponderada.

### Baseada em casos (`recommender.py`)

- **`CaseBasedRetrieval.rank`** — `similarity(p, REQ) = Σ w_r·sim(p, r) / Σ w_r`.
  A similaridade local depende do desejo do cliente: **MIB** (mais é melhor,
  ex.: megapixels), **LIB** (menos é melhor, ex.: preço), valor-alvo, ou
  igualdade para atributos yes/no. O slide só dá a fórmula geral; as
  similaridades locais são as clássicas do livro.

- **Críticas** (slides 20-22): `unit_critique` ("mais barata") escolhe, entre
  os itens que atendem à crítica, o mais parecido com o item atual.
  `compound_critiques` implementa o `CompoundCritiques` do Algoritmo 4.4: o
  padrão de diferença de cada candidato (`preço <`, `mpix >`...) passa por um
  **Apriori** com suporte mínimo σmin, e as `k` críticas compostas de menor
  suporte são oferecidas. (O slide não fixa o critério de seleção; usamos o
  de menor suporte, o mais discriminante.) `apply_critique` aplica a escolhida.

## Exemplo do capítulo (`example_slide.py`)

`python example_slide.py` reproduz os resultados do slide e **confere cada um**
com o valor impresso; termina com `OK` se tudo bate:

| Slide | Resultado |
|---|---|
| 9 | `σ[mpix≥10, price<300](P) = {P4, P7}` |
| 13 | conflitos `{r1,r2}`, `{r2,r4}`, `{r1,r3}`; diagnósticos `d1`, `d2`, `d3` |
| 16 | reparos `(278, 10×)`, `(182, sem água)`, `(4×, sem som)` |
| 18 | utilidades de cu1 e cu2 das 8 câmeras |

**Divergência no slide 18:** para P5 (LCD 3,0") o slide imprime as notas de LCD
como 10 (qualidade) e 6 (economia), mas a tabela do slide 17 dá 9 e 5 — e é o
que o próprio slide usa em P6, P7 e P8, que também têm LCD 3,0". Pela tabela,
P5 fica com Σqualidade=52 e Σeconomia=59 (utilidade 53,4 / 56,2), e não
53 / 60 (54,4 / 57,2). Além disso, P4 e P6 empatam em utilidade e o slide as
ordena de forma diferente em cu1 e cu2; aqui o desempate é pelo id.

## Executar

Na pasta `implementations/knowledge-based_recommendation` (ou use
`py ../menu.py` e escolha o algoritmo 4):

```powershell
py example_slide.py
py present.py
```

`present.py` roda o exemplo do capítulo e depois abre um diálogo: você informa
requisitos (preço máximo, mpix, zoom, LCD, filmes, som, à prova d'água) e
recebe as câmeras que servem, ranqueadas por utilidade; dá para navegar por
críticas a partir do primeiro item. Cada pergunta já vem com um **valor padrão**
(preço ≤ 250 €, ≥ 8 mpix, ≥ 4× de zoom, LCD ≥ 2,5", grava filmes e som; à
prova d'água "tanto faz"): basta apertar Enter, e `-` ignora o requisito. Os
padrões ficam em `DEFAULTS`, no topo da seção de requisitos do `present.py`. Se nenhuma câmera serve, ele mostra o
conflito (QuickXPlain), os diagnósticos e os reparos.

## Limitações (também citadas no slide)

- **Custo de aquisição de conhecimento**: a tabela de utilidade e as regras
  vêm de especialistas; aqui são fixas, do slide.
- **Precisão do modelo de preferências**: perfis muito finos exigem muitos
  ciclos de interação.
- **Independência**: a soma de notas por atributo assume que as preferências
  são independentes entre si.
