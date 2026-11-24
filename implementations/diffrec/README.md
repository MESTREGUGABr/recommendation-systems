# Reprodução experimental do DiffRec

Implementação do protocolo de treino e avaliação de *Diffusion Recommender Model*
(Wang et al., SIGIR 2023). O treinamento parte de pesos aleatórios. O
projeto usa as redes e o processo de difusão da revisão oficial `e42532f`,
com carregamento esparso por lote, seleção por validação e saída em JSON.
[Artigo](https://arxiv.org/abs/2304.04971) · [Código dos autores](https://github.com/YiyanXu/DiffRec/tree/e42532f5f329a5f1c5230f05ad7e2930d77b9f75)

## Preparar no Windows

Execute na raiz do repositório com Python 3.11, Git e o `tar` do Windows
(bsdtar, com suporte a RAR). O último comando baixa a revisão fixada, verifica
os seis arquivos RAR por SHA-256 e extrai suas partições em `datasets/`.

```powershell
python -m venv .venv
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install -r implementations/diffrec/requirements.txt
& $py -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
& $py implementations/diffrec/prepare.py
```

Em uma máquina sem CUDA, use `https://download.pytorch.org/whl/cpu` como
índice na instalação do PyTorch. Para extrair um único conjunto, informe seu
nome após `prepare.py`, por exemplo `ml-1m_clean`.

## Dados oficiais

Cada partição contém pares de IDs inteiros usuário–item. Os cenários
`clean` e `noisy` correspondem, respectivamente, a interações positivas
e treinamento com ruído natural. As duas versões de cada dataset usam o
mesmo conjunto de teste. O cenário temporal usa a ordem das interações de
cada usuário na partição de treino e aplica os pesos lineares dos autores.
Os IDs de item são remapeados separadamente em cada cenário.

| Dataset | Usuários | Itens | Treino | Validação | Teste |
|---|---:|---:|---:|---:|---:|
| `ml-1m_clean` | 5.949 | 2.810 | 403.277 | 110.722 | 57.532 |
| `ml-1m_noisy` | 5.949 | 3.494 | 429.993 | 130.772 | 57.532 |
| `yelp_clean` | 54.574 | 34.395 | 1.014.486 | 246.676 | 141.574 |
| `yelp_noisy` | 54.574 | 77.405 | 1.076.433 | 253.668 | 141.574 |
| `amazon-book_clean` | 108.822 | 94.949 | 2.283.281 | 544.594 | 318.381 |
| `amazon-book_noisy` | 108.822 | 178.181 | 2.279.538 | 547.304 | 318.381 |

A versão `yelp_noisy` contém 3.453 pares em comum entre treino e
validação, 1.055 entre treino e teste e 684 entre validação e teste.
Preservamos as partições originais. Se um alvo também estiver no histórico
mascarado, ele continua no denominador da métrica e fica fora do ranking,
como no código dos autores.

As variantes L-DiffRec e LT-DiffRec precisam dos embeddings de itens do
LightGCN disponibilizados pelos autores. Para Yelp e MovieLens-1M eles vêm
nos RARs. Para Amazon-book, `prepare.py` busca o ZIP separado, extrai os
embeddings e verifica seus SHA-256. Para usar uma cópia do ZIP já baixada,
passe seu caminho em `prepare.py --amazon-embeddings <arquivo.zip>`.
[Embeddings oficiais de Amazon-book](https://pan.ustc.edu.cn/share/index/2a4185182c9940c18e7f)

## Executar

```powershell
# Testes de pesos temporais, histórico da difusão e avaliação
& $py -m unittest discover -s implementations/diffrec -v

# Teste de ponta a ponta com uma época e dados reais
& $py -u implementations/diffrec/train.py --epochs 1

# Experimento DiffRec, MovieLens-1M limpo, Tabela 2
& $py -u implementations/diffrec/train.py --dataset ml-1m_clean

# Experimento L-DiffRec, MovieLens-1M com ruído natural, Tabela 4
& $py -u implementations/diffrec/train.py --model l-diffrec --dataset ml-1m_noisy

# Experimentos temporais, Tabela 7
& $py -u implementations/diffrec/train.py --model t-diffrec --dataset yelp_clean
& $py -u implementations/diffrec/train.py --model lt-diffrec --dataset yelp_clean

# DiffRec com injeção de 10% de ruído aleatório, cenário da Figura 4
& $py -u implementations/diffrec/train.py --dataset amazon-book_clean --random-noise 0.1
```

O padrão é semente 1, GPU quando disponível e limite de 1.000 épocas com
parada antecipada. Cada execução salva o melhor checkpoint e um
`metrics.json` com configuração, histórico, métricas de validação e teste,
valores do artigo e diferenças em relação ao artigo. Use `--device cpu`
para forçar CPU e `--output` para escolher um novo diretório de saída.
Para Amazon-book, o otimizador do DiffRec usa memória proporcional ao
número de parâmetros, além do espaço para cada lote. `--batch-size`
reduz o espaço por lote e muda o número de atualizações por época.
Para as variantes, o JSON registra o número de parâmetros e, em GPU,
o pico de memória alocada pelo PyTorch em MiB.

| Modelo | Presets | Referência |
|---|---|---|
| DiffRec | Três datasets, limpo e ruído natural | Tabelas 2 e 3 |
| L-DiffRec | Três datasets, limpo e ruído natural | Tabelas 6 e 4 |
| T-DiffRec e LT-DiffRec | Amazon-book e Yelp, temporal | Tabela 7 |

### Protocolo

- A validação ranqueia o catálogo e mascara itens do treino. O teste
  usa os vetores do treino como entrada e mascara itens do treino e da
  validação, conforme o modo padrão do código oficial. A opção
  `--test-with-validation` inclui as interações de validação na entrada
  do teste, como o argumento `--tst_w_val` dos autores.
- A média de Recall@10/20 e NDCG@10/20 inclui todos os usuários. Na
  validação, Recall@20 é arredondado a quatro casas antes da seleção do
  checkpoint, como no código oficial. O teste roda uma vez com os pesos da
  melhor época. Quando o limite de épocas cai entre duas avaliações, o
  script também valida na última época pedida.
- As configurações vêm dos checkpoints citados em `inference.py`.
  Aplicamos as duas correções de `reweight` declaradas pelos autores para
  Amazon-book e Yelp limpos no DiffRec. Para os demais valores seguimos
  o nome do checkpoint. O script permite `--reweight` e `--no-reweight`
  e calcula MSE sem pesos quando solicitado, corrigindo uma variável
  de perda que ficava indefinida no código original.
- O código oficial usa Python 3.8 e PyTorch 1.12. Aqui usamos Python 3.11,
  PyTorch 2.5.1 e lotes montados a partir de matrizes esparsas. A ordem
  aleatória dos lotes pode variar em relação à implementação oficial.
- O ruído aleatório é gerado localmente. Para cada usuário e partição de
  treino ou validação, a quantidade adicionada é a parte inteira da
  proporção vezes o número de interações. A amostragem é uniforme, sem
  reposição, sobre itens ausentes das três partições daquele usuário.
  O teste mantém as interações oficiais.

## Resultados locais

Windows, Python 3.11.5, PyTorch 2.5.1 com CUDA 11.8 e RTX 3050 de 6 GB.
Cada linha abaixo é um treino a partir de pesos aleatórios com semente 1
e parada antecipada. Valores no formato **local / artigo**.

| Modelo | Dataset | Tabela | Melhor época | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 |
|---|---|---:|---:|---:|---:|---:|---:|
| DiffRec | MovieLens-1M limpo | 2 | 10 | 0,1017 / 0,1058 | 0,1745 / 0,1787 | 0,0874 / 0,0901 | 0,1118 / 0,1148 |
| DiffRec | MovieLens-1M com ruído | 3 | 5 | 0,0636 / 0,0658 | 0,1242 / 0,1236 | 0,0470 / 0,0488 | 0,0696 / 0,0703 |
| L-DiffRec | MovieLens-1M limpo | 6 | 50 | 0,0916 / 0,1060 | 0,1611 / 0,1809 | 0,0774 / 0,0868 | 0,1017 / 0,1122 |
| L-DiffRec | MovieLens-1M com ruído | 4 | 30 | 0,0637 / 0,0665 | 0,1172 / 0,1272 | 0,0493 / 0,0493 | 0,0688 / 0,0710 |

Os cinco testes automatizados passaram, incluindo a comparação da
perda ponderada com a revisão oficial. As partições e os embeddings dos
três datasets foram extraídos e conferidos.

Também completamos treino, validação, recuperação do checkpoint e teste
em uma época com estas combinações.

| Modelo | Dataset | Hardware |
|---|---|---|
| DiffRec | MovieLens-1M limpo | CPU |
| DiffRec | MovieLens-1M com 10% de ruído aleatório | GPU |
| T-DiffRec | Yelp limpo | GPU |
| LT-DiffRec | Yelp limpo | GPU |
| L-DiffRec | Amazon-book limpo | GPU |

Em um diagnóstico do L-DiffRec no MovieLens-1M limpo, com o checkpoint
equivalente da época 50, `--test-with-validation` elevou Recall@20 de
0,1611 para 0,2094 e NDCG@20 de 0,1017 para 0,1405. A tabela usa o
protocolo padrão com histórico de treino.

Os baselines, os testes estatísticos, a varredura da Figura 4 e os treinos
completos em Yelp e Amazon-book ainda precisam de execução. Para as
ablações da difusão, `--help` lista opções para passos, escala de ruído,
alvo `eps` e número de categorias.
