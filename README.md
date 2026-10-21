# Sistemas de Recomendação

Repositório da disciplina **Sistemas de Recomendação** (CC3), do curso de Bacharelado em Ciência da Computação da **UFAPE** (Universidade Federal do Agreste de Pernambuco), período letivo **2026.2**.

- **Professora:** Maria Aparecida Amorim Sibaldo de Carvalho
- **Carga horária:** 60h (4 créditos)
- **Horário:** Terças-feiras, 14h–18h

## Ementa

Função do Sistema de Recomendação. Coleta de Informações. Estratégias de Recomendação. Descoberta do Conhecimento (tipos de descoberta, classificação, agrupamento). Filtragem de informações (baseada em conteúdo, colaborativa, híbrida). Técnicas de Recomendação. Aplicação e avaliação. Sistemas de Recomendação e IHC. Processamento de Dados, Classificação, Análise de Cluster, Regras associativas de recomendação. Exemplos de Sistemas de Recomendação. Projeto.

## Conteúdo programático

1. Introdução a sistemas de recomendação
2. Tipos de sistemas de recomendação
   1. Filtragem colaborativa
   2. Filtragem baseada em conteúdo
   3. Filtragem baseada em agrupamento
3. Algoritmos aplicados em sistemas de recomendação
   1. Filtragem colaborativa baseada em usuário
   2. Filtragem colaborativa baseada em item
   3. SVD (Singular Value Decomposition)
4. Métricas de avaliação de resultados
   1. MSE (Mean Squared Error)
   2. NDCG (Normalized Discounted Cumulative Gain)
5. Trabalhos atuais na área
6. Projeto

## Cronograma

| Data | Conteúdo |
|---|---|
| 13/10/2026 | Apresentação da ementa e da disciplina. Introdução a Sistemas de Recomendação |
| 20/10/2026 | Tipos de Sistemas de Recomendação. Filtragem colaborativa |
| 27/10/2026 | Filtragem baseada em conteúdo. Filtragem baseada em agrupamento |
| 03/11/2026 | Algoritmos aplicados em Sistemas de Recomendação |
| 10/11/2026 | Filtragem colaborativa baseada em conteúdo. Filtragem colaborativa baseada em item |
| 17/11/2026 | SVD (Singular Value Decomposition) |
| 24/11/2026 | Métricas de avaliação de resultados. MSE. NDCG |
| 01/12/2026 | Escolha do artigo a ser trabalhado — **1ª VA** |
| 08/12/2026 a 23/02/2027 | Acompanhamento do projeto — **2ª VA** em 23/02 |
| 09/03/2027 | 3ª VA |
| 16/03/2027 | Avaliação Final |

## Avaliação

- **1ª VA:** acompanhamento das implementações feitas em sala, aula a aula.
- **2ª VA:** implementação do artigo escolhido para replicar.
- **3ª VA e Avaliação Final:** projeto final da disciplina.

## Estrutura do repositório

```
.
├── materials/                     # plano de ensino oficial e anotações de aula
├── papers/                        # artigos científicos trabalhados em sala
├── books/                         # bibliografia de apoio
├── slides/                        # slides das aulas
├── implementations/               # implementações dos algoritmos vistos em aula
│   ├── userbased-nearestneighbor/ # filtragem colaborativa baseada em usuário
│   └── itembased-nearestneighbor/ # filtragem colaborativa baseada em item
└── README.md
```

### Implementações (`implementations/`)

Implementações em Python (biblioteca padrão apenas) dos algoritmos de
filtragem colaborativa, com exemplos reproduzindo os slides da disciplina e
testes automatizados. Veja o [README de `implementations/`](implementations/README.md)
para a lista de métodos, o status de cada um e como preparar o ambiente para
rodar o código e os testes.

### Artigos (`papers/`)

| Arquivo | Título | Resumo |
|---|---|---|
| `2304.04971.pdf` | Diffusion Recommender Model (DiffRec) | Aplica modelos de difusão a sistemas de recomendação para modelar interações do usuário lidando com ruído, predição em larga escala e mudanças temporais de preferência. |
| `Paper marcado.pdf` | Diffusion Recommender Model (DiffRec) | Cópia anotada do artigo acima, usada para discussão em sala. |

### Livros (`books/`)

- `Recommender_Systems.pdf` — material de apoio da bibliografia complementar (Ricci, Rokach, Shapira & Kantor; Jannach et al.; Aggarwal).

### Materiais (`materials/`)

- `2026.2-SistemasDeRecomendacao-planoDeEnsino.pdf` — plano de ensino oficial.
- `notations.md` — notações/referências usadas em aula.

### Bibliografia complementar

- RICCI, F.; ROKACH, L.; SHAPIRA, B.; KANTOR, P. *Recommender Systems Handbook*. Springer, 2011.
- JANNACH, D.; ZANKER, M.; FELFERNIG, A.; FRIEDRICH, G. *Recommender Systems: An Introduction*. Cambridge University Press, 2010.
- AGGARWAL, Charu C. *Recommender Systems: The Textbook*. Springer, 2016.
- AGARWAL, Deepak K.; CHEN, Bee-Chung. *Statistical Methods for Recommender Systems*. Cambridge University Press, 2016.
- RUSSELL, M. *Mining the Social Web*. O'Reilly, 2011.
- UCHYIGIT, G.; MA, M. Y. *Personalization Techniques and Recommender Systems*. World Scientific Publishing Company.
- MUSIAL, K. *Recommender System for Online Social Network*. Lambert Academic Publishing, 2009.
- YOO, Kyung-Hyan; GRETZEL, Ulrike; ZANKER, Markus. *Persuasive Recommender Systems: Conceptual Background and Implications*. Springer-Verlag New York, 2013.
