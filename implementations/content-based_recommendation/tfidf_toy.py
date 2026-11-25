import math

LOG_BASE = 10

# ---------------------------------------------------------------- dataset toy
documentos = ["Antony and Cleopatra", "Julius Caesar", "The Tempest",
              "Hamlet", "Othello", "Macbeth"]

# TF bruto: termo -> contagem em cada documento (mesma ordem de `documentos`)
tf = {
    "Antony":    [157,  73, 0, 0, 0, 0],
    "Brutus":    [  4, 157, 0, 1, 0, 0],
    "Caesar":    [232, 227, 0, 2, 1, 1],
    "Calpurnia": [  0,  10, 0, 0, 0, 0],
    "Cleopatra": [ 57,   0, 0, 0, 0, 0],
    "mercy":     [1.51,  0, 3, 5, 5, 1],   # 1.51 vem do slide (nao e inteiro)
    "worser":    [1.37,  0, 1, 1, 1, 0],   # 1.37 vem do slide (nao e inteiro)
}

N = len(documentos)


# ------------------------------------------------------------------ formulas
def n_i(termo):
    """n(i): em quantos documentos o termo aparece (TF > 0)."""
    return sum(1 for c in tf[termo] if c > 0)


def idf(termo):
    """IDF(i) = log(N / n(i))."""
    return math.log(N / n_i(termo), LOG_BASE)


def tf_idf(termo, j):
    """TF-IDF(i, j) = TF(i, j) * IDF(i)."""
    return tf[termo][j] * idf(termo)


# ------------------------------------------------------------------- impressao
def imprimir_tabela(titulo, linhas, fmt):
    print(f"\n{titulo}")
    larg = max(len(t) for t in tf) + 2
    cab = " " * larg + "".join(f"{d[:12]:>14}" for d in documentos)
    print(cab)
    print("-" * len(cab))
    for termo, valores in linhas.items():
        print(f"{termo:<{larg}}" + "".join(f"{fmt(v):>14}" for v in valores))


if __name__ == "__main__":
    print(f"N = {N} documentos | log base {LOG_BASE}")

    # 1) n(i) e IDF
    print("\nPasso 1 - n(i) e IDF(i) = log(N / n(i))")
    print(f"{'termo':<12}{'n(i)':>6}{'N/n(i)':>10}{'IDF':>10}")
    for t in tf:
        print(f"{t:<12}{n_i(t):>6}{N / n_i(t):>10.2f}{idf(t):>10.3f}")

    # 2) TF bruto
    imprimir_tabela("Passo 2 - TF(i, j) (contagem bruta)", tf,
                    lambda v: f"{v:g}")

    # 3) TF-IDF
    resultado = {t: [tf_idf(t, j) for j in range(N)] for t in tf}
    imprimir_tabela("Passo 3 - TF-IDF(i, j) = TF(i, j) * IDF(i)", resultado,
                    lambda v: f"{v:.2f}")

    # 4) Exemplo passo a passo
    t, j = "Antony", 0
    print(f"\nExemplo a mao: termo '{t}' no documento '{documentos[j]}'")
    print(f"  TF      = {tf[t][j]}")
    print(f"  n(i)    = {n_i(t)}  ->  IDF = log({N}/{n_i(t)}) = {idf(t):.3f}")
    print(f"  TF-IDF  = {tf[t][j]} * {idf(t):.3f} = {tf_idf(t, j):.2f}")
