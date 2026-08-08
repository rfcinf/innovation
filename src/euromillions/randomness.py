"""
Bateria de testes de aleatoriedade e deteção de viés físico.

Este módulo existe para responder a uma pergunta concreta e falsificável:

    "As bolas e as máquinas do EuroMillions desviam-se, de forma
     mensurável e explorável, do sorteio uniforme?"

É a única pergunta desta família que alguma vez rendeu dinheiro a alguém.
Joseph Jagger (Monte Carlo, 1873) e a equipa de Ed Thorp não previram o
futuro: mediram o desgaste físico de um equipamento e apostaram no desvio.
Se existe uma brecha do lado dos *números*, é aqui — e só aqui — que ela
aparece.

Duas obrigações metodológicas que este módulo cumpre e que quase toda a
literatura amadora de lotarias ignora:

1. Correção para testes múltiplos. Com 50 números, 12 estrelas, 1225 pares
   e centenas de janelas temporais, aparecem "anomalias a p < 0.05" às
   centenas *por construção*. Sem controlo de FDR, encontrar padrões é
   trivial e não significa absolutamente nada. Aqui aplica-se
   Benjamini-Hochberg a tudo.

2. Análise de potência. "Não rejeitámos H0" não é o mesmo que "não há
   viés". `detection_power()` diz qual é o menor viés que este volume de
   dados conseguiria detetar — e portanto qual é o tamanho da brecha que
   ainda poderia estar escondida.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .dataset import MAIN_COLS, STAR_COLS, mains_matrix, stars_matrix


@dataclass
class TestResult:
    name: str
    statistic: float
    pvalue: float
    dof: int | None = None
    detail: str = ""

    def __str__(self) -> str:
        star = ""
        if self.pvalue < 0.001:
            star = " ***"
        elif self.pvalue < 0.01:
            star = " **"
        elif self.pvalue < 0.05:
            star = " *"
        return f"{self.name:<52} stat={self.statistic:10.3f}  p={self.pvalue:.4f}{star}"


# ---------------------------------------------------------------------------
# Frequências
# ---------------------------------------------------------------------------

def ball_counts(df: pd.DataFrame, pool: int | None = None) -> np.ndarray:
    """Contagem de cada número (índice 0 = número 1)."""
    pool = pool or config.MAIN_POOL
    flat = mains_matrix(df).ravel()
    return np.bincount(flat, minlength=pool + 1)[1:]


def star_counts(df: pd.DataFrame, pool: int) -> np.ndarray:
    flat = stars_matrix(df).ravel()
    return np.bincount(flat, minlength=pool + 1)[1:]


def chi_square_uniform(counts: np.ndarray, label: str) -> TestResult:
    """Teste qui-quadrado de aderência à uniforme."""
    n = counts.sum()
    k = len(counts)
    expected = np.full(k, n / k)
    stat = float(((counts - expected) ** 2 / expected).sum())
    dof = k - 1
    p = float(stats.chi2.sf(stat, dof))
    return TestResult(
        name=f"Qui-quadrado uniformidade — {label}",
        statistic=stat,
        pvalue=p,
        dof=dof,
        detail=f"n={n}, k={k}, esperado/célula={n/k:.1f}",
    )


def per_ball_binomial(counts: np.ndarray, n_draws: int, picks: int) -> pd.DataFrame:
    """
    Teste binomial para cada bola individualmente, com correção FDR.

    Sem a coluna `p_ajustado` esta tabela é uma máquina de produzir
    superstições: 50 testes a 5% dão ~2.5 "anomalias" garantidas.
    """
    k = len(counts)
    p0 = picks / k
    rows = []
    for i, c in enumerate(counts, start=1):
        res = stats.binomtest(int(c), n_draws, p0, alternative="two-sided")
        rows.append(
            {
                "numero": i,
                "vezes": int(c),
                "esperado": n_draws * p0,
                "desvio_pct": 100 * (c - n_draws * p0) / (n_draws * p0),
                "p": res.pvalue,
            }
        )
    out = pd.DataFrame(rows)
    out["p_ajustado"] = benjamini_hochberg(out["p"].to_numpy())
    out["significativo_fdr5"] = out["p_ajustado"] < 0.05
    return out.sort_values("p").reset_index(drop=True)


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """Controlo da taxa de falsas descobertas (FDR). Devolve q-valores."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


# ---------------------------------------------------------------------------
# Independência temporal
# ---------------------------------------------------------------------------

def indicator_matrix(df: pd.DataFrame, pool: int = config.MAIN_POOL) -> np.ndarray:
    """(n_sorteios, pool) binária: saiu / não saiu."""
    mains = mains_matrix(df)
    out = np.zeros((len(df), pool), dtype=np.int8)
    rows = np.repeat(np.arange(len(df)), config.MAIN_PICK)
    out[rows, mains.ravel() - 1] = 1
    return out


def serial_correlation(df: pd.DataFrame, max_lag: int = 10) -> pd.DataFrame:
    """
    Autocorrelação da série indicadora de cada bola, para vários lags.

    Se sair um 7 hoje altera a probabilidade de sair um 7 no próximo
    sorteio, tem de aparecer aqui. É o teste direto à ideia de "números
    quentes/frios" — e o único que a pode validar.
    """
    ind = indicator_matrix(df)
    n, pool = ind.shape
    rows = []
    for lag in range(1, max_lag + 1):
        a = ind[:-lag]
        b = ind[lag:]
        corrs = []
        for j in range(pool):
            x, y = a[:, j], b[:, j]
            if x.std() == 0 or y.std() == 0:
                continue
            corrs.append(np.corrcoef(x, y)[0, 1])
        corrs = np.array(corrs)
        # Sob H0, r ~ N(0, 1/sqrt(m)); a média de `pool` correlações
        # independentes tem erro-padrão 1/sqrt(m*pool).
        m = len(a)
        se = 1.0 / np.sqrt(m * len(corrs))
        z = corrs.mean() / se
        rows.append(
            {
                "lag": lag,
                "r_medio": corrs.mean(),
                "r_abs_max": np.abs(corrs).max(),
                "z": z,
                "p": 2 * stats.norm.sf(abs(z)),
            }
        )
    out = pd.DataFrame(rows)
    out["p_ajustado"] = benjamini_hochberg(out["p"].to_numpy())
    return out


def gap_analysis(df: pd.DataFrame, min_expected: float = 5.0) -> pd.DataFrame:
    """
    Distribuição de intervalos entre aparições de cada número.

    Sob independência os intervalos são geométricos com p = 5/50 = 0.1
    (a extração sem reposição dentro de um sorteio não altera a
    probabilidade marginal de um número sair: é exatamente 5/50).

    NOTA METODOLÓGICA — porque é que aqui não se usa Kolmogorov-Smirnov:
    os intervalos são inteiros com muitos empates, e o KS pressupõe uma
    distribuição contínua. Aplicá-lo a dados discretos produz p-valores
    inválidos: numa versão anterior deste módulo "detetou" desvios em 47
    dos 50 números — um artefacto puro do teste errado, não um padrão do
    EuroMillions. É precisamente assim que nascem os sistemas de lotaria
    que prometem padrões. Usa-se qui-quadrado com classes agrupadas, que é
    o teste correto para dados discretos.
    """
    ind = indicator_matrix(df)
    n, pool = ind.shape
    p0 = config.MAIN_PICK / pool
    rows = []
    for j in range(pool):
        idx = np.flatnonzero(ind[:, j])
        if len(idx) < 30:
            continue
        gaps = np.diff(idx)
        m = len(gaps)

        # Classes 1, 2, ... até que o esperado caia abaixo de `min_expected`;
        # o resto agrupa-se numa classe de cauda.
        probs: list[float] = []
        g = 1
        while True:
            pk = (1 - p0) ** (g - 1) * p0
            if m * pk < min_expected or g > 200:
                break
            probs.append(pk)
            g += 1
        tail_p = 1.0 - sum(probs)
        edges = list(range(1, len(probs) + 1))

        observed = np.array([(gaps == e).sum() for e in edges] + [(gaps > edges[-1]).sum()])
        expected = np.array(probs + [tail_p]) * m
        stat = float(((observed - expected) ** 2 / expected).sum())
        dof = len(observed) - 1          # sem parâmetros estimados: p0 é teórico
        rows.append(
            {
                "numero": j + 1,
                "aparicoes": len(idx),
                "gap_medio": gaps.mean(),
                "gap_esperado": 1 / p0,
                "gap_max": int(gaps.max()),
                "classes": len(observed),
                "chi2": stat,
                "p": float(stats.chi2.sf(stat, dof)),
            }
        )
    out = pd.DataFrame(rows)
    out["p_ajustado"] = benjamini_hochberg(out["p"].to_numpy())
    out["significativo_fdr5"] = out["p_ajustado"] < 0.05
    return out.sort_values("p").reset_index(drop=True)


def stability_check(df: pd.DataFrame, number: int, n_folds: int = 4) -> pd.DataFrame:
    """
    Um desvio real é estável; um acaso não é.

    Divide o histórico em partes iguais e mede o desvio do número em cada
    uma. Se um número está genuinamente enviesado (bola mais leve, posição
    na máquina), o sinal repete-se em todos os troços. Se aparece só num,
    era ruído — e é o teste que separa uma descoberta de uma superstição.
    """
    folds = np.array_split(np.arange(len(df)), n_folds)
    p0 = config.MAIN_PICK / config.MAIN_POOL
    rows = []
    for k, fold in enumerate(folds, start=1):
        sub = df.iloc[fold]
        hits = int((mains_matrix(sub) == number).any(axis=1).sum())
        exp = len(sub) * p0
        res = stats.binomtest(hits, len(sub), p0)
        rows.append(
            {
                "troco": k,
                "de": sub["date"].iloc[0],
                "a": sub["date"].iloc[-1],
                "sorteios": len(sub),
                "saiu": hits,
                "esperado": round(exp, 1),
                "desvio_pct": round(100 * (hits - exp) / exp, 1),
                "p": round(res.pvalue, 4),
            }
        )
    return pd.DataFrame(rows)


def runs_test(df: pd.DataFrame) -> TestResult:
    """Wald-Wolfowitz sobre a paridade da soma dos 5 números."""
    sums = mains_matrix(df).sum(axis=1)
    seq = (sums % 2 == 0).astype(int)
    n1 = int(seq.sum())
    n0 = len(seq) - n1
    runs = 1 + int((seq[1:] != seq[:-1]).sum())
    mu = 2 * n0 * n1 / (n0 + n1) + 1
    var = (2 * n0 * n1 * (2 * n0 * n1 - n0 - n1)) / ((n0 + n1) ** 2 * (n0 + n1 - 1))
    z = (runs - mu) / np.sqrt(var)
    return TestResult(
        name="Teste de sequências (paridade da soma)",
        statistic=float(z),
        pvalue=float(2 * stats.norm.sf(abs(z))),
        detail=f"runs={runs}, esperado={mu:.1f}",
    )


# ---------------------------------------------------------------------------
# Estrutura conjunta
# ---------------------------------------------------------------------------

def pair_cooccurrence(df: pd.DataFrame) -> TestResult:
    """
    Os 1225 pares de números aparecem juntos com a frequência esperada?

    Se uma máquina agrupasse bolas por peso ou posição de carregamento,
    certos pares sairiam juntos mais vezes. Este teste apanha isso.
    """
    pool = config.MAIN_POOL
    mains = mains_matrix(df)
    counts = np.zeros((pool, pool), dtype=int)
    for row in mains:
        for a_i in range(len(row)):
            for b_i in range(a_i + 1, len(row)):
                a, b = row[a_i] - 1, row[b_i] - 1
                counts[a, b] += 1
    iu = np.triu_indices(pool, k=1)
    observed = counts[iu].astype(float)
    n_draws = len(df)
    n_pairs_per_draw = config.MAIN_PICK * (config.MAIN_PICK - 1) / 2
    total_pairs = pool * (pool - 1) / 2
    expected = n_draws * n_pairs_per_draw / total_pairs
    stat = float(((observed - expected) ** 2 / expected).sum())
    dof = len(observed) - 1
    return TestResult(
        name="Qui-quadrado co-ocorrência de pares (1225 pares)",
        statistic=stat,
        pvalue=float(stats.chi2.sf(stat, dof)),
        dof=dof,
        detail=f"esperado/par={expected:.2f}",
    )


def positional_test(df: pd.DataFrame) -> list[TestResult]:
    """
    A i-ésima bola *por ordem de saída* seria informativa — mas os dados
    públicos vêm ordenados por valor. O que se pode testar é se as
    estatísticas de ordem batem certo com a teoria da amostragem sem
    reposição, o que valida a integridade do conjunto de dados.
    """
    mains = mains_matrix(df)
    out = []
    n = len(df)
    for i in range(config.MAIN_PICK):
        obs = mains[:, i]
        # E[i-ésima menor de 5 sem reposição de 1..50] = i*(51)/6
        expected = (i + 1) * (config.MAIN_POOL + 1) / (config.MAIN_PICK + 1)
        se = obs.std(ddof=1) / np.sqrt(n)
        z = (obs.mean() - expected) / se
        out.append(
            TestResult(
                name=f"Estatística de ordem {i+1} (média {obs.mean():.2f} vs {expected:.2f})",
                statistic=float(z),
                pvalue=float(2 * stats.norm.sf(abs(z))),
            )
        )
    return out


def sum_distribution_test(df: pd.DataFrame) -> TestResult:
    """A soma dos 5 números segue a distribuição teórica?"""
    obs = mains_matrix(df).sum(axis=1)
    rng = np.random.default_rng(12345)
    sim = np.array(
        [rng.choice(config.MAIN_POOL, config.MAIN_PICK, replace=False).sum()
         for _ in range(200_000)]
    ) + config.MAIN_PICK
    ks = stats.ks_2samp(obs, sim)
    return TestResult(
        name="KS soma dos 5 números vs teórica",
        statistic=float(ks.statistic),
        pvalue=float(ks.pvalue),
        detail=f"média obs={obs.mean():.2f}, sim={sim.mean():.2f}",
    )


def entropy_test(df: pd.DataFrame) -> TestResult:
    """Entropia empírica das bolas vs máximo teórico log2(50)."""
    counts = ball_counts(df)
    p = counts / counts.sum()
    h = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    hmax = np.log2(len(counts))
    # G-test (razão de verosimilhanças) equivalente
    n = counts.sum()
    expected = n / len(counts)
    g = 2 * float((counts[counts > 0] * np.log(counts[counts > 0] / expected)).sum())
    return TestResult(
        name=f"Entropia {h:.5f} bits (máx {hmax:.5f}) — G-test",
        statistic=g,
        pvalue=float(stats.chi2.sf(g, len(counts) - 1)),
        dof=len(counts) - 1,
        detail=f"défice de entropia = {hmax - h:.6f} bits",
    )


# ---------------------------------------------------------------------------
# Deteção de viés localizado no tempo (equipamento a degradar-se)
# ---------------------------------------------------------------------------

def sliding_bias_scan(
    df: pd.DataFrame, window: int = 200, step: int = 25
) -> pd.DataFrame:
    """
    Varre o histórico com uma janela deslizante à procura de períodos em que
    a uniformidade falha.

    É assim que se apanharia uma máquina ou um conjunto de bolas a
    degradar-se: o viés não seria visível no agregado de 20 anos, apenas
    num troço. Esta é a versão honesta da caça ao padrão — e por isso vem
    com correção FDR sobre todas as janelas testadas.
    """
    rows = []
    for start in range(0, len(df) - window + 1, step):
        sub = df.iloc[start : start + window]
        res = chi_square_uniform(ball_counts(sub), "janela")
        rows.append(
            {
                "inicio": sub["date"].iloc[0],
                "fim": sub["date"].iloc[-1],
                "n": len(sub),
                "chi2": res.statistic,
                "p": res.pvalue,
            }
        )
    out = pd.DataFrame(rows)
    if len(out):
        out["p_ajustado"] = benjamini_hochberg(out["p"].to_numpy())
        out["significativo_fdr5"] = out["p_ajustado"] < 0.05
    return out


# ---------------------------------------------------------------------------
# O teste decisivo: persistência fora da amostra
# ---------------------------------------------------------------------------

def _deviation_vector(df: pd.DataFrame) -> np.ndarray:
    """Desvio relativo de cada número face ao esperado, neste troço."""
    counts = ball_counts(df).astype(float)
    expected = len(df) * config.MAIN_PICK / config.MAIN_POOL
    return (counts - expected) / expected


def out_of_sample_persistence(
    df: pd.DataFrame, split: float = 0.5, n_sim: int = 20_000, seed: int = 7
) -> dict:
    """
    A pergunta que decide tudo: os desvios do passado continuam no futuro?

    Um viés físico (bola gasta, câmara desequilibrada) é uma propriedade
    persistente do equipamento — tem de aparecer nas duas metades do
    histórico e correlacionar-se entre elas. Ruído amostral não faz isso:
    por construção, os desvios de duas metades independentes têm
    correlação zero.

    Mede-se a correlação entre os vetores de desvio das duas metades e
    compara-se com a distribuição nula obtida por simulação de sorteios
    genuinamente uniformes. Este teste é imune ao problema de seleção
    ("escolhi o número mais extremo dos 50"), porque avalia os 50 de uma
    vez e valida-se contra a sua própria distribuição nula.
    """
    cut = int(len(df) * split)
    train, test = df.iloc[:cut], df.iloc[cut:]
    dev_train = _deviation_vector(train)
    dev_test = _deviation_vector(test)
    r_obs = float(np.corrcoef(dev_train, dev_test)[0, 1])

    rng = np.random.default_rng(seed)
    pool, picks = config.MAIN_POOL, config.MAIN_PICK
    n1, n2 = len(train), len(test)
    exp1 = n1 * picks / pool
    exp2 = n2 * picks / pool

    sims = np.empty(n_sim)
    for i in range(n_sim):
        c1 = np.zeros(pool)
        c2 = np.zeros(pool)
        # Amostragem sem reposição dentro de cada sorteio, sorteios independentes.
        idx1 = rng.random((n1, pool)).argsort(axis=1)[:, :picks]
        idx2 = rng.random((n2, pool)).argsort(axis=1)[:, :picks]
        c1 = np.bincount(idx1.ravel(), minlength=pool).astype(float)
        c2 = np.bincount(idx2.ravel(), minlength=pool).astype(float)
        sims[i] = np.corrcoef((c1 - exp1) / exp1, (c2 - exp2) / exp2)[0, 1]

    p = float((np.abs(sims) >= abs(r_obs)).mean())
    return {
        "r_observado": r_obs,
        "p_simulado": p,
        "n_treino": n1,
        "n_teste": n2,
        "nulo_media": float(sims.mean()),
        "nulo_dp": float(sims.std()),
        "corte": train["date"].iloc[-1],
    }


def forward_selection_test(
    df: pd.DataFrame, split: float = 0.5, k: int = 5
) -> pd.DataFrame:
    """
    Simula a estratégia ingénua: escolher os k números "mais quentes" (e os
    mais frios) na primeira metade e ver como se portam na segunda.

    É o que qualquer sistema de padrões faz implicitamente. Aqui mede-se o
    resultado com honestidade, em vez de se assumir que funciona.
    """
    cut = int(len(df) * split)
    train, test = df.iloc[:cut], df.iloc[cut:]
    dev_train = _deviation_vector(train)
    dev_test = _deviation_vector(test)
    order = np.argsort(dev_train)
    coldest = order[:k]
    hottest = order[-k:]
    rows = []
    for label, sel in (("mais quentes no treino", hottest), ("mais frios no treino", coldest)):
        rows.append(
            {
                "seleção": label,
                "números": ", ".join(str(i + 1) for i in sorted(sel)),
                "desvio_treino_%": round(100 * dev_train[sel].mean(), 2),
                "desvio_teste_%": round(100 * dev_test[sel].mean(), 2),
                "manteve_direção": bool(
                    np.sign(dev_train[sel].mean()) == np.sign(dev_test[sel].mean())
                ),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Potência: que tamanho de brecha é que estes dados conseguiriam ver?
# ---------------------------------------------------------------------------

def detection_power(
    n_draws: int, effect_pct: float, pool: int = config.MAIN_POOL,
    picks: int = config.MAIN_PICK, alpha: float = 0.05,
) -> float:
    """
    Potência para detetar uma bola com probabilidade `effect_pct`% acima do
    normal, usando um teste binomial bilateral.
    """
    p0 = picks / pool
    p1 = p0 * (1 + effect_pct / 100)
    se0 = np.sqrt(p0 * (1 - p0) / n_draws)
    se1 = np.sqrt(p1 * (1 - p1) / n_draws)
    zcrit = stats.norm.isf(alpha / 2)
    return float(
        stats.norm.sf((zcrit * se0 - (p1 - p0)) / se1)
        + stats.norm.cdf((-zcrit * se0 - (p1 - p0)) / se1)
    )


def minimum_detectable_bias(
    n_draws: int, power: float = 0.80, alpha: float = 0.05, **kw
) -> float:
    """Menor viés (em %) detetável com a potência pedida. Busca binária."""
    lo, hi = 0.01, 500.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if detection_power(n_draws, mid, alpha=alpha, **kw) < power:
            lo = mid
        else:
            hi = mid
    return hi


def edge_value(bias_pct: float, star_pool: int = 12) -> float:
    """
    Se uma bola fosse `bias_pct`% mais provável, quanto valeria isso?

    Traduz um viés físico em vantagem sobre o preço da aposta, para se
    perceber se a brecha, mesmo a existir, pagaria a despesa.
    """
    p_jackpot = config.JACKPOT.probability(star_pool)
    # Jogar 5 bolas enviesadas multiplica a probabilidade por (1+b)^5.
    lift = (1 + bias_pct / 100) ** config.MAIN_PICK
    return lift - 1.0


# ---------------------------------------------------------------------------
# Relatório completo
# ---------------------------------------------------------------------------

def full_battery(df: pd.DataFrame) -> dict:
    """Corre tudo e devolve resultados estruturados."""
    results: dict = {"global": [], "por_era": {}, "tabelas": {}}

    results["global"].append(chi_square_uniform(ball_counts(df), "números 1-50 (todo o histórico)"))
    results["global"].append(runs_test(df))
    results["global"].append(pair_cooccurrence(df))
    results["global"].append(sum_distribution_test(df))
    results["global"].append(entropy_test(df))
    results["global"].extend(positional_test(df))

    for era in config.ERAS:
        sub = df[df["era"] == era.name]
        if len(sub) < 30:
            continue
        era_res = [
            chi_square_uniform(ball_counts(sub), f"números — era {era.name}"),
            chi_square_uniform(star_counts(sub, era.star_pool), f"estrelas 1-{era.star_pool} — era {era.name}"),
        ]
        results["por_era"][era.name] = era_res

    results["tabelas"]["bolas"] = per_ball_binomial(
        ball_counts(df), len(df), config.MAIN_PICK
    )
    results["tabelas"]["gaps"] = gap_analysis(df)
    results["tabelas"]["autocorr"] = serial_correlation(df)
    results["tabelas"]["janelas"] = sliding_bias_scan(df)
    return results
