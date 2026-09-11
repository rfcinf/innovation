"""
Simulação completa de uma noite de sorteio.

O valor esperado é um número só, e é o número que menos informa. Diz
quanto se ganharia *em média* se a mesma noite se repetisse um milhão de
vezes — mas a noite acontece uma vez, e a média cai num ponto onde a
distribuição quase nunca aterra. Com uma carteira de 5 apostas, o valor
esperado é ~€6,7 e a probabilidade de ganhar exatamente €6,7 é zero: o
resultado real é €0 em 64% das noites, €3-€10 em quase todo o resto, e
uma cauda que vale milhões com probabilidade de 1 em dezenas de milhões.

Este módulo devolve a distribuição toda, em três partes:

  CORPO    — por simulação de Monte Carlo. É onde a noite realmente
             aterra: P(nada), percentis, P(recuperar o custo). Os
             bilhetes partilham o mesmo sorteio e portanto estão
             correlacionados; a simulação trata isso exatamente, uma
             fórmula de independência não trataria.

  CAUDA    — por combinatória exata. Os escalões de 5 acertos têm
             probabilidade de 1 em milhões: uma simulação de 200 mil
             noites nunca os vê, e se os visse uma vez seria ruído. O
             número de acertos esperados por escalão é n × p por
             linearidade da esperança — exato, sem supor independência
             entre bilhetes.

  CHEQUE   — o que a cauda paga depois de partilha e imposto. O jackpot
             bruto não é o que se recebe: divide-se por 1+K com K
             ~ Poisson(λ), e o Imposto do Selo leva 20% acima de €5.000.

UM DETALHE QUE MUDA A CONTA
---------------------------
Para escalões de 3 ou mais acertos, dois bilhetes com números disjuntos
NÃO PODEM ganhar os dois: seriam precisos 6 números sorteados e só saem
5. Por isso, nesses escalões, P(pelo menos um bilhete acerta) = n × p
exatamente, e não 1-(1-p)^n. A diferença é minúscula em valor absoluto e
é gratuita em rigor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, ev

# Acima de 5 acertos de 5, dois bilhetes disjuntos não podem ganhar ambos.
_DISJOINT_EXCLUSIVE_FROM = 3


# ---------------------------------------------------------------------------
# O cheque: partilha pari-mutuel + Imposto do Selo
# ---------------------------------------------------------------------------

def jackpot_check(jackpot_eur: float, lam: float, k_max: int = 80) -> dict:
    """
    Quanto se recebe, líquido, ao acertar no jackpot.

    Não é `jackpot_eur`. É E[líquido(J / (1+K))] com K ~ Poisson(λ), onde
    λ = vendas × p_jackpot × popularidade. O imposto é aplicado *depois*
    da divisão porque é sobre o que cada vencedor recebe — aplicá-lo
    antes sobrestimaria a carga fiscal quando o prémio se divide.
    """
    from scipy import stats

    k = np.arange(k_max + 1)
    w = stats.poisson.pmf(k, lam)
    # Trunca-se em k_max; renormalizar devolve a massa perdida à cauda
    # comprimida. Para os λ que interessam (< 1) a massa perdida é ~1e-100.
    w = w / w.sum()

    bruto = jackpot_eur / (1.0 + k)
    liquido = np.array([config.net_prize(float(g)) for g in bruto])
    return {
        "lambda": float(lam),
        "p_sozinho": float(np.exp(-lam)),
        "bruto_se_sozinho": float(jackpot_eur),
        "liquido_se_sozinho": float(config.net_prize(jackpot_eur)),
        "cheque_esperado": float((w * liquido).sum()),
        "vencedores_esperados_alem_de_nos": float(lam),
    }


# ---------------------------------------------------------------------------
# Matriz de prémios indexada por (acertos, estrelas)
# ---------------------------------------------------------------------------

def prize_matrix(tier_prizes: dict[str, float], jackpot_value: float) -> np.ndarray:
    P = np.zeros((config.MAIN_PICK + 1, config.STAR_PICK + 1))
    for t in config.TIERS:
        P[t.mains, t.stars] = float(tier_prizes.get(t.label, 0.0))
    P[config.MAIN_PICK, config.STAR_PICK] = float(jackpot_value)
    return P


# ---------------------------------------------------------------------------
# O corpo: Monte Carlo vetorizado
# ---------------------------------------------------------------------------

def simulate_night(
    tickets: list[tuple[list[int], list[int]]],
    prize_mat: np.ndarray,
    n_sim: int = 400_000,
    star_pool: int = 12,
    seed: int = 7,
    chunk: int = 25_000,
) -> dict:
    """
    Simula `n_sim` noites e devolve a distribuição empírica do ganho.

    Vetorizado por multiplicação de matrizes booleanas: em vez de comparar
    conjuntos bilhete a bilhete num ciclo de Python, constrói a matriz de
    pertença dos sorteios (n_sim × 50) e multiplica pela dos bilhetes
    (n_bilhetes × 50). O produto dá diretamente o nº de acertos de cada
    bilhete em cada noite.
    """
    rng = np.random.default_rng(seed)
    n_t = len(tickets)

    T_main = np.zeros((n_t, config.MAIN_POOL), dtype=np.uint8)
    T_star = np.zeros((n_t, star_pool), dtype=np.uint8)
    for j, (m, s) in enumerate(tickets):
        T_main[j, np.array(m) - 1] = 1
        T_star[j, np.array(s) - 1] = 1

    custo = n_t * config.TICKET_PRICE_EUR
    ganhos = np.empty(n_sim, dtype=np.float64)
    n_nada = 0
    feito = 0

    while feito < n_sim:
        b = min(chunk, n_sim - feito)

        # b sorteios de 5 em 50 e 2 em `star_pool`, sem reposição
        idx_m = np.argpartition(rng.random((b, config.MAIN_POOL)),
                                config.MAIN_PICK, axis=1)[:, :config.MAIN_PICK]
        idx_s = np.argpartition(rng.random((b, star_pool)),
                                config.STAR_PICK, axis=1)[:, :config.STAR_PICK]

        D_main = np.zeros((b, config.MAIN_POOL), dtype=np.uint8)
        D_star = np.zeros((b, star_pool), dtype=np.uint8)
        np.put_along_axis(D_main, idx_m, 1, axis=1)
        np.put_along_axis(D_star, idx_s, 1, axis=1)

        km = D_main @ T_main.T          # (b, n_t) acertos de números
        ks = D_star @ T_star.T          # (b, n_t) acertos de estrelas

        pago = prize_mat[km, ks]        # (b, n_t) prémio de cada bilhete
        total = pago.sum(axis=1)

        ganhos[feito:feito + b] = total
        n_nada += int((total == 0).sum())
        feito += b

    p_nada = n_nada / n_sim
    lucro = ganhos - custo

    def p_ge(x: float) -> float:
        return float((ganhos >= x).mean())

    ganhou = ganhos[ganhos > 0]

    return {
        "n_bilhetes": n_t,
        "custo": custo,
        "n_sim": n_sim,
        "p_nada": p_nada,
        "p_nada_se": float(np.sqrt(p_nada * (1 - p_nada) / n_sim)),
        "p_algum_premio": 1 - p_nada,
        "p_recupera_custo": p_ge(custo),
        "p_ge_25": p_ge(25.0),
        "p_ge_100": p_ge(100.0),
        "p_ge_1000": p_ge(1000.0),
        "ganho_medio": float(ganhos.mean()),
        "ganho_medio_se": float(ganhos.std(ddof=1) / np.sqrt(n_sim)),
        "lucro_medio": float(lucro.mean()),
        "percentis": {q: float(np.quantile(ganhos, q / 100))
                      for q in (50, 60, 70, 75, 80, 90, 95, 99, 99.9)},
        "dado_que_ganha": {
            "n": int(ganhou.size),
            "media": float(ganhou.mean()) if ganhou.size else 0.0,
            "mediana": float(np.median(ganhou)) if ganhou.size else 0.0,
            "p90": float(np.quantile(ganhou, 0.90)) if ganhou.size else 0.0,
            "p99": float(np.quantile(ganhou, 0.99)) if ganhou.size else 0.0,
            "max": float(ganhou.max()) if ganhou.size else 0.0,
        },
        "_ganhos": ganhos,
    }


# ---------------------------------------------------------------------------
# A cauda: combinatória exata
# ---------------------------------------------------------------------------

def tier_table(
    n_tickets: int,
    tier_prizes: dict[str, float],
    jackpot_value: float,
    star_pool: int = 12,
) -> pd.DataFrame:
    """
    Cada escalão, com a probabilidade exata e o que paga.

    A coluna `1_em` é 1/(n × p): o número esperado de acertos é exato por
    linearidade, sem supor nada sobre a correlação entre bilhetes. Para
    escalões de 3+ acertos com bilhetes disjuntos é também exatamente a
    probabilidade, porque dois bilhetes não podem lá chegar na mesma noite.
    """
    linhas = []
    for t in sorted(config.TIERS, key=lambda x: -(x.mains * 10 + x.stars)):
        p = t.probability(star_pool)
        esperado = n_tickets * p
        premio = (jackpot_value if (t.mains, t.stars) == (config.MAIN_PICK, config.STAR_PICK)
                  else float(tier_prizes.get(t.label, 0.0)))
        linhas.append({
            "escalão": t.label,
            "1_em": round(1.0 / esperado, 1) if esperado else float("inf"),
            "prob_%": 100 * esperado,
            "prémio_€": premio,
            "contrib_EV_€": esperado * premio,
            "exclusivo": t.mains >= _DISJOINT_EXCLUSIVE_FROM,
        })
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
# A noite inteira, enquadrada
# ---------------------------------------------------------------------------

def frame_night(
    tickets: list,
    jackpot_eur: float,
    sales: float,
    tier_prizes: dict[str, float],
    ev_m1lhao: float = 0.0,
    star_pool: int = 12,
    n_sim: int = 400_000,
    seed: int = 7,
    popularities: list[float] | None = None,
) -> dict:
    """
    Junta corpo, cauda e cheque para uma carteira concreta numa noite
    concreta.

    `tickets` pode ser uma lista de opt.Ticket (que já traz a popularidade
    medida) ou de pares (números, estrelas). No segundo caso é OBRIGATÓRIO
    passar `popularities`, ou assume-se 1.0 — e 1.0 não é um valor neutro:
    é a popularidade média do mercado, que faz o cheque do jackpot descer
    ~5% em relação a uma carteira otimizada. Deixar isto por omissão numa
    comparação faz a carteira parecer pior do que é.
    """
    pares, pops = [], []
    for i, t in enumerate(tickets):
        if hasattr(t, "mains"):
            pares.append((list(t.mains), list(t.stars)))
            pops.append(float(t.popularity))
        else:
            pares.append((list(t[0]), list(t[1])))
            pops.append(1.0 if popularities is None else float(popularities[i]))

    n = len(pares)
    p_jack = config.TIER_BY_LABEL["5+2"].probability(star_pool)

    cheques = [jackpot_check(jackpot_eur, ev.jackpot_lambda(sales, pi, star_pool))
               for pi in pops]
    cheque_medio = float(np.mean([c["cheque_esperado"] for c in cheques]))
    p_solo_medio = float(np.mean([c["p_sozinho"] for c in cheques]))

    P = prize_matrix(tier_prizes, cheque_medio)
    corpo = simulate_night(pares, P, n_sim=n_sim, star_pool=star_pool, seed=seed)

    tab = tier_table(n, tier_prizes, cheque_medio, star_pool)
    ev_total = float(tab["contrib_EV_€"].sum()) + n * ev_m1lhao

    return {
        "n_bilhetes": n,
        "custo": n * config.TICKET_PRICE_EUR,
        "jackpot_eur": jackpot_eur,
        "popularidade_media": float(np.mean(pops)),
        "corpo": corpo,
        "escaloes": tab,
        "cheque": {
            "lambda_medio": float(np.mean([c["lambda"] for c in cheques])),
            "p_sozinho_dado_que_acerta": p_solo_medio,
            "liquido_se_sozinho": float(config.net_prize(jackpot_eur)),
            "cheque_esperado": cheque_medio,
            "p_jackpot_carteira": n * p_jack,
            "1_em_jackpot": 1.0 / (n * p_jack),
            "1_em_jackpot_sozinho": 1.0 / (n * p_jack * p_solo_medio),
        },
        "ev_total": ev_total,
        "ev_m1lhao": n * ev_m1lhao,
        "retorno_por_euro": ev_total / (n * config.TICKET_PRICE_EUR),
    }


def concentrated_portfolio(
    base_mains: list[int],
    base_stars: list[int],
    n_tickets: int,
    popularity_model=None,
    star_pool: int = 12,
) -> list[tuple[list[int], list[int]]]:
    """
    A carteira do extremo oposto: todos os bilhetes partilham quase todos os
    números, variando um só.

    NÃO é a carteira recomendada por omissão, e é importante perceber
    porquê — e porque mesmo assim existe.

    Dispersar (25 números distintos) e concentrar (9) são objetivos em
    OPOSIÇÃO DIRETA. Medido em 400 mil noites, 5 apostas, jackpot €111M:

                            dispersa   concentrada
        P(não ganhar nada)    0,6446        0,8302
        P(recuperar o custo)  0,0142        0,0540
        P(ganhar >= EUR 25)   0,0012        0,0212
        percentil 99          EUR 14,18     EUR 31,26
        valor esperado        EUR 6,40      EUR 6,44

    Concentrar piora a probabilidade de sair de mãos vazias em 19 pontos
    e melhora por um fator de ~4 a probabilidade de a noite pagar o
    bilhete. A razão é a correlação: se os números partilhados saírem,
    VÁRIOS bilhetes ganham ao mesmo tempo e os prémios somam-se; se não
    saírem, nenhum ganha.

    O valor esperado é praticamente idêntico nas duas — a esperança é
    linear e não vê correlação. Só muda a forma da distribuição. Qual das
    duas é "melhor" não é uma questão estatística: depende de se preferir
    ganhar pouco muitas vezes ou raramente e mais de cada vez.
    """
    base = list(base_mains)
    pool = [x for x in range(1, config.MAIN_POOL + 1) if x not in base]

    if popularity_model is not None:
        cand = []
        for pos in range(config.MAIN_PICK):
            for novo in pool:
                t = sorted(base[:pos] + [novo] + base[pos + 1:])
                pi = float(popularity_model.predict_popularity(np.array(t))[0])
                cand.append((pi, t))
        cand.sort(key=lambda x: x[0])
        escolhidos, vistos = [], {tuple(sorted(base))}
        for _, t in cand:
            if tuple(t) not in vistos:
                escolhidos.append(t)
                vistos.add(tuple(t))
            if len(escolhidos) >= n_tickets - 1:
                break
    else:
        escolhidos = []
        for i in range(n_tickets - 1):
            t = base.copy()
            t[i % config.MAIN_PICK] = pool[i % len(pool)]
            escolhidos.append(sorted(t))

    return [(sorted(base), list(base_stars))] + [
        (t, list(base_stars)) for t in escolhidos[: n_tickets - 1]
    ]


def sweep(
    carteiras: dict[str, list],
    jackpot_eur: float,
    sales: float,
    tier_prizes: dict[str, float],
    ev_m1lhao: float = 0.0,
    star_pool: int = 12,
    n_sim: int = 200_000,
    seed: int = 7,
    popularities: dict[str, list[float]] | None = None,
) -> pd.DataFrame:
    """Compara carteiras de tamanhos diferentes na mesma noite."""
    linhas = []
    pops_por_carteira = popularities or {}
    for nome, tks in carteiras.items():
        r = frame_night(tks, jackpot_eur, sales, tier_prizes, ev_m1lhao,
                        star_pool, n_sim=n_sim, seed=seed,
                        popularities=pops_por_carteira.get(nome))
        c = r["corpo"]
        linhas.append({
            "carteira": nome,
            "apostas": r["n_bilhetes"],
            "custo_€": r["custo"],
            "nºs": int(len(set().union(*[set(p[0]) for p in
                                         [(t.mains, t.stars) if hasattr(t, "mains")
                                          else t for t in tks]]))),
            "popul.": round(r["popularidade_media"], 3),
            "P(nada)": round(c["p_nada"], 4),
            "P(recupera)": round(c["p_recupera_custo"], 4),
            "P(≥€100)": round(c["p_ge_100"], 5),
            "1_em_jackpot": round(r["cheque"]["1_em_jackpot"] / 1e6, 1),
            "1_em_sozinho": round(r["cheque"]["1_em_jackpot_sozinho"] / 1e6, 1),
            "cheque_€M": round(r["cheque"]["cheque_esperado"] / 1e6, 1),
            "EV_€": round(r["ev_total"], 2),
            "€_por_€": round(r["retorno_por_euro"], 3),
        })
    return pd.DataFrame(linhas)
