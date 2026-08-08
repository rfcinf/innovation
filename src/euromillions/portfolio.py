"""
Construção de carteira: reduzir ao máximo a probabilidade de não ganhar nada.

A SEGUNDA ALAVANCA
------------------
`popularity.py` e `ev.py` atacam o *tamanho* do prémio. Este módulo ataca
uma coisa diferente e independente: a **probabilidade de sair de mãos a
abanar**.

Uma aposta isolada tem P(ganhar alguma coisa) ≈ 1/13,06 = 7,66%. Para N
apostas, o resultado depende de como elas se relacionam entre si, e isto é
onde quase toda a gente perde valor sem dar por isso:

* Bilhetes que **partilham números** falham em conjunto. Se o sorteio evita
  a zona que ambos cobrem, ambos perdem. É correlação positiva.
* Bilhetes **disjuntos** cobrem mais do universo. Quando um falha, o outro
  tem mais hipóteses de acertar. É correlação negativa.

O efeito é real e grátis: não custa um cêntimo a mais escolher 25 números
distintos em vez de 5 bilhetes sobrepostos. E, ao contrário da vantagem de
popularidade, esta não depende de modelo nenhum — é combinatória pura,
verificável por simulação direta.

NOTA IMPORTANTE SOBRE O QUE ISTO NÃO É
--------------------------------------
Reduzir P(não ganhar nada) **não aumenta o valor esperado**. O EV de N
apostas é N × EV(1 aposta), independentemente de como se escolhem. O que
muda é a *distribuição*: menos variância, prémios pequenos mais frequentes.

Quem quiser maximizar a hipótese de tocar no jackpot deve ignorar este
módulo. Quem quiser que €12,50 não desapareçam sem nada com tanta
frequência, ganha aqui — e o ganho é mensurável.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# Avaliação por simulação
# ---------------------------------------------------------------------------

def _prize_tier(n_main: int, n_star: int) -> str | None:
    label = f"{n_main}+{n_star}"
    return label if label in config.TIER_BY_LABEL else None


def simulate_portfolio(
    tickets: list[tuple[list[int], list[int]]],
    n_sim: int = 200_000,
    star_pool: int = 12,
    tier_prizes: dict[str, float] | None = None,
    seed: int = 42,
) -> dict:
    """
    Simula `n_sim` sorteios e mede o comportamento real da carteira.

    Não usa aproximações nem independência assumida: sorteia, compara,
    conta. É lento comparado com uma fórmula, e é a única forma de acertar
    quando os bilhetes estão correlacionados entre si.
    """
    rng = np.random.default_rng(seed)
    prizes = dict(tier_prizes or {})

    main_sets = [set(t[0]) for t in tickets]
    star_sets = [set(t[1]) for t in tickets]
    n_t = len(tickets)

    n_nothing = 0
    winnings = np.zeros(n_sim)
    tier_hits: dict[str, int] = {}

    pool_main = np.arange(1, config.MAIN_POOL + 1)
    pool_star = np.arange(1, star_pool + 1)

    for i in range(n_sim):
        drawn_m = set(rng.choice(pool_main, config.MAIN_PICK, replace=False).tolist())
        drawn_s = set(rng.choice(pool_star, config.STAR_PICK, replace=False).tolist())
        total = 0.0
        won_any = False
        for j in range(n_t):
            k = len(main_sets[j] & drawn_m)
            s = len(star_sets[j] & drawn_s)
            label = _prize_tier(k, s)
            if label is not None:
                won_any = True
                tier_hits[label] = tier_hits.get(label, 0) + 1
                total += prizes.get(label, 0.0)
        if not won_any:
            n_nothing += 1
        winnings[i] = total

    cost = n_t * config.TICKET_PRICE_EUR
    p_nada = n_nothing / n_sim
    return {
        "n_bilhetes": n_t,
        "numeros_distintos": len(set().union(*main_sets)) if main_sets else 0,
        "p_nada": p_nada,
        "p_nada_se": float(np.sqrt(p_nada * (1 - p_nada) / n_sim)),
        "p_algum_premio": 1 - p_nada,
        "ganho_medio_eur": float(winnings.mean()),
        # O erro-padrão da média é enorme e tem de ser mostrado. A média dos
        # ganhos é dominada por escalões raríssimos (5+1 vale €300k com
        # p = 1/7.000.000): um único acerto desses numa simulação de 150 mil
        # sorteios desloca a média em €2. Sem esta coluna, o utilizador leria
        # ruído como se fosse diferença entre carteiras — e concluiria, ao
        # contrário da teoria, que uma carteira sobreposta "rende mais".
        "ganho_medio_se": float(winnings.std(ddof=1) / np.sqrt(n_sim)),
        "ganho_mediano_eur": float(np.median(winnings)),
        "custo_eur": cost,
        "escaloes": tier_hits,
        "n_sim": n_sim,
    }


def expected_winnings_analytic(
    n_tickets: int, tier_prizes: dict[str, float], star_pool: int = 12
) -> float:
    """
    Ganho médio exato, por cálculo em vez de simulação.

    O valor esperado de N apostas é sempre N × EV(1 aposta), qualquer que
    seja a sobreposição entre elas — a esperança é linear, e a correlação
    entre bilhetes não a altera. Serve de referência para detetar quando a
    média simulada está apenas a exibir ruído de cauda.
    """
    per_ticket = sum(
        t.probability(star_pool) * tier_prizes.get(t.label, 0.0) for t in config.TIERS
    )
    return n_tickets * per_ticket


def probability_no_prize_independent(n_tickets: int, star_pool: int = 12) -> float:
    """
    Referência teórica: P(nada) se os N bilhetes fossem independentes.

    Comparar a simulação com este valor mostra exatamente quanto é que a
    estrutura da carteira ganha (ou perde) face ao acaso puro.
    """
    p_win = config.probability_any_prize(star_pool)
    return float((1 - p_win) ** n_tickets)


# ---------------------------------------------------------------------------
# Construção
# ---------------------------------------------------------------------------

def coverage_score(tickets: list[tuple[list[int], list[int]]]) -> dict:
    """Diagnóstico da sobreposição de uma carteira."""
    mains = [set(t[0]) for t in tickets]
    distinct = set().union(*mains) if mains else set()
    overlaps = [
        len(mains[i] & mains[j])
        for i in range(len(mains))
        for j in range(i + 1, len(mains))
    ]
    return {
        "numeros_distintos": len(distinct),
        "maximo_possivel": min(len(tickets) * config.MAIN_PICK, config.MAIN_POOL),
        "cobertura_pct": round(100 * len(distinct) / config.MAIN_POOL, 1),
        "sobreposicao_media": round(float(np.mean(overlaps)), 3) if overlaps else 0.0,
        "sobreposicao_maxima": max(overlaps) if overlaps else 0,
    }


def build_disjoint_portfolio(
    candidates: pd.DataFrame,
    combos: np.ndarray,
    star_combos: list[list[int]],
    n_tickets: int = 5,
    popularity_quantile: float = 0.25,
) -> list[int]:
    """
    Escolhe uma carteira que é simultaneamente impopular e bem dispersa.

    Estratégia: restringe-se primeiro ao quartil mais impopular dos
    candidatos (é aí que está o valor), e só depois se otimiza a cobertura
    dentro desse conjunto — por seleção gulosa do bilhete que acrescenta
    mais números novos.

    A ordem importa. Otimizar cobertura primeiro e popularidade depois dá
    carteiras bem dispersas mas caras em partilha; o valor está na
    popularidade, a cobertura é o bónus.
    """
    thresh = candidates["popularidade"].quantile(popularity_quantile)
    eligible = candidates.index[candidates["popularidade"] <= thresh].tolist()
    if not eligible:
        eligible = candidates.index.tolist()

    chosen: list[int] = []
    used: set[int] = set()

    # Primeiro bilhete: o mais impopular de todos.
    eligible.sort(key=lambda i: candidates.loc[i, "popularidade"])
    chosen.append(eligible[0])
    used |= set(int(x) for x in combos[eligible[0]])

    while len(chosen) < n_tickets:
        best, best_key = None, None
        for i in eligible:
            if i in chosen:
                continue
            new = len(set(int(x) for x in combos[i]) - used)
            # Desempate por popularidade: entre bilhetes que acrescentam a
            # mesma cobertura, fica o mais impopular.
            key = (new, -float(candidates.loc[i, "popularidade"]))
            if best_key is None or key > best_key:
                best, best_key = i, key
        if best is None:
            break
        chosen.append(best)
        used |= set(int(x) for x in combos[best])

    return chosen


# ---------------------------------------------------------------------------
# Comparação de estratégias de carteira
# ---------------------------------------------------------------------------

def compare_portfolios(
    portfolios: dict[str, list[tuple[list[int], list[int]]]],
    n_sim: int = 100_000,
    tier_prizes: dict[str, float] | None = None,
    star_pool: int = 12,
    seed: int = 42,
) -> pd.DataFrame:
    """Corre a simulação em cada carteira e devolve a tabela comparativa."""
    rows = []
    for name, tickets in portfolios.items():
        r = simulate_portfolio(
            tickets, n_sim=n_sim, star_pool=star_pool,
            tier_prizes=tier_prizes, seed=seed,
        )
        cov = coverage_score(tickets)
        rows.append(
            {
                "carteira": name,
                "nºs_distintos": cov["numeros_distintos"],
                "sobrep_média": cov["sobreposicao_media"],
                "P(nada)": round(r["p_nada"], 4),
                "±": round(1.96 * r["p_nada_se"], 4),
                "P(algum prémio)": round(r["p_algum_premio"], 4),
                "ganho_médio_€": round(r["ganho_medio_eur"], 2),
                "±_ganho": round(1.96 * r["ganho_medio_se"], 2),
            }
        )
    out = pd.DataFrame(rows)
    base = out["P(nada)"].max()
    out["redução_P(nada)_%"] = (100 * (base - out["P(nada)"]) / base).round(2)
    if tier_prizes:
        n_t = len(next(iter(portfolios.values())))
        out.attrs["ganho_medio_teorico"] = round(
            expected_winnings_analytic(n_t, tier_prizes, star_pool), 3
        )
    return out
