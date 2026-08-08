"""
Motor de valor esperado.

Junta as três forças que determinam quanto vale realmente uma aposta:

  (+) o jackpot acumulado — sobe a cada rollover, até ao teto de €250M;
  (-) a partilha pari-mutuel — quanto mais popular a combinação, mais o
      prémio se divide;
  (-) o Imposto do Selo português — 20% sobre a parcela acima de €5.000.

O utilizador pediu para usar os fatores negativos *com* os positivos. É
exatamente isto: a brecha não está em nenhum deles isoladamente, está no
ponto onde os três se cruzam de forma mais favorável.

A MATEMÁTICA DA PARTILHA
------------------------
Se S apostas são vendidas e a nossa combinação tem popularidade relativa
π, o nº de *outros* bilhetes vencedores do jackpot é aproximadamente

    K ~ Poisson(λ),  λ = S · p_jackpot · π

e o prémio esperado dado que ganhamos é

    E[J / (1+K)] = J · (1 - e^(-λ)) / λ

Esta é a alavanca. p_jackpot é intocável; λ não é, porque π depende da
combinação escolhida. Reduzir π de 3.0 para 0.5 não muda em nada a
probabilidade de ganhar — multiplica o cheque por um fator que, em
jackpots grandes, chega a ser superior a 2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config

JACKPOT_CAP_EUR = 250_000_000.0
"""Teto do jackpot em vigor. Acima disto o excedente escorre para os
escalões inferiores, o que limita quanto a acumulação pode compensar."""


# ---------------------------------------------------------------------------
# Partilha
# ---------------------------------------------------------------------------

def expected_share_factor(lam: float) -> float:
    """
    E[1/(1+K)] com K ~ Poisson(λ): a fração do bolo que esperamos receber.

    λ→0  => 1.0 (recebemos tudo)
    λ=1  => 0.632
    λ=5  => 0.199
    """
    if lam <= 1e-12:
        return 1.0
    return float((1.0 - np.exp(-lam)) / lam)


def jackpot_lambda(sales: float, popularity: float, star_pool: int) -> float:
    """Nº esperado de *outros* vencedores do jackpot."""
    return sales * config.JACKPOT.probability(star_pool) * popularity


# ---------------------------------------------------------------------------
# Prémios típicos dos escalões inferiores
# ---------------------------------------------------------------------------

def empirical_tier_prizes(breakdown: pd.DataFrame, since_year: int = 2020) -> dict[str, float]:
    """
    Prémio mediano observado por escalão (exceto jackpot), em euros.

    Usa-se a mediana e não a média porque a distribuição é fortemente
    assimétrica — uma única semana atípica distorce a média.
    """
    bd = breakdown.copy()
    bd["year"] = [d.year for d in bd["date"]]
    bd = bd[(bd["year"] >= since_year) & bd["prize_eur"].notna()]
    bd = bd[bd["prize_eur"] > 0]
    med = bd.groupby("tier")["prize_eur"].median()
    return {t: float(v) for t, v in med.items() if t != "5+2"}


# Valores de recurso, caso não haja dados descarregados (era atual).
FALLBACK_TIER_PRIZES: dict[str, float] = {
    "5+1": 300_000.0, "5+0": 25_000.0, "4+2": 2_500.0, "4+1": 160.0,
    "3+2": 80.0, "4+0": 45.0, "2+2": 20.0, "3+1": 14.0,
    "3+0": 11.0, "1+2": 9.0, "2+1": 7.0, "2+0": 4.0,
}


# ---------------------------------------------------------------------------
# Elasticidade da partilha por escalão
# ---------------------------------------------------------------------------

# Quanto é que a popularidade da nossa combinação afeta o nº de vencedores
# de cada escalão. Nos escalões de 5 números o efeito é total (é a mesma
# combinação exata); nos escalões baixos é diluído, porque acertar 2 ou 3
# números não fixa a combinação toda.
#
# ESTES VALORES SÃO MEDIDOS, NÃO ARBITRADOS.
#
# A versão anterior desta tabela (1,0 / 0,7 / 0,4 / 0,2 / 0,1) foi escrita à
# mão por analogia — o mesmo pecado que foi corrigido nas estrelas e que
# ficou aqui por corrigir. `elasticity.measure_elasticities()` estima-a
# regredindo log(vencedores do escalão) na popularidade PREVISTA da
# combinação sorteada, que depende apenas das suas características e não
# contém vencedores nem vendas estimadas — e portanto não sofre da
# correlação espúria que a popularidade observada introduziria.
#
# Validação do método: o escalão 5+0 mede 0,956 quando a teoria exige
# exatamente 1,0 (acertar os 5 números *é* ter a nossa combinação). Os
# escalões de 5 números ficam fixados em 1,0 por essa razão teórica; os
# restantes usam a medição.
#
# O resultado é que a tabela anterior estava sistematicamente inflacionada,
# sobrestimando o EV da combinação otimizada em ~4,5%.
#
# Os escalões de 2 números medem elasticidade nula, e o 1+2 mede-a negativa
# (-0,24, p = 5e-5). Não é ruído: é um efeito de composição. Quando a
# combinação sorteada é popular, os bilhetes populares tendem a acertar
# *mais* números, e a massa desloca-se dos escalões baixos para os altos —
# esvaziando-os.
TIER_ELASTICITY: dict[str, float] = {
    "5+2": 1.000, "5+1": 1.000, "5+0": 1.000,   # teoria, confirmada por 5+0 = 0,956
    "4+2": 0.523, "4+1": 0.572, "4+0": 0.601,
    "3+2": 0.193, "3+1": 0.234, "3+0": 0.258,
    "2+2": 0.000, "2+1": 0.000, "2+0": 0.009,
    "1+2": 0.000,
}

TIER_ELASTICITY_HANDMADE: dict[str, float] = {
    "5+2": 1.00, "5+1": 1.00, "5+0": 1.00,
    "4+2": 0.70, "4+1": 0.70, "4+0": 0.70,
    "3+2": 0.40, "3+1": 0.40, "3+0": 0.40,
    "2+2": 0.20, "2+1": 0.20, "2+0": 0.20,
    "1+2": 0.10,
}
"""Guardada para comparação — é o que o sistema usava antes de medir."""


@dataclass
class EVResult:
    jackpot_eur: float
    sales: float
    popularity: float
    star_pool: int
    ev_bruto: float
    ev_liquido: float
    ev_jackpot: float
    ev_inferiores: float
    custo: float
    retorno_por_euro: float
    lambda_jackpot: float
    fator_partilha: float
    ev_m1lhao: float = 0.0

    def __str__(self) -> str:
        linha_m1 = (
            f"EV M1lhão        €{self.ev_m1lhao:.4f}\n" if self.ev_m1lhao else ""
        )
        return (
            f"Jackpot          €{self.jackpot_eur:,.0f}\n"
            f"Popularidade     {self.popularity:.3f}x\n"
            f"λ (concorrentes) {self.lambda_jackpot:.2f}  "
            f"→ recebemos {100*self.fator_partilha:.1f}% do bolo se ganharmos\n"
            f"EV jackpot       €{self.ev_jackpot:.4f}\n"
            f"EV outros esc.   €{self.ev_inferiores:.4f}\n"
            + linha_m1 +
            f"EV líquido total €{self.ev_liquido:.4f}  (custo €{self.custo:.2f})\n"
            f"Retorno por euro €{self.retorno_por_euro:.4f}"
        )


def expected_value(
    jackpot_eur: float,
    sales: float,
    popularity: float = 1.0,
    star_pool: int = 12,
    tier_prizes: dict[str, float] | None = None,
    apply_tax: bool = True,
    ev_m1lhao: float = 0.0,
) -> EVResult:
    """
    Valor esperado líquido de uma aposta, em euros.

    `popularity` é o multiplicador devolvido pelo modelo de popularidade:
    1.0 = combinação banal, 0.4 = combinação que quase ninguém joga.

    `ev_m1lhao` é a parcela do sorteio português do M1lhão (ver
    `m1lhao.expected_value_per_bet`). É uma constante por aposta: o código é
    gerado pelo sistema, ninguém o escolhe, e por isso não há popularidade
    nem partilha a otimizar. Some-se sempre que se queira o valor real de um
    bilhete comprado em Portugal — omiti-la subestima o retorno em ~25%.
    """
    prizes = dict(FALLBACK_TIER_PRIZES)
    if tier_prizes:
        prizes.update(tier_prizes)

    jackpot_eur = min(jackpot_eur, JACKPOT_CAP_EUR)

    lam = jackpot_lambda(sales, popularity, star_pool)
    share = expected_share_factor(lam)
    p_jack = config.JACKPOT.probability(star_pool)

    gross_jackpot = jackpot_eur * share
    net_jackpot = config.net_prize(gross_jackpot) if apply_tax else gross_jackpot
    ev_jackpot = p_jack * net_jackpot

    ev_lower = 0.0
    for tier in config.TIERS:
        if tier.label == "5+2":
            continue
        base = prizes.get(tier.label)
        if base is None:
            continue
        # Escalões inferiores também são pari-mutuel: uma combinação
        # impopular recebe uma fatia maior do bolo do escalão.
        elasticity = TIER_ELASTICITY.get(tier.label, 0.5)
        adj = popularity ** (-elasticity)
        gross = base * adj
        net = config.net_prize(gross) if apply_tax else gross
        ev_lower += tier.probability(star_pool) * net

    ev_gross_total = ev_jackpot + ev_lower + ev_m1lhao
    cost = config.TICKET_PRICE_EUR
    return EVResult(
        jackpot_eur=jackpot_eur,
        sales=sales,
        popularity=popularity,
        star_pool=star_pool,
        ev_bruto=p_jack * gross_jackpot + ev_lower + ev_m1lhao,
        ev_liquido=ev_gross_total,
        ev_jackpot=ev_jackpot,
        ev_inferiores=ev_lower,
        custo=cost,
        retorno_por_euro=ev_gross_total / cost,
        lambda_jackpot=lam,
        fator_partilha=share,
        ev_m1lhao=ev_m1lhao,
    )


# ---------------------------------------------------------------------------
# O ponto de equilíbrio
# ---------------------------------------------------------------------------

def breakeven_jackpot(
    sales: float,
    popularity: float = 1.0,
    star_pool: int = 12,
    tier_prizes: dict[str, float] | None = None,
    apply_tax: bool = True,
    hi: float = 5e9,
    ev_m1lhao: float = 0.0,
) -> float | None:
    """
    Jackpot a partir do qual o EV iguala o preço da aposta.

    Devolve None se for inatingível — que é a resposta mais importante que
    esta função dá, porque mostra quando a acumulação *não* chega para
    compensar a partilha e o imposto.
    """
    def f(j: float) -> float:
        return expected_value(
            j, sales, popularity, star_pool, tier_prizes, apply_tax, ev_m1lhao
        ).ev_liquido - config.TICKET_PRICE_EUR

    if f(hi) < 0:
        return None
    lo = 0.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def ev_curve(
    sales: float,
    popularity_levels: tuple[float, ...] = (0.35, 1.0, 3.0),
    jackpots: tuple[float, ...] | None = None,
    star_pool: int = 12,
    tier_prizes: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Tabela de EV em função do jackpot e da popularidade da combinação."""
    jackpots = jackpots or (17e6, 50e6, 100e6, 150e6, 200e6, 250e6)
    rows = []
    for j in jackpots:
        row = {"jackpot_M€": j / 1e6}
        for pop in popularity_levels:
            r = expected_value(j, sales, pop, star_pool, tier_prizes)
            row[f"EV π={pop}"] = round(r.ev_liquido, 4)
        row["ganho_impopular_%"] = round(
            100
            * (
                expected_value(j, sales, min(popularity_levels), star_pool, tier_prizes).ev_liquido
                / expected_value(j, sales, 1.0, star_pool, tier_prizes).ev_liquido
                - 1
            ),
            1,
        )
        rows.append(row)
    return pd.DataFrame(rows)


def tax_impact(gross_values: tuple[float, ...] = (5_000, 25_000, 1e6, 50e6, 250e6)) -> pd.DataFrame:
    """Quanto é que o Imposto do Selo retira, por escalão de prémio."""
    rows = []
    for g in gross_values:
        n = config.net_prize(g)
        rows.append(
            {
                "prémio_bruto_€": g,
                "líquido_€": round(n, 2),
                "imposto_€": round(g - n, 2),
                "taxa_efetiva_%": round(100 * (g - n) / g, 2) if g else 0.0,
            }
        )
    return pd.DataFrame(rows)
