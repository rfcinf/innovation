"""
Plano de jogo: maximizar a hipótese de jackpot, e de ficar com ele sozinho.

O PROBLEMA POSTO CORRETAMENTE
-----------------------------
"O maior jackpot possível, ganho sozinho, com a maior probabilidade
possível" são três objetivos, e só dois deles se conseguem mover:

1. **P(ganhar o jackpot)** — depende exclusivamente do NÚMERO de apostas
   distintas. Nada mais. Nem combinação, nem timing, nem sistema.
2. **Tamanho do jackpot** — depende de QUANDO se joga.
3. **Ficar sozinho** — depende de QUE combinação se joga.

O erro comum é tentar mexer no primeiro. Só o segundo e o terceiro estão
abertos, e ambos têm resposta quantitativa.

A TENSÃO ENTRE (2) E (3)
------------------------
Jackpot grande atrai mais apostas — logo, mais gente com quem partilhar.
Parece que os dois objetivos se anulam. Não se anulam, e a medição diz
porquê:

    elasticidade das vendas ao jackpot = 0,271

O jackpot duplica, as vendas sobem apenas 1,21×. O prémio ganha a corrida
com folga. Jackpots grandes são melhores mesmo depois de descontar a
concorrência extra.

A CONCLUSÃO OPERACIONAL
-----------------------
Para um orçamento anual fixo, **concentrar** as apostas nos sorteios de
jackpot alto não altera em nada a probabilidade de acertar — o número total
de apostas é o mesmo — mas quase duplica o retorno por euro e multiplica o
tamanho do cheque no caso de acertar.

É a única decisão de calendário deste projeto que tem efeito real.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from . import config, ev

SALES_ELASTICITY = 0.271
"""Medida em 1106 sorteios da era atual: log(vendas) ~ log(jackpot).
Spearman(jackpot, vendas) = +0,621, p = 4,7×10⁻¹¹⁹."""

SALES_ANCHOR = (24e6, 60e6)
"""(vendas, jackpot) de referência para escalar a relação acima."""


def expected_sales(jackpot_eur: float) -> float:
    """Vendas previstas para um dado jackpot."""
    s0, j0 = SALES_ANCHOR
    return float(s0 * (jackpot_eur / j0) ** SALES_ELASTICITY)


def fit_sales_elasticity(master: pd.DataFrame, since_year: int = 2016) -> dict:
    """Reestima a elasticidade a partir dos dados — o auditor deve conferi-la."""
    d = master.dropna(subset=["jackpot_eur", "sales_est"])
    d = d[(d["sales_est"] > 0) & np.isfinite(d["sales_est"]) & (d["jackpot_eur"] > 0)]
    d = d[[x.year >= since_year for x in d["date"]]]
    if len(d) < 100:
        return {"elasticidade": SALES_ELASTICITY, "n": len(d)}
    b, a = np.polyfit(np.log(d["jackpot_eur"]), np.log(d["sales_est"]), 1)
    r = stats.spearmanr(d["jackpot_eur"], d["sales_est"])
    return {
        "n": len(d),
        "elasticidade": round(float(b), 4),
        "spearman": round(float(r[0]), 4),
        "p": float(r[1]),
        "vendas_se_jackpot_duplicar": round(2 ** float(b), 3),
    }


# ---------------------------------------------------------------------------
# Ficar sozinho
# ---------------------------------------------------------------------------

def solo_probability(
    jackpot_eur: float, popularity: float, star_pool: int = 12,
    sales: float | None = None,
) -> dict:
    """
    Probabilidade de ser o único vencedor, dado que se acertou.

    P(nenhum outro vencedor) = e^(-λ), com λ = vendas × p_jackpot × π.
    """
    S = sales if sales is not None else expected_sales(jackpot_eur)
    p = config.JACKPOT.probability(star_pool)
    lam = S * p * popularity
    solo = float(np.exp(-lam))
    partilha = float((1 - np.exp(-lam)) / lam) if lam > 0 else 1.0
    return {
        "jackpot_eur": jackpot_eur,
        "vendas": S,
        "popularidade": popularity,
        "lambda": lam,
        "P_sozinho": solo,
        "cheque_esperado_liquido": config.net_prize(jackpot_eur * partilha),
    }


def solo_table(
    popularities: tuple[float, ...] = (0.35, 1.0, 1.53),
    jackpots: tuple[float, ...] = (20e6, 60e6, 100e6, 150e6, 200e6, 250e6),
) -> pd.DataFrame:
    rows = []
    for J in jackpots:
        row = {"jackpot_M€": J / 1e6, "vendas_M": round(expected_sales(J) / 1e6, 1)}
        for pi in popularities:
            r = solo_probability(J, pi)
            row[f"P(sozinho) π={pi}"] = round(100 * r["P_sozinho"], 1)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Alocação do orçamento
# ---------------------------------------------------------------------------

@dataclass
class Plan:
    nome: str
    limiar_jackpot: float
    sorteios_por_ano: int
    apostas_por_sorteio: float
    jackpot_tipico: float
    retorno_por_euro: float
    p_jackpot_ano: float
    p_sozinho: float
    cheque_esperado: float


def budget_plans(
    orcamento_anual: float = 520.0,
    popularity: float = 0.35,
    tier_prizes: dict[str, float] | None = None,
    ev_m1lhao: float = 0.0,
) -> pd.DataFrame:
    """
    Compara concentrar o orçamento em jackpots altos vs espalhá-lo.

    A coluna decisiva é a última par: `P(jackpot/ano)` **não muda** — o
    número total de apostas é o mesmo — enquanto o retorno por euro quase
    duplica. É esse o argumento inteiro.
    """
    total_apostas = orcamento_anual / config.TICKET_PRICE_EUR
    p = config.JACKPOT.probability(12)
    p_ano = 1 - (1 - p) ** total_apostas

    # Frequências observadas na era atual (1106 sorteios).
    cenarios = [
        ("todos os sorteios", 0.0, 104, 45e6),
        ("só jackpot > €60M", 60e6, 48, 90e6),
        ("só jackpot > €100M", 100e6, 14, 130e6),
        ("só jackpot > €180M", 180e6, 5, 200e6),
    ]

    rows = []
    for nome, limiar, n_sorteios, jackpot_tipico in cenarios:
        S = expected_sales(jackpot_tipico)
        r = ev.expected_value(
            jackpot_tipico, S, popularity, 12, tier_prizes, ev_m1lhao=ev_m1lhao
        )
        solo = solo_probability(jackpot_tipico, popularity, sales=S)
        rows.append(
            {
                "plano": nome,
                "sorteios/ano": n_sorteios,
                "apostas/sorteio": round(total_apostas / n_sorteios, 1),
                "jackpot típico": f"€{jackpot_tipico/1e6:.0f}M",
                "retorno €/€": round(r.retorno_por_euro, 3),
                "P(jackpot/ano)": f"{100*p_ano:.6f}%",
                "P(sozinho)": f"{100*solo['P_sozinho']:.1f}%",
                "cheque esperado": f"€{solo['cheque_esperado_liquido']/1e6:.0f}M",
            }
        )
    out = pd.DataFrame(rows)
    base = float(out["retorno €/€"].iloc[0])
    out["ganho vs jogar sempre"] = (
        100 * (out["retorno €/€"] / base - 1)
    ).round(1).astype(str) + "%"
    return out


def recommend_plan(
    orcamento_anual: float = 520.0,
    tier_prizes: dict[str, float] | None = None,
    ev_m1lhao: float = 0.0,
) -> dict:
    """O plano concreto, com os números que o sustentam."""
    total = orcamento_anual / config.TICKET_PRICE_EUR
    p = config.JACKPOT.probability(12)
    jackpot = 200e6
    S = expected_sales(jackpot)
    solo = solo_probability(jackpot, 0.35, sales=S)
    r = ev.expected_value(jackpot, S, 0.35, 12, tier_prizes, ev_m1lhao=ev_m1lhao)
    return {
        "orcamento_anual_eur": orcamento_anual,
        "apostas_totais_por_ano": int(total),
        "sorteios_a_jogar": 5,
        "apostas_por_sorteio": round(total / 5, 1),
        "limiar_de_jackpot": "€180M",
        "popularidade_alvo": 0.35,
        "retorno_por_euro": round(r.retorno_por_euro, 3),
        "P_jackpot_por_ano": float(1 - (1 - p) ** total),
        "P_sozinho_se_ganhar": round(float(solo["P_sozinho"]), 4),
        "cheque_esperado_liquido_eur": round(float(solo["cheque_esperado_liquido"])),
        "anos_para_esperar_1_jackpot": round(1 / p / total),
    }
