"""
Regras estruturais do EuroMillions.

Tudo o que aqui está é facto verificável sobre o jogo (matriz, escalões,
preço, fiscalidade), não opinião. O resto do sistema depende destas
constantes, por isso estão isoladas num só ficheiro e cobertas por testes.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from math import comb

# ---------------------------------------------------------------------------
# Matriz do jogo e eras
# ---------------------------------------------------------------------------

MAIN_POOL = 50          # bolas principais: 1..50 (nunca mudou desde 2004)
MAIN_PICK = 5
STAR_PICK = 2

FIRST_DRAW = dt.date(2004, 2, 13)


@dataclass(frozen=True)
class Era:
    """Período com matriz de estrelas constante."""

    name: str
    start: dt.date
    end: dt.date | None      # None = em vigor
    star_pool: int
    note: str

    def contains(self, day: dt.date) -> bool:
        if day < self.start:
            return False
        return self.end is None or day <= self.end


# O número de estrelas mudou duas vezes. As datas abaixo são a hipótese
# nominal; `dataset.validate_eras()` confirma-as contra os dados reais e
# grita se a realidade discordar.
ERAS: tuple[Era, ...] = (
    Era(
        name="E1",
        start=dt.date(2004, 2, 13),
        end=dt.date(2011, 5, 6),
        star_pool=9,
        note="Apenas sextas-feiras. 5/50 + 2/9.",
    ),
    Era(
        name="E2",
        start=dt.date(2011, 5, 10),
        end=dt.date(2016, 9, 23),
        star_pool=11,
        note="Introdução do sorteio de terça-feira. 5/50 + 2/11.",
    ),
    Era(
        name="E3",
        start=dt.date(2016, 9, 24),
        end=None,
        star_pool=12,
        note="Matriz atual. 5/50 + 2/12.",
    ),
)


def era_for(day: dt.date) -> Era:
    for era in ERAS:
        if era.contains(day):
            return era
    raise ValueError(f"Sem era definida para {day!r} (primeiro sorteio: {FIRST_DRAW})")


def star_pool_for(day: dt.date) -> int:
    return era_for(day).star_pool


# ---------------------------------------------------------------------------
# Escalões de prémio
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Tier:
    """Um escalão: acertar `mains` números e `stars` estrelas."""

    mains: int
    stars: int

    @property
    def label(self) -> str:
        return f"{self.mains}+{self.stars}"

    def probability(self, star_pool: int) -> float:
        """Probabilidade exata por aposta simples, calculada de raiz."""
        p_main = (
            comb(MAIN_PICK, self.mains)
            * comb(MAIN_POOL - MAIN_PICK, MAIN_PICK - self.mains)
            / comb(MAIN_POOL, MAIN_PICK)
        )
        p_star = (
            comb(STAR_PICK, self.stars)
            * comb(star_pool - STAR_PICK, STAR_PICK - self.stars)
            / comb(star_pool, STAR_PICK)
        )
        return p_main * p_star

    def odds(self, star_pool: int) -> float:
        return 1.0 / self.probability(star_pool)


# Os 13 escalões premiados, do maior para o menor.
TIERS: tuple[Tier, ...] = (
    Tier(5, 2), Tier(5, 1), Tier(5, 0),
    Tier(4, 2), Tier(4, 1), Tier(3, 2),
    Tier(4, 0), Tier(2, 2), Tier(3, 1),
    Tier(3, 0), Tier(1, 2), Tier(2, 1),
    Tier(2, 0),
)

TIER_BY_LABEL = {t.label: t for t in TIERS}

JACKPOT = TIERS[0]


def probability_any_prize(star_pool: int) -> float:
    return sum(t.probability(star_pool) for t in TIERS)


def total_combinations(star_pool: int) -> int:
    """Espaço amostral completo: quantas apostas distintas existem."""
    return comb(MAIN_POOL, MAIN_PICK) * comb(star_pool, STAR_PICK)


MAIN_COMBINATIONS = comb(MAIN_POOL, MAIN_PICK)   # 2 118 760


# ---------------------------------------------------------------------------
# Economia do jogo em Portugal
# ---------------------------------------------------------------------------

TICKET_PRICE_EUR = 2.50
"""Preço de uma aposta simples em Portugal (inclui o código M1lhão)."""

STAMP_DUTY_RATE = 0.20
"""Imposto do Selo (verba 11.2.2 da TGIS) sobre prémios de jogo."""

STAMP_DUTY_THRESHOLD_EUR = 5_000.00
"""O imposto incide apenas sobre a parcela que excede este valor."""

PRIZE_FUND_SHARE = 0.50
"""Fração das vendas que reverte para prémios (~50% no EuroMillions)."""


def net_prize(gross_eur: float) -> float:
    """Prémio líquido em Portugal, após retenção na fonte do Imposto do Selo."""
    if gross_eur <= STAMP_DUTY_THRESHOLD_EUR:
        return gross_eur
    excess = gross_eur - STAMP_DUTY_THRESHOLD_EUR
    return gross_eur - STAMP_DUTY_RATE * excess


def gross_for_net(net_eur: float) -> float:
    """Inverso de `net_prize` — útil para calcular limiares."""
    if net_eur <= STAMP_DUTY_THRESHOLD_EUR:
        return net_eur
    return (net_eur - STAMP_DUTY_RATE * STAMP_DUTY_THRESHOLD_EUR) / (1 - STAMP_DUTY_RATE)


# ---------------------------------------------------------------------------
# Recolha de dados
# ---------------------------------------------------------------------------

SOURCE_BASE = "https://www.euro-millions.com"
YEAR_URL = SOURCE_BASE + "/results-history-{year}"
DRAW_URL = SOURCE_BASE + "/results/{date}"      # date = DD-MM-YYYY

DATA_DIR = "data"
DRAWS_CSV = "data/draws.csv"
BREAKDOWN_CSV = "data/breakdown.csv"
