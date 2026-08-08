"""
O M1lhão — a parcela de valor que faltava.

O ERRO QUE ESTE MÓDULO CORRIGE
------------------------------
O bilhete português do EuroMillions custa €2,50 e **inclui um código
M1lhão**: um sorteio exclusivo de Portugal, com prémio de €1.000.000,
realizado às sextas-feiras. Participam todos os códigos gerados pelas
apostas registadas em Portugal entre sábado e a sexta-feira do sorteio —
ou seja, uma semana inteira, o que inclui as apostas do sorteio de terça.

O motor de valor esperado deste projeto contava os 13 escalões do
EuroMillions e ignorava isto por completo. Como o M1lhão vale ~€0,21 por
aposta depois de imposto, o EV publicado estava subestimado em cerca de um
quarto. Nada disto altera a conclusão de fundo — continua a ser um jogo de
soma negativa — mas os números estavam errados e passaram a estar certos.

DUAS PROPRIEDADES QUE IMPORTAM PARA A ESTRATÉGIA
------------------------------------------------
1. O código é **gerado pelo sistema**, não escolhido pelo apostador. Não há
   viés humano, não há combinações "populares", não há nada a otimizar.

2. Por isso mesmo, o M1lhão é uma parcela de EV **fixa e igual para todos**
   — e **dilui a vantagem relativa** da otimização de combinações. Os
   ganhos percentuais medidos sobre a parte do EuroMillions encolhem quando
   calculados sobre o total. Ignorar isto seria inflacionar a vantagem do
   sistema, que é o oposto do que este projeto se propõe fazer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

PRIZE_EUR = 1_000_000.0
"""Prémio do M1lhão. Um único código vencedor por sorteio semanal."""

DRAWS_PER_WEEK = 2
"""Terça e sexta: as apostas de ambos os sorteios entram no M1lhão de sexta."""


def portuguese_share(
    breakdown: pd.DataFrame, tier: str = "2+0", since_year: int = 2025
) -> float:
    """
    Quota portuguesa do total de apostas europeias.

    Estimada pelo escalão mais frequente (2+0, ~1 em 22): a razão entre
    vencedores portugueses e vencedores totais é uma estimativa direta da
    quota de apostas, e nesse escalão os números são grandes o suficiente
    para o ruído ser desprezável.

    Usa-se o período recente por defeito porque a quota tem vindo a descer
    de forma sistemática (11,8% em 2019 → 7,9% em 2026), e é a quota atual
    que interessa para decidir hoje.
    """
    bd = breakdown[breakdown["tier"] == tier].dropna(
        subset=["winners_pt", "winners_total"]
    )
    bd = bd[bd["winners_total"] > 0].copy()
    bd["year"] = [d.year for d in bd["date"]]
    recent = bd[bd["year"] >= since_year]
    if len(recent) < 20:
        recent = bd
    return float((recent["winners_pt"] / recent["winners_total"]).median())


def share_trend(breakdown: pd.DataFrame, tier: str = "2+0") -> pd.DataFrame:
    """Evolução anual da quota portuguesa — relevante porque não é estável."""
    bd = breakdown[breakdown["tier"] == tier].dropna(
        subset=["winners_pt", "winners_total"]
    )
    bd = bd[bd["winners_total"] > 0].copy()
    bd["ano"] = [d.year for d in bd["date"]]
    bd["quota"] = bd["winners_pt"] / bd["winners_total"]
    out = bd.groupby("ano")["quota"].agg(sorteios="size", quota_mediana="median")
    return out.reset_index()


def codes_per_draw(sales_total: float, pt_share: float) -> float:
    """
    Nº de códigos que concorrem a cada M1lhão.

    Uma semana de apostas portuguesas: o sorteio de terça e o de sexta.
    """
    return sales_total * pt_share * DRAWS_PER_WEEK


def expected_value_per_bet(
    sales_total: float,
    pt_share: float,
    apply_tax: bool = True,
) -> float:
    """
    Valor esperado do M1lhão por aposta simples, em euros.

    Cada aposta gera um código; um código é sorteado por semana. Logo,
    P(ganhar) = 1 / nº de códigos, e o EV é o prémio líquido a dividir por
    esse número.
    """
    codes = codes_per_draw(sales_total, pt_share)
    if codes <= 0:
        return 0.0
    prize = config.net_prize(PRIZE_EUR) if apply_tax else PRIZE_EUR
    return prize / codes


def summary(
    breakdown: pd.DataFrame,
    sales_tuesday: float,
    sales_friday: float,
    pt_share: float | None = None,
) -> dict:
    """Relatório completo, usando vendas separadas por dia (são diferentes)."""
    pt_share = pt_share if pt_share is not None else portuguese_share(breakdown)
    codes = pt_share * (sales_tuesday + sales_friday)
    gross = PRIZE_EUR
    net = config.net_prize(gross)
    return {
        "quota_pt": round(pt_share, 4),
        "vendas_terca": sales_tuesday,
        "vendas_sexta": sales_friday,
        "codigos_por_sorteio_semanal": round(codes),
        "probabilidade_por_aposta": codes and 1 / codes,
        "premio_bruto_eur": gross,
        "premio_liquido_eur": round(net, 2),
        "ev_por_aposta_eur": round(net / codes, 4) if codes else 0.0,
    }


def blend_with_euromillions(
    ev_euromillions: float, ev_m1lhao: float
) -> dict:
    """
    Junta as duas parcelas e mostra o efeito de diluição.

    A vantagem da otimização vive toda na parte do EuroMillions. Somar uma
    parcela fixa ao numerador e ao denominador reduz o ganho percentual —
    e é essa a percentagem verdadeira.
    """
    total = ev_euromillions + ev_m1lhao
    return {
        "ev_euromillions": round(ev_euromillions, 4),
        "ev_m1lhao": round(ev_m1lhao, 4),
        "ev_total": round(total, 4),
        "peso_m1lhao_%": round(100 * ev_m1lhao / total, 1) if total else 0.0,
        "retorno_por_euro": round(total / config.TICKET_PRICE_EUR, 4),
    }
