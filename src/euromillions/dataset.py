"""
Carregamento e normalização do histórico.

Fornece duas estruturas centrais:

    load_draws()      -> DataFrame, um sorteio por linha
    load_breakdown()  -> DataFrame, um escalão por linha

E, criticamente, `estimate_sales()`, que reconstrói o volume de apostas de
cada sorteio a partir do total de vencedores. Sem essa estimativa não é
possível medir popularidade de combinações.
"""

from __future__ import annotations

import datetime as dt
import os

import numpy as np
import pandas as pd

from . import config

MAIN_COLS = ["n1", "n2", "n3", "n4", "n5"]
STAR_COLS = ["s1", "s2"]


# ---------------------------------------------------------------------------
# Carregamento
# ---------------------------------------------------------------------------

def load_draws(path: str = config.DRAWS_CSV) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} não existe. Corre primeiro:  python -m euromillions.cli fetch"
        )
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = df["date"].dt.date
    df = df.sort_values("date").reset_index(drop=True)
    df["draw_index"] = np.arange(1, len(df) + 1)
    df["star_pool"] = [config.star_pool_for(d) for d in df["date"]]
    df["era"] = [config.era_for(d).name for d in df["date"]]
    df["year"] = [d.year for d in df["date"]]
    df["dow"] = [d.strftime("%a") for d in df["date"]]
    return df


def load_breakdown(path: str = config.BREAKDOWN_CSV) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} não existe. Corre:  python -m euromillions.cli fetch --breakdown"
        )
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = df["date"].dt.date
    return df.drop_duplicates(subset=["date", "tier"]).reset_index(drop=True)


def mains_matrix(df: pd.DataFrame) -> np.ndarray:
    """(n_sorteios, 5) com os números principais ordenados."""
    return df[MAIN_COLS].to_numpy(dtype=int)


def stars_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[STAR_COLS].to_numpy(dtype=int)


# ---------------------------------------------------------------------------
# Validação
# ---------------------------------------------------------------------------

def validate(df: pd.DataFrame) -> list[str]:
    """Verificações de integridade. Devolve lista de problemas (vazia = ok)."""
    problems: list[str] = []

    mains = mains_matrix(df)
    stars = stars_matrix(df)

    if mains.min() < 1 or mains.max() > config.MAIN_POOL:
        problems.append(f"números fora de 1..{config.MAIN_POOL}")
    for i, row in enumerate(mains):
        if len(set(row)) != config.MAIN_PICK:
            problems.append(f"números repetidos em {df['date'].iloc[i]}")
    for i, row in enumerate(stars):
        if len(set(row)) != config.STAR_PICK:
            problems.append(f"estrelas repetidas em {df['date'].iloc[i]}")

    # As estrelas nunca podem exceder o pool da era em vigor.
    over = df[stars.max(axis=1) > df["star_pool"]]
    if len(over):
        problems.append(
            f"{len(over)} sorteios com estrela acima do pool declarado "
            f"(primeiro: {over['date'].iloc[0]}, estrela "
            f"{stars[over.index[0]].max()})"
        )

    if df["date"].duplicated().any():
        problems.append("datas duplicadas")

    return problems


def validate_eras(df: pd.DataFrame) -> pd.DataFrame:
    """Confronta as eras declaradas em config com a realidade observada."""
    rows = []
    for era in config.ERAS:
        mask = df["era"] == era.name
        sub = df[mask]
        if not len(sub):
            continue
        observed = int(stars_matrix(sub).max())
        rows.append(
            {
                "era": era.name,
                "inicio": sub["date"].min(),
                "fim": sub["date"].max(),
                "sorteios": len(sub),
                "pool_declarado": era.star_pool,
                "estrela_max_observada": observed,
                "coerente": observed <= era.star_pool,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reconstrução do volume de vendas
# ---------------------------------------------------------------------------

def estimate_sales(
    draws: pd.DataFrame, breakdown: pd.DataFrame
) -> pd.DataFrame:
    """
    Estima o número de apostas vendidas em cada sorteio.

    Método: o total de vencedores em todos os escalões é, para uma
    população de apostas, aproximadamente

        E[vencedores] = S * P(ganhar alguma coisa)

    P(ganhar) depende só da matriz (≈ 1/13 na era atual), por isso

        Ŝ = total_vencedores / P(ganhar)

    O enviesamento humano na escolha de números distorce isto, mas os
    escalões baixos (2+0, 2+1) dominam a contagem e são pouco sensíveis à
    combinação escolhida — acertar 2 números é uma restrição fraca. O erro
    residual é de poucos por cento e, para o que interessa (comparar a
    popularidade de combinações *entre* sorteios), cancela-se em grande
    medida.
    """
    winners = (
        breakdown.groupby("date")["winners_total"].sum().rename("winners_all_tiers")
    )
    out = draws.merge(winners, left_on="date", right_index=True, how="left")
    p_any = np.array([config.probability_any_prize(sp) for sp in out["star_pool"]])
    out["p_any_prize"] = p_any
    out["sales_est"] = out["winners_all_tiers"] / p_any
    return out


def exact_main_set_winners(breakdown: pd.DataFrame) -> pd.Series:
    """
    Nº de apostas que jogaram *exatamente* o conjunto de 5 números sorteado.

    É a soma dos escalões 5+2, 5+1 e 5+0: qualquer aposta que acerte os 5
    números cai num destes três, independentemente das estrelas. Este é o
    único observável público que mede diretamente a popularidade de uma
    combinação concreta de 5 números.
    """
    five = breakdown[breakdown["tier"].str.startswith("5+")]
    return five.groupby("date")["winners_total"].sum().rename("winners_main5")


def both_stars_winners(breakdown: pd.DataFrame) -> pd.Series:
    """
    Nº de apostas que acertaram ambas as estrelas *e* pelo menos 1 número.

    Soma dos escalões x+2. Não inclui 0+2 (não é premiado), por isso é um
    limite inferior da popularidade do par de estrelas — mas cobre ~91% dos
    casos, o que chega para modelar.
    """
    both = breakdown[breakdown["tier"].str.endswith("+2")]
    return both.groupby("date")["winners_total"].sum().rename("winners_star2")


def build_master(
    draws_path: str = config.DRAWS_CSV,
    breakdown_path: str = config.BREAKDOWN_CSV,
) -> pd.DataFrame:
    """Tabela mestra: sorteios + vendas estimadas + popularidade observada."""
    draws = load_draws(draws_path)
    bd = load_breakdown(breakdown_path)
    master = estimate_sales(draws, bd)
    master = master.merge(
        exact_main_set_winners(bd), left_on="date", right_index=True, how="left"
    )
    master = master.merge(
        both_stars_winners(bd), left_on="date", right_index=True, how="left"
    )

    # Popularidade = quantas vezes mais gente jogou esta combinação do que
    # se todos escolhessem ao acaso. 1.0 = combinação perfeitamente banal.
    expected_main5 = master["sales_est"] / config.MAIN_COMBINATIONS
    master["popularity_main5"] = master["winners_main5"] / expected_main5

    jackpot_row = bd[bd["tier"] == "5+2"].set_index("date")
    master = master.merge(
        jackpot_row[["winners_total"]].rename(columns={"winners_total": "jackpot_winners"}),
        left_on="date", right_index=True, how="left",
    )
    return master
