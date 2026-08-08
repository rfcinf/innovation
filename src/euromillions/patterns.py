"""
O mapa completo do espaço de combinações.

A PERGUNTA QUE ESTE MÓDULO RESPONDE
-----------------------------------
"Há coisas que nunca aconteceram em 22 anos. Não deveríamos concluir que
são menos prováveis, e evitá-las para reduzir o erro?"

É a pergunta certa, e é testável. A resposta divide-se em duas partes, e a
segunda é a que interessa.

PARTE 1 — "nunca aconteceu" quase nunca significa o que parece
--------------------------------------------------------------
O exemplo clássico é a repetição exata dos 5 números. Parece impossível: em
1970 sorteios nunca se viu... exceto que **se viu**. A combinação
4-30-31-38-42 saiu a 2 de maio de 2014 e outra vez a 31 de agosto de 2018.

E o número esperado de repetições era 0,915.

O cálculo é o do "paradoxo dos aniversários": não se compara um sorteio com
um alvo, comparam-se **todos os pares de sorteios entre si**. São
C(1970,2) = 1.939.465 pares, cada um com probabilidade 1/2.118.760 de
coincidir. Esperado: 0,92. Observado: 1.

Antes de concluir que algo é improvável por não ter acontecido, é preciso
calcular quantas vezes deveria ter acontecido. Quase sempre a resposta é
"menos de uma vez", e nesse caso a ausência não é informação nenhuma.

PARTE 2 — mas há não-uniformidade real, e é enorme
--------------------------------------------------
Aqui a intuição acerta em cheio. **As combinações individuais são todas
igualmente prováveis, mas as FAMÍLIAS de combinações não são.**

Só 8,02% dos sorteios têm os 5 números ≤ 31. Não por viés nenhum: porque
só 8,02% das combinações possíveis têm essa propriedade — são 169.911 em
2.118.760.

Este módulo enumera **as 2.118.760 combinações todas** e calcula a
probabilidade exata de cada família de padrões, sem simulação e sem
aproximação. Depois compara com o que 1970 sorteios realmente fizeram.

A DISTINÇÃO QUE MUDA TUDO
-------------------------
Uma família rara **não torna o seu bilhete menos provável**. Se escolher
uma combinação com os 5 números ≤ 31, a sua probabilidade continua a ser
exatamente 1 em 2.118.760 — igual à de qualquer outra. A família é rara
precisamente porque tem poucos membros, e o seu bilhete é um deles.

Confundir "esta família sai poucas vezes" com "este bilhete ganha menos
vezes" é o erro que sustenta metade dos sistemas de lotaria à venda.

O que a raridade de uma família **muda mesmo** é outra coisa: quantas
pessoas jogam lá dentro. E isso é a vantagem real deste projeto
(`popularity.py`), que nada tem a ver com probabilidade.
"""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .dataset import MAIN_COLS, mains_matrix
from .randomness import benjamini_hochberg

_UNIVERSE: np.ndarray | None = None


def universe() -> np.ndarray:
    """
    As 2.118.760 combinações de 5 números em 50, todas.

    Em memória são ~10 MB. Enumerar é preferível a simular: dá
    probabilidades exatas em vez de estimativas, e o espaço é pequeno o
    suficiente para caber num computador portátil.
    """
    global _UNIVERSE
    if _UNIVERSE is None:
        _UNIVERSE = np.array(
            list(combinations(range(1, config.MAIN_POOL + 1), config.MAIN_PICK)),
            dtype=np.int8,
        )
    return _UNIVERSE


# ---------------------------------------------------------------------------
# Famílias de padrões
# ---------------------------------------------------------------------------

def _feature_frame(M: np.ndarray) -> dict[str, np.ndarray]:
    """Calcula todas as propriedades de uma matriz (n, 5) de combinações."""
    M = M.astype(np.int16)
    diffs = np.diff(M, axis=1)
    return {
        "nums_ate_31": (M <= 31).sum(axis=1),
        "nums_ate_12": (M <= 12).sum(axis=1),
        "pares_consecutivos": (diffs == 1).sum(axis=1),
        "impares": (M % 2 == 1).sum(axis=1),
        "amplitude": M[:, -1] - M[:, 0],
        "soma": M.sum(axis=1),
        "decadas_cobertas": np.array(
            [len(np.unique((row - 1) // 10)) for row in M]
        ) if len(M) < 100_000 else _decades_fast(M),
        "max_mesma_dezena": _max_same_decade(M),
    }


def _decades_fast(M: np.ndarray) -> np.ndarray:
    dec = (M - 1) // 10
    out = np.zeros(len(M), dtype=np.int8)
    for d in range(5):
        out += (dec == d).any(axis=1)
    return out


def _max_same_decade(M: np.ndarray) -> np.ndarray:
    dec = (M - 1) // 10
    counts = np.zeros((len(M), 5), dtype=np.int8)
    for d in range(5):
        counts[:, d] = (dec == d).sum(axis=1)
    return counts.max(axis=1)


def exact_distribution(feature: str, bins: list | None = None) -> pd.DataFrame:
    """
    Distribuição exata de uma propriedade, por enumeração completa.

    Não é uma estimativa: é a contagem de quantas das 2.118.760 combinações
    possuem cada valor.
    """
    U = universe()
    vals = _feature_frame(U)[feature]
    if bins is not None:
        vals = np.digitize(vals, bins)
    uniq, counts = np.unique(vals, return_counts=True)
    total = counts.sum()
    return pd.DataFrame(
        {
            "valor": uniq,
            "combinacoes": counts,
            "probabilidade": counts / total,
            "1_em": total / counts,
        }
    )


def observed_vs_exact(
    draws: pd.DataFrame, feature: str, bins: list | None = None
) -> tuple[pd.DataFrame, dict]:
    """
    Compara o que saiu com o que a enumeração exata prevê.

    É aqui que se veria uma lacuna real: se alguma família de padrões
    aparecesse sistematicamente menos vezes do que o espaço permite.
    """
    exact = exact_distribution(feature, bins).set_index("valor")
    M = mains_matrix(draws)
    obs_vals = _feature_frame(M)[feature]
    if bins is not None:
        obs_vals = np.digitize(obs_vals, bins)

    n = len(draws)
    table = exact.copy()
    table["observado"] = pd.Series(
        {v: int((obs_vals == v).sum()) for v in table.index}
    )
    table["observado"] = table["observado"].fillna(0).astype(int)
    table["esperado"] = table["probabilidade"] * n
    table["desvio_%"] = 100 * (table["observado"] - table["esperado"]) / table["esperado"]

    # Qui-quadrado apenas sobre classes com esperado suficiente.
    keep = table["esperado"] >= 5
    o = table.loc[keep, "observado"].to_numpy(float)
    e = table.loc[keep, "esperado"].to_numpy(float)
    # Reagrupar a cauda para não perder massa.
    o_tail = table.loc[~keep, "observado"].sum()
    e_tail = table.loc[~keep, "esperado"].sum()
    if e_tail > 0:
        o = np.append(o, o_tail)
        e = np.append(e, e_tail)
    stat = float(((o - e) ** 2 / e).sum())
    dof = len(o) - 1
    return table.reset_index(), {
        "propriedade": feature,
        "classes": len(o),
        "chi2": stat,
        "gl": dof,
        "p": float(stats.chi2.sf(stat, dof)),
    }


FEATURES_TO_TEST: tuple[tuple[str, list | None], ...] = (
    ("nums_ate_31", None),
    ("nums_ate_12", None),
    ("pares_consecutivos", None),
    ("impares", None),
    ("decadas_cobertas", None),
    ("max_mesma_dezena", None),
    ("amplitude", [10, 20, 25, 30, 35, 40, 44, 47]),
    ("soma", [60, 90, 105, 120, 135, 150, 165, 190]),
)


def full_pattern_battery(draws: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Testa todas as famílias de padrões contra a enumeração exata, com
    correção para testes múltiplos.

    Se existisse uma lacuna real — uma família que a máquina evita — teria
    de aparecer aqui.
    """
    rows = []
    detail = {}
    for feat, bins in FEATURES_TO_TEST:
        tab, res = observed_vs_exact(draws, feat, bins)
        rows.append(res)
        detail[feat] = tab
    out = pd.DataFrame(rows)
    out["p_ajustado"] = benjamini_hochberg(out["p"].to_numpy())
    out["significativo_fdr5"] = out["p_ajustado"] < 0.05
    return out.sort_values("p").reset_index(drop=True), detail


# ---------------------------------------------------------------------------
# Coincidências entre sorteios
# ---------------------------------------------------------------------------

def overlap_analysis(draws: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Quantos números partilham, dois a dois, todos os pares de sorteios da
    história — comparado com o que o acaso puro exige.

    Inclui a repetição total (5 números em comum), que é a pergunta
    original: aconteceu, uma vez, quando o esperado era 0,92.
    """
    M = mains_matrix(draws)
    n = len(M)
    A = np.zeros((n, config.MAIN_POOL + 1), dtype=np.int16)
    for i, row in enumerate(M):
        A[i, row] = 1
    G = A @ A.T
    iu = np.triu_indices(n, k=1)
    vals = G[iu]
    n_pairs = len(vals)

    rows = []
    for k in range(config.MAIN_PICK + 1):
        p = comb(5, k) * comb(45, 5 - k) / comb(50, 5)
        e = n_pairs * p
        o = int((vals == k).sum())
        rows.append(
            {
                "numeros_em_comum": k,
                "observado": o,
                "esperado": round(e, 2),
                "desvio_%": round(100 * (o - e) / e, 2) if e > 0 else np.nan,
                "1_em": round(1 / p),
            }
        )
    table = pd.DataFrame(rows)
    stat = float(
        ((table["observado"] - table["esperado"]) ** 2 / table["esperado"]).sum()
    )
    dof = len(table) - 1
    return table, {
        "pares_de_sorteios": n_pairs,
        "chi2": round(stat, 3),
        "gl": dof,
        "p": float(stats.chi2.sf(stat, dof)),
    }


def repeat_report(draws: pd.DataFrame) -> dict:
    """A resposta direta: houve repetições exatas, e quantas se esperavam?"""
    M = mains_matrix(draws)
    n = len(M)
    sets = [frozenset(r.tolist()) for r in M]
    seen: dict[frozenset, list[int]] = {}
    for i, s in enumerate(sets):
        seen.setdefault(s, []).append(i)
    repeats = {k: v for k, v in seen.items() if len(v) > 1}

    n_pairs = n * (n - 1) / 2
    expected = n_pairs / config.MAIN_COMBINATIONS

    detail = []
    for k, idxs in repeats.items():
        detail.append(
            {
                "combinacao": sorted(int(x) for x in k),
                "datas": [str(draws["date"].iloc[i]) for i in idxs],
                "sorteios_de_intervalo": int(idxs[-1] - idxs[0]),
            }
        )

    return {
        "sorteios": n,
        "pares_comparados": int(n_pairs),
        "repeticoes_observadas": len(repeats),
        "repeticoes_esperadas": round(expected, 3),
        "P_zero_repeticoes": round(float(np.exp(-expected)), 3),
        "detalhe": detail,
    }


def never_happened_yet(draws: pd.DataFrame) -> pd.DataFrame:
    """
    Famílias de padrões que ainda não saíram — e quantas vezes deveriam ter
    saído.

    A coluna que interessa é a última: quando o esperado é inferior a 1, a
    ausência não é evidência de nada. É o teste que separa uma lacuna real
    de uma ilusão de raridade.
    """
    U = universe()
    uf = _feature_frame(U)
    M = mains_matrix(draws)
    of = _feature_frame(M)
    n = len(draws)

    rows = []
    for feat in uf:
        vals_u = uf[feat]
        vals_o = set(np.unique(of[feat]).tolist())
        for v in np.unique(vals_u):
            if int(v) in vals_o:
                continue
            p = float((vals_u == v).mean())
            rows.append(
                {
                    "propriedade": feat,
                    "valor_nunca_visto": int(v),
                    "probabilidade": p,
                    "1_em": round(1 / p) if p > 0 else np.inf,
                    "esperado_em_1970_sorteios": round(n * p, 3),
                    "ausencia_e_informativa": bool(n * p > 3),
                }
            )
    return pd.DataFrame(rows).sort_values(
        "esperado_em_1970_sorteios", ascending=False
    ).reset_index(drop=True)
