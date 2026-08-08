"""
Elasticidade da partilha por escalão — medida, não arbitrada.

O QUE ESTAVA ERRADO
-------------------
`ev.TIER_ELASTICITY` traduzia "quanto é que a popularidade da nossa
combinação afeta o número de vencedores de cada escalão". Nos escalões de 5
números o efeito é total; nos baixos é diluído, porque acertar 2 ou 3
números não fixa a combinação toda.

A tabela estava correta em espírito e **inventada em número**: 1,0 / 0,7 /
0,4 / 0,2 / 0,1, atribuídos por mim. É exatamente o mesmo pecado que foi
corrigido nas estrelas — substituir palpite por medição — e que ficou por
corrigir aqui.

A ARMADILHA NA MEDIÇÃO
----------------------
O caminho óbvio é regredir o nº de vencedores do escalão na popularidade
observada da combinação sorteada. Não serve: a popularidade observada é
calculada dividindo por vendas estimadas, e os vencedores de qualquer
escalão também escalam com as vendas. O erro da estimativa de vendas entra
dos dois lados e produz correlação espúria — inflacionando todas as
elasticidades.

A solução é usar a popularidade **prevista pelo modelo**, que depende
apenas das características da combinação (quantos números ≤31, espaçamento,
geometria do boletim) e não contém qualquer vencedor nem qualquer
estimativa de vendas. Fica assim livre de acoplamento mecânico.

    log(W_t) = a_t + b_t · log(π̂) + c_t · log(vencedores_totais) + ε

`b_t` é a elasticidade. Como π̂ é um indicador ruidoso da popularidade
verdadeira, todos os `b_t` sofrem a mesma atenuação — que se cancela ao
normalizar pelo escalão de 5 números, cuja elasticidade é teoricamente 1,0
(o escalão *é* a combinação).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .dataset import MAIN_COLS


def measure_elasticities(
    master: pd.DataFrame,
    breakdown: pd.DataFrame,
    model,
    min_obs: int = 200,
    normalise: bool = True,
) -> pd.DataFrame:
    """
    Estima a elasticidade de cada escalão à popularidade da combinação.

    Devolve uma tabela com o coeficiente, o erro-padrão e a comparação com
    a tabela que estava escrita à mão.
    """
    from .ev import TIER_ELASTICITY as OLD

    d = master.dropna(subset=["winners_all_tiers"]).copy()
    d = d[d["winners_all_tiers"] > 0]
    combos = d[MAIN_COLS].to_numpy(int)
    d["pi_hat"] = model.predict_popularity(combos)
    d = d[np.isfinite(d["pi_hat"]) & (d["pi_hat"] > 0)]

    piv = breakdown.pivot_table(
        index="date", columns="tier", values="winners_total", aggfunc="first"
    )

    rows = []
    for tier in config.TIERS:
        label = tier.label
        if label not in piv:
            continue
        w = piv[label].reindex(d["date"]).to_numpy(dtype=float)
        ok = np.isfinite(w) & (w > 0)
        if ok.sum() < min_obs:
            continue

        y = np.log(w[ok])
        x1 = np.log(d["pi_hat"].to_numpy()[ok])
        x2 = np.log(d["winners_all_tiers"].to_numpy(dtype=float)[ok])
        X = np.column_stack([np.ones(ok.sum()), x1, x2])

        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        dof = max(len(y) - X.shape[1], 1)
        cov = np.linalg.pinv(X.T @ X) * float(resid @ resid / dof)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))

        rows.append(
            {
                "escalao": label,
                "n": int(ok.sum()),
                "b_bruto": float(beta[1]),
                "se": float(se[1]),
                "z": float(beta[1] / se[1]) if se[1] > 0 else np.nan,
                "p": float(2 * stats.norm.sf(abs(beta[1] / se[1]))) if se[1] > 0 else np.nan,
                "escrito_a_mao": OLD.get(label),
            }
        )

    out = pd.DataFrame(rows)
    if not len(out):
        return out

    if normalise:
        # O escalão 5+0 é o mais limpo dos três de 5 números: tem contagens
        # maiores do que 5+2 e 5+1 (não exige acertar estrelas), logo menos
        # ruído. A sua elasticidade teórica é 1,0.
        ref = out.loc[out["escalao"] == "5+0", "b_bruto"]
        scale = float(ref.iloc[0]) if len(ref) and ref.iloc[0] != 0 else 1.0
        out["elasticidade"] = (out["b_bruto"] / scale).round(3)
    else:
        out["elasticidade"] = out["b_bruto"].round(3)

    out["diferenca"] = (out["elasticidade"] - out["escrito_a_mao"]).round(3)
    return out


def fitted_table(elasticities: pd.DataFrame, clip: tuple[float, float] = (0.0, 1.0)) -> dict[str, float]:
    """Converte a tabela medida no formato que `ev.py` consome."""
    lo, hi = clip
    return {
        r.escalao: float(np.clip(r.elasticidade, lo, hi))
        for r in elasticities.itertuples()
        if np.isfinite(r.elasticidade)
    }


def impact_on_ev(
    measured: dict[str, float],
    jackpot_eur: float = 60e6,
    sales: float = 24e6,
    popularity: float = 0.62,
    tier_prizes: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Quanto é que a correção das elasticidades muda o valor esperado."""
    from . import ev as _ev

    old = _ev.expected_value(jackpot_eur, sales, popularity, 12, tier_prizes)
    saved = dict(_ev.TIER_ELASTICITY)
    try:
        _ev.TIER_ELASTICITY.update(measured)
        new = _ev.expected_value(jackpot_eur, sales, popularity, 12, tier_prizes)
    finally:
        _ev.TIER_ELASTICITY.clear()
        _ev.TIER_ELASTICITY.update(saved)

    return pd.DataFrame(
        [
            {
                "tabela": "escrita à mão",
                "EV_€": round(old.ev_liquido, 4),
                "escalões_baixos_€": round(old.ev_inferiores, 4),
            },
            {
                "tabela": "medida",
                "EV_€": round(new.ev_liquido, 4),
                "escalões_baixos_€": round(new.ev_inferiores, 4),
            },
        ]
    )
