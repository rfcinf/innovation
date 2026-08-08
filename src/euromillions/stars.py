"""
Popularidade das estrelas, medida — não adivinhada.

PORQUE É QUE ESTE MÓDULO EXISTE
-------------------------------
A primeira versão do sistema tratava as estrelas com um palpite escrito à
mão: "as baixas devem ser mais jogadas porque parecem datas". Direção
certa, magnitude errada, e errado em pelo menos um caso concreto (a estrela
1, que o palpite dava como popular, é das menos jogadas).

Isso era a parte mais fraca do sistema, e é evitável: a popularidade das
estrelas **é observável**, pelo mesmo tipo de raciocínio que revelou a
popularidade dos números.

O IDENTIFICADOR
---------------
Entre as apostas que acertaram exatamente k números principais, a
repartição por estrelas deveria seguir, sob escolha uniforme:

    P(2 estrelas) : P(0 estrelas)  =  1 : C(pool-2, 2)

(pool 12 → 1:45, pool 11 → 1:36, pool 9 → 1:21)

O desvio dessa razão mede diretamente quantas pessoas escolheram *aquele
par de estrelas*:

    π_estrelas  ≈  C(pool-2, 2) · W(k,2) / W(k,0)

Fazendo isto para k = 2, 3, 4 obtêm-se ~3 observações por sorteio,
independentes do lado dos números. Com 1936 sorteios, são mais de 5000
observações para estimar apenas 12 parâmetros.

O RESULTADO
-----------
Todas as 12 estrelas diferem entre si de forma esmagadora
(χ² = 4386, 11 g.l., p ≈ 0; R² = 0,59). O padrão é o que a psicologia
prevê, e a estrela 7 — o "número da sorte" — é a mais sobre-jogada de
todas.

O par (7,5) é escolhido cerca de **2,4 vezes mais** do que o par (12,10),
com exatamente a mesma probabilidade de sair.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb

import numpy as np
import pandas as pd
from scipy import stats

from . import config


# ---------------------------------------------------------------------------
# Construção das observações
# ---------------------------------------------------------------------------

def star_observations(
    draws: pd.DataFrame, breakdown: pd.DataFrame, ks: tuple[int, ...] = (2, 3, 4)
) -> pd.DataFrame:
    """
    Uma linha por (sorteio, k): estimativa bruta da popularidade do par de
    estrelas sorteado.
    """
    piv = breakdown.pivot_table(
        index="date", columns="tier", values="winners_total", aggfunc="first"
    )
    d = draws.set_index("date")
    rows = []
    for k in ks:
        top, bot = f"{k}+2", f"{k}+0"
        if top not in piv or bot not in piv:
            continue
        sub = piv[[top, bot]].dropna()
        sub = sub[(sub[bot] > 0) & (sub[top] > 0)]
        for date, r in sub.iterrows():
            if date not in d.index:
                continue
            pool = int(d.loc[date, "star_pool"])
            rows.append(
                {
                    "date": date,
                    "k": k,
                    "pool": pool,
                    "s1": int(d.loc[date, "s1"]),
                    "s2": int(d.loc[date, "s2"]),
                    # razão esperada W(k,0)/W(k,2) sob escolha uniforme
                    "pi_bruto": comb(pool - 2, 2) * r[top] / r[bot],
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

@dataclass
class StarPopularity:
    """
    Popularidade multiplicativa por estrela, estimada em log-escala.

        log π(a,b) = α_a + α_b

    Normalizada para que a média geométrica sobre todos os pares seja 1.
    """

    pool: int = 12
    alpha: np.ndarray = field(default_factory=lambda: np.array([]))
    se: np.ndarray = field(default_factory=lambda: np.array([]))
    r2: float = 0.0
    n_obs: int = 0
    joint_chi2: float = 0.0
    joint_p: float = 1.0

    def fit(self, obs: pd.DataFrame) -> "StarPopularity":
        cur = obs[obs["pool"] == self.pool]
        if len(cur) < 100:
            raise ValueError(f"observações insuficientes para pool={self.pool}")

        y = np.log(cur["pi_bruto"].to_numpy(dtype=float))
        S = self.pool
        ks = sorted(cur["k"].unique())

        # Desenho: intercepto + contagem por estrela + efeitos fixos de k.
        # A estrela 1 é a referência (coluna omitida) para evitar
        # colinearidade com o intercepto.
        X = np.zeros((len(cur), 1 + (S - 1) + (len(ks) - 1)))
        X[:, 0] = 1.0
        s1 = cur["s1"].to_numpy(int)
        s2 = cur["s2"].to_numpy(int)
        for j in range(2, S + 1):
            X[:, j - 1] = (s1 == j).astype(float) + (s2 == j).astype(float)
        for i, kk in enumerate(ks[1:], start=S):
            X[:, i] = (cur["k"].to_numpy() == kk).astype(float)

        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        dof = max(len(y) - X.shape[1], 1)
        cov = np.linalg.pinv(X.T @ X) * float(resid @ resid / dof)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))

        # Teste conjunto: todas as estrelas têm a mesma popularidade?
        idx = list(range(1, S))
        b = beta[idx]
        W = float(b @ np.linalg.pinv(cov[np.ix_(idx, idx)]) @ b)

        # Coeficiente por estrela, com a referência a 0 e recentrado.
        alpha = np.zeros(S)
        alpha[1:] = beta[1:S]
        alpha -= alpha.mean()

        self.alpha = alpha
        self.se = np.concatenate([[0.0], se[1:S]])
        self.r2 = float(1 - resid.var() / y.var())
        self.n_obs = len(y)
        self.joint_chi2 = W
        self.joint_p = float(stats.chi2.sf(W, S - 1))
        return self

    # -- utilização -------------------------------------------------------

    def star_factor(self, star: int) -> float:
        return float(np.exp(self.alpha[star - 1]))

    def pair_popularity(self, pair: tuple[int, int]) -> float:
        a, b = pair
        return float(np.exp(self.alpha[a - 1] + self.alpha[b - 1]))

    def table(self) -> pd.DataFrame:
        rows = []
        for j in range(1, self.pool + 1):
            z = self.alpha[j - 1] / self.se[j - 1] if self.se[j - 1] > 0 else np.nan
            rows.append(
                {
                    "estrela": j,
                    "popularidade": round(self.star_factor(j), 4),
                    "z_vs_referencia": round(float(z), 2) if np.isfinite(z) else None,
                }
            )
        return pd.DataFrame(rows).sort_values("popularidade", ascending=False)

    def best_pairs(self, n: int = 8) -> pd.DataFrame:
        pairs = [
            (a, b)
            for a in range(1, self.pool + 1)
            for b in range(a + 1, self.pool + 1)
        ]
        rows = [
            {"par": f"{a}-{b}", "popularidade": round(self.pair_popularity((a, b)), 4)}
            for a, b in pairs
        ]
        out = pd.DataFrame(rows).sort_values("popularidade").reset_index(drop=True)
        return pd.concat([out.head(n), out.tail(n)])


# ---------------------------------------------------------------------------
# Validação fora da amostra
# ---------------------------------------------------------------------------

def validate_out_of_sample(obs: pd.DataFrame, pool: int = 12, train_frac: float = 0.7) -> dict:
    """
    Estima em dados antigos, avalia em dados novos que nunca viu.

    O mesmo padrão de rigor aplicado ao modelo dos números: sem isto, um R²
    alto dentro da amostra não prova nada.
    """
    cur = obs[obs["pool"] == pool].sort_values("date").reset_index(drop=True)
    dates = cur["date"].unique()
    cut_date = dates[int(len(dates) * train_frac)]

    train = cur[cur["date"] < cut_date]
    test = cur[cur["date"] >= cut_date]

    model = StarPopularity(pool=pool).fit(train)

    pred = np.array(
        [model.pair_popularity((int(r.s1), int(r.s2))) for r in test.itertuples()]
    )
    actual = test["pi_bruto"].to_numpy(dtype=float)

    ok = np.isfinite(pred) & np.isfinite(actual) & (actual > 0)
    pred, actual = pred[ok], actual[ok]

    rho = stats.spearmanr(pred, actual)
    r_log = stats.pearsonr(np.log(pred), np.log(actual))

    q = pd.qcut(pred, 4, labels=False, duplicates="drop")
    grp = pd.DataFrame({"q": q, "obs": actual}).groupby("q")["obs"].agg(
        ["size", "mean", "median"]
    )

    return {
        "n_treino": len(train),
        "n_teste": len(test),
        "corte": str(cut_date),
        "spearman_rho": round(float(rho[0]), 4),
        "spearman_p": float(rho[1]),
        "pearson_log_r": round(float(r_log[0]), 4),
        "pearson_p": float(r_log[1]),
        "quartis": grp,
        "racio_alto_baixo": round(float(grp["median"].iloc[-1] / grp["median"].iloc[0]), 3),
    }


def fit_star_model(
    draws: pd.DataFrame, breakdown: pd.DataFrame, pool: int = 12
) -> StarPopularity:
    return StarPopularity(pool=pool).fit(star_observations(draws, breakdown))
