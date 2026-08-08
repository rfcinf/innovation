"""
Backtest honesto.

Um backtest só vale alguma coisa se puder falhar. Este testa duas
afirmações opostas e deixa os dados decidir:

  H_previsão  — "existe informação no histórico que ajuda a acertar nos
                 números do próximo sorteio."
  H_valor     — "existe informação no histórico que ajuda a receber mais
                 dinheiro quando se acerta."

A primeira é a que toda a gente tenta e é testada aqui com honestidade
suficiente para poder falhar (e falha). A segunda é a proposta deste
sistema e é validada fora da amostra: o modelo é estimado numa parte do
histórico e avaliado noutra que nunca viu.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .dataset import MAIN_COLS
from .popularity import design_matrix, fit_popularity_model


# ---------------------------------------------------------------------------
# H_previsão — as estratégias de números
# ---------------------------------------------------------------------------

def _hits(pick: set[int], drawn: np.ndarray) -> int:
    return len(pick & set(int(x) for x in drawn))


def strategy_backtest(
    df: pd.DataFrame, window: int = 200, seed: int = 11
) -> pd.DataFrame:
    """
    Percorre o histórico em modo "tempo real": em cada sorteio, escolhe 5
    números usando apenas informação anterior, e conta quantos acertou.

    Estratégias testadas:
      quentes   — os 5 mais frequentes na janela recente
      frios     — os 5 menos frequentes
      atrasados — os 5 com maior intervalo desde a última aparição
      aleatório — controlo

    Sob independência, todas têm de convergir para a mesma média: 0.5
    acertos por sorteio (5 escolhas × 5/50). Qualquer estratégia que
    funcionasse teria de se destacar aqui.
    """
    rng = np.random.default_rng(seed)
    mains = df[MAIN_COLS].to_numpy(dtype=int)
    n = len(df)
    pool = config.MAIN_POOL
    results = {k: [] for k in ("quentes", "frios", "atrasados", "aleatório")}

    for i in range(window, n):
        hist = mains[i - window : i]
        drawn = mains[i]

        counts = np.bincount(hist.ravel(), minlength=pool + 1)[1:]
        order = np.argsort(counts)
        results["frios"].append(_hits(set((order[:5] + 1).tolist()), drawn))
        results["quentes"].append(_hits(set((order[-5:] + 1).tolist()), drawn))

        last_seen = np.full(pool, -1)
        for t in range(len(hist)):
            for v in hist[t]:
                last_seen[v - 1] = t
        overdue = np.argsort(last_seen)[:5]
        results["atrasados"].append(_hits(set((overdue + 1).tolist()), drawn))

        rnd = rng.choice(np.arange(1, pool + 1), 5, replace=False)
        results["aleatório"].append(_hits(set(rnd.tolist()), drawn))

    expected = config.MAIN_PICK * config.MAIN_PICK / pool
    rows = []
    for name, hits in results.items():
        arr = np.array(hits, dtype=float)
        se = arr.std(ddof=1) / np.sqrt(len(arr))
        z = (arr.mean() - expected) / se
        rows.append(
            {
                "estratégia": name,
                "sorteios": len(arr),
                "acertos_médios": round(arr.mean(), 4),
                "esperado_se_aleatório": expected,
                "z": round(float(z), 2),
                "p": round(float(2 * stats.norm.sf(abs(z))), 4),
                "melhor_que_acaso": bool(z > 1.96),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# H_valor — validação fora da amostra do modelo de popularidade
# ---------------------------------------------------------------------------

def popularity_out_of_sample(
    master: pd.DataFrame, train_frac: float = 0.70
) -> dict:
    """
    O teste que interessa.

    Estima o modelo de popularidade nos primeiros `train_frac` sorteios e
    prevê a popularidade das combinações do período seguinte, que nunca
    viu. Se a correlação entre previsto e observado for positiva e
    significativa, o modelo sabe genuinamente prever quantas pessoas
    partilham uma combinação — e é isso, e só isso, que o sistema promete.
    """
    d = master.dropna(subset=["popularity_main5", "sales_est", "winners_main5"]).copy()
    d = d[np.isfinite(d["popularity_main5"]) & (d["sales_est"] > 0)]
    d = d.sort_values("date").reset_index(drop=True)

    cut = int(len(d) * train_frac)
    train, test = d.iloc[:cut], d.iloc[cut:]

    model, _ = fit_popularity_model(train)

    combos_test = test[MAIN_COLS].to_numpy(dtype=int)
    pred = model.predict_popularity(combos_test)
    obs = test["popularity_main5"].to_numpy(dtype=float)

    ok = np.isfinite(pred) & np.isfinite(obs)
    pred, obs = pred[ok], obs[ok]

    r_p = stats.pearsonr(np.log(pred), np.log1p(obs))
    r_s = stats.spearmanr(pred, obs)

    # Teste prático: dividir o período de teste em decis de popularidade
    # prevista e comparar a popularidade realmente observada nos extremos.
    q = pd.qcut(pred, 5, labels=False, duplicates="drop")
    grp = pd.DataFrame({"q": q, "obs": obs}).groupby("q")["obs"].agg(["size", "mean", "median"])

    lowest = float(grp["mean"].iloc[0])
    highest = float(grp["mean"].iloc[-1])

    return {
        "n_treino": len(train),
        "n_teste": len(test),
        "corte": str(train["date"].iloc[-1]),
        "pearson_log_r": round(float(r_p[0]), 4),
        "pearson_p": float(r_p[1]),
        "spearman_rho": round(float(r_s[0]), 4),
        "spearman_p": float(r_s[1]),
        "quintis": grp,
        "popularidade_obs_quintil_baixo": round(lowest, 3),
        "popularidade_obs_quintil_alto": round(highest, 3),
        "racio_alto_sobre_baixo": round(highest / lowest, 2) if lowest > 0 else float("nan"),
        "ganho_esperado_no_cheque_%": round(100 * (highest / lowest - 1), 1) if lowest > 0 else float("nan"),
    }


def jackpot_sharing_reality(master: pd.DataFrame) -> pd.DataFrame:
    """
    A partilha não é hipótese: aconteceu, repetidamente.

    Distribuição do nº de vencedores do jackpot nos sorteios em que o
    jackpot saiu. Cada vencedor adicional é o prémio a dividir.
    """
    d = master.dropna(subset=["jackpot_winners"])
    won = d[d["jackpot_winners"] > 0]
    dist = won["jackpot_winners"].value_counts().sort_index()
    out = pd.DataFrame(
        {
            "vencedores_do_jackpot": dist.index.astype(int),
            "nº_de_sorteios": dist.values,
        }
    )
    out["% dos jackpots"] = (100 * out["nº_de_sorteios"] / out["nº_de_sorteios"].sum()).round(1)
    out["fração_recebida"] = (1 / out["vencedores_do_jackpot"]).round(3)
    return out


def date_bias_evidence(master: pd.DataFrame) -> pd.DataFrame:
    """
    Evidência direta e sem modelo do viés de datas.

    Compara a popularidade observada de combinações com muitos números
    ≤ 31 (compatíveis com aniversários) contra as que têm poucos.
    """
    d = master.dropna(subset=["popularity_main5"]).copy()
    d = d[np.isfinite(d["popularity_main5"])]
    combos = d[MAIN_COLS].to_numpy(dtype=int)
    d["n_ate_31"] = (combos <= 31).sum(axis=1)

    grp = d.groupby("n_ate_31")["popularity_main5"].agg(
        sorteios="size", popularidade_media="mean", mediana="median"
    ).reset_index()

    lo = d[d["n_ate_31"] <= 2]["popularity_main5"]
    hi = d[d["n_ate_31"] >= 4]["popularity_main5"]
    if len(lo) > 5 and len(hi) > 5:
        t = stats.mannwhitneyu(hi, lo, alternative="greater")
        grp.attrs["teste"] = {
            "comparação": "≥4 números ≤31  vs  ≤2 números ≤31",
            "n_alto": len(hi),
            "n_baixo": len(lo),
            "média_alto": round(float(hi.mean()), 3),
            "média_baixo": round(float(lo.mean()), 3),
            "racio": round(float(hi.mean() / lo.mean()), 2),
            "mann_whitney_p": float(t.pvalue),
        }
    return grp
