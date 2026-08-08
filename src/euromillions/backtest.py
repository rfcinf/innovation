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


# ---------------------------------------------------------------------------
# Backtest preditivo walk-forward, sobre sorteios reais
# ---------------------------------------------------------------------------

def _make_portfolio(strategy: str, rng: np.random.Generator, n_tickets: int,
                    star_pool: int, filters=None) -> tuple[np.ndarray, np.ndarray]:
    """Gera uma carteira segundo a estratégia indicada."""
    from .optimizer import Filters

    pool = np.arange(1, config.MAIN_POOL + 1)
    spool = np.arange(1, star_pool + 1)

    if strategy == "datas":
        # O erro mais comum: só números ≤31, estrelas baixas.
        mains = np.array([
            np.sort(rng.choice(np.arange(1, 32), config.MAIN_PICK, replace=False))
            for _ in range(n_tickets)
        ])
        stars = np.array([
            np.sort(rng.choice(np.arange(1, 10), config.STAR_PICK, replace=False))
            for _ in range(n_tickets)
        ])
        return mains, stars

    if strategy == "sobreposta":
        # Variar 1-2 números a partir de uma base — muito frequente na prática.
        base = np.sort(rng.choice(pool, config.MAIN_PICK, replace=False))
        rows = []
        for i in range(n_tickets):
            t = base.copy()
            alt = rng.choice([x for x in pool if x not in t])
            t[i % config.MAIN_PICK] = alt
            rows.append(np.sort(t))
        mains = np.array(rows)
        stars = np.array([
            np.sort(rng.choice(spool, config.STAR_PICK, replace=False))
            for _ in range(n_tickets)
        ])
        return mains, stars

    if strategy == "aleatoria":
        mains = np.array([
            np.sort(rng.choice(pool, config.MAIN_PICK, replace=False))
            for _ in range(n_tickets)
        ])
        stars = np.array([
            np.sort(rng.choice(spool, config.STAR_PICK, replace=False))
            for _ in range(n_tickets)
        ])
        return mains, stars

    if strategy == "otimizada":
        # Cobertura disjunta + filtros ditados pelos dados + estrelas altas.
        filters = filters or Filters()
        need = n_tickets * config.MAIN_PICK
        for _ in range(400):
            perm = rng.permutation(pool)[:need].reshape(n_tickets, config.MAIN_PICK)
            perm = np.sort(perm, axis=1)
            if all(filters.accepts(row) for row in perm):
                mains = perm
                break
        else:
            mains = np.sort(rng.permutation(pool)[:need].reshape(n_tickets, -1), axis=1)
        # Estrelas: as menos jogadas segundo o modelo medido (10, 11, 12).
        high = spool[spool >= max(1, star_pool - 2)]
        stars = np.array([
            np.sort(rng.choice(high, config.STAR_PICK, replace=False))
            for _ in range(n_tickets)
        ])
        return mains, stars

    raise ValueError(f"estratégia desconhecida: {strategy}")


def walk_forward(
    draws: pd.DataFrame,
    breakdown: pd.DataFrame,
    strategies: tuple[str, ...] = ("datas", "sobreposta", "aleatoria", "otimizada"),
    n_tickets: int = 5,
    since: str = "2016-09-27",
    n_reps: int = 400,
    seed: int = 2024,
) -> pd.DataFrame:
    """
    Backtest preditivo sobre sorteios reais.

    Para cada estratégia gera `n_reps` carteiras independentes e confronta
    cada uma com **todos** os sorteios reais do período, usando os prémios
    históricos efetivamente pagos em cada escalão e em cada sorteio.

    O QUE ESTE BACKTEST PODE E NÃO PODE MEDIR — e é essencial perceber a
    diferença:

    * PODE medir a taxa de acerto por escalão e a probabilidade de sair sem
      nada. São eventos frequentes (P(algum prémio) ≈ 1/13 por aposta), há
      dados que cheguem, e as diferenças entre estratégias saem com
      precisão.

    * NÃO PODE medir a vantagem de popularidade. O prémio histórico de cada
      escalão é um número fixo, já dividido pelos vencedores que realmente
      existiram. Não há forma de o histórico revelar quanto teríamos
      recebido com uma combinação diferente. Essa vantagem vive quase toda
      nos escalões altos, que num backtest de algumas centenas de sorteios
      nunca são atingidos.

    A vantagem de popularidade está validada noutro sítio e de outra forma:
    em `popularity_out_of_sample()`, que prevê fora da amostra quantas
    pessoas partilham cada combinação. Confundir as duas coisas levaria a
    concluir, erradamente, que a vantagem não existe só porque este teste
    não a consegue ver.
    """
    d = draws[draws["date"].astype(str) >= since].reset_index(drop=True)
    if len(d) < 50:
        raise ValueError("período de teste demasiado curto")

    drawn_m = d[MAIN_COLS].to_numpy(int)
    drawn_s = d[["s1", "s2"]].to_numpy(int)
    star_pool = int(d["star_pool"].iloc[-1])

    # Prémios reais pagos, por sorteio e escalão.
    piv = breakdown.pivot_table(
        index="date", columns="tier", values="prize_eur", aggfunc="first"
    )
    prize_lookup = {
        t.label: piv[t.label].reindex(d["date"]).to_numpy(dtype=float)
        if t.label in piv else np.full(len(d), np.nan)
        for t in config.TIERS
    }
    tier_index = {t.label: t for t in config.TIERS}

    rng = np.random.default_rng(seed)
    rows = []

    for strat in strategies:
        n_nothing = 0
        n_port = 0
        winnings: list[float] = []
        hits: dict[str, int] = {}
        n_ticket_draws = 0
        # Estatísticas POR RÉPLICA. São estas que dão a incerteza correta:
        # os 1030 sorteios enfrentados por uma mesma carteira não são
        # observações independentes — partilham os mesmos números. A unidade
        # independente é a carteira, e há apenas `n_reps` delas.
        rep_p_nada: list[float] = []
        rep_retorno: list[float] = []
        rep_taxa: list[float] = []

        for _ in range(n_reps):
            mains, stars = _make_portfolio(strat, rng, n_tickets, star_pool)

            # Interseções vetorizadas contra todos os sorteios do período.
            # (T, n_tickets)
            match_m = np.zeros((len(d), n_tickets), dtype=np.int8)
            match_s = np.zeros((len(d), n_tickets), dtype=np.int8)
            for j in range(n_tickets):
                match_m[:, j] = np.isin(drawn_m, mains[j]).sum(axis=1)
                match_s[:, j] = np.isin(drawn_s, stars[j]).sum(axis=1)

            payout = np.zeros(len(d))
            any_win = np.zeros(len(d), dtype=bool)
            rep_hits = 0
            for label, tier in tier_index.items():
                hit = (match_m == tier.mains) & (match_s == tier.stars)
                if not hit.any():
                    continue
                cnt = hit.sum(axis=1)
                hits[label] = hits.get(label, 0) + int(cnt.sum())
                rep_hits += int(cnt.sum())
                prizes = np.nan_to_num(prize_lookup[label], nan=0.0)
                payout += cnt * prizes
                any_win |= cnt > 0

            n_nothing += int((~any_win).sum())
            n_port += len(d)
            n_ticket_draws += len(d) * n_tickets
            winnings.append(float(payout.sum()))

            rep_cost = len(d) * n_tickets * config.TICKET_PRICE_EUR
            rep_p_nada.append(float((~any_win).mean()))
            rep_retorno.append(float(payout.sum() / rep_cost))
            rep_taxa.append(rep_hits / (len(d) * n_tickets))

        total_win = float(np.sum(winnings))
        cost = n_ticket_draws * config.TICKET_PRICE_EUR

        def _ci(vals: list[float]) -> float:
            a = np.asarray(vals, dtype=float)
            return float(1.96 * a.std(ddof=1) / np.sqrt(len(a)))

        rows.append(
            {
                "estratégia": strat,
                "carteiras": n_reps,
                "apostas": n_ticket_draws,
                "P(nada)": round(float(np.mean(rep_p_nada)), 4),
                "±95%": round(_ci(rep_p_nada), 4),
                "taxa_acerto": round(float(np.mean(rep_taxa)), 5),
                "±95%_taxa": round(_ci(rep_taxa), 5),
                "retorno_€/€": round(total_win / cost, 4),
                "±95%_ret": round(_ci(rep_retorno), 4),
                "custo_€": round(cost, 0),
            }
        )

    out = pd.DataFrame(rows)
    out.attrs["hits"] = hits
    out.attrs["periodo"] = f"{d['date'].min()} → {d['date'].max()} ({len(d)} sorteios)"
    return out


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
