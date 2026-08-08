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
from .ev import TIER_ELASTICITY as _EV_ELASTICITY
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


# ---------------------------------------------------------------------------
# Comparativo sobre TODO o histórico: modelo preditivo vs apostas avulsas
# ---------------------------------------------------------------------------

def full_history_comparison(
    draws: pd.DataFrame,
    breakdown: pd.DataFrame,
    strategies: tuple[str, ...] = ("datas", "sobreposta", "aleatoria", "otimizada"),
    n_tickets: int = 5,
    n_reps: int = 300,
    seed: int = 7,
    popularity: dict[str, float] | None = None,
) -> dict:
    """
    Confronta cada estratégia com os 1970 sorteios reais desde 2004.

    CORREÇÃO CRÍTICA face ao backtest da era atual: o número de estrelas
    mudou duas vezes (9 → 11 → 12). Gerar bilhetes com a estrela 12 num
    sorteio de 2005 produziria apostas impossíveis, que nunca ganhariam
    nada nos escalões com estrelas — e enviesaria a comparação inteira a
    favor de quem calhasse jogar estrelas baixas. Aqui o histórico é
    processado era a era, cada uma com a matriz que estava em vigor.

    Devolve duas contabilidades, e a diferença entre elas é o ponto central:

    * **bruta** — usa o prémio histórico efetivamente pago em cada escalão
      de cada sorteio. É o que teria acontecido *literalmente*. Não
      distingue estratégias no valor do prémio, porque esses prémios já
      estão divididos pelos vencedores que realmente existiram.

    * **ajustada** — corrige cada prémio pelo fator de partilha que a
      popularidade da nossa combinação implicaria (prémio × π^-elasticidade,
      como em `ev.py`). É um contrafactual modelado, não uma observação, e
      está identificado como tal. É a única forma de responder a "quanto
      teríamos recebido", já que o histórico não o pode revelar.
    """
    popularity = popularity or {}
    rng = np.random.default_rng(seed)

    piv_prize = breakdown.pivot_table(
        index="date", columns="tier", values="prize_eur", aggfunc="first"
    )
    tiers = list(config.TIERS)

    agg: dict[str, dict] = {
        s: {
            "custo": 0.0, "bruto": 0.0, "ajustado": 0.0,
            "n_nada": 0, "n_carteiras": 0, "n_apostas": 0,
            "hits": {}, "maior_premio": 0.0, "melhor_escalao": None,
            "rep_ret": [], "rep_nada": [],
        }
        for s in strategies
    }
    por_era: list[dict] = []

    for era in config.ERAS:
        d = draws[draws["era"] == era.name].reset_index(drop=True)
        if len(d) == 0:
            continue
        pool = era.star_pool
        drawn_m = d[MAIN_COLS].to_numpy(int)
        drawn_s = d[["s1", "s2"]].to_numpy(int)

        prize_lookup = {
            t.label: (
                np.nan_to_num(piv_prize[t.label].reindex(d["date"]).to_numpy(float), nan=0.0)
                if t.label in piv_prize else np.zeros(len(d))
            )
            for t in tiers
        }

        for strat in strategies:
            pi = popularity.get(strat, 1.0)
            era_bruto = era_ajust = 0.0
            era_nada = 0

            for _ in range(n_reps):
                mains, stars = _make_portfolio(strat, rng, n_tickets, pool)

                match_m = np.zeros((len(d), n_tickets), dtype=np.int8)
                match_s = np.zeros((len(d), n_tickets), dtype=np.int8)
                for j in range(n_tickets):
                    match_m[:, j] = np.isin(drawn_m, mains[j]).sum(axis=1)
                    match_s[:, j] = np.isin(drawn_s, stars[j]).sum(axis=1)

                bruto = np.zeros(len(d))
                ajust = np.zeros(len(d))
                any_win = np.zeros(len(d), dtype=bool)

                for t in tiers:
                    hit = (match_m == t.mains) & (match_s == t.stars)
                    if not hit.any():
                        continue
                    cnt = hit.sum(axis=1)
                    a = agg[strat]
                    a["hits"][t.label] = a["hits"].get(t.label, 0) + int(cnt.sum())
                    prizes = prize_lookup[t.label]
                    bruto += cnt * prizes
                    # Contrafactual: a nossa combinação teria sido partilhada
                    # por mais ou menos gente, conforme a sua popularidade.
                    elast = _EV_ELASTICITY.get(t.label, 0.5)
                    ajust += cnt * prizes * (pi ** (-elast))
                    any_win |= cnt > 0
                    best = float((cnt * prizes).max())
                    if best > a["maior_premio"]:
                        a["maior_premio"] = best
                        a["melhor_escalao"] = t.label

                a = agg[strat]
                cost = len(d) * n_tickets * config.TICKET_PRICE_EUR
                a["custo"] += cost
                a["bruto"] += float(bruto.sum())
                a["ajustado"] += float(ajust.sum())
                a["n_nada"] += int((~any_win).sum())
                a["n_carteiras"] += len(d)
                a["n_apostas"] += len(d) * n_tickets
                a["rep_ret"].append(float(bruto.sum() / cost))
                a["rep_nada"].append(float((~any_win).mean()))

                era_bruto += float(bruto.sum())
                era_ajust += float(ajust.sum())
                era_nada += int((~any_win).sum())

            por_era.append(
                {
                    "era": era.name,
                    "estrelas": pool,
                    "periodo": f"{d['date'].min()} → {d['date'].max()}",
                    "sorteios": len(d),
                    "estratégia": strat,
                    "retorno_€/€": round(era_bruto / (len(d) * n_tickets * config.TICKET_PRICE_EUR * n_reps), 4),
                    "P(nada)": round(era_nada / (len(d) * n_reps), 4),
                }
            )

    rows = []
    for strat in strategies:
        a = agg[strat]
        ret = np.asarray(a["rep_ret"])
        nada = np.asarray(a["rep_nada"])
        rows.append(
            {
                "estratégia": strat,
                "apostas": a["n_apostas"],
                "custo_€": round(a["custo"], 0),
                "ganho_bruto_€": round(a["bruto"], 0),
                "retorno_bruto_€/€": round(a["bruto"] / a["custo"], 4),
                "±95%": round(1.96 * ret.std(ddof=1) / np.sqrt(len(ret)), 4),
                "retorno_ajustado_€/€": round(a["ajustado"] / a["custo"], 4),
                "P(nada)": round(float(nada.mean()), 4),
                "±95%_nada": round(1.96 * nada.std(ddof=1) / np.sqrt(len(nada)), 4),
                "prémios": sum(a["hits"].values()),
                "maior_prémio_€": round(a["maior_premio"], 2),
                "melhor_escalão": a["melhor_escalao"],
            }
        )

    return {
        "resumo": pd.DataFrame(rows),
        "por_era": pd.DataFrame(por_era),
        "escaloes": pd.DataFrame(
            {s: agg[s]["hits"] for s in strategies}
        ).reindex([t.label for t in tiers]).fillna(0).astype(int),
        "n_reps": n_reps,
        "n_tickets": n_tickets,
    }


def analytic_comparison(
    breakdown: pd.DataFrame,
    popularity: dict[str, float],
    star_pool: int = 12,
    jackpot_eur: float = 60e6,
    sales: float = 24e6,
    n_bets: int = 2_462_500,
) -> pd.DataFrame:
    """
    Comparação com variância reduzida — a que consegue mesmo separar as
    estratégias.

    PORQUE É PRECISA ESTA SEGUNDA TABELA
    ------------------------------------
    O retorno realizado num backtest é dominado por acontecimentos
    raríssimos. Em 2,46 milhões de apostas simuladas por estratégia, o
    escalão 5+1 (€300 mil a €500 mil) foi atingido **uma vez ou nenhuma**, e
    o jackpot **nunca**. Um único acerto desses desloca o retorno total em
    mais de 0,08 €/€ — mais do que toda a diferença que queremos medir.

    Foi exatamente isso que aconteceu: a estratégia aleatória apanhou um
    5+1 de €508 mil e "ganhou" o comparativo bruto, com uma margem de erro
    de ±0,19 que cobre todas as outras. Ler essa tabela como se mostrasse
    superioridade seria confundir sorte com método — precisamente o erro
    que este projeto existe para não cometer.

    A solução é padrão em simulação: substituir o resultado realizado pelo
    seu valor esperado condicional, calculado analiticamente a partir das
    probabilidades exatas de cada escalão. Elimina-se a variância da cauda
    sem introduzir enviesamento, e o efeito da popularidade — que é o que
    distingue as estratégias — passa a ser visível.
    """
    from .ev import empirical_tier_prizes, expected_value

    try:
        prizes = empirical_tier_prizes(breakdown)
    except Exception:
        prizes = None

    rows = []
    for strat, pi in popularity.items():
        r = expected_value(jackpot_eur, sales, pi, star_pool, prizes)
        rows.append(
            {
                "estratégia": strat,
                "popularidade": round(pi, 4),
                "EV_por_aposta_€": round(r.ev_liquido, 4),
                "retorno_€/€": round(r.retorno_por_euro, 4),
                "fatia_do_jackpot_%": round(100 * r.fator_partilha, 1),
                "EV_total_€": round(r.ev_liquido * n_bets, 0),
            }
        )
    out = pd.DataFrame(rows)
    base = out.loc[out["estratégia"] == "datas", "EV_por_aposta_€"]
    if len(base):
        out["ganho_vs_datas_%"] = (100 * (out["EV_por_aposta_€"] / base.iloc[0] - 1)).round(1)
    return out


def measure_strategy_popularity(
    model,
    star_model=None,
    strategies: tuple[str, ...] = ("datas", "sobreposta", "aleatoria", "otimizada"),
    n_tickets: int = 5,
    n_reps: int = 300,
    star_pool: int = 12,
    seed: int = 99,
) -> dict[str, float]:
    """
    Mede a popularidade média das carteiras que cada estratégia produz.

    Não se assume um valor: gera-se centenas de carteiras por estratégia e
    passa-se cada bilhete pelos modelos estimados (números e estrelas).
    """
    from .optimizer import star_pair_popularity

    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    for strat in strategies:
        pops = []
        for _ in range(n_reps):
            m, s = _make_portfolio(strat, rng, n_tickets, star_pool)
            pm = model.predict_popularity(m)
            ps = np.array(
                [star_pair_popularity(tuple(x), star_pool, star_model) for x in s]
            )
            pops.append(float((pm * ps).mean()))
        out[strat] = float(np.mean(pops))
    return out


def jackpot_expectation(n_bets: int, star_pool: int = 12) -> dict:
    """
    Quanto tempo é preciso para *esperar* um jackpot — a escala real do
    problema, que nenhum backtest de 22 anos consegue mostrar.
    """
    p = config.JACKPOT.probability(star_pool)
    esperados = n_bets * p
    return {
        "apostas_simuladas": n_bets,
        "jackpots_esperados": round(esperados, 4),
        "P(pelo menos um)": round(1 - (1 - p) ** n_bets, 4),
        "apostas_para_1_esperado": round(1 / p),
        "anos_jogando_5_por_sorteio": round(1 / p / (5 * 104)),
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
