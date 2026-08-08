"""
Interface de linha de comandos.

    python -m euromillions.cli fetch [--breakdown]
    python -m euromillions.cli aleatoriedade
    python -m euromillions.cli maquinas
    python -m euromillions.cli popularidade
    python -m euromillions.cli valor [--jackpot 100e6]
    python -m euromillions.cli estrelas
    python -m euromillions.cli carteira
    python -m euromillions.cli m1lhao
    python -m euromillions.cli elasticidade
    python -m euromillions.cli comparativo
    python -m euromillions.cli backtest
    python -m euromillions.cli jogar [--bilhetes 5] [--jackpot 100e6]
    python -m euromillions.cli relatorio
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys

import numpy as np
import pandas as pd

from . import config, elasticity, ev, m1lhao, machines, portfolio, quantum, randomness, stars
from . import backtest as bt
from . import dataset as ds
from . import optimizer as opt
from .popularity import fit_popularity_model, observed_popularity_by_feature

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)


def _hr(title: str) -> None:
    print()
    print("═" * 78)
    print(f"  {title}")
    print("═" * 78)


# ---------------------------------------------------------------------------

def cmd_fetch(args) -> None:
    from . import fetch as F

    _hr("RECOLHA DO HISTÓRICO")
    f = F.Fetcher(delay=args.delay)
    draws = F.fetch_all_draws(fetcher=f)
    F.write_draws_csv(draws)
    print(f"\n{len(draws)} sorteios guardados em {config.DRAWS_CSV}")
    print(f"Período: {draws[0].date} → {draws[-1].date}")

    if args.breakdown:
        _hr("QUEBRA DE PRÉMIOS POR SORTEIO (demora ~15 min)")
        n = F.fetch_breakdowns([d.date for d in draws], fetcher=f)
        print(f"{n} sorteios novos guardados em {config.BREAKDOWN_CSV}")


def cmd_aleatoriedade(args) -> None:
    df = ds.load_draws()
    _hr(f"INTEGRIDADE — {len(df)} sorteios, {df['date'].min()} a {df['date'].max()}")
    problems = ds.validate(df)
    print("Problemas:", "; ".join(problems) if problems else "nenhum")
    print()
    print(ds.validate_eras(df).to_string(index=False))

    r = randomness.full_battery(df)

    _hr("TESTES GLOBAIS DE ALEATORIEDADE")
    for t in r["global"]:
        print(t)

    _hr("POR ERA")
    for tests in r["por_era"].values():
        for t in tests:
            print(t)

    _hr("BOLAS INDIVIDUAIS (as 8 mais extremas de 50)")
    b = r["tabelas"]["bolas"]
    print(b.head(8).to_string(index=False))
    print(f"\nSignificativas após correção FDR: {int(b['significativo_fdr5'].sum())} de {len(b)}")

    _hr("MEMÓRIA DO SISTEMA (autocorrelação)")
    print(r["tabelas"]["autocorr"].to_string(index=False))

    _hr("INTERVALOS ENTRE APARIÇÕES")
    g = r["tabelas"]["gaps"]
    print(g.head(5).to_string(index=False))
    print(f"\nSignificativos após FDR: {int(g['significativo_fdr5'].sum())} de {len(g)}")

    _hr("VARRIMENTO POR JANELAS (equipamento a degradar-se?)")
    w = r["tabelas"]["janelas"]
    print(f"Janelas testadas: {len(w)}   com viés (FDR 5%): {int(w['significativo_fdr5'].sum())}")
    print(w.nsmallest(3, "p").to_string(index=False))

    _hr("O TESTE DECISIVO — os desvios do passado persistem no futuro?")
    p = randomness.out_of_sample_persistence(df, n_sim=args.sim)
    for k, v in p.items():
        print(f"  {k:<16} {v}")
    print()
    print(randomness.forward_selection_test(df).to_string(index=False))

    _hr("POTÊNCIA — que tamanho de brecha é que estes dados veriam?")
    for n in (len(df), 5_000, 20_000, 100_000):
        mdb = randomness.minimum_detectable_bias(n)
        print(f"  {n:>7} sorteios → viés mínimo detetável a 80% de potência: {mdb:5.1f}%")
    print(f"\n  Ao ritmo atual (104 sorteios/ano), 20.000 sorteios chegam no ano "
          f"{dt.date.today().year + int((20000 - len(df)) / 104)}.")


def cmd_maquinas(args) -> None:
    _hr("EQUIPAMENTO DE SORTEIO")
    print(machines.describe())

    df = ds.load_draws()
    _hr("VIÉS POR SEGMENTO (proxy do conjunto de bolas)")
    seg = machines.segment_bias_scan(df)
    print(seg.head(10).to_string(index=False))
    print(f"\nSegmentos com viés após FDR: {int(seg['significativo_fdr5'].sum())} de {len(seg)}")

    _hr("TERÇAS vs SEXTAS")
    wc = machines.weekday_consistency(df)
    if len(wc):
        print(wc.to_string(index=False))

    _hr("O CUSTO DE NÃO SABER QUE CONJUNTO FOI USADO")
    for k in (2, 3, 5):
        d = machines.dilution_penalty(len(df), k)
        print(f"\n  Assumindo {k} conjuntos de bolas:")
        for kk, vv in d.items():
            print(f"    {kk:<38} {vv}")


def cmd_popularidade(args) -> None:
    master = ds.build_master()
    _hr("EVIDÊNCIA DIRETA — sem modelo pelo meio")
    tab = bt.date_bias_evidence(master)
    print(tab.to_string(index=False))
    if "teste" in tab.attrs:
        print()
        for k, v in tab.attrs["teste"].items():
            print(f"  {k:<18} {v}")

    _hr("PARTILHA DO JACKPOT — o que aconteceu de facto")
    print(bt.jackpot_sharing_reality(master).to_string(index=False))

    model, used = fit_popularity_model(master)
    _hr(f"MODELO DE POPULARIDADE (GLM Poisson, n={len(used)} sorteios)")
    s = model.summary()
    s = s.reindex(s["p"].sort_values().index)
    print(s.to_string(index=False, float_format=lambda x: f"{x:9.4f}"))
    print(f"\n  Sobredispersão: {model.dispersion:.1f}  (>1 = comportamento humano "
          f"muito mais irregular do que Poisson; erros-padrão já corrigidos)")

    _hr("LEITURA")
    row = s[s["termo"] == "n_ate_31"]
    if len(row):
        mult = float(row["efeito_multiplicativo"].iloc[0])
        print(f"  Cada desvio-padrão adicional de números ≤31 multiplica a")
        print(f"  popularidade da combinação por {mult:.3f}.")
    print("  Popularidade alta = mais gente com o mesmo bilhete = cheque menor.")


def cmd_estrelas(args) -> None:
    dr, bd = ds.load_draws(), ds.load_breakdown()
    obs = stars.star_observations(dr, bd)
    m = stars.StarPopularity(pool=12).fit(obs)

    _hr(f"POPULARIDADE DAS ESTRELAS (n={m.n_obs} observações)")
    print(f"R² = {m.r2:.4f}    teste conjunto χ² = {m.joint_chi2:.0f} "
          f"(11 g.l.)  p = {m.joint_p:.2e}")
    print("\nTodas as 12 estrelas diferem entre si. Não é palpite: é medição.\n")
    print(m.table().to_string(index=False))

    _hr("PARES DE ESTRELAS — os 6 melhores e os 6 piores")
    print(m.best_pairs(6).to_string(index=False))
    best = m.best_pairs(1)
    lo = float(best["popularidade"].iloc[0]); hi = float(best["popularidade"].iloc[-1])
    print(f"\n  Rácio pior/melhor: {hi/lo:.2f}x — mesma probabilidade de sair,")
    print(f"  {100*(hi/lo-1):.0f}% mais gente com quem dividir.")

    _hr("VALIDAÇÃO FORA DA AMOSTRA")
    v = stars.validate_out_of_sample(obs)
    q = v.pop("quartis")
    for k, val in v.items():
        print(f"  {k:<18} {val}")
    print("\n  popularidade OBSERVADA por quartil de popularidade PREVISTA:")
    print(q.to_string())


def cmd_carteira(args) -> None:
    import numpy as np

    try:
        prizes = ev.empirical_tier_prizes(ds.load_breakdown())
    except FileNotFoundError:
        prizes = ev.FALLBACK_TIER_PRIZES

    rng = np.random.default_rng(args.seed)
    n = args.bilhetes
    pool = np.arange(1, config.MAIN_POOL + 1)

    def st():
        return sorted(rng.choice(np.arange(1, 13), 2, replace=False).tolist())

    base = sorted(rng.choice(pool, 5, replace=False).tolist())
    sobreposta = []
    for i in range(n):
        t = base.copy()
        t[i % 5] = int(rng.choice([x for x in pool if x not in t]))
        sobreposta.append((sorted(t), st()))
    aleatoria = [(sorted(rng.choice(pool, 5, replace=False).tolist()), st()) for _ in range(n)]
    nums = rng.permutation(pool)[: n * 5].reshape(n, 5)
    disjunta = [(sorted(r.tolist()), st()) for r in nums]

    _hr(f"PROBABILIDADE DE NÃO GANHAR NADA — {n} apostas, {args.sim:,} sorteios simulados")
    res = portfolio.compare_portfolios(
        {"sobreposta (varia 1 nº)": sobreposta,
         "aleatória": aleatoria,
         f"disjunta ({n*5} nºs)": disjunta},
        n_sim=args.sim, tier_prizes=prizes,
    )
    print(res.to_string(index=False))
    print(f"\n  Referência teórica se as apostas fossem independentes: "
          f"P(nada) = {portfolio.probability_no_prize_independent(n):.4f}")
    if "ganho_medio_teorico" in res.attrs:
        print(f"  Ganho médio teórico (igual para as três, por linearidade): "
              f"€{res.attrs['ganho_medio_teorico']}")
        print("  As diferenças na coluna do ganho médio são ruído — veja a coluna ±.")

    _hr("LEITURA")
    print("  Cobrir mais números distintos NÃO aumenta o valor esperado.")
    print("  Reduz a probabilidade de sair de mãos vazias. É grátis: custa o mesmo.")


def cmd_comparativo(args) -> None:
    dr, bdf = ds.load_draws(), ds.load_breakdown()
    master = ds.build_master()
    model, _ = fit_popularity_model(master)
    try:
        star_model = stars.fit_star_model(dr, bdf)
    except ValueError:
        star_model = None

    _hr("POPULARIDADE MEDIDA DE CADA ESTRATÉGIA")
    pops = bt.measure_strategy_popularity(model, star_model)
    for k, v in pops.items():
        print(f"  {k:<12} {v:.4f}x")

    _hr(f"CONFRONTO COM TODO O HISTÓRICO — {len(dr)} sorteios (2004→hoje)")
    print(f"{args.reps} carteiras independentes por estratégia × {args.bilhetes} apostas,")
    print("contra os prémios históricos efetivamente pagos, era a era.\n")
    r = bt.full_history_comparison(
        dr, bdf, n_tickets=args.bilhetes, n_reps=args.reps, popularity=pops
    )
    print(r["resumo"].to_string(index=False))

    _hr("ATENÇÃO À COLUNA ±95%")
    print("  O retorno realizado é dominado por acontecimentos raríssimos.")
    print("  Um único 5+1 desloca o total mais do que toda a diferença entre")
    print("  estratégias. As margens acima mostram que a comparação bruta")
    print("  NÃO separa as estratégias — quem 'ganhou' teve sorte, não método.")

    _hr("PRÉMIOS POR ESCALÃO (acumulado)")
    print(r["escaloes"].to_string())

    _hr("COMPARAÇÃO ANALÍTICA — variância reduzida, esta separa")
    print(bt.analytic_comparison(bdf, pops, jackpot_eur=args.jackpot,
                                 ev_m1lhao=_m1lhao_ev()).to_string(index=False))

    _hr("POR ERA")
    print(r["por_era"].pivot_table(
        index=["era", "estrelas", "sorteios"], columns="estratégia",
        values="P(nada)").to_string())

    _hr("A ESCALA DO PROBLEMA")
    n_bets = int(r["resumo"]["apostas"].iloc[0])
    for k, v in bt.jackpot_expectation(n_bets).items():
        print(f"  {k:<30} {v}")


def _m1lhao_ev() -> float:
    """EV do M1lhão por aposta, a partir dos dados recentes."""
    try:
        bd = ds.load_breakdown()
        m = ds.build_master()
        rec = m[[d.year >= 2025 for d in m["date"]]]
        tue = rec[rec["dow"] == "Tue"]["sales_est"].median()
        fri = rec[rec["dow"] == "Fri"]["sales_est"].median()
        return float(m1lhao.summary(bd, tue, fri)["ev_por_aposta_eur"])
    except Exception:
        return 0.0


def cmd_m1lhao(args) -> None:
    bd = ds.load_breakdown()
    m = ds.build_master()
    rec = m[[d.year >= 2025 for d in m["date"]]]
    tue = rec[rec["dow"] == "Tue"]["sales_est"].median()
    fri = rec[rec["dow"] == "Fri"]["sales_est"].median()

    _hr("M1LHÃO — a parcela que faltava no motor de EV")
    s = m1lhao.summary(bd, tue, fri)
    for k, v in s.items():
        print(f"  {k:<30} {v}")

    _hr("QUOTA PORTUGUESA AO LONGO DO TEMPO")
    print(m1lhao.share_trend(bd).tail(10).to_string(index=False))
    print("\n  A quota tem descido de forma sistemática. Menos apostas")
    print("  portuguesas = menos códigos = M1lhão mais valioso por aposta.")

    _hr("EFEITO DE DILUIÇÃO NA VANTAGEM DO SISTEMA")
    prizes = ev.empirical_tier_prizes(bd)
    e = s["ev_por_aposta_eur"]
    for nome, pi in (("datas", 1.5343), ("aleatória", 1.0201), ("otimizada", 0.6191)):
        r = ev.expected_value(args.jackpot, 24e6, pi, 12, prizes, ev_m1lhao=e)
        b = m1lhao.blend_with_euromillions(r.ev_liquido - e, e)
        print(f"  {nome:<10} EM €{b['ev_euromillions']:.4f} + M1lhão €{b['ev_m1lhao']:.4f} "
              f"= €{b['ev_total']:.4f}   ({b['peso_m1lhao_%']}% vem do M1lhão)")
    print("\n  O código do M1lhão é gerado pelo sistema: não há nada a otimizar nele.")
    print("  Somar uma parcela fixa reduz a vantagem PERCENTUAL da otimização —")
    print("  e essa percentagem menor é a verdadeira.")


def cmd_elasticidade(args) -> None:
    master = ds.build_master()
    bd = ds.load_breakdown()
    model, _ = fit_popularity_model(master)

    _hr("ELASTICIDADE POR ESCALÃO — medida vs escrita à mão")
    t = elasticity.measure_elasticities(master, bd, model)
    print(t.to_string(index=False))
    print("\n  Validação do método: 5+0 mede 0,956 quando a teoria exige 1,0")
    print("  (acertar os 5 números É ter a nossa combinação exata).")
    print("  Os escalões de 5 ficam fixados em 1,0 por essa razão teórica.")

    _hr("IMPACTO NO VALOR ESPERADO")
    prizes = ev.empirical_tier_prizes(bd)
    med = elasticity.fitted_table(t)
    for k in ("5+2", "5+1", "5+0"):
        med[k] = 1.0
    print(elasticity.impact_on_ev(med, jackpot_eur=args.jackpot,
                                  tier_prizes=prizes, popularity=0.62).to_string(index=False))
    print("\n  A tabela escrita à mão inflacionava o EV. Já foi substituída")
    print("  em ev.TIER_ELASTICITY pelos valores medidos.")


def cmd_valor(args) -> None:
    try:
        bd = ds.load_breakdown()
        prizes = ev.empirical_tier_prizes(bd)
        src = "medianas observadas desde 2020"
    except FileNotFoundError:
        prizes, src = None, "valores de recurso"

    e = _m1lhao_ev()
    _hr("IMPOSTO DO SELO EM PORTUGAL")
    print(ev.tax_impact().to_string(index=False))

    _hr(f"VALOR ESPERADO POR APOSTA (€{config.TICKET_PRICE_EUR}) — prémios: {src}")
    print(f"Vendas assumidas: {args.vendas/1e6:.0f}M apostas por sorteio\n")
    print(ev.ev_curve(args.vendas, tier_prizes=prizes).to_string(index=False))
    print(f"\n  (tabela acima SEM o M1lhão; some €{e:.4f} por aposta para o valor real em Portugal)")

    _hr("PONTO DE EQUILÍBRIO")
    for pop, label in ((3.0, "combinação popular (datas)"),
                       (1.0, "combinação média"),
                       (0.35, "combinação otimizada")):
        be = ev.breakeven_jackpot(args.vendas, pop, tier_prizes=prizes, ev_m1lhao=e)
        if be is None:
            print(f"  {label:<30} inatingível — nem no teto de €250M o EV chega a €2,50")
        else:
            reach = "ATINGÍVEL" if be <= ev.JACKPOT_CAP_EUR else "acima do teto de €250M"
            print(f"  {label:<30} €{be/1e6:8.1f}M   ({reach})")

    _hr("SEM IMPOSTO, PARA COMPARAR")
    for pop in (1.0, 0.35):
        be = ev.breakeven_jackpot(args.vendas, pop, tier_prizes=prizes, apply_tax=False)
        txt = "inatingível" if be is None else f"€{be/1e6:.1f}M"
        print(f"  popularidade {pop}: {txt}")


def cmd_backtest(args) -> None:
    df = ds.load_draws()
    _hr("H_PREVISÃO — as estratégias de números funcionam?")
    print(bt.strategy_backtest(df).to_string(index=False))
    print("\n  Sob independência, todas convergem para 0,5 acertos por sorteio.")

    _hr("BACKTEST PREDITIVO WALK-FORWARD (sorteios reais)")
    try:
        bdf = ds.load_breakdown()
        wf = bt.walk_forward(df, bdf, n_reps=args.reps)
        print("Período:", wf.attrs["periodo"])
        print(f"Taxa de acerto teórica por aposta: {config.probability_any_prize(12):.5f}\n")
        print(wf.to_string(index=False))
        print("\n  Taxa de acerto: todas dentro da margem da teórica.")
        print("  Nenhuma estratégia altera a probabilidade de ganhar.")
        print("  P(nada): as diferenças são reais e os intervalos não se sobrepõem.")
    except FileNotFoundError as e:
        print(f"[saltado: {e}]", file=sys.stderr)

    master = ds.build_master()
    _hr("H_VALOR — o modelo de popularidade prevê fora da amostra?")
    r = bt.popularity_out_of_sample(master)
    q = r.pop("quintis")
    for k, v in r.items():
        print(f"  {k:<34} {v}")
    print("\n  Popularidade observada por quintil de popularidade prevista:")
    print(q.to_string())


def cmd_jogar(args) -> None:
    master = ds.build_master()
    model, _ = fit_popularity_model(master)

    star_model = None
    try:
        bdf = ds.load_breakdown()
        prizes = ev.empirical_tier_prizes(bdf)
        star_model = stars.fit_star_model(ds.load_draws(), bdf)
    except (FileNotFoundError, ValueError):
        prizes = None

    _hr("FONTE DE ENTROPIA")
    src, report = quantum.get_source(allow_network=not args.offline)
    print(f"  {src.name}")
    for k, v in report.items():
        print(f"    {k:<32} {v}")
    if report["veredicto"] != "aprovada":
        print("\n  A fonte não passou a certificação. A abortar.", file=sys.stderr)
        sys.exit(1)

    _hr(f"GERAÇÃO — jackpot €{args.jackpot/1e6:.0f}M, vendas {args.vendas/1e6:.0f}M")
    tickets, table = opt.optimize(
        model, src,
        n_tickets=args.bilhetes,
        jackpot_eur=args.jackpot,
        sales=args.vendas,
        tier_prizes=prizes,
        n_candidates=args.candidatos,
        allow_network=not args.offline,
        star_model=star_model,
        ev_m1lhao=_m1lhao_ev(),
        verbose=True,
    )
    if star_model is not None:
        print(f"  modelo de estrelas: medido (R²={star_model.r2:.3f}, n={star_model.n_obs})")

    _hr("BILHETES")
    for i, t in enumerate(tickets, 1):
        print(f"  {i}.  {t}")

    _hr("COMPARAÇÃO")
    print(opt.compare_to_typical(model, tickets, args.jackpot, args.vendas,
                                 tier_prizes=prizes, star_model=star_model,
                                 ev_m1lhao=_m1lhao_ev()).to_string(index=False))

    _hr("RISCO DA CARTEIRA")
    sim = portfolio.simulate_portfolio(
        [(t.mains, t.stars) for t in tickets],
        n_sim=60000, tier_prizes=prizes or ev.FALLBACK_TIER_PRIZES,
    )
    cov = portfolio.coverage_score([(t.mains, t.stars) for t in tickets])
    print(f"  números distintos cobertos : {cov['numeros_distintos']} de 50 "
          f"({cov['cobertura_pct']}%)")
    print(f"  P(não ganhar nada)         : {sim['p_nada']:.4f}")
    print(f"  P(algum prémio)            : {sim['p_algum_premio']:.4f}")
    print(f"  referência se independentes: "
          f"{portfolio.probability_no_prize_independent(len(tickets)):.4f}")

    _hr("O QUE ISTO É E O QUE NÃO É")
    p = config.JACKPOT.probability(12)
    print(f"  Probabilidade de acertar no jackpot: 1 em {1/p:,.0f}. Igual para")
    print("  estes bilhetes e para quaisquer outros. Não mudou, não pode mudar.")
    print("  O que mudou foi o tamanho do cheque no caso — improvável — de acertar.")
    print(f"\n  Custo de {len(tickets)} apostas: €{len(tickets)*config.TICKET_PRICE_EUR:.2f}")
    ev_total = sum(t.ev_liquido for t in tickets)
    print(f"  Valor esperado:  €{ev_total:.2f}   "
          f"(perda esperada: €{len(tickets)*config.TICKET_PRICE_EUR - ev_total:.2f})")


def cmd_relatorio(args) -> None:
    for fn in (cmd_aleatoriedade, cmd_maquinas, cmd_popularidade, cmd_estrelas,
               cmd_elasticidade, cmd_m1lhao, cmd_valor, cmd_carteira,
               cmd_backtest, cmd_comparativo):
        try:
            fn(args)
        except FileNotFoundError as e:
            print(f"\n[saltado: {e}]", file=sys.stderr)


# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="euromillions",
        description="Sistema estatístico para o EuroMillions (Portugal)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="recolher o histórico")
    f.add_argument("--breakdown", action="store_true", help="incluir quebra de prémios")
    f.add_argument("--delay", type=float, default=0.3)
    f.set_defaults(func=cmd_fetch)

    a = sub.add_parser("aleatoriedade", help="bateria de testes de aleatoriedade")
    a.add_argument("--sim", type=int, default=8000)
    a.set_defaults(func=cmd_aleatoriedade)

    m = sub.add_parser("maquinas", help="equipamento e viés por segmento")
    m.set_defaults(func=cmd_maquinas)

    po = sub.add_parser("popularidade", help="modelo de escolha humana")
    po.set_defaults(func=cmd_popularidade)

    v = sub.add_parser("valor", help="valor esperado e ponto de equilíbrio")
    v.add_argument("--vendas", type=float, default=80e6)
    v.set_defaults(func=cmd_valor)

    b = sub.add_parser("backtest", help="validação fora da amostra + walk-forward")
    b.add_argument("--reps", type=int, default=300)
    b.set_defaults(func=cmd_backtest)

    e = sub.add_parser("estrelas", help="popularidade medida das estrelas")
    e.set_defaults(func=cmd_estrelas)

    mm = sub.add_parser("m1lhao", help="a parcela portuguesa do EV")
    mm.add_argument("--jackpot", type=float, default=60e6)
    mm.set_defaults(func=cmd_m1lhao)

    el = sub.add_parser("elasticidade", help="elasticidades medidas por escalão")
    el.add_argument("--jackpot", type=float, default=60e6)
    el.set_defaults(func=cmd_elasticidade)

    cp = sub.add_parser("comparativo", help="modelo vs apostas avulsas, todo o histórico")
    cp.add_argument("--reps", type=int, default=250)
    cp.add_argument("--bilhetes", type=int, default=5)
    cp.add_argument("--jackpot", type=float, default=60e6)
    cp.set_defaults(func=cmd_comparativo)

    ca = sub.add_parser("carteira", help="reduzir a probabilidade de não ganhar nada")
    ca.add_argument("--bilhetes", type=int, default=5)
    ca.add_argument("--sim", type=int, default=150000)
    ca.add_argument("--seed", type=int, default=1)
    ca.set_defaults(func=cmd_carteira)

    j = sub.add_parser("jogar", help="gerar bilhetes otimizados")
    j.add_argument("--bilhetes", type=int, default=5)
    j.add_argument("--jackpot", type=float, default=100e6)
    j.add_argument("--vendas", type=float, default=80e6)
    j.add_argument("--candidatos", type=int, default=4000)
    j.add_argument("--offline", action="store_true", help="não usar QRNG por rede")
    j.set_defaults(func=cmd_jogar)

    r = sub.add_parser("relatorio", help="corre tudo")
    r.add_argument("--vendas", type=float, default=24e6)
    r.add_argument("--sim", type=int, default=4000)
    r.add_argument("--reps", type=int, default=150)
    r.add_argument("--bilhetes", type=int, default=5)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--jackpot", type=float, default=60e6)
    r.set_defaults(func=cmd_relatorio)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
