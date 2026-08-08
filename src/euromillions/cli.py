"""
Interface de linha de comandos.

    python -m euromillions.cli fetch [--breakdown]
    python -m euromillions.cli aleatoriedade
    python -m euromillions.cli maquinas
    python -m euromillions.cli popularidade
    python -m euromillions.cli valor [--jackpot 100e6]
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

from . import config, ev, machines, quantum, randomness
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


def cmd_valor(args) -> None:
    try:
        bd = ds.load_breakdown()
        prizes = ev.empirical_tier_prizes(bd)
        src = "medianas observadas desde 2020"
    except FileNotFoundError:
        prizes, src = None, "valores de recurso"

    _hr("IMPOSTO DO SELO EM PORTUGAL")
    print(ev.tax_impact().to_string(index=False))

    _hr(f"VALOR ESPERADO POR APOSTA (€{config.TICKET_PRICE_EUR}) — prémios: {src}")
    print(f"Vendas assumidas: {args.vendas/1e6:.0f}M apostas por sorteio\n")
    print(ev.ev_curve(args.vendas, tier_prizes=prizes).to_string(index=False))

    _hr("PONTO DE EQUILÍBRIO")
    for pop, label in ((3.0, "combinação popular (datas)"),
                       (1.0, "combinação média"),
                       (0.35, "combinação otimizada")):
        be = ev.breakeven_jackpot(args.vendas, pop, tier_prizes=prizes)
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

    try:
        prizes = ev.empirical_tier_prizes(ds.load_breakdown())
    except FileNotFoundError:
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
        verbose=True,
    )

    _hr("BILHETES")
    for i, t in enumerate(tickets, 1):
        print(f"  {i}.  {t}")

    _hr("COMPARAÇÃO")
    print(opt.compare_to_typical(model, tickets, args.jackpot, args.vendas,
                                 tier_prizes=prizes).to_string(index=False))

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
    for fn in (cmd_aleatoriedade, cmd_maquinas, cmd_popularidade, cmd_valor, cmd_backtest):
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

    b = sub.add_parser("backtest", help="validação fora da amostra")
    b.set_defaults(func=cmd_backtest)

    j = sub.add_parser("jogar", help="gerar bilhetes otimizados")
    j.add_argument("--bilhetes", type=int, default=5)
    j.add_argument("--jackpot", type=float, default=100e6)
    j.add_argument("--vendas", type=float, default=80e6)
    j.add_argument("--candidatos", type=int, default=4000)
    j.add_argument("--offline", action="store_true", help="não usar QRNG por rede")
    j.set_defaults(func=cmd_jogar)

    r = sub.add_parser("relatorio", help="corre tudo")
    r.add_argument("--vendas", type=float, default=80e6)
    r.add_argument("--sim", type=int, default=4000)
    r.set_defaults(func=cmd_relatorio)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
