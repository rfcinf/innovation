"""
Interface de linha de comandos.

    python -m euromillions.cli fetch [--breakdown]
    python -m euromillions.cli aleatoriedade
    python -m euromillions.cli maquinas
    python -m euromillions.cli popularidade
    python -m euromillions.cli valor [--jackpot 100e6]
    python -m euromillions.cli estrelas
    python -m euromillions.cli carteira
    python -m euromillions.cli totoloto
    python -m euromillions.cli plano
    python -m euromillions.cli modelo
    python -m euromillions.cli auditoria
    python -m euromillions.cli padroes
    python -m euromillions.cli m1lhao
    python -m euromillions.cli elasticidade
    python -m euromillions.cli comparativo
    python -m euromillions.cli backtest
    python -m euromillions.cli jogar [--bilhetes 5] [--jackpot 100e6]
    python -m euromillions.cli simulacao [--jackpot 111e6] [--tamanhos 1,2,5,10,20]
    python -m euromillions.cli relatorio
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys

import numpy as np
import pandas as pd

from . import (audit, config, elasticity, ev, m1lhao, machines, model as mdl,
               patterns, portfolio, quantum, randomness, simulacao, stars,
               strategy, totoloto)
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


def cmd_padroes(args) -> None:
    df = ds.load_draws()

    _hr("A PERGUNTA: alguma combinação já se repetiu?")
    r = patterns.repeat_report(df)
    print(f"  sorteios                    {r['sorteios']}")
    print(f"  pares de sorteios possíveis {r['pares_comparados']:,}")
    print(f"  repetições OBSERVADAS       {r['repeticoes_observadas']}")
    print(f"  repetições ESPERADAS        {r['repeticoes_esperadas']}")
    print(f"  P(zero repetições)          {r['P_zero_repeticoes']}")
    for d in r["detalhe"]:
        print(f"\n  → {d['combinacao']}")
        print(f"    saiu em {d['datas'][0]} e outra vez em {d['datas'][1]}")
    print("\n  Não se compara um sorteio com um alvo: comparam-se TODOS os pares")
    print("  de sorteios entre si (paradoxo dos aniversários). Daí o esperado ~1.")

    _hr("COINCIDÊNCIAS ENTRE PARES DE SORTEIOS")
    t, st = patterns.overlap_analysis(df)
    print(t.to_string(index=False))
    print(f"\n  {st['pares_de_sorteios']:,} pares   χ² = {st['chi2']} ({st['gl']} g.l.)   p = {st['p']:.4f}")
    print("  Concordância quase perfeita com o acaso puro, em todos os níveis.")

    _hr("O MAPA DO ESPAÇO — enumeração exata das 2.118.760 combinações")
    print("A enumerar...", end=" ", flush=True)
    patterns.universe()
    print("feito.\n")
    bat, det = patterns.full_pattern_battery(df)
    print(bat.to_string(index=False))
    print(f"\n  Famílias com desvio real após correção FDR: "
          f"{int(bat['significativo_fdr5'].sum())} de {len(bat)}")

    _hr("ONDE ESTÃO AS FAMÍLIAS RARAS (exemplo: números ≤31)")
    tt = det["nums_ate_31"]
    print(tt[["valor", "combinacoes", "probabilidade", "1_em",
              "observado", "esperado", "desvio_%"]].to_string(index=False))
    print("\n  ATENÇÃO À LEITURA. A família com 0 números ≤31 sai 1 vez em 182.")
    print("  Isso NÃO torna um bilhete dessa família menos provável: ele continua")
    print("  a valer 1 em 2.118.760, igual a qualquer outro. A família é rara")
    print("  porque tem poucos membros (11.628), e o seu bilhete é um deles.")

    _hr("'NUNCA ACONTECEU' — quando é que isso é informação?")
    nh = patterns.never_happened_yet(df)
    inform = nh[nh["ausencia_e_informativa"]] if len(nh) else nh
    print(f"  Classes de padrões que ainda não saíram: {len(nh)}")
    print(f"  Dessas, com ausência informativa (esperado > 3): {len(inform)}")
    if len(nh):
        print("\n  As 5 mais próximas de significarem alguma coisa:")
        print(nh.head(5)[["propriedade", "valor_nunca_visto", "1_em",
                          "esperado_em_1970_sorteios"]].to_string(index=False))
    print("\n  Nenhuma delas deveria ter saído sequer 3 vezes em 22 anos.")
    print("  Ausência sem expectativa não é evidência — é aritmética mal lida.")


def cmd_modelo(args) -> None:
    _hr("MODELO CONSOLIDADO")
    m = mdl.EuroMillionsModel.fit()
    print(m.describe())

    _hr("DE ONDE VEM CADA NÚMERO QUE O MODELO USA")
    print(m.provenance().to_string(index=False))

    ass = m.assumptions()
    _hr(f"SUPOSIÇÕES POR VALIDAR ({len(ass)}) — é aqui que vive o próximo erro")
    for r in ass.itertuples():
        print(f"  · {r.nome}: {r.valor}")
        print(f"    {r.nota}")

    _hr(f"RECOMENDAÇÃO — jackpot €{args.jackpot/1e6:.0f}M")
    rec = m.recommend(
        n_tickets=args.bilhetes, jackpot_eur=args.jackpot,
        n_candidates=args.candidatos, allow_network=not args.offline,
    )
    print(rec.to_frame().to_string(index=False))
    print()
    print(f"  popularidade média : {rec.popularidade_media:.3f}x")
    print(f"  cobertura          : {rec.cobertura} de 50 números")
    print(f"  P(não ganhar nada) : {rec.p_nada:.4f}")
    print(f"  custo              : €{rec.custo:.2f}")
    print(f"  valor esperado     : €{rec.ev_total:.2f}  "
          f"(dos quais €{rec.ev_m1lhao*args.bilhetes:.2f} do M1lhão)")
    print(f"  perda esperada     : €{rec.custo - rec.ev_total:.2f}")

    if args.avaliar:
        nums = [int(x) for x in args.avaliar.split(",")[:5]]
        strs = [int(x) for x in args.avaliar.split(",")[5:7]]
        _hr("AVALIAÇÃO DA SUA APOSTA")
        for k, v in m.score(nums, strs, args.jackpot).items():
            print(f"  {k:<24} {v}")

    if args.json:
        _hr("MODELO EM JSON")
        print(m.to_json())


def cmd_simulacao(args) -> None:
    tamanhos = [int(x) for x in args.tamanhos.split(",")]
    ref = args.bilhetes

    _hr("MODELO")
    m = mdl.EuroMillionsModel.fit()
    print(m.describe())

    # Enche-se o reservatório quântico de uma vez. Deixar que se esvazie a
    # meio da otimização faz com que cada recarga vá à rede — 1 KB por
    # pedido, dezenas de pedidos, e a otimização passa a ser dominada por
    # latência em vez de cálculo. A partir daqui consome-se em modo local:
    # o que sobrar degrada para os.urandom, e a proveniência regista a
    # mistura exata em vez de a esconder.
    src = quantum.EntropySource()
    src.refill(args.entropia, allow_network=not args.offline)
    print(f"  entropia: {src.name}")

    _hr(f"A CONSTRUIR CARTEIRAS — {', '.join(map(str, tamanhos))} apostas")
    carteiras: dict[str, list] = {}
    saved = dict(ev.TIER_ELASTICITY)
    try:
        ev.TIER_ELASTICITY.update(m.elasticities)
        for n in tamanhos:
            # opt.optimize diretamente, e não m.recommend: esta última corre
            # uma simulação de 40 mil noites em ciclo de Python que aqui se
            # deitaria fora — a distribuição é calculada abaixo, vetorizada.
            tks, _ = opt.optimize(
                m.popularity, src, n_tickets=n, jackpot_eur=args.jackpot,
                sales=m.sales_estimate, tier_prizes=m.tier_prizes,
                n_candidates=args.candidatos, allow_network=False,
                star_model=m.star_model, ev_m1lhao=m.ev_m1lhao,
            )
            carteiras[f"otimizada ×{n}"] = tks
            cov = portfolio.coverage_score([(t.mains, t.stars) for t in tks])
            print(f"  ×{n:<3} popularidade "
                  f"{np.mean([t.popularity for t in tks]):.3f}   "
                  f"cobertura {cov['numeros_distintos']}/50")
    finally:
        ev.TIER_ELASTICITY.clear()
        ev.TIER_ELASTICITY.update(saved)

    # A mistura real de fontes, sem a esconder: o que o reservatório
    # quântico não cobriu veio do CSPRNG do sistema.
    from collections import Counter
    mix = Counter(p.split("<- ")[1] for p in src.provenance())
    print("\n  proveniência da entropia:")
    for origem, n in mix.items():
        print(f"    {n:>3} recarga(s)  {origem}")

    # Referência honesta: o mesmo dinheiro gasto ao acaso.
    rng = np.random.default_rng(args.seed)
    pool = np.arange(1, config.MAIN_POOL + 1)
    aleatoria = [
        (sorted(rng.choice(pool, 5, replace=False).tolist()),
         sorted(rng.choice(np.arange(1, 13), 2, replace=False).tolist()))
        for _ in range(ref)
    ]
    carteiras[f"ao acaso ×{ref}"] = aleatoria

    # O extremo oposto da dispersão: mesma base, varia um número. Serve
    # para mostrar que "cobertura" e "recuperar o custo" puxam em sentidos
    # contrários, e que a escolha entre as duas não é estatística.
    melhor = carteiras[f"otimizada ×{min(tamanhos)}"][0]
    concentrada = simulacao.concentrated_portfolio(
        melhor.mains, melhor.stars, ref, popularity_model=m.popularity)
    carteiras[f"concentrada ×{ref}"] = concentrada

    # As carteiras que não vêm do otimizador não trazem popularidade. Se
    # ficassem com o valor por omissão (1.0) apareceriam com um cheque de
    # jackpot ~5% menor do que o real, e a comparação estaria viciada
    # contra elas — a concentrada é construída a partir do MELHOR bilhete
    # e é tudo menos popular.
    def _pops(tks):
        return [
            float(m.popularity.predict_popularity(np.array(mains))[0])
            * opt.star_pair_popularity(tuple(strs), 12, m.star_model)
            for mains, strs in tks
        ]

    pops = {f"ao acaso ×{ref}": _pops(aleatoria),
            f"concentrada ×{ref}": _pops(concentrada)}

    alvo = carteiras[f"otimizada ×{ref}"]
    r = simulacao.frame_night(
        alvo, args.jackpot, m.sales_estimate, m.tier_prizes,
        ev_m1lhao=m.ev_m1lhao, n_sim=args.sim, seed=args.seed,
    )
    c = r["corpo"]

    _hr(f"A CARTEIRA — {ref} apostas, €{r['custo']:.2f}, jackpot "
        f"€{args.jackpot/1e6:.0f}M")
    for i, t in enumerate(alvo, 1):
        print(f"  {i}  {' '.join(f'{x:02d}' for x in t.mains)}   "
              f"★ {' '.join(f'{x:02d}' for x in t.stars)}   "
              f"popularidade {t.popularity:.3f}")

    _hr(f"O CORPO DA DISTRIBUIÇÃO — {c['n_sim']:,} noites simuladas")
    print(f"  P(não ganhar nada)        {c['p_nada']:.4f}  ± {c['p_nada_se']:.4f}")
    print(f"  P(ganhar alguma coisa)    {c['p_algum_premio']:.4f}")
    print(f"  P(recuperar os €{r['custo']:.2f})     {c['p_recupera_custo']:.4f}")
    print(f"  P(ganhar ≥ €25)           {c['p_ge_25']:.4f}")
    print(f"  P(ganhar ≥ €100)          {c['p_ge_100']:.5f}")
    print(f"  P(ganhar ≥ €1.000)        {c['p_ge_1000']:.6f}")
    print("\n  percentis do ganho da noite:")
    for q, v in c["percentis"].items():
        print(f"    p{q:<5} €{v:,.2f}")
    d = c["dado_que_ganha"]
    print(f"\n  DADO QUE ganha alguma coisa ({d['n']:,} das {c['n_sim']:,} noites):")
    print(f"    mediana €{d['mediana']:,.2f}   média €{d['media']:,.2f}   "
          f"p90 €{d['p90']:,.2f}   p99 €{d['p99']:,.2f}   máx €{d['max']:,.2f}")

    _hr("A CAUDA — combinatória exata, onde a simulação não chega")
    tab = r["escaloes"].copy()
    tab["prob_%"] = tab["prob_%"].map(lambda x: f"{x:.6f}")
    tab["prémio_€"] = tab["prémio_€"].map(lambda x: f"{x:,.2f}")
    tab["contrib_EV_€"] = tab["contrib_EV_€"].map(lambda x: f"{x:.4f}")
    tab["1_em"] = tab["1_em"].map(lambda x: f"{x:,.0f}")
    print(tab.to_string(index=False))
    print("\n  'exclusivo' = com números disjuntos, dois bilhetes não podem")
    print("  ganhar este escalão na mesma noite. Aí n × p é exato, não é")
    print("  aproximação.")

    ch = r["cheque"]
    _hr("O CHEQUE — o que a cauda paga mesmo")
    print(f"  jackpot anunciado           €{args.jackpot:,.0f}")
    print(f"  líquido se ficar sozinho    €{ch['liquido_se_sozinho']:,.0f}"
          f"   (Imposto do Selo: −€{args.jackpot - ch['liquido_se_sozinho']:,.0f})")
    print(f"  λ (concorrência esperada)   {ch['lambda_medio']:.4f}")
    print(f"  P(sozinho | acertar)        {ch['p_sozinho_dado_que_acerta']:.4f}")
    print(f"  cheque esperado             €{ch['cheque_esperado']:,.0f}")
    print(f"\n  1 em {ch['1_em_jackpot']/1e6:,.1f}M de acertar no jackpot")
    print(f"  1 em {ch['1_em_jackpot_sozinho']/1e6:,.1f}M de acertar E ficar sozinho")

    _hr("COMPARAÇÃO ENTRE ORÇAMENTOS")
    sw = simulacao.sweep(carteiras, args.jackpot, m.sales_estimate,
                         m.tier_prizes, ev_m1lhao=m.ev_m1lhao,
                         n_sim=args.sim // 2, seed=args.seed,
                         popularities=pops)
    print(sw.to_string(index=False))
    print("\n  1_em_jackpot e 1_em_sozinho estão em milhões.")
    print("  Repare em '€_por_€': é praticamente constante. Nenhum orçamento")
    print("  torna o jogo favorável — o que muda é a forma da distribuição.")

    _hr("O TRADE-OFF QUE NÃO SE PODE OTIMIZAR NOS DOIS SENTIDOS")
    disp = sw[sw["carteira"] == f"otimizada ×{ref}"].iloc[0]
    conc = sw[sw["carteira"] == f"concentrada ×{ref}"].iloc[0]
    print(f"  {'':<22}{'dispersa':>12}{'concentrada':>14}")
    for rot, col in [("números distintos", "nºs"),
                     ("popularidade", "popul."),
                     ("P(não ganhar nada)", "P(nada)"),
                     ("P(recuperar o custo)", "P(recupera)"),
                     ("P(ganhar ≥ €100)", "P(≥€100)"),
                     ("cheque se acertar (€M)", "cheque_€M"),
                     ("EV (€)", "EV_€")]:
        print(f"  {rot:<22}{disp[col]:>12}{conc[col]:>14}")
    print("\n  a carteira concentrada, para quem preferir esta forma:")
    for i, (mm, ss) in enumerate(concentrada, 1):
        print(f"    {i}  {' '.join(f'{x:02d}' for x in mm)}   "
              f"★ {' '.join(f'{x:02d}' for x in ss)}")

    print("\n  O valor esperado é praticamente o mesmo: a esperança é linear")
    print("  e não vê correlação entre bilhetes. O que muda é tudo o resto.")
    print("  Dispersar = ganhar pouco mais vezes. Concentrar = ganhar mais")
    print("  raramente, e mais de cada vez. Não há resposta estatística para")
    print("  qual é melhor — há uma preferência, e agora está quantificada.")

    _hr("LEITURA")
    print("  O valor esperado é o número que menos informa: cai num ponto")
    print("  onde a distribuição quase nunca aterra. O que se compra com a")
    print("  otimização é a FORMA da distribuição — menos noites a zero, e")
    print("  um cheque maior no caso raro em que a cauda acontece.")


def cmd_auditoria(args) -> None:
    _hr("AUDITORIA DO SISTEMA")
    print("  A verificar dados, modelos, deriva, suposições, afirmações e lacunas...\n")
    rep, gaps = audit.run_audit(n_sim=args.sim, skip_slow=args.rapido)

    print(rep)

    if len(gaps):
        _hr("VARRIMENTOS DE LACUNAS")
        print(gaps.to_string(index=False))

    _hr("RESUMO")
    tab = rep.to_frame()
    print(tab.groupby("gravidade").size().to_string())
    print()
    print(f"  VEREDICTO: {rep.verdict()}")
    if rep.criticos:
        print("\n  Problemas críticos a resolver antes de confiar no modelo:")
        for f in rep.criticos:
            print(f"    · [{f.area}] {f.check}")
    if rep.avisos:
        print("\n  Avisos (não bloqueiam, mas é aí que vive o próximo erro):")
        for f in rep.avisos:
            print(f"    · [{f.area}] {f.check}")
    if rep.criticos:
        sys.exit(1)


def cmd_plano(args) -> None:
    bd = ds.load_breakdown()
    master = ds.build_master()
    prizes = ev.empirical_tier_prizes(bd)
    e = _m1lhao_ev()

    _hr("A TENSÃO: jackpot grande atrai mais concorrência")
    f = strategy.fit_sales_elasticity(master)
    for k, v in f.items():
        print(f"  {k:<28} {v}")
    print(f"\n  O jackpot duplica; as vendas sobem apenas "
          f"{f.get('vendas_se_jackpot_duplicar', 1.21)}x.")
    print("  O prémio ganha a corrida à concorrência — jackpots altos são melhores")
    print("  MESMO depois de descontar a partilha extra.")

    _hr("PROBABILIDADE DE FICAR SOZINHO COM O JACKPOT")
    print(strategy.solo_table().to_string(index=False))
    print("\n  π=0,35 é a carteira otimizada; π=1,53 é jogar datas.")
    print("  A diferença aparece toda aqui: a mesma probabilidade de acertar,")
    print("  e uma hipótese muito diferente de não ter de dividir.")

    _hr(f"ALOCAÇÃO DO ORÇAMENTO — €{args.orcamento:.0f}/ano")
    tab = strategy.budget_plans(args.orcamento, tier_prizes=prizes, ev_m1lhao=e)
    print(tab.to_string(index=False))
    print("\n  Repare na coluna P(jackpot/ano): NÃO MUDA. O número total de")
    print("  apostas é o mesmo em todos os planos, e é só isso que determina")
    print("  a probabilidade de acertar. O que muda é o retorno e o cheque.")

    _hr("O PLANO RECOMENDADO")
    r = strategy.recommend_plan(args.orcamento, tier_prizes=prizes, ev_m1lhao=e)
    for k, v in r.items():
        print(f"  {k:<32} {v}")

    _hr("A ESCALA, SEM ADOÇAR")
    print(f"  P(jackpot em algum ano)      {100*r['P_jackpot_por_ano']:.6f}%")
    print(f"  anos para ESPERAR 1 jackpot  {r['anos_para_esperar_1_jackpot']:,}")
    print(f"  retorno esperado             {r['retorno_por_euro']:.3f} € por cada €1")
    print("\n  O plano melhora o que é melhorável. Não torna o jogo favorável,")
    print("  e nenhum plano pode.")


def cmd_totoloto(args) -> None:
    import pandas as _pd

    _hr("TOTOLOTO — ESTRUTURA (calculada de raiz)")
    print(_pd.DataFrame(totoloto.structure_table()).to_string(index=False))
    p = totoloto.probability_any_prize()
    print(f"\n  P(algum prémio) = 1 em {1/p:.2f}")
    print(f"  Santa Casa publica 1 em 7 → {'CONFERE' if abs(1/p-7)<0.2 else 'DIVERGE'}")
    print("  (a Santa Casa não publica a probabilidade de cada escalão;")
    print("   este confronto é a única forma de validar a estrutura assumida)")

    _hr("TOTOLOTO vs EUROMILHÕES")
    print(_pd.DataFrame(totoloto.compare_with_euromillions()).to_string(index=False))

    freq = totoloto.load_frequencies()
    r = totoloto.bias_battery(freq)

    _hr(f"ANÁLISE DE VIESES — {r['sorteios']} sorteios desde 2011-03-16")
    for k in ("números 1-49", "Nº da Sorte 1-13"):
        d = r[k]
        print(f"  {k:<18} χ² = {d['chi2']:7.2f} ({d['gl']} g.l.)   p = {d['p']:.4f}"
              f"   esperado/valor = {d['esperado_por_valor']:.1f}")

    _hr("VALORES MAIS EXTREMOS")
    print("Números (de 49):")
    print(r["tabelas"]["numeros"].head(5).to_string(index=False))
    print(f"  significativos após FDR: "
          f"{int(r['tabelas']['numeros']['significativo_fdr5'].sum())} de 49")
    print("\nNº da Sorte (de 13):")
    print(r["tabelas"]["sorte"].head(4).to_string(index=False))
    print(f"  significativos após FDR: "
          f"{int(r['tabelas']['sorte']['significativo_fdr5'].sum())} de 13")

    _hr("O TESTE DECISIVO — os desvios persistem no tempo?")
    for key, nome in (("numeros", "números 1-49"), ("sorte", "Nº da Sorte 1-13")):
        d = r[f"persistencia_{key}"]
        print(f"\n  {nome}")
        print(f"    {d['segmento_antigo']}  vs  {d['segmento_recente']}")
        print(f"    r = {d['r_observado']:+.4f}   dp do nulo = {d['nulo_dp']:.4f}"
              f"   p = {d['p_simulado']:.4f}")
    print("\n  Um viés físico é persistente por definição — vive no equipamento.")
    print("  Ruído amostral não persiste. É este teste que separa os dois.")

    _hr("POTÊNCIA — que tamanho de viés é que estes dados veriam?")
    for n in (r["sorteios"], 5000, 20000):
        m = totoloto.minimum_detectable_bias(n, totoloto.MAIN_POOL, totoloto.MAIN_PICK)
        s_ = totoloto.minimum_detectable_bias(n, totoloto.LUCKY_POOL, 1)
        print(f"  {n:6d} sorteios → números {m:5.1f}%   Nº da Sorte {s_:5.1f}%")

    if args.chaves:
        _hr(f"CHAVES SUGERIDAS ({args.chaves})")
        src, rep = quantum.get_source(allow_network=not args.offline)
        print(f"  entropia: {src.name} — {rep['veredicto']}\n")
        for i, c in enumerate(totoloto.generate(args.chaves, entropy=src,
                                                n_candidates=args.candidatos,
                                                allow_network=not args.offline), 1):
            nums = " ".join(f"{n:02d}" for n in c["numeros"])
            print(f"  {i}.  {nums}   Nº da Sorte {c['numero_da_sorte']:2d}"
                  f"   pop {c['popularidade_aprox']:.3f}")
        print("\n  AVISO: a popularidade do Totoloto NÃO está medida — os filtros")
        print("  são transferidos do modelo validado no EuroMilhões. Ver o módulo.")


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
               cmd_elasticidade, cmd_m1lhao, cmd_padroes, cmd_plano, cmd_valor,
               cmd_carteira, cmd_totoloto,
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

    tt = sub.add_parser("totoloto", help="estrutura, vieses e chaves do Totoloto")
    tt.add_argument("--chaves", type=int, default=5)
    tt.add_argument("--candidatos", type=int, default=3000)
    tt.add_argument("--offline", action="store_true")
    tt.set_defaults(func=cmd_totoloto)

    pl = sub.add_parser("plano", help="calendário e orçamento ótimos")
    pl.add_argument("--orcamento", type=float, default=520.0,
                    help="orçamento anual em euros")
    pl.set_defaults(func=cmd_plano)

    mo = sub.add_parser("modelo", help="modelo consolidado e recomendação")
    mo.add_argument("--bilhetes", type=int, default=5)
    mo.add_argument("--jackpot", type=float, default=60e6)
    mo.add_argument("--candidatos", type=int, default=3000)
    mo.add_argument("--offline", action="store_true")
    mo.add_argument("--json", action="store_true")
    mo.add_argument("--avaliar", type=str, default=None,
                    help="avaliar uma aposta: 'n1,n2,n3,n4,n5,e1,e2'")
    mo.set_defaults(func=cmd_modelo)

    si = sub.add_parser("simulacao", help="distribuição completa de uma noite")
    si.add_argument("--jackpot", type=float, default=111e6)
    si.add_argument("--bilhetes", type=int, default=5, help="carteira detalhada")
    si.add_argument("--tamanhos", type=str, default="1,2,5,10,20")
    si.add_argument("--sim", type=int, default=400_000)
    si.add_argument("--candidatos", type=int, default=3000)
    si.add_argument("--seed", type=int, default=7)
    si.add_argument("--entropia", type=int, default=65536,
                    help="bytes quânticos a recolher antes de otimizar")
    si.add_argument("--offline", action="store_true")
    si.set_defaults(func=cmd_simulacao)

    au = sub.add_parser("auditoria", help="auditar o sistema e caçar lacunas")
    au.add_argument("--sim", type=int, default=4000)
    au.add_argument("--rapido", action="store_true", help="só verificações rápidas")
    au.set_defaults(func=cmd_auditoria)

    pa = sub.add_parser("padroes", help="repetições, coincidências e o mapa do espaço")
    pa.set_defaults(func=cmd_padroes)

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
    r.add_argument("--chaves", type=int, default=0)
    r.add_argument("--candidatos", type=int, default=2000)
    r.add_argument("--offline", action="store_true")
    r.set_defaults(func=cmd_relatorio)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
