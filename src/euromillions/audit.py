"""
Auditor — o sistema que verifica o sistema.

PORQUE É QUE ISTO EXISTE
------------------------
Ao longo deste projeto cometi cinco erros que só apareceram porque alguém
(ou algum teste) foi verificar:

1. um teste de Kolmogorov-Smirnov contínuo aplicado a dados discretos, que
   "descobriu" desvios em 47 dos 50 números;
2. filtros que proibiam números consecutivos por folclore, quando os dados
   mostram que são sub-jogados;
3. barras de erro que tratavam 309.000 observações correlacionadas como
   independentes;
4. um otimizador que produzia carteiras com P(nada) pior do que jogar ao
   acaso;
5. o M1lhão em falta no motor de EV — 21% do valor de um bilhete português.

Todos me favoreciam ou faziam o sistema parecer melhor do que era. Nenhum
foi detetado por olhar para o resultado: foram detetados por verificar o
processo.

Este módulo automatiza essa verificação. Corre sozinho, reprova quando há
motivo, e trata as suposições como suspeitas até prova em contrário.

O QUE AUDITA
------------
A. integridade e frescura dos dados
B. validade dos modelos, revalidados fora da amostra
C. deriva — a vantagem está a decair?
D. suposições por validar (é aí que vive o próximo erro)
E. verificação das afirmações publicadas contra o que o código calcula
F. caça a lacunas novas, com controlo de falsas descobertas
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from . import config, ev, patterns, randomness, stars
from . import backtest as bt
from . import dataset as ds
from .popularity import fit_popularity_model

CRITICO, AVISO, INFO, OK = "CRÍTICO", "AVISO", "INFO", "OK"


@dataclass
class Finding:
    severity: str
    area: str
    check: str
    detail: str
    passed: bool = True

    def __str__(self) -> str:
        mark = {CRITICO: "✗✗", AVISO: "✗ ", INFO: "· ", OK: "✓ "}[self.severity]
        return f"  {mark} [{self.area}] {self.check}\n      {self.detail}"


@dataclass
class AuditReport:
    findings: list[Finding] = field(default_factory=list)
    started: str = ""
    finished: str = ""

    def add(self, severity, area, check, detail, passed=True) -> None:
        self.findings.append(Finding(severity, area, check, detail, passed))

    @property
    def criticos(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == CRITICO]

    @property
    def avisos(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == AVISO]

    def verdict(self) -> str:
        if self.criticos:
            return f"REPROVADO — {len(self.criticos)} problema(s) crítico(s)"
        if self.avisos:
            return f"APROVADO COM RESERVAS — {len(self.avisos)} aviso(s)"
        return "APROVADO"

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"gravidade": f.severity, "área": f.area,
                 "verificação": f.check, "passou": f.passed}
                for f in self.findings
            ]
        )

    def __str__(self) -> str:
        order = {CRITICO: 0, AVISO: 1, INFO: 2, OK: 3}
        out = [str(f) for f in sorted(self.findings, key=lambda f: order[f.severity])]
        return "\n".join(out)


# ---------------------------------------------------------------------------
# A. Dados
# ---------------------------------------------------------------------------

def audit_data(rep: AuditReport) -> dict:
    ctx: dict = {}
    try:
        draws = ds.load_draws()
        ctx["draws"] = draws
    except FileNotFoundError as e:
        rep.add(CRITICO, "dados", "histórico de sorteios", str(e), False)
        return ctx

    problems = ds.validate(draws)
    rep.add(
        CRITICO if problems else OK, "dados", "integridade dos sorteios",
        "; ".join(problems) if problems else
        f"{len(draws)} sorteios, sem duplicados nem valores fora de domínio",
        not problems,
    )

    eras = ds.validate_eras(draws)
    coerente = bool(eras["coerente"].all())
    rep.add(
        OK if coerente else CRITICO, "dados", "coerência das eras",
        "estrelas observadas dentro do pool declarado em todas as eras"
        if coerente else "estrela observada acima do pool declarado",
        coerente,
    )

    # Frescura: um modelo estimado sobre dados velhos decide sobre um mundo
    # que já mudou.
    last = draws["date"].max()
    idade = (dt.date.today() - last).days
    if idade > 30:
        sev, ok = AVISO, False
        det = f"último sorteio há {idade} dias ({last}) — correr `fetch`"
    else:
        sev, ok = OK, True
        det = f"último sorteio {last} ({idade} dias)"
    rep.add(sev, "dados", "frescura", det, ok)

    # Buracos na sequência: sorteios em falta enviesam qualquer contagem.
    #
    # A primeira versão desta verificação sinalizava 377 "anomalias" e estava
    # errada: até maio de 2011 havia um único sorteio por semana, à sexta, e
    # intervalos de 7 dias eram o normal. Assumir dois sorteios semanais em
    # todo o histórico transformava a regra do jogo numa avaria dos dados.
    #
    # Fica registado porque é o mesmo tipo de erro que o auditor existe para
    # apanhar — e desta vez o erro era do auditor.
    anomalias = 0
    detalhe = []
    for era in config.ERAS:
        sub = draws[draws["era"] == era.name]
        if len(sub) < 2:
            continue
        d = pd.to_datetime(pd.Series(list(sub["date"])))
        gaps = d.diff().dt.days.dropna()
        # E1: só sextas → 7 dias. E2/E3: terças e sextas → 3 ou 4 dias.
        limite = 9 if era.star_pool == 9 else 5
        n = int((gaps > limite).sum())
        anomalias += n
        detalhe.append(f"{era.name}: {n} acima de {limite}d")
    rep.add(
        OK if anomalias == 0 else AVISO, "dados", "continuidade da série",
        f"{anomalias} intervalos anómalos ({', '.join(detalhe)})",
        anomalias == 0,
    )

    try:
        bd = ds.load_breakdown()
        ctx["breakdown"] = bd
        cobertura = bd["date"].nunique() / len(draws)
        rep.add(
            OK if cobertura > 0.95 else AVISO, "dados",
            "cobertura das quebras de prémios",
            f"{bd['date'].nunique()} de {len(draws)} sorteios ({cobertura:.1%})",
            cobertura > 0.95,
        )
    except FileNotFoundError:
        rep.add(CRITICO, "dados", "quebras de prémios",
                "ausentes — sem elas não há modelo de popularidade", False)
    return ctx


# ---------------------------------------------------------------------------
# B. Modelos, revalidados fora da amostra
# ---------------------------------------------------------------------------

def audit_models(rep: AuditReport, ctx: dict) -> None:
    if "breakdown" not in ctx:
        return
    master = ds.build_master()

    r = bt.popularity_out_of_sample(master)
    rho, p = r["spearman_rho"], r["spearman_p"]
    ok = rho > 0.15 and p < 0.01
    rep.add(
        OK if ok else CRITICO, "modelos",
        "popularidade dos números — fora da amostra",
        f"ρ = {rho:.4f}, p = {p:.2e} "
        f"(treino {r['n_treino']}, teste {r['n_teste']})",
        ok,
    )

    # Monotonia dos quintis: se o modelo é real, popularidade prevista mais
    # alta tem de corresponder a popularidade observada mais alta.
    q = r["quintis"]["mean"].to_numpy()
    mono = bool(np.all(np.diff(q) > -0.02))
    rep.add(
        OK if mono else AVISO, "modelos", "monotonia dos quintis",
        f"{np.round(q, 3).tolist()}", mono,
    )

    try:
        obs = stars.star_observations(ctx["draws"], ctx["breakdown"])
        v = stars.validate_out_of_sample(obs)
        ok_s = v["spearman_rho"] > 0.3 and v["spearman_p"] < 0.01
        rep.add(
            OK if ok_s else CRITICO, "modelos",
            "popularidade das estrelas — fora da amostra",
            f"ρ = {v['spearman_rho']:.4f}, p = {v['spearman_p']:.2e}",
            ok_s,
        )
    except (ValueError, KeyError) as e:
        rep.add(AVISO, "modelos", "modelo de estrelas", f"não avaliado: {e}", False)

    # Auto-teste da medição de elasticidades: o escalão 5+0 tem elasticidade
    # teórica exatamente 1,0. Se a medição independente se afastar disso, o
    # método está partido — e não se saberia por nenhum outro caminho.
    from . import elasticity as el

    model, _ = fit_popularity_model(master)
    tab = el.measure_elasticities(master, ctx["breakdown"], model, normalise=False)
    linha = tab[tab["escalao"] == "5+0"]
    if len(linha):
        b = float(linha["b_bruto"].iloc[0])
        ok_e = 0.75 <= b <= 1.25
        rep.add(
            OK if ok_e else CRITICO, "modelos",
            "auto-teste do método de elasticidades",
            f"5+0 mede {b:.3f}; a teoria exige 1,0 "
            f"({'dentro' if ok_e else 'FORA'} de ±25%)",
            ok_e,
        )


# ---------------------------------------------------------------------------
# C. Deriva — a vantagem está a decair?
# ---------------------------------------------------------------------------

def audit_drift(rep: AuditReport, ctx: dict, window: int = 400) -> None:
    if "breakdown" not in ctx:
        return
    master = ds.build_master()
    m = master[np.isfinite(master["popularity_main5"])].copy()
    c = m[["n1", "n2", "n3", "n4", "n5"]].to_numpy(int)
    m["n31"] = (c <= 31).sum(axis=1)

    recent = m.tail(window)
    hi = recent[recent["n31"] >= 4]["popularity_main5"]
    lo = recent[recent["n31"] <= 2]["popularity_main5"]
    if len(hi) > 20 and len(lo) > 20:
        u = stats.mannwhitneyu(hi, lo, alternative="greater")
        ratio = float(hi.mean() / lo.mean())
        ok = u.pvalue < 0.05 and ratio > 1.05
        rep.add(
            OK if ok else CRITICO, "deriva",
            f"viés de datas ainda ativo (últimos {window} sorteios)",
            f"rácio = {ratio:.3f}, p = {u.pvalue:.2e}"
            + ("" if ok else "  ← a vantagem pode ter fechado"),
            ok,
        )

    # A vantagem depende de a popularidade das estrelas se manter. Compara
    # a estimativa recente com a histórica.
    try:
        obs = stars.star_observations(ctx["draws"], ctx["breakdown"])
        cur = obs[obs["pool"] == 12].sort_values("date")
        half = len(cur) // 2
        m1 = stars.StarPopularity(12).fit(cur.iloc[:half])
        m2 = stars.StarPopularity(12).fit(cur.iloc[half:])
        r = float(np.corrcoef(m1.alpha, m2.alpha)[0, 1])
        ok = r > 0.5
        rep.add(
            OK if ok else AVISO, "deriva",
            "estabilidade da popularidade das estrelas",
            f"correlação entre as duas metades = {r:.3f}",
            ok,
        )
    except (ValueError, KeyError):
        pass


# ---------------------------------------------------------------------------
# D. Suposições por validar
# ---------------------------------------------------------------------------

def audit_assumptions(rep: AuditReport) -> None:
    """
    As suposições declaradas no modelo são tratadas como suspeitas.

    Não é pedantismo: dos cinco erros deste projeto, quatro estavam
    exatamente aqui — números escritos à mão que ninguém tinha ido medir.
    """
    from .model import PROVENANCE

    for inp in PROVENANCE:
        if inp.origem == "assumido":
            rep.add(
                AVISO, "suposições", inp.nome,
                f"{inp.valor} — {inp.nota}", False,
            )

    # Verificação ativa da suposição mais perigosa: o estimador de vendas.
    try:
        master = ds.build_master()
        rho = stats.spearmanr(master["draw_index"], master["popularity_main5"],
                              nan_policy="omit")
        forte = abs(rho[0]) > 0.2 and rho[1] < 0.01
        rep.add(
            AVISO if forte else OK, "suposições",
            "deriva do estimador de vendas",
            f"ρ(tempo, popularidade global) = {rho[0]:+.3f}, p = {rho[1]:.1e}"
            + ("  ← deriva não explicada; contamina a escala absoluta"
               if forte else ""),
            not forte,
        )
    except Exception:
        pass

    # Parâmetros oficiais que podem mudar sem aviso do lado de fora.
    rep.add(INFO, "suposições", "parâmetros oficiais a confirmar periodicamente",
            f"preço €{config.TICKET_PRICE_EUR}, teto €{ev.JACKPOT_CAP_EUR/1e6:.0f}M, "
            f"imposto {config.STAMP_DUTY_RATE:.0%} acima de "
            f"€{config.STAMP_DUTY_THRESHOLD_EUR:.0f}")


# ---------------------------------------------------------------------------
# E. As afirmações publicadas continuam verdadeiras?
# ---------------------------------------------------------------------------

CLAIMS: tuple[tuple[str, str, float, float], ...] = (
    # (nome, unidade, valor publicado, tolerância relativa)
    ("odds do jackpot", "1 em", 139_838_160, 0.0),
    ("P(algum prémio) por aposta", "prob", 0.07708, 0.01),
    ("EV otimizada @ €60M com M1lhão", "€", 1.0162, 0.10),
    ("vantagem otimizada vs datas", "%", 7.8, 0.25),
    ("P(nada) com 5 apostas disjuntas", "prob", 0.6415, 0.03),
)


def audit_claims(rep: AuditReport, ctx: dict) -> None:
    """
    Recalcula as afirmações do README e compara com o que está publicado.

    A documentação e o código afastam-se com o tempo, e o resultado é uma
    página que promete um número que o programa já não produz. Esta secção
    torna esse afastamento impossível de passar despercebido.
    """
    live: dict[str, float] = {}

    live["odds do jackpot"] = config.JACKPOT.odds(12)
    live["P(algum prémio) por aposta"] = config.probability_any_prize(12)

    if "breakdown" in ctx:
        from .model import EuroMillionsModel

        try:
            mdl = EuroMillionsModel.fit()
            saved = dict(ev.TIER_ELASTICITY)
            try:
                ev.TIER_ELASTICITY.update(mdl.elasticities)
                otim = ev.expected_value(60e6, 24e6, 0.6191, 12,
                                         mdl.tier_prizes, ev_m1lhao=mdl.ev_m1lhao)
                datas = ev.expected_value(60e6, 24e6, 1.5343, 12,
                                          mdl.tier_prizes, ev_m1lhao=mdl.ev_m1lhao)
            finally:
                ev.TIER_ELASTICITY.clear()
                ev.TIER_ELASTICITY.update(saved)
            live["EV otimizada @ €60M com M1lhão"] = otim.ev_liquido
            live["vantagem otimizada vs datas"] = 100 * (
                otim.ev_liquido / datas.ev_liquido - 1
            )
        except Exception as e:  # pragma: no cover
            rep.add(AVISO, "afirmações", "recálculo do EV", str(e), False)

    from . import portfolio as pf

    rng = np.random.default_rng(1)
    nums = rng.permutation(np.arange(1, 51))[:25].reshape(5, 5)
    tickets = [
        (sorted(r.tolist()),
         sorted(rng.choice(np.arange(1, 13), 2, replace=False).tolist()))
        for r in nums
    ]
    live["P(nada) com 5 apostas disjuntas"] = pf.simulate_portfolio(
        tickets, n_sim=40_000, seed=1
    )["p_nada"]

    for nome, unidade, publicado, tol in CLAIMS:
        if nome not in live:
            continue
        atual = live[nome]
        desvio = abs(atual - publicado) / publicado if publicado else 0.0
        ok = desvio <= tol
        rep.add(
            OK if ok else CRITICO, "afirmações", nome,
            f"publicado {publicado:g} {unidade} · calculado {atual:.5g} "
            f"(desvio {desvio:.1%}, tolerância {tol:.0%})",
            ok,
        )


# ---------------------------------------------------------------------------
# F. Caça a lacunas
# ---------------------------------------------------------------------------

def hunt_gaps(rep: AuditReport, ctx: dict, n_sim: int = 4000) -> pd.DataFrame:
    """
    Procura ativamente sinal novo — e reporta honestamente quando não o há.

    Todo o varrimento passa por correção de falsas descobertas. Sem isso,
    testar dezenas de hipóteses garante "achados" por construção, e o
    auditor passaria a ser uma fábrica de superstições em vez de um travão.
    """
    if "draws" not in ctx:
        return pd.DataFrame()
    draws = ctx["draws"]
    rows = []

    r = randomness.full_battery(draws)
    b = r["tabelas"]["bolas"]
    n_sig = int(b["significativo_fdr5"].sum())
    rows.append({"varrimento": "bolas individuais", "testes": len(b), "achados": n_sig})
    rep.add(
        INFO if n_sig == 0 else AVISO, "lacunas", "viés nas bolas",
        f"{n_sig} de {len(b)} após correção FDR", n_sig == 0,
    )

    g = r["tabelas"]["gaps"]
    ng = int(g["significativo_fdr5"].sum())
    rows.append({"varrimento": "intervalos", "testes": len(g), "achados": ng})

    w = r["tabelas"]["janelas"]
    nw = int(w["significativo_fdr5"].sum()) if len(w) else 0
    rows.append({"varrimento": "janelas temporais", "testes": len(w), "achados": nw})

    pers = randomness.out_of_sample_persistence(draws, n_sim=n_sim)
    rep.add(
        INFO if pers["p_simulado"] > 0.05 else CRITICO, "lacunas",
        "persistência dos desvios fora da amostra",
        f"r = {pers['r_observado']:.4f}, p = {pers['p_simulado']:.3f}"
        + ("" if pers["p_simulado"] > 0.05 else "  ← SINAL REAL a investigar"),
        pers["p_simulado"] > 0.05,
    )

    bat, _ = patterns.full_pattern_battery(draws)
    npat = int(bat["significativo_fdr5"].sum())
    rows.append({"varrimento": "famílias de padrões", "testes": len(bat), "achados": npat})
    rep.add(
        INFO if npat == 0 else AVISO, "lacunas", "famílias de padrões",
        f"{npat} de {len(bat)} desviam da enumeração exata (após FDR)",
        npat == 0,
    )

    ov, st = patterns.overlap_analysis(draws)
    rep.add(
        INFO if st["p"] > 0.05 else AVISO, "lacunas",
        "coincidências entre sorteios",
        f"χ² = {st['chi2']}, p = {st['p']:.4f} sobre "
        f"{st['pares_de_sorteios']:,} pares",
        st["p"] > 0.05,
    )

    mdb = randomness.minimum_detectable_bias(len(draws))
    rep.add(
        INFO, "lacunas", "poder de deteção atual",
        f"com {len(draws)} sorteios só se detetaria um viés acima de "
        f"{mdb:.1f}% — brechas menores continuariam invisíveis",
    )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Execução completa
# ---------------------------------------------------------------------------

def run_audit(n_sim: int = 4000, skip_slow: bool = False) -> tuple[AuditReport, pd.DataFrame]:
    rep = AuditReport(started=dt.datetime.now().isoformat(timespec="seconds"))
    ctx = audit_data(rep)
    if not skip_slow:
        audit_models(rep, ctx)
        audit_drift(rep, ctx)
    audit_assumptions(rep)
    if not skip_slow:
        audit_claims(rep, ctx)
    gaps = hunt_gaps(rep, ctx, n_sim=n_sim) if not skip_slow else pd.DataFrame()
    rep.finished = dt.datetime.now().isoformat(timespec="seconds")
    return rep, gaps
