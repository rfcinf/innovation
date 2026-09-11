"""
Testes do núcleo.

As probabilidades e a fiscalidade são calculadas de raiz em config.py; se
estiverem erradas, todo o resto do sistema é ficção. Estes testes fixam-nas
contra valores oficiais publicados.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from euromillions import config, ev  # noqa: E402
from euromillions.optimizer import Filters  # noqa: E402
from euromillions.popularity import features_main, slip_col, slip_row  # noqa: E402
from euromillions.randomness import benjamini_hochberg  # noqa: E402


# ---------------------------------------------------------------------------
# Probabilidades
# ---------------------------------------------------------------------------

def test_odds_jackpot_era_atual():
    """1 em 139.838.160 — valor oficial da matriz 5/50 + 2/12."""
    assert round(config.JACKPOT.odds(12)) == 139_838_160


def test_odds_jackpot_eras_anteriores():
    assert round(config.JACKPOT.odds(9)) == 76_275_360      # 2004-2011
    assert round(config.JACKPOT.odds(11)) == 116_531_800    # 2011-2016


@pytest.mark.parametrize(
    "label,esperado",
    [
        ("5+1", 6_991_908),
        ("5+0", 3_107_515),
        ("4+2", 621_503),
        ("2+0", 22),
    ],
)
def test_odds_escaloes(label, esperado):
    tier = config.TIER_BY_LABEL[label]
    assert round(tier.odds(12)) == pytest.approx(esperado, rel=0.001)


def test_probabilidade_de_ganhar_alguma_coisa():
    """Aproximadamente 1 em 13."""
    p = config.probability_any_prize(12)
    assert 1 / p == pytest.approx(13.0, abs=0.2)


def test_treze_escaloes():
    assert len(config.TIERS) == 13
    assert len({t.label for t in config.TIERS}) == 13


def test_espaco_de_combinacoes():
    assert config.MAIN_COMBINATIONS == 2_118_760
    assert config.total_combinations(12) == 139_838_160


# ---------------------------------------------------------------------------
# Eras
# ---------------------------------------------------------------------------

def test_eras_cobrem_todo_o_historico_sem_sobreposicao():
    dia = config.FIRST_DRAW
    hoje = dt.date.today()
    while dia <= hoje:
        matches = [e for e in config.ERAS if e.contains(dia)]
        assert len(matches) == 1, f"{dia}: {len(matches)} eras"
        dia += dt.timedelta(days=31)


def test_pool_de_estrelas_por_data():
    assert config.star_pool_for(dt.date(2005, 6, 3)) == 9
    assert config.star_pool_for(dt.date(2013, 1, 8)) == 11
    assert config.star_pool_for(dt.date(2024, 5, 10)) == 12


# ---------------------------------------------------------------------------
# Imposto do Selo
# ---------------------------------------------------------------------------

def test_isencao_ate_5000():
    assert config.net_prize(5_000.0) == 5_000.0
    assert config.net_prize(100.0) == 100.0


def test_imposto_incide_so_sobre_o_excedente():
    """€20.000 → recebe €17.000 (20% sobre os €15.000 acima do limiar)."""
    assert config.net_prize(20_000.0) == pytest.approx(17_000.0)


def test_imposto_num_jackpot():
    bruto = 100_000_000.0
    liquido = config.net_prize(bruto)
    assert liquido == pytest.approx(bruto - 0.20 * (bruto - 5_000))


def test_inverso_do_imposto():
    for bruto in (1_000.0, 5_000.0, 42_000.0, 9e6):
        assert config.gross_for_net(config.net_prize(bruto)) == pytest.approx(bruto)


# ---------------------------------------------------------------------------
# Partilha pari-mutuel
# ---------------------------------------------------------------------------

def test_fator_de_partilha_limites():
    assert ev.expected_share_factor(0.0) == 1.0
    assert ev.expected_share_factor(1.0) == pytest.approx(0.6321, abs=1e-3)
    assert ev.expected_share_factor(1e-9) == pytest.approx(1.0, abs=1e-6)


def test_fator_de_partilha_decresce():
    lams = [0.1, 0.5, 1, 2, 5, 10]
    vals = [ev.expected_share_factor(x) for x in lams]
    assert all(a > b for a, b in zip(vals, vals[1:]))


def test_combinacao_impopular_vale_mais():
    """O coração do sistema: mesma probabilidade, cheque maior."""
    caro = ev.expected_value(150e6, 80e6, popularity=3.0)
    barato = ev.expected_value(150e6, 80e6, popularity=0.35)
    assert barato.ev_liquido > caro.ev_liquido
    assert barato.fator_partilha > caro.fator_partilha


def test_jackpot_limitado_pelo_teto():
    a = ev.expected_value(250e6, 80e6)
    b = ev.expected_value(900e6, 80e6)
    assert a.ev_liquido == pytest.approx(b.ev_liquido)


def test_ev_nunca_positivo_em_condicoes_normais():
    """
    O resultado mais importante do projeto: mesmo no melhor cenário
    realista, o valor esperado fica abaixo do preço da aposta.
    """
    melhor = ev.expected_value(ev.JACKPOT_CAP_EUR, 40e6, popularity=0.3)
    assert melhor.ev_liquido < config.TICKET_PRICE_EUR


# ---------------------------------------------------------------------------
# Características das combinações
# ---------------------------------------------------------------------------

def test_geometria_do_boletim():
    assert (slip_row(1), slip_col(1)) == (0, 0)
    assert (slip_row(10), slip_col(10)) == (0, 9)
    assert (slip_row(11), slip_col(11)) == (1, 0)
    assert (slip_row(50), slip_col(50)) == (4, 9)


def test_caracteristicas_de_uma_combinacao_de_datas():
    f = dict(zip(
        ("n_ate_31", "n_ate_12", "n_consecutivos", "soma_norm", "amplitude_norm",
         "n_impares", "mesma_coluna", "mesma_linha", "prog_aritmetica",
         "n_multiplos_7", "tem_7", "espaco_regular"),
        features_main(np.array([3, 7, 11, 19, 24])),
    ))
    assert f["n_ate_31"] == 5
    assert f["n_ate_12"] == 3
    assert f["tem_7"] == 1.0


def test_progressao_aritmetica_detetada():
    f = features_main(np.array([1, 6, 11, 16, 21]))
    assert f[8] == 1.0
    assert features_main(np.array([1, 6, 11, 16, 22]))[8] == 0.0


def test_consecutivos():
    assert features_main(np.array([1, 2, 3, 20, 40]))[2] == 2


# ---------------------------------------------------------------------------
# Filtros
# ---------------------------------------------------------------------------

def test_filtro_rejeita_aposta_de_datas():
    assert not Filters().accepts(np.array([3, 7, 11, 19, 24]))


def test_filtro_rejeita_progressao():
    assert not Filters().accepts(np.array([5, 15, 25, 35, 45]))


def test_filtro_rejeita_sequencia():
    assert not Filters().accepts(np.array([1, 2, 3, 4, 5]))


def test_filtro_aceita_combinacao_dispersa():
    assert Filters().accepts(np.array([4, 17, 33, 41, 48]))


def test_filtro_nao_proibe_consecutivos():
    """
    O modelo mostra que consecutivos sao SUB-jogados (beta<0, p<0.0001).
    Proibi-los, como faz o folclore das apostas, deitaria fora combinacoes
    valiosas. Este teste impede que a intuicao volte a entrar pela porta
    das traseiras.
    """
    assert Filters().accepts(np.array([2, 33, 34, 45, 50]))


def test_filtro_rejeita_espacamento_regular():
    """`espaco_regular` e o preditor mais forte: regular = popular = mau."""
    assert not Filters().accepts(np.array([2, 13, 24, 35, 47]))


# ---------------------------------------------------------------------------
# Correção para testes múltiplos
# ---------------------------------------------------------------------------

def test_bh_monotono_e_maior_que_p():
    p = np.array([0.001, 0.008, 0.02, 0.2, 0.6])
    q = benjamini_hochberg(p)
    assert np.all(q >= p - 1e-12)
    assert np.all(np.diff(q) >= -1e-12)
    assert np.all(q <= 1.0)


def test_bh_controla_falsas_descobertas():
    """50 testes de ruído puro não devem produzir descobertas."""
    rng = np.random.default_rng(0)
    p = rng.uniform(size=50)
    assert (benjamini_hochberg(p) < 0.05).sum() == 0


# ---------------------------------------------------------------------------
# Entropia
# ---------------------------------------------------------------------------

def test_entropia_sem_vies_de_modulo():
    """
    `byte % 50` introduz viés; a amostragem por rejeição não. Testa-se com
    uma fonte offline para não depender da rede.
    """
    from euromillions.quantum import EntropySource

    src = EntropySource()
    src.refill(200_000, allow_network=False)
    vals = [src.uniform_int(1, 50, allow_network=False) for _ in range(20_000)]
    counts = np.bincount(vals, minlength=51)[1:]
    chi2 = ((counts - counts.mean()) ** 2 / counts.mean()).sum()
    assert chi2 < 100  # 49 gl: valor crítico a 0.1% ≈ 85


def test_bilhete_quantico_bem_formado():
    from euromillions.quantum import EntropySource

    src = EntropySource()
    src.refill(50_000, allow_network=False)
    for _ in range(50):
        m, s = src.draw_ticket(12, allow_network=False)
        assert len(m) == 5 and len(set(m)) == 5
        assert len(s) == 2 and len(set(s)) == 2
        assert all(1 <= x <= 50 for x in m)
        assert all(1 <= x <= 12 for x in s)


# ---------------------------------------------------------------------------
# Estrelas
# ---------------------------------------------------------------------------

def test_modelo_estrelas_e_multiplicativo():
    """log pi(a,b) = alpha_a + alpha_b, logo pi(a,b) = f(a)*f(b)."""
    from euromillions.stars import StarPopularity

    m = StarPopularity(pool=12)
    m.alpha = np.log(np.linspace(0.7, 1.3, 12))
    assert m.pair_popularity((3, 9)) == pytest.approx(
        m.star_factor(3) * m.star_factor(9)
    )


def test_normalizacao_do_estimador_de_estrelas():
    """
    A razao esperada W(k,0)/W(k,2) depende do pool e tem de mudar com a era:
    12 estrelas -> 45, 11 -> 36, 9 -> 21. Errar isto contaminaria as tres
    eras num so numero e inventaria um efeito onde nao ha.
    """
    from math import comb

    assert comb(12 - 2, 2) == 45
    assert comb(11 - 2, 2) == 36
    assert comb(9 - 2, 2) == 21


# ---------------------------------------------------------------------------
# Carteira
# ---------------------------------------------------------------------------

def test_cobertura_disjunta_reduz_prob_de_nada():
    """
    Bilhetes disjuntos falham menos vezes em conjunto do que bilhetes
    sobrepostos. E combinatoria pura, nao depende de modelo nenhum.
    """
    from euromillions import portfolio as pf

    disj = [([1, 2, 3, 4, 5], [1, 2]), ([6, 7, 8, 9, 10], [3, 4]),
            ([11, 12, 13, 14, 15], [5, 6])]
    over = [([1, 2, 3, 4, 5], [1, 2]), ([1, 2, 3, 4, 6], [3, 4]),
            ([1, 2, 3, 4, 7], [5, 6])]
    a = pf.simulate_portfolio(disj, n_sim=40_000, seed=5)
    b = pf.simulate_portfolio(over, n_sim=40_000, seed=5)
    assert a["p_nada"] < b["p_nada"]


def test_simulacao_bate_certo_com_a_teoria():
    """Bilhetes disjuntos ~ independentes: a simulacao tem de bater na teoria."""
    from euromillions import portfolio as pf

    tickets = [([1, 2, 3, 4, 5], [1, 2]), ([6, 7, 8, 9, 10], [3, 4])]
    sim = pf.simulate_portfolio(tickets, n_sim=120_000, seed=9)
    teorico = pf.probability_no_prize_independent(2)
    assert abs(sim["p_nada"] - teorico) < 0.01


def test_ganho_medio_e_linear_e_independente_da_sobreposicao():
    """
    O valor esperado de N apostas e sempre N x EV(1), qualquer que seja a
    sobreposicao. Se esta identidade falhar, o motor de EV esta errado.
    """
    from euromillions import portfolio as pf

    prizes = ev.FALLBACK_TIER_PRIZES
    um = pf.expected_winnings_analytic(1, prizes)
    cinco = pf.expected_winnings_analytic(5, prizes)
    assert cinco == pytest.approx(5 * um)


def test_cobertura_maxima_com_5_bilhetes():
    from euromillions import portfolio as pf

    tickets = [(list(range(1 + 5 * i, 6 + 5 * i)), [1, 2]) for i in range(5)]
    cov = pf.coverage_score(tickets)
    assert cov["numeros_distintos"] == 25
    assert cov["sobreposicao_maxima"] == 0


# ---------------------------------------------------------------------------
# Comparativo sobre todo o historico
# ---------------------------------------------------------------------------

def test_carteiras_respeitam_o_pool_de_estrelas_da_era():
    """
    Em 2004-2011 so existiam 9 estrelas. Gerar um bilhete com a estrela 12
    nessa era produz uma aposta impossivel, que nunca ganharia nos escaloes
    com estrelas — e enviesaria o comparativo a favor de quem calhasse
    jogar estrelas baixas. Este teste garante a correcao por era.
    """
    from euromillions.backtest import _make_portfolio

    rng = np.random.default_rng(3)
    for pool in (9, 11, 12):
        for strat in ("datas", "sobreposta", "aleatoria", "otimizada"):
            _, stars_arr = _make_portfolio(strat, rng, 5, pool)
            assert stars_arr.max() <= pool, f"{strat} com pool={pool}"
            assert stars_arr.min() >= 1


def test_carteiras_sao_apostas_validas():
    from euromillions.backtest import _make_portfolio

    rng = np.random.default_rng(4)
    for strat in ("datas", "sobreposta", "aleatoria", "otimizada"):
        mains, stars_arr = _make_portfolio(strat, rng, 5, 12)
        assert mains.shape == (5, config.MAIN_PICK)
        assert stars_arr.shape == (5, config.STAR_PICK)
        for row in mains:
            assert len(set(row.tolist())) == config.MAIN_PICK
            assert 1 <= row.min() and row.max() <= config.MAIN_POOL
        for row in stars_arr:
            assert len(set(row.tolist())) == config.STAR_PICK


def test_estrategia_datas_so_usa_numeros_ate_31():
    from euromillions.backtest import _make_portfolio

    rng = np.random.default_rng(5)
    mains, stars_arr = _make_portfolio("datas", rng, 5, 12)
    assert mains.max() <= 31
    assert stars_arr.max() <= 9


def test_comparacao_analitica_ordena_pela_popularidade():
    """
    Menor popularidade tem de dar maior EV — se esta monotonia falhar, o
    motor de EV ou a tabela de elasticidades estao errados.
    """
    import pandas as pd

    from euromillions.backtest import analytic_comparison

    pops = {"popular": 2.0, "media": 1.0, "impopular": 0.5}
    # Sem dados de quebra de premios, recorre aos precos de recurso.
    out = analytic_comparison(pd.DataFrame(), pops, jackpot_eur=100e6)
    evs = out.set_index("estratégia")["EV_por_aposta_€"]
    assert evs["impopular"] > evs["media"] > evs["popular"]


def test_escala_do_jackpot():
    """
    Numero que da a dimensao real: jogando 5 apostas por sorteio, sao
    precisos centenas de milhares de anos para ESPERAR um jackpot.
    """
    from euromillions.backtest import jackpot_expectation

    r = jackpot_expectation(2_000_000)
    assert r["apostas_para_1_esperado"] == 139_838_160
    assert r["anos_jogando_5_por_sorteio"] > 100_000
    assert r["jackpots_esperados"] < 0.02


# ---------------------------------------------------------------------------
# M1lhao
# ---------------------------------------------------------------------------

def test_m1lhao_ev_por_aposta():
    """
    Cada aposta gera um codigo; um codigo ganha por semana. O EV e o premio
    liquido a dividir pelo numero de codigos em circulacao.
    """
    from euromillions import m1lhao

    codigos = 4_000_000.0
    e = m1lhao.expected_value_per_bet(codigos / m1lhao.DRAWS_PER_WEEK, 1.0)
    assert e == pytest.approx(config.net_prize(1_000_000.0) / codigos)


def test_m1lhao_conta_uma_semana_de_apostas():
    """O sorteio e a sexta e inclui as apostas de terca: duas por semana."""
    from euromillions import m1lhao

    assert m1lhao.DRAWS_PER_WEEK == 2
    assert m1lhao.codes_per_draw(1_000_000, 0.10) == pytest.approx(200_000)


def test_m1lhao_e_tributado():
    """€1M excede o limiar dos €5.000: 20% sobre o excedente."""
    from euromillions import m1lhao

    com = m1lhao.expected_value_per_bet(1e6, 0.1, apply_tax=True)
    sem = m1lhao.expected_value_per_bet(1e6, 0.1, apply_tax=False)
    assert com < sem
    assert com / sem == pytest.approx(801_000 / 1_000_000)


def test_m1lhao_soma_ao_ev_e_nao_e_otimizavel():
    """
    O codigo e gerado pelo sistema: a parcela e identica seja qual for a
    combinacao jogada. Tem de somar igual aos dois lados.
    """
    e = 0.2133
    pop = ev.expected_value(60e6, 24e6, 1.5, ev_m1lhao=e)
    imp = ev.expected_value(60e6, 24e6, 0.6, ev_m1lhao=e)
    assert pop.ev_m1lhao == imp.ev_m1lhao == e
    sem_pop = ev.expected_value(60e6, 24e6, 1.5)
    assert pop.ev_liquido - sem_pop.ev_liquido == pytest.approx(e)


def test_m1lhao_dilui_a_vantagem_percentual():
    """
    Somar uma parcela fixa e igual para todos reduz o ganho PERCENTUAL da
    otimizacao. Ignorar isto inflacionaria a vantagem anunciada.
    """
    e = 0.2133
    sem = (ev.expected_value(60e6, 24e6, 0.62).ev_liquido
           / ev.expected_value(60e6, 24e6, 1.53).ev_liquido - 1)
    com = (ev.expected_value(60e6, 24e6, 0.62, ev_m1lhao=e).ev_liquido
           / ev.expected_value(60e6, 24e6, 1.53, ev_m1lhao=e).ev_liquido - 1)
    assert com < sem


def test_ev_continua_negativo_mesmo_com_m1lhao():
    """A correcao muda os numeros, nao a conclusao."""
    melhor = ev.expected_value(
        ev.JACKPOT_CAP_EUR, 20e6, popularity=0.3, ev_m1lhao=0.25
    )
    assert melhor.ev_liquido < config.TICKET_PRICE_EUR


# ---------------------------------------------------------------------------
# Elasticidades medidas
# ---------------------------------------------------------------------------

def test_elasticidade_dos_escaloes_de_cinco_e_um():
    """
    Acertar os 5 numeros E ter a nossa combinacao exata: a elasticidade tem
    de ser 1.0 por definicao. A medicao independente deu 0.956 no 5+0, o que
    valida o metodo.
    """
    for label in ("5+2", "5+1", "5+0"):
        assert ev.TIER_ELASTICITY[label] == 1.0


def test_elasticidade_decresce_com_numeros_acertados():
    """
    Menos numeros acertados = combinacao menos fixada = menos sensivel a
    popularidade. A monotonia e a estrutura essencial da tabela.
    """
    cinco = ev.TIER_ELASTICITY["5+0"]
    quatro = ev.TIER_ELASTICITY["4+0"]
    tres = ev.TIER_ELASTICITY["3+0"]
    dois = ev.TIER_ELASTICITY["2+0"]
    assert cinco > quatro > tres > dois


def test_tabela_escrita_a_mao_estava_inflacionada():
    """
    Registo do erro: os valores medidos sao inferiores aos que estavam
    escritos a mao em todos os escaloes abaixo de 5 numeros.
    """
    for label in ("4+2", "4+1", "4+0", "3+2", "3+1", "3+0", "2+2", "2+1", "2+0", "1+2"):
        assert ev.TIER_ELASTICITY[label] < ev.TIER_ELASTICITY_HANDMADE[label]


# ---------------------------------------------------------------------------
# Padroes e o espaco de combinacoes
# ---------------------------------------------------------------------------

def test_universo_tem_o_tamanho_certo():
    from euromillions import patterns

    U = patterns.universe()
    assert len(U) == 2_118_760
    assert U.shape[1] == config.MAIN_PICK


def test_probabilidade_de_familia_e_contagem_de_membros():
    """
    A ideia central: uma familia e "rara" exatamente na proporcao dos seus
    membros. P(familia) = nº de membros / 2.118.760. Nao ha nada alem disso,
    e por isso escolher dentro de uma familia rara NAO baixa a probabilidade
    do bilhete.
    """
    from euromillions import patterns

    d = patterns.exact_distribution("nums_ate_31")
    assert d["combinacoes"].sum() == 2_118_760
    assert d["probabilidade"].sum() == pytest.approx(1.0)
    for r in d.itertuples():
        assert r.probabilidade == pytest.approx(r.combinacoes / 2_118_760)


def test_familia_de_aniversarios_bate_com_a_combinatoria():
    """Os 5 numeros <=31: C(31,5)/C(50,5) = 8,02%."""
    from math import comb

    from euromillions import patterns

    d = patterns.exact_distribution("nums_ate_31").set_index("valor")
    assert d.loc[5, "combinacoes"] == comb(31, 5)
    assert d.loc[5, "probabilidade"] == pytest.approx(comb(31, 5) / comb(50, 5))


def test_repeticao_esperada_usa_pares_e_nao_sorteios():
    """
    O erro classico e comparar 1970 com 2.118.760 e concluir "impossivel".
    O correto e comparar os C(1970,2) = 1.939.465 PARES. Da esperado ~0,92,
    e observou-se 1 repeticao real.
    """
    n = 1970
    pares = n * (n - 1) / 2
    assert pares == pytest.approx(1_939_465)
    esperado = pares / config.MAIN_COMBINATIONS
    assert esperado == pytest.approx(0.915, abs=0.01)


def test_sobreposicao_soma_um():
    """As probabilidades de partilhar 0..5 numeros tem de somar 1."""
    from math import comb

    total = sum(comb(5, k) * comb(45, 5 - k) / comb(50, 5) for k in range(6))
    assert total == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Modelo consolidado
# ---------------------------------------------------------------------------

def test_proveniencia_declara_a_origem_de_tudo():
    """
    Cada input do modelo tem de dizer de onde vem. Um numero sem origem
    declarada e um palpite a espera de ser descoberto — foi assim que
    nasceram o prior das estrelas e a tabela de elasticidades.
    """
    from euromillions.model import PROVENANCE

    assert len(PROVENANCE) >= 8
    validas = {"medido", "oficial", "assumido"}
    for inp in PROVENANCE:
        assert inp.origem in validas, inp.nome
        assert inp.valor


def test_suposicoes_estao_explicitas_e_nao_escondidas():
    """
    O estimador de vendas e a fracao de apostas aleatorias sao conhecidos
    pontos fracos. Tem de continuar marcados como suposicoes, para o auditor
    os sinalizar.
    """
    from euromillions.model import PROVENANCE

    assumidas = {i.nome for i in PROVENANCE if i.origem == "assumido"}
    assert any("vendas" in n for n in assumidas)
    assert any("aleat" in n for n in assumidas)


# ---------------------------------------------------------------------------
# Auditor
# ---------------------------------------------------------------------------

def test_auditor_reprova_quando_ha_critico():
    from euromillions import audit

    rep = audit.AuditReport()
    rep.add(audit.OK, "x", "tudo bem", "ok")
    assert rep.verdict() == "APROVADO"
    rep.add(audit.AVISO, "x", "duvida", "hmm", False)
    assert "RESERVAS" in rep.verdict()
    rep.add(audit.CRITICO, "x", "avaria", "partido", False)
    assert rep.verdict().startswith("REPROVADO")


def test_auditor_verifica_afirmacoes_publicadas():
    """
    As afirmacoes do README sao recalculadas e comparadas. Se a documentacao
    se afastar do codigo, tem de falhar — uma pagina que promete um numero
    que o programa ja nao produz e pior do que nao ter pagina nenhuma.
    """
    from euromillions.audit import CLAIMS

    nomes = {c[0] for c in CLAIMS}
    assert "odds do jackpot" in nomes
    for nome, unidade, valor, tol in CLAIMS:
        assert valor > 0
        assert 0.0 <= tol < 1.0


def test_odds_publicadas_batem_com_o_calculo():
    """A afirmacao central do projeto, verificada de raiz."""
    from euromillions.audit import CLAIMS

    publicado = dict((c[0], c[2]) for c in CLAIMS)
    assert round(config.JACKPOT.odds(12)) == publicado["odds do jackpot"]
    assert config.probability_any_prize(12) == pytest.approx(
        publicado["P(algum prémio) por aposta"], rel=0.01
    )


def test_continuidade_respeita_uma_ou_duas_sextas_por_semana():
    """
    Ate maio de 2011 havia UM sorteio por semana. Uma verificacao de
    continuidade que exija intervalos curtos em todo o historico sinaliza
    377 falsas anomalias — foi o que a primeira versao do auditor fez.
    """
    from euromillions import config as cfg

    e1 = [e for e in cfg.ERAS if e.star_pool == 9][0]
    assert "sextas" in e1.note.lower()


# ---------------------------------------------------------------------------
# Plano de jogo
# ---------------------------------------------------------------------------

def test_jackpot_cresce_mais_depressa_que_a_concorrencia():
    """
    Elasticidade das vendas ao jackpot = 0,271: o jackpot duplica, as vendas
    sobem 1,21x. Se fosse >= 1, jackpots grandes nao compensariam e a
    recomendacao inverter-se-ia.
    """
    from euromillions import strategy

    assert 0 < strategy.SALES_ELASTICITY < 1
    assert 2 ** strategy.SALES_ELASTICITY < 1.5
    assert strategy.expected_sales(120e6) < 2 * strategy.expected_sales(60e6)


def test_p_jackpot_depende_so_do_numero_de_apostas():
    """
    O ponto central do plano: concentrar o orcamento NAO altera a
    probabilidade de acertar. Se este teste falhar, a recomendacao esta a
    prometer o que nao pode.
    """
    from euromillions import strategy

    p = config.JACKPOT.probability(12)
    total = 208
    espalhado = 1 - (1 - p) ** total          # 2 apostas x 104 sorteios
    concentrado = 1 - (1 - p) ** total        # 41.6 apostas x 5 sorteios
    assert espalhado == pytest.approx(concentrado)

    tab = strategy.budget_plans(520.0)
    assert tab["P(jackpot/ano)"].nunique() == 1


def test_concentrar_aumenta_o_retorno():
    """O que a concentracao muda de facto: retorno por euro e tamanho do cheque."""
    from euromillions import strategy

    tab = strategy.budget_plans(520.0)
    ret = tab["retorno €/€"].to_numpy()
    assert ret[-1] > ret[0]
    assert ret[-1] / ret[0] > 1.5


def test_impopular_aumenta_a_hipotese_de_ficar_sozinho():
    from euromillions import strategy

    otim = strategy.solo_probability(200e6, 0.35)
    datas = strategy.solo_probability(200e6, 1.53)
    assert otim["P_sozinho"] > datas["P_sozinho"]
    assert otim["cheque_esperado_liquido"] > datas["cheque_esperado_liquido"]


def test_plano_nao_promete_lucro():
    """Mesmo o melhor plano possivel continua a devolver menos do que custa."""
    from euromillions import strategy

    tab = strategy.budget_plans(520.0)
    assert tab["retorno €/€"].max() < 1.0


# ---------------------------------------------------------------------------
# Totoloto
# ---------------------------------------------------------------------------

def test_totoloto_matriz_e_odds():
    from math import comb

    from euromillions import totoloto as T

    assert T.MAIN_COMBINATIONS == comb(49, 5) == 1_906_884
    assert T.TOTAL_COMBINATIONS == 24_789_492


def test_totoloto_estrutura_bate_com_o_oficial():
    """
    A Santa Casa publica "1 em 7" para ganhar algum premio, mas nao publica
    a probabilidade de cada escalao. Reconstruir os escaloes de raiz e obter
    1 em 6,86 e a unica forma de confirmar que a estrutura assumida esta
    certa — sem este confronto seria um palpite.
    """
    from euromillions import totoloto as T

    p = T.probability_any_prize()
    assert 1 / p == pytest.approx(7.0, abs=0.2)


def test_totoloto_numero_da_sorte_acumula():
    """
    O Nº da Sorte e um premio a parte que acumula com os escaloes de
    numeros. Somar probabilidades daria um valor errado; o calculo tem de
    ser pelo complementar.
    """
    from euromillions import totoloto as T

    soma_ingenua = sum(t.probability() for t in T.TIERS) + 1 / T.LUCKY_POOL
    correto = T.probability_any_prize()
    assert correto < soma_ingenua


def test_totoloto_melhor_por_euro_que_euromilhoes():
    """5,6x melhores odds a 2,5x menos preco = 14x mais hipoteses por euro."""
    from euromillions import totoloto as T

    por_euro_toto = (1 / T.TOTAL_COMBINATIONS) / T.TICKET_PRICE_EUR
    por_euro_em = config.JACKPOT.probability(12) / config.TICKET_PRICE_EUR
    assert por_euro_toto / por_euro_em > 10


def test_totoloto_filtros_seguem_os_dados_e_nao_o_folclore():
    from euromillions import totoloto as T

    assert not T.accepts(np.array([3, 7, 11, 19, 24]))     # datas
    assert not T.accepts(np.array([5, 15, 25, 35, 45]))    # progressao
    assert T.accepts(np.array([6, 41, 44, 45, 47]))        # consecutivos: OK


def test_totoloto_prior_do_numero_da_sorte_e_transferido_nao_medido():
    """
    O prior vem do modelo das estrelas do EuroMilhoes. Esta limitacao tem de
    ficar visivel: e exatamente o tipo de numero escrito a mao que ja errou
    uma vez neste projeto.
    """
    from euromillions import totoloto as T

    w = T.lucky_number_prior()
    assert len(w) == 13
    assert w[6] > w[11]      # 7 mais jogado que 12
    assert w[12] == min(w)   # 13 assumido como o menos jogado
    assert "transferida" in T.lucky_number_prior.__doc__.lower()


def test_totoloto_dados_de_frequencia_sao_validados():
    """
    A soma das contagens tem de dar exatamente 5 x sorteios (numeros) e
    1 x sorteios (Nº da Sorte). Se nao der, a extracao leu a tabela errada
    e os dados nao servem — e a unica forma de o saber sem os ver a olho.
    """
    from euromillions import totoloto as T

    freq = T.load_frequencies()
    for periodo, d in freq.items():
        n = d["sorteios"]
        assert sum(d["numeros"]) == T.MAIN_PICK * n, periodo
        assert abs(sum(d["sorte"]) - n) <= 1, periodo
        assert len(d["numeros"]) == 49 and len(d["sorte"]) == 13


def test_totoloto_segmentos_sao_disjuntos():
    """
    Os periodos da fonte sao cumulativos. Subtrai-los da segmentos disjuntos,
    que e o que permite o teste de persistencia — comparar dois troços que
    nao partilham sorteios.
    """
    from euromillions import totoloto as T

    freq = T.load_frequencies()
    antigo = T.segment(freq, "2011-03-16", "2022-01-01")
    recente = T.segment(freq, "2022-01-01", None)
    assert antigo["sorteios"] + recente["sorteios"] == freq["2011-03-16"]["sorteios"]
    assert all(c >= 0 for c in antigo["numeros"])


def test_totoloto_sem_vies_detetavel():
    """
    Resultado da analise: nenhum numero nem Nº da Sorte sobrevive a correcao
    FDR, e os desvios nao persistem entre segmentos. Se algum dia isto
    falhar, ha sinal novo a investigar — nao um teste partido.
    """
    from euromillions import totoloto as T

    r = T.bias_battery(T.load_frequencies())
    assert r["números 1-49"]["p"] > 0.05
    assert r["Nº da Sorte 1-13"]["p"] > 0.05
    assert int(r["tabelas"]["numeros"]["significativo_fdr5"].sum()) == 0
    assert int(r["tabelas"]["sorte"]["significativo_fdr5"].sum()) == 0
    assert r["persistencia_numeros"]["p_simulado"] > 0.05


# ---------------------------------------------------------------------------
# Simulacao da noite de sorteio
# ---------------------------------------------------------------------------

def test_simulacao_bate_com_a_esperanca_analitica():
    """
    A media simulada tem de convergir para N x EV(1 aposta), qualquer que
    seja a sobreposicao entre bilhetes: a esperanca e linear e a correlacao
    entre bilhetes nao a altera.

    Este e o teste que apanha um erro na matriz de premios ou na contagem
    de acertos — os dois sitios onde uma simulacao vetorizada se engana
    silenciosamente e continua a produzir numeros plausiveis.
    """
    from euromillions import portfolio, simulacao as S

    prizes = {t.label: 10.0 * (i + 1) for i, t in enumerate(config.TIERS)}
    prizes["5+2"] = 0.0
    tk = [([1, 2, 3, 4, 5], [1, 2]), ([6, 7, 8, 9, 10], [3, 4]),
          ([11, 12, 13, 14, 15], [5, 6])]

    P = S.prize_matrix(prizes, 0.0)
    r = S.simulate_night(tk, P, n_sim=120_000, seed=3)
    teor = portfolio.expected_winnings_analytic(3, prizes)

    z = abs(r["ganho_medio"] - teor) / r["ganho_medio_se"]
    assert z < 4, f"media simulada {r['ganho_medio']} vs analitica {teor} ({z:.1f} EP)"


def test_carteira_disjunta_ganha_a_formula_de_independencia():
    """
    Bilhetes com numeros disjuntos tem P(nada) MENOR do que apostas
    independentes: nao podem falhar todos da mesma maneira. Se esta
    desigualdade inverter, a cobertura deixou de ser calculada.
    """
    from euromillions import portfolio, simulacao as S

    prizes = {t.label: 1.0 for t in config.TIERS}
    tk = [(list(range(1 + 5 * i, 6 + 5 * i)), [1, 2]) for i in range(5)]
    r = S.simulate_night(tk, S.prize_matrix(prizes, 0.0), n_sim=120_000, seed=5)

    assert r["p_nada"] < portfolio.probability_no_prize_independent(5)


def test_imposto_do_jackpot_aplica_se_depois_da_partilha():
    """
    O Imposto do Selo incide sobre o que CADA vencedor recebe. Aplica-lo ao
    premio bruto antes de dividir sobrestimaria a carga fiscal.

    Com lambda grande o premio parte-se em fatias pequenas; se alguma fatia
    descer abaixo de EUR 5.000 fica isenta, e o cheque esperado tem de ser
    ESTRITAMENTE MAIOR do que a versao que tributa antes de dividir.
    """
    from euromillions import simulacao as S

    j = 111e6
    correto = S.jackpot_check(j, lam=0.5)["cheque_esperado"]
    errado = config.net_prize(j) * ev.expected_share_factor(0.5)
    assert correto > errado


def test_cheque_do_jackpot_limites():
    """lambda->0 devolve o premio liquido inteiro; lambda grande divide-o."""
    from euromillions import simulacao as S

    j = 111e6
    sem_concorrencia = S.jackpot_check(j, lam=1e-9)
    assert sem_concorrencia["cheque_esperado"] == pytest.approx(
        config.net_prize(j), rel=1e-6)
    assert sem_concorrencia["p_sozinho"] == pytest.approx(1.0, abs=1e-6)

    muita = S.jackpot_check(j, lam=5.0)
    assert muita["cheque_esperado"] < 0.30 * config.net_prize(j)
    assert muita["p_sozinho"] < 0.01


def test_tabela_de_escaloes_conta_acertos_esperados():
    """
    A coluna 1_em e 1/(n x p). Para 5 apostas, o jackpot tem de dar
    139.838.160 / 5 — e a soma das contribuicoes tem de reproduzir o EV.
    """
    from euromillions import simulacao as S

    prizes = {t.label: 0.0 for t in config.TIERS}
    prizes["5+0"] = 1000.0
    tab = S.tier_table(5, prizes, jackpot_value=0.0)

    jack = tab[tab["escalão"] == "5+2"].iloc[0]
    assert jack["1_em"] == pytest.approx(139_838_160 / 5, rel=1e-6)

    p5 = config.TIER_BY_LABEL["5+0"].probability(12)
    total = tab["contrib_EV_€"].sum()
    assert total == pytest.approx(5 * p5 * 1000.0, rel=1e-9)


def test_concentrar_e_dispersar_puxam_em_sentidos_contrarios():
    """
    Cobertura e P(recuperar o custo) sao objetivos OPOSTOS, e o sistema tem
    de continuar a saber disso.

    Concentrar (bilhetes que partilham quase todos os numeros) piora
    P(nada) e melhora muito P(recuperar o custo), porque os bilhetes ganham
    em bloco. Dispersar faz o contrario. O valor esperado e igual nos dois
    casos — a esperanca e linear e nao ve correlacao.

    Se algum dia este teste falhar nas duas desigualdades ao mesmo tempo, e
    porque a simulacao deixou de captar a correlacao entre bilhetes.
    """
    from euromillions import simulacao as S

    prizes = {t.label: 0.0 for t in config.TIERS}
    prizes["2+0"] = 4.0
    prizes["2+1"] = 6.0
    prizes["3+0"] = 10.0
    P = S.prize_matrix(prizes, 0.0)

    dispersa = [(list(range(1 + 5 * i, 6 + 5 * i)), [1, 2]) for i in range(5)]
    concentrada = S.concentrated_portfolio([1, 2, 3, 4, 5], [1, 2], 5)

    n_disp = len(set().union(*[set(t[0]) for t in dispersa]))
    n_conc = len(set().union(*[set(t[0]) for t in concentrada]))
    assert n_disp == 25 and n_conc <= 10

    d = S.simulate_night(dispersa, P, n_sim=150_000, seed=9)
    c = S.simulate_night(concentrada, P, n_sim=150_000, seed=9)

    assert d["p_nada"] < c["p_nada"]
    assert c["p_recupera_custo"] > d["p_recupera_custo"]

    # ... e mesmo assim o ganho medio e o mesmo, dentro do ruido
    z = abs(d["ganho_medio"] - c["ganho_medio"]) / np.hypot(
        d["ganho_medio_se"], c["ganho_medio_se"])
    assert z < 4, f"medias divergem em {z:.1f} erros-padrao"
