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
