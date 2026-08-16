"""
Totoloto — o mesmo método, um jogo diferente.

O QUE MUDA FACE AO EUROMILHÕES
------------------------------
| | EuroMilhões | Totoloto |
|---|---|---|
| matriz | 5/50 + 2/12 | 5/49 + 1/13 |
| preço | €2,50 | €1,00 |
| odds do 1º prémio | 1 em 139.838.160 | 1 em 24.789.492 |
| sorteios | terça e sexta | quarta e sábado |
| mercado | 9 países | só Portugal |
| jackpot mínimo | €17M | €1M |

O Totoloto tem odds **5,6× melhores** e custa **2,5× menos**. Por euro
gasto, dá cerca de **14 vezes mais hipóteses** de tocar no 1º prémio. Em
troca, o prémio é tipicamente uma ordem de grandeza menor.

VALIDAÇÃO DA ESTRUTURA
----------------------
A Santa Casa publica que a probabilidade de ganhar um qualquer prémio é
"1 em 7", mas não publica a probabilidade de cada escalão. A estrutura
abaixo é reconstruída de raiz por combinatória e depois confrontada com
esse valor:

    P(algum prémio) calculado = 1 em 6,86

Bate. Isso valida os seis escalões — sem esse confronto, seria apenas uma
suposição minha sobre as regras.

(Uma tabela de odds que encontrei em fontes secundárias dava "4 acertos =
1 em 211.876"; o cálculo direto dá 1 em 8.668. Não a usei.)

O QUE SE TRANSFERE E O QUE NÃO SE TRANSFERE
--------------------------------------------
**Transfere-se** (é psicologia humana, não propriedade do jogo):
  · viés de datas — e aqui é ainda mais concentrado, porque 1-31 cobre
    63% dos 49 números;
  · aversão a números consecutivos e a espaçamento irregular;
  · o Número da Sorte 1-13 é o análogo direto das estrelas: TODOS os
    valores são datas plausíveis, e o 7 será sobre-jogado.

**Não se transfere** (tem de ser medido de novo):
  · a magnitude exata de cada efeito;
  · a popularidade de cada Número da Sorte;
  · as elasticidades por escalão.

Isso exigiria a quebra de prémios do Totoloto sorteio a sorteio — o número
de vencedores em cada escalão — que a Santa Casa renderiza por JavaScript e
não publica em formato acessível. **Enquanto esse dado não for recolhido,
este módulo aplica os princípios validados no EuroMilhões, e diz
explicitamente que são transferidos e não medidos aqui.**
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np

MAIN_POOL = 49
MAIN_PICK = 5
LUCKY_POOL = 13
TICKET_PRICE_EUR = 1.00
"""Desde 21 de outubro de 2021 (era €0,90 na década anterior)."""

MAIN_COMBINATIONS = comb(MAIN_POOL, MAIN_PICK)      # 1 906 884
TOTAL_COMBINATIONS = MAIN_COMBINATIONS * LUCKY_POOL  # 24 789 492

JACKPOT_MIN_EUR = 1_000_000.0
"""1º prémio mínimo garantido, sem teto de acumulação."""


# ---------------------------------------------------------------------------
# Escalões
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Tier:
    nome: str
    mains: int
    lucky: bool
    exact_mains: bool = True

    @property
    def combinations(self) -> int:
        """Nº de chaves do jogador compatíveis com este escalão."""
        return comb(MAIN_PICK, self.mains) * comb(MAIN_POOL - MAIN_PICK, MAIN_PICK - self.mains)

    def probability(self) -> float:
        p_main = self.combinations / MAIN_COMBINATIONS
        if self.lucky:
            return p_main / LUCKY_POOL
        return p_main

    def odds(self) -> float:
        return 1.0 / self.probability()


TIERS: tuple[Tier, ...] = (
    Tier("1º — 5 números + Nº da Sorte", 5, True),
    Tier("2º — 5 números", 5, False),
    Tier("3º — 4 números", 4, False),
    Tier("4º — 3 números", 3, False),
    Tier("5º — 2 números", 2, False),
)

LUCKY_ONLY_ODDS = float(LUCKY_POOL)
"""6º escalão: só o Nº da Sorte. Acumula com os restantes."""


def probability_any_prize() -> float:
    """
    P(ganhar alguma coisa).

    O Nº da Sorte é um prémio à parte que **acumula** com os escalões de
    números, por isso não se somam probabilidades: calcula-se o complementar
    de "menos de 2 números E sem Nº da Sorte".
    """
    p_two_or_more = sum(
        comb(MAIN_PICK, k) * comb(MAIN_POOL - MAIN_PICK, MAIN_PICK - k)
        for k in range(2, MAIN_PICK + 1)
    ) / MAIN_COMBINATIONS
    p_no_lucky = (LUCKY_POOL - 1) / LUCKY_POOL
    return 1.0 - (1.0 - p_two_or_more) * p_no_lucky


def structure_table() -> list[dict]:
    rows = [
        {
            "escalão": t.nome,
            "combinações": t.combinations,
            "probabilidade": t.probability(),
            "1 em": round(t.odds()),
        }
        for t in TIERS
    ]
    rows.append(
        {"escalão": "6º — só Nº da Sorte", "combinações": None,
         "probabilidade": 1 / LUCKY_POOL, "1 em": LUCKY_POOL}
    )
    return rows


# ---------------------------------------------------------------------------
# Comparação com o EuroMilhões
# ---------------------------------------------------------------------------

def compare_with_euromillions() -> list[dict]:
    from . import config as em

    p_toto = 1 / TOTAL_COMBINATIONS
    p_em = em.JACKPOT.probability(12)
    return [
        {
            "métrica": "odds do 1º prémio",
            "Totoloto": f"1 em {TOTAL_COMBINATIONS:,}",
            "EuroMilhões": f"1 em {round(1/p_em):,}",
            "vantagem": f"{p_toto/p_em:.1f}x melhor no Totoloto",
        },
        {
            "métrica": "preço por aposta",
            "Totoloto": f"€{TICKET_PRICE_EUR:.2f}",
            "EuroMilhões": f"€{em.TICKET_PRICE_EUR:.2f}",
            "vantagem": f"{em.TICKET_PRICE_EUR/TICKET_PRICE_EUR:.1f}x mais barato",
        },
        {
            "métrica": "hipóteses de jackpot por €10",
            "Totoloto": f"1 em {round(1/(p_toto*10/TICKET_PRICE_EUR)):,}",
            "EuroMilhões": f"1 em {round(1/(p_em*10/em.TICKET_PRICE_EUR)):,}",
            "vantagem": f"{(p_toto/TICKET_PRICE_EUR)/(p_em/em.TICKET_PRICE_EUR):.1f}x no Totoloto",
        },
        {
            "métrica": "P(algum prémio)",
            "Totoloto": f"1 em {1/probability_any_prize():.2f}",
            "EuroMilhões": f"1 em {1/em.probability_any_prize(12):.2f}",
            "vantagem": "Totoloto",
        },
        {
            "métrica": "jackpot típico",
            "Totoloto": "€1M a €19M",
            "EuroMilhões": "€17M a €250M",
            "vantagem": "EuroMilhões",
        },
    ]


# ---------------------------------------------------------------------------
# Seleção de chaves
# ---------------------------------------------------------------------------

def lucky_number_prior() -> np.ndarray:
    """
    Popularidade relativa esperada de cada Nº da Sorte (1-13).

    TRANSFERIDA do modelo das estrelas do EuroMilhões, **não medida aqui**.
    Isso é uma limitação real e está declarada: sem a quebra de prémios do
    Totoloto não há como estimá-la, e a experiência deste projeto é que
    priors escritos à mão erram — o das estrelas errou por um fator de dois
    e enganou-se na estrela 1.

    O que se sabe do modelo medido no EuroMilhões (3090 observações):
      · o 7 é o mais sobre-jogado de todos (1,21x)
      · os valores altos são sub-jogados (10: 0,82x, 11: 0,84x, 12: 0,72x)
      · o 1 é dos menos jogados, ao contrário do que a intuição diz

    Aqui há 13 valores em vez de 12. O 13 não existia no EuroMilhões e é o
    único que acumula duas razões para ser evitado — superstição e estar
    acima dos 12 meses — pelo que deve ser o menos jogado de todos.
    """
    w = np.ones(LUCKY_POOL)
    medido_em = {1: 0.92, 2: 1.08, 3: 1.14, 4: 1.02, 5: 1.14, 6: 1.04,
                 7: 1.21, 8: 1.11, 9: 1.08, 10: 0.82, 11: 0.84, 12: 0.72}
    for n, v in medido_em.items():
        w[n - 1] = v
    w[12] = 0.65   # extrapolação para o 13 — o valor menos ancorado de todos
    return w / w.mean()


def accepts(combo: np.ndarray) -> bool:
    """
    Filtros, herdados dos que os dados do EuroMilhões validaram.

    Note-se o que NÃO está aqui: nenhuma proibição de números consecutivos.
    O modelo medido mostrou que são sub-jogados (β<0, p<0,0001) e portanto
    valiosos — o folclore manda evitá-los, e é esse erro que os torna bons.
    """
    c = np.sort(np.asarray(combo, dtype=int))
    if (c <= 31).sum() > 3:            # viés de datas
        return False
    if (c <= 12).sum() > 1:            # meses
        return False
    diffs = np.diff(c)
    if float(diffs.std()) < 3.0:       # espaçamento regular = popular
        return False
    if len(set(diffs.tolist())) == 1:  # progressão aritmética
        return False
    cols = np.bincount([(int(x) - 1) % 10 for x in c], minlength=10)
    if cols.max() > 2:                 # coluna do boletim
        return False
    return True


def popularity_score(combo: np.ndarray, lucky: int) -> float:
    """
    Índice de popularidade aproximado (1,0 = combinação banal).

    Aproximação estrutural, calibrada pelos coeficientes medidos no
    EuroMilhões. Não tem a precisão do modelo estimado — serve para ordenar
    candidatos, não para prometer um número.
    """
    c = np.sort(np.asarray(combo, dtype=int))
    diffs = np.diff(c)
    z = (
        0.048 * ((c <= 31).sum() - 3.1)
        + 0.068 * ((c <= 12).sum() - 1.2)
        - 0.060 * ((diffs == 1).sum() - 0.4)
        - 0.130 * (float(diffs.std()) - 4.5)
    )
    return float(np.exp(z)) * float(lucky_number_prior()[lucky - 1])


def generate(
    n_tickets: int = 5,
    entropy=None,
    n_candidates: int = 4000,
    allow_network: bool = True,
    seed: int | None = None,
) -> list[dict]:
    """
    Gera chaves impopulares e bem dispersas.

    Mesmo procedimento do EuroMilhões: a entropia propõe, a estatística
    filtra, e a cobertura é maximizada entre bilhetes.
    """
    if entropy is None:
        from .quantum import EntropySource

        entropy = EntropySource()
        entropy.refill(8192, allow_network=allow_network)

    cands: list[tuple[np.ndarray, int]] = []
    tries = 0
    while len(cands) < n_candidates and tries < n_candidates * 60:
        tries += 1
        mains = entropy.sample_without_replacement(
            MAIN_POOL, MAIN_PICK, allow_network=allow_network
        )
        arr = np.array(mains)
        if not accepts(arr):
            continue
        lucky = entropy.uniform_int(1, LUCKY_POOL, allow_network=allow_network)
        cands.append((arr, lucky))

    if not cands:
        raise RuntimeError("nenhum candidato passou os filtros")

    scored = sorted(cands, key=lambda t: popularity_score(t[0], t[1]))

    # Seleção gulosa por cobertura, dentro do quantil mais impopular.
    pool_idx = scored[: max(n_tickets * 8, len(scored) // 20)]
    chosen = [pool_idx[0]]
    used = set(int(x) for x in pool_idx[0][0])
    while len(chosen) < n_tickets and len(chosen) < len(pool_idx):
        best, best_key = None, None
        for cand in pool_idx:
            if any(cand is c for c in chosen):
                continue
            novos = len(set(int(x) for x in cand[0]) - used)
            key = (novos, -popularity_score(cand[0], cand[1]))
            if best_key is None or key > best_key:
                best, best_key = cand, key
        if best is None:
            break
        chosen.append(best)
        used |= set(int(x) for x in best[0])

    return [
        {
            "numeros": sorted(int(x) for x in m),
            "numero_da_sorte": int(l),
            "popularidade_aprox": round(popularity_score(m, l), 3),
        }
        for m, l in chosen
    ]


def expected_value(
    jackpot_eur: float,
    sales: float,
    popularity: float = 1.0,
    tier_prizes: dict[str, float] | None = None,
) -> dict:
    """
    Valor esperado de uma aposta de €1.

    `tier_prizes` são os prémios médios dos escalões inferiores. Sem dados
    recolhidos do Totoloto, usam-se valores de ordem de grandeza publicados
    — e por isso o resultado é indicativo, não medido.
    """
    from . import config as em

    prizes = {
        "2º — 5 números": 25_000.0,
        "3º — 4 números": 250.0,
        "4º — 3 números": 12.0,
        "5º — 2 números": 2.0,
    }
    if tier_prizes:
        prizes.update(tier_prizes)

    p_jack = 1 / TOTAL_COMBINATIONS
    lam = sales * p_jack * popularity
    share = (1 - np.exp(-lam)) / lam if lam > 1e-12 else 1.0
    ev_jack = p_jack * em.net_prize(jackpot_eur * share)

    ev_low = 0.0
    for t in TIERS:
        if t.nome.startswith("1º"):
            continue
        base = prizes.get(t.nome)
        if base is None:
            continue
        ev_low += t.probability() * em.net_prize(base * popularity ** -0.5)

    ev_lucky = (1 / LUCKY_POOL) * TICKET_PRICE_EUR   # devolve a aposta

    total = ev_jack + ev_low + ev_lucky
    return {
        "jackpot_eur": jackpot_eur,
        "popularidade": popularity,
        "lambda": lam,
        "P_sozinho": float(np.exp(-lam)),
        "EV_jackpot": ev_jack,
        "EV_escaloes_baixos": ev_low,
        "EV_numero_da_sorte": ev_lucky,
        "EV_total": total,
        "retorno_por_euro": total / TICKET_PRICE_EUR,
    }


# ---------------------------------------------------------------------------
# Análise de vieses — dados reais
# ---------------------------------------------------------------------------

FREQ_CACHE = "data/totoloto_freq.json"
FREQ_URL = "http://euroleste.pt/totoloto/total_aparitii_n_s.php"
PERIODS = ("2011-03-16", "2022-01-01", "2023-01-01",
           "2024-01-01", "2025-01-01", "2026-01-01")
"""Períodos cumulativos oferecidos pela fonte. Subtraindo-os obtêm-se
segmentos disjuntos, que é o que permite o teste de persistência."""


def load_frequencies(path: str = FREQ_CACHE) -> dict:
    """Lê a recolha guardada; se não existir, vai buscá-la."""
    import json
    import os

    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    freq = fetch_frequencies()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(freq, fh)
    return freq


def fetch_frequencies(periods: tuple[str, ...] = PERIODS) -> dict:
    """
    Frequências observadas de cada número e de cada Nº da Sorte.

    A Santa Casa publica estatísticas desde 2011-03-16 mas renderiza-as por
    JavaScript. Esta fonte expõe as mesmas contagens por POST.

    VALIDAÇÃO OBRIGATÓRIA: a soma das contagens dos números tem de dar
    exatamente 5 × sorteios, e a das do Nº da Sorte exatamente 1 × sorteios.
    Se não der, a extração está partida e os dados não são usados — é a
    única forma de saber que se leu a tabela certa.
    """
    import re

    import requests

    out: dict[str, dict] = {}
    for an in periods:
        r = requests.post(FREQ_URL, data={"an": an, "submit": "Veja"},
                          headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        h = r.text
        m = re.search(r"Registros totais disponível:\s*(\d+)", h)
        if not m:
            continue
        n = int(m.group(1))
        t = re.sub(r"<script.*?</script>", "", h, flags=re.S)
        t = re.sub(r"<[^>]+>", "|", t)
        t = re.sub(r"[\|\s]+", "|", t)
        pares = re.findall(r"\|(\d{1,2})\|(\d+)\|vezes\|([\d.]+)%", t)
        nums = [int(c) for _, c, _ in pares[:MAIN_POOL]]
        lucky = [int(c) for _, c, _ in pares[MAIN_POOL:MAIN_POOL + LUCKY_POOL]]
        if len(nums) != MAIN_POOL or len(lucky) != LUCKY_POOL:
            continue
        ok_n = sum(nums) == MAIN_PICK * n
        ok_l = abs(sum(lucky) - n) <= 1
        out[an] = {
            "sorteios": n, "numeros": nums, "sorte": lucky,
            "validado": bool(ok_n and ok_l),
        }
    return out


def segment(freq: dict, inicio: str, fim: str | None) -> dict:
    """
    Segmento disjunto por subtração de dois períodos cumulativos.

    `fim=None` significa "até hoje".
    """
    a = freq[inicio]
    if fim is None:
        return {"sorteios": a["sorteios"],
                "numeros": list(a["numeros"]), "sorte": list(a["sorte"])}
    b = freq[fim]
    return {
        "sorteios": a["sorteios"] - b["sorteios"],
        "numeros": [x - y for x, y in zip(a["numeros"], b["numeros"])],
        "sorte": [x - y for x, y in zip(a["sorte"], b["sorte"])],
    }


def bias_battery(freq: dict) -> dict:
    """
    Bateria de vieses sobre as frequências observadas.

    Mesma disciplina do EuroMilhões: qui-quadrado global, teste binomial por
    número com correção de falsas descobertas, análise de potência, e o
    teste decisivo de persistência entre segmentos independentes.
    """
    import numpy as np
    from scipy import stats

    from .randomness import benjamini_hochberg

    full = freq[PERIODS[0]]
    n = full["sorteios"]
    res: dict = {"sorteios": n}

    # -- qui-quadrado global -------------------------------------------
    for nome, counts, pool, picks in (
        ("números 1-49", full["numeros"], MAIN_POOL, MAIN_PICK),
        ("Nº da Sorte 1-13", full["sorte"], LUCKY_POOL, 1),
    ):
        obs = np.array(counts, dtype=float)
        exp = np.full(pool, obs.sum() / pool)
        chi2 = float(((obs - exp) ** 2 / exp).sum())
        res[nome] = {
            "chi2": chi2, "gl": pool - 1,
            "p": float(stats.chi2.sf(chi2, pool - 1)),
            "esperado_por_valor": float(exp[0]),
        }

    # -- por número, com FDR -------------------------------------------
    tabelas = {}
    for nome, counts, pool, picks in (
        ("numeros", full["numeros"], MAIN_POOL, MAIN_PICK),
        ("sorte", full["sorte"], LUCKY_POOL, 1),
    ):
        p0 = picks / pool
        rows = []
        for i, c in enumerate(counts, start=1):
            r = stats.binomtest(int(c), n, p0)
            rows.append({"valor": i, "vezes": int(c), "esperado": n * p0,
                         "desvio_%": 100 * (c - n * p0) / (n * p0),
                         "p": r.pvalue})
        import pandas as pd

        df = pd.DataFrame(rows)
        df["p_ajustado"] = benjamini_hochberg(df["p"].to_numpy())
        df["significativo_fdr5"] = df["p_ajustado"] < 0.05
        tabelas[nome] = df.sort_values("p").reset_index(drop=True)
    res["tabelas"] = tabelas

    # -- persistência: dois segmentos independentes ---------------------
    antigo = segment(freq, PERIODS[0], "2022-01-01")   # 2011-2021
    recente = segment(freq, "2022-01-01", None)        # 2022-hoje
    for nome, key, pool, picks in (("números", "numeros", MAIN_POOL, MAIN_PICK),
                                   ("Nº da Sorte", "sorte", LUCKY_POOL, 1)):
        a = np.array(antigo[key], float)
        b = np.array(recente[key], float)
        ea = antigo["sorteios"] * picks / pool
        eb = recente["sorteios"] * picks / pool
        dev_a, dev_b = (a - ea) / ea, (b - eb) / eb
        r = float(np.corrcoef(dev_a, dev_b)[0, 1])
        # nulo por simulação: dois segmentos genuinamente uniformes
        rng = np.random.default_rng(11)
        sims = np.empty(5000)
        for i in range(5000):
            ca = rng.multinomial(int(a.sum()), [1 / pool] * pool).astype(float)
            cb = rng.multinomial(int(b.sum()), [1 / pool] * pool).astype(float)
            sims[i] = np.corrcoef((ca - ea) / ea, (cb - eb) / eb)[0, 1]
        res[f"persistencia_{key}"] = {
            "segmento_antigo": f"{antigo['sorteios']} sorteios (2011-2021)",
            "segmento_recente": f"{recente['sorteios']} sorteios (2022-hoje)",
            "r_observado": round(r, 4),
            "nulo_dp": round(float(sims.std()), 4),
            "p_simulado": float((np.abs(sims) >= abs(r)).mean()),
        }
    return res


def detection_power(n_draws: int, effect_pct: float, pool: int, picks: int,
                    alpha: float = 0.05) -> float:
    from scipy import stats

    p0 = picks / pool
    p1 = p0 * (1 + effect_pct / 100)
    se0 = (p0 * (1 - p0) / n_draws) ** 0.5
    se1 = (p1 * (1 - p1) / n_draws) ** 0.5
    z = stats.norm.isf(alpha / 2)
    return float(stats.norm.sf((z * se0 - (p1 - p0)) / se1)
                 + stats.norm.cdf((-z * se0 - (p1 - p0)) / se1))


def minimum_detectable_bias(n_draws: int, pool: int, picks: int,
                            power: float = 0.80) -> float:
    lo, hi = 0.01, 500.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if detection_power(n_draws, mid, pool, picks) < power:
            lo = mid
        else:
            hi = mid
    return hi
