"""
Seleção de bilhetes: entropia quântica como motor, estatística como filtro.

O procedimento:

  1. A fonte quântica propõe milhares de combinações candidatas. Nenhuma
     delas carrega viés humano, por construção.
  2. O modelo de popularidade estima, para cada uma, quantas outras pessoas
     a jogariam.
  3. O motor de EV traduz isso no cheque esperado, dado o jackpot atual.
  4. Escolhem-se as melhores, com dispersão entre bilhetes.

Um aviso que faz parte da engenharia, não da retórica: se este método se
tornasse popular, deixaria de funcionar — combinações "impopulares"
passariam a ser jogadas por toda a gente que corresse o mesmo código, e a
partilha voltaria. A aleatoriedade da proposta (passo 1) é o que protege
contra isso: dois utilizadores deste sistema, com sementes quânticas
diferentes, obtêm bilhetes diferentes com popularidade igualmente baixa.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .ev import expected_value
from .popularity import PoissonGLM, design_matrix, features_main, slip_col, slip_row


# ---------------------------------------------------------------------------
# Filtros duros
# ---------------------------------------------------------------------------

@dataclass
class Filters:
    """
    Restrições que eliminam as famílias de combinações sobre-jogadas.

    ATENÇÃO — ESTES FILTROS SÃO DITADOS PELOS DADOS, NÃO POR INTUIÇÃO.

    A primeira versão deste ficheiro proibia números consecutivos, porque é
    isso que "toda a gente sabe" sobre apostas. O modelo estimado sobre
    1936 sorteios diz o contrário, e com força (β = -0.060, p < 0.0001):
    combinações com pares consecutivos são **menos** populares. A razão é
    psicológica e bem conhecida — as pessoas acham que 23-24 "não parece
    aleatório" e evitam-no. Isso torna os consecutivos sub-jogados, e
    portanto valiosos para quem quer partilhar o prémio com menos gente.

    O mesmo se aplica ao espaçamento: `espaco_regular` é o preditor mais
    forte de todos (β = -0.130, p < 10⁻¹¹), e o sinal negativo significa
    que as pessoas preferem combinações visualmente regulares. Espaçamento
    irregular é impopular — logo, bom.

    O filtro da soma foi removido: `soma_norm` não tem significância
    estatística (p = 0.55). Era folclore, e restringir sem evidência só
    encolhe o espaço de busca e cria uma assinatura partilhada por todos os
    utilizadores deste código.
    """

    max_ate_31: int = 3           # β>0: datas de nascimento — o efeito clássico
    max_ate_12: int = 1           # β>0 (p=0.002): meses, ainda mais concentrado
    min_consecutivos: int = 0     # β<0: consecutivos são SUB-jogados; não proibir
    max_mesma_coluna: int = 2     # geometria do boletim
    max_mesma_linha: int = 2
    min_espaco_irregular: float = 3.0   # β<0: exigir espaçamento pouco "certinho"
    proibir_prog_aritmetica: bool = True

    def accepts(self, combo: np.ndarray) -> bool:
        c = np.sort(np.asarray(combo, dtype=int))
        if (c <= 31).sum() > self.max_ate_31:
            return False
        if (c <= 12).sum() > self.max_ate_12:
            return False
        diffs = np.diff(c)
        if (diffs == 1).sum() < self.min_consecutivos:
            return False
        cols = np.bincount([slip_col(int(x)) for x in c], minlength=10)
        if cols.max() > self.max_mesma_coluna:
            return False
        rows = np.bincount([slip_row(int(x)) for x in c], minlength=5)
        if rows.max() > self.max_mesma_linha:
            return False
        if float(diffs.std()) < self.min_espaco_irregular:
            return False
        if self.proibir_prog_aritmetica and len(set(diffs.tolist())) == 1:
            return False
        return True


# ---------------------------------------------------------------------------
# Estrelas
# ---------------------------------------------------------------------------

def star_popularity_prior(star_pool: int = 12) -> np.ndarray:
    """
    Recurso APENAS para quando não há dados de quebra de prémios.

    Isto é um palpite, e um palpite medíocre: captava cerca de metade do
    efeito real e errava a estrela 1 (que supunha popular, e é das menos
    jogadas). Use-se `stars.StarPopularity`, que mede em vez de adivinhar.
    """
    w = np.ones(star_pool)
    for i in range(star_pool):
        n = i + 1
        w[i] = 1.12 if n <= 9 else 0.80
    return w / w.mean()


def star_pair_popularity(
    pair: tuple[int, int], star_pool: int = 12, star_model=None
) -> float:
    """
    Popularidade de um par de estrelas.

    Com `star_model` (um `stars.StarPopularity` estimado a partir de ~5000
    observações), isto é uma medição. Sem ele, é o palpite acima.
    """
    if star_model is not None:
        return star_model.pair_popularity((int(pair[0]), int(pair[1])))
    w = star_popularity_prior(star_pool)
    return float(w[pair[0] - 1] * w[pair[1] - 1])


# ---------------------------------------------------------------------------
# Otimização
# ---------------------------------------------------------------------------

@dataclass
class Ticket:
    mains: list[int]
    stars: list[int]
    popularity: float
    ev_liquido: float
    retorno_por_euro: float

    def __str__(self) -> str:
        m = " ".join(f"{n:2d}" for n in self.mains)
        s = " ".join(f"{n:2d}" for n in self.stars)
        return (
            f"{m}   ★ {s}   "
            f"popularidade {self.popularity:5.2f}x   "
            f"EV €{self.ev_liquido:.3f}  ({self.retorno_por_euro:.1%} do custo)"
        )


def optimize(
    model: PoissonGLM,
    entropy,
    n_tickets: int = 5,
    jackpot_eur: float = 100e6,
    sales: float = 80e6,
    star_pool: int = 12,
    tier_prizes: dict[str, float] | None = None,
    n_candidates: int = 6000,
    filters: Filters | None = None,
    min_disjoint: int = 3,
    allow_network: bool = True,
    star_model=None,
    maximize_coverage: bool = True,
    popularity_quantile: float = 0.05,
    ev_m1lhao: float = 0.0,
    verbose: bool = False,
) -> tuple[list[Ticket], pd.DataFrame]:
    """
    Gera candidatos com entropia quântica e escolhe a melhor carteira.

    `maximize_coverage`: otimiza as duas alavancas (popularidade E
    cobertura) em vez de só o valor esperado por bilhete. Ver o comentário
    extenso no corpo da função — desligar isto produz carteiras com melhor
    EV individual e P(não ganhar nada) pior do que jogar ao acaso.

    `popularity_quantile`: fração dos melhores candidatos por EV dentro da
    qual se procura cobertura.

    `min_disjoint`: usado apenas quando `maximize_coverage=False`.
    """
    filters = filters or Filters()

    candidates: list[np.ndarray] = []
    star_candidates: list[list[int]] = []
    attempts = 0
    max_attempts = n_candidates * 60

    while len(candidates) < n_candidates and attempts < max_attempts:
        attempts += 1
        mains, stars = entropy.draw_ticket(star_pool, allow_network=allow_network)
        arr = np.array(mains)
        if not filters.accepts(arr):
            continue
        candidates.append(arr)
        star_candidates.append(stars)

    if not candidates:
        raise RuntimeError("nenhum candidato passou os filtros — relaxe as restrições")

    combos = np.array(candidates)
    pop_main = model.predict_popularity(combos)
    pop_star = np.array(
        [star_pair_popularity(tuple(s), star_pool, star_model) for s in star_candidates]
    )
    popularity = pop_main * pop_star

    evs = np.array(
        [
            expected_value(
                jackpot_eur, sales, float(p), star_pool, tier_prizes,
                ev_m1lhao=ev_m1lhao
            ).ev_liquido
            for p in popularity
        ]
    )

    table = pd.DataFrame(
        {
            "combo": [" ".join(f"{x:02d}" for x in c) for c in combos],
            "estrelas": [" ".join(f"{x:02d}" for x in s) for s in star_candidates],
            "popularidade": popularity,
            "ev": evs,
        }
    ).sort_values("ev", ascending=False).reset_index(drop=True)

    if verbose:
        print(f"  {len(candidates)} candidatos aceites em {attempts} propostas quânticas")

    # ------------------------------------------------------------------
    # Seleção: as duas alavancas ao mesmo tempo.
    #
    # Ordenar apenas por EV parece óbvio e é um erro. As combinações de
    # baixa popularidade concentram-se nos números altos (>31), por isso os
    # melhores bilhetes individuais repetem os mesmos números entre si. O
    # resultado é uma carteira com EV ótimo por bilhete mas com apenas ~19
    # números distintos — mais sobreposta do que uma escolha aleatória, e
    # com P(não ganhar nada) PIOR do que jogar ao acaso.
    #
    # As duas alavancas são independentes e ambas valem dinheiro:
    #   popularidade baixa -> cheque maior quando se acerta
    #   cobertura alta     -> menos vezes a sair de mãos vazias
    #
    # A solução é hierárquica: restringir primeiro ao quantil mais
    # impopular (é aí que está o valor, e dentro dele o EV varia pouco), e
    # só depois maximizar a cobertura por seleção gulosa.
    # ------------------------------------------------------------------
    order = np.argsort(-evs)
    if maximize_coverage:
        cutoff = max(n_tickets * 8, int(len(order) * popularity_quantile))
        pool_idx = list(order[:cutoff])
        chosen_idx: list[int] = [pool_idx[0]]
        used_nums: set[int] = set(int(x) for x in combos[pool_idx[0]])
        while len(chosen_idx) < n_tickets and len(chosen_idx) < len(pool_idx):
            best, best_key = None, None
            for i in pool_idx:
                if i in chosen_idx:
                    continue
                new = len(set(int(x) for x in combos[i]) - used_nums)
                key = (new, float(evs[i]))    # desempate pelo melhor EV
                if best_key is None or key > best_key:
                    best, best_key = i, key
            if best is None:
                break
            chosen_idx.append(best)
            used_nums |= set(int(x) for x in combos[best])
        selected = chosen_idx
    else:
        selected = []
        used: list[set[int]] = []
        for idx in order:
            cs = set(int(x) for x in combos[idx])
            if any(len(cs - u) < min_disjoint for u in used):
                continue
            selected.append(int(idx))
            used.append(cs)
            if len(selected) >= n_tickets:
                break

    chosen: list[Ticket] = []
    for idx in selected:
        r = expected_value(
            jackpot_eur, sales, float(popularity[idx]), star_pool, tier_prizes,
            ev_m1lhao=ev_m1lhao,
        )
        chosen.append(
            Ticket(
                mains=[int(x) for x in combos[idx]],
                stars=[int(x) for x in star_candidates[idx]],
                popularity=float(popularity[idx]),
                ev_liquido=r.ev_liquido,
                retorno_por_euro=r.retorno_por_euro,
            )
        )

    return chosen, table


def compare_to_typical(
    model: PoissonGLM,
    tickets: list[Ticket],
    jackpot_eur: float,
    sales: float,
    star_pool: int = 12,
    tier_prizes: dict[str, float] | None = None,
    star_model=None,
    ev_m1lhao: float = 0.0,
) -> pd.DataFrame:
    """
    Compara a carteira escolhida com dois pontos de referência: uma aposta
    típica (popularidade 1.0) e uma aposta "de datas" (o erro mais comum).
    """
    date_combo = np.array([3, 7, 11, 19, 24])   # todos ≤31, padrão de aniversários
    pop_date = float(model.predict_popularity(date_combo)[0]) * star_pair_popularity(
        (3, 7), star_pool, star_model
    )
    pop_ours = float(np.mean([t.popularity for t in tickets]))

    rows = []
    for label, pop in (
        ("aposta de datas (todos ≤31)", pop_date),
        ("aposta média / aleatória", 1.0),
        ("carteira otimizada", pop_ours),
    ):
        r = expected_value(
            jackpot_eur, sales, pop, star_pool, tier_prizes, ev_m1lhao=ev_m1lhao
        )
        rows.append(
            {
                "estratégia": label,
                "popularidade": round(pop, 3),
                "EV_líquido_€": round(r.ev_liquido, 4),
                "retorno_por_euro": round(r.retorno_por_euro, 4),
                "λ_partilha": round(r.lambda_jackpot, 2),
                "fatia_do_jackpot_%": round(100 * r.fator_partilha, 1),
            }
        )
    out = pd.DataFrame(rows)
    base = out.loc[out["estratégia"] == "aposta de datas (todos ≤31)", "EV_líquido_€"].iloc[0]
    out["ganho_vs_datas_%"] = (100 * (out["EV_líquido_€"] / base - 1)).round(1)
    return out
