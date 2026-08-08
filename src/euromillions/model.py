"""
O modelo consolidado — tudo o que o histórico ensinou, num só objeto.

O QUE "MELHOR SEGUNDO O HISTÓRICO" SIGNIFICA AQUI
-------------------------------------------------
Convém ser exato, porque a expressão admite duas leituras e uma delas está
morta:

* **Leitura que não funciona:** "jogar os números que mais saíram". Testada
  em 1970 sorteios com 12 famílias de testes e um backtest de 2 milhões de
  apostas. Não funciona, e a razão é estrutural — as bolas não têm memória
  (`docs/04`, `docs/08`).

* **Leitura que funciona:** "usar o histórico para saber o que as OUTRAS
  PESSOAS escolhem, e evitar essas escolhas". Aí o histórico é
  informativo, mensurável e validado fora da amostra.

Este módulo implementa a segunda. Junta os cinco componentes estimados a
partir de 1970 sorteios e 1936 quebras de prémios, e produz a carteira com
melhor valor esperado.

O QUE ENTRA
-----------
| componente | origem | validação |
|---|---|---|
| popularidade dos números | GLM Poisson, 1936 sorteios | ρ = 0,411 fora da amostra |
| popularidade das estrelas | 3090 observações, 12 parâmetros | ρ = 0,775 fora da amostra |
| elasticidades por escalão | regressão, 1936 sorteios | 5+0 mede 0,956 vs teoria 1,0 |
| M1lhão | quota PT medida no escalão 2+0 | aritmética direta |
| cobertura da carteira | combinatória exata | simulação a 150k sorteios |

Cada um destes números foi **medido**. Onde ainda há suposições, elas estão
declaradas em `PROVENANCE` e o auditor (`audit.py`) sinaliza-as — porque é
aí que vive o próximo erro.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from . import config, elasticity, ev, m1lhao, portfolio, quantum, stars
from . import dataset as ds
from . import optimizer as opt
from .popularity import PoissonGLM, fit_popularity_model

MODEL_VERSION = "3.0"


# ---------------------------------------------------------------------------
# Proveniência: de onde vem cada número que o modelo usa
# ---------------------------------------------------------------------------

@dataclass
class Input:
    """Um input do modelo, com a sua origem declarada."""

    nome: str
    origem: str          # "medido" | "oficial" | "assumido"
    valor: str
    nota: str = ""


PROVENANCE: tuple[Input, ...] = (
    Input("probabilidades dos 13 escalões", "oficial",
          "combinatória exata", "calculadas de raiz, verificadas em testes"),
    Input("preço da aposta", "oficial", "€2,50", "Portugal, inclui código M1lhão"),
    Input("Imposto do Selo", "oficial", "20% acima de €5.000", "verba 11.2.2 da TGIS"),
    Input("teto do jackpot", "oficial", "€250M", "regra em vigor"),
    Input("popularidade dos números", "medido",
          "GLM Poisson, 12 características", "ρ=0,411 fora da amostra"),
    Input("popularidade das estrelas", "medido",
          "12 parâmetros, 3090 obs.", "ρ=0,775 fora da amostra"),
    Input("elasticidades por escalão", "medido",
          "regressão com π previsto", "5+0 mede 0,956 vs teoria 1,0"),
    Input("quota portuguesa", "medido", "8,2% via escalão 2+0", "período 2025+"),
    Input("vendas por sorteio", "assumido",
          "vencedores totais ÷ P(prémio)", "estimador circular — ver audit"),
    Input("fração de apostas aleatórias", "assumido",
          "não estimada", "comprime a popularidade medida; teto desconhecido"),
    Input("preços dos escalões baixos", "assumido",
          "mediana desde 2020", "variam com vendas e acumulação"),
)


# ---------------------------------------------------------------------------
# O modelo
# ---------------------------------------------------------------------------

@dataclass
class Recommendation:
    tickets: list[opt.Ticket]
    jackpot_eur: float
    sales: float
    ev_total: float
    custo: float
    p_nada: float
    cobertura: int
    popularidade_media: float
    ev_m1lhao: float

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "nº": i,
                    "números": " ".join(f"{x:02d}" for x in t.mains),
                    "estrelas": " ".join(f"{x:02d}" for x in t.stars),
                    "popularidade": round(t.popularity, 3),
                    "EV_€": round(t.ev_liquido, 4),
                }
                for i, t in enumerate(self.tickets, 1)
            ]
        )


@dataclass
class EuroMillionsModel:
    """Modelo de produção. Estima-se uma vez, usa-se muitas."""

    popularity: PoissonGLM | None = None
    star_model: stars.StarPopularity | None = None
    tier_prizes: dict[str, float] = field(default_factory=dict)
    elasticities: dict[str, float] = field(default_factory=dict)
    ev_m1lhao: float = 0.0
    sales_estimate: float = 24e6
    n_draws: int = 0
    n_breakdowns: int = 0
    last_draw: str = ""
    fitted_at: str = ""

    # -- estimação --------------------------------------------------------

    @classmethod
    def fit(cls, verbose: bool = False) -> "EuroMillionsModel":
        draws = ds.load_draws()
        bd = ds.load_breakdown()
        master = ds.build_master()

        pop, used = fit_popularity_model(master)

        try:
            sm = stars.fit_star_model(draws, bd)
        except ValueError:
            sm = None

        prizes = ev.empirical_tier_prizes(bd)

        el_tab = elasticity.measure_elasticities(master, bd, pop)
        el = elasticity.fitted_table(el_tab)
        # Os escalões de 5 números têm elasticidade 1,0 por definição:
        # acertar os 5 números É ter a nossa combinação exata. A medição
        # independente (0,956 no 5+0) confirma-o e serve de auto-teste.
        for lbl in ("5+2", "5+1", "5+0"):
            el[lbl] = 1.0

        recent = master[[d.year >= 2025 for d in master["date"]]]
        tue = float(recent[recent["dow"] == "Tue"]["sales_est"].median())
        fri = float(recent[recent["dow"] == "Fri"]["sales_est"].median())
        m1 = float(m1lhao.summary(bd, tue, fri)["ev_por_aposta_eur"])

        model = cls(
            popularity=pop,
            star_model=sm,
            tier_prizes=prizes,
            elasticities=el,
            ev_m1lhao=m1,
            sales_estimate=float(recent["sales_est"].median()),
            n_draws=len(draws),
            n_breakdowns=int(master["winners_main5"].notna().sum()),
            last_draw=str(draws["date"].max()),
            fitted_at=dt.datetime.now().isoformat(timespec="seconds"),
        )
        if verbose:
            print(model.describe())
        return model

    # -- utilização -------------------------------------------------------

    def recommend(
        self,
        n_tickets: int = 5,
        jackpot_eur: float = 60e6,
        sales: float | None = None,
        n_candidates: int = 4000,
        allow_network: bool = True,
        entropy=None,
        seed: int | None = None,
    ) -> Recommendation:
        """A carteira com melhor valor esperado, dadas as condições atuais."""
        if self.popularity is None:
            raise RuntimeError("modelo não estimado — use EuroMillionsModel.fit()")

        sales = sales if sales is not None else self.sales_estimate
        src = entropy or quantum.EntropySource()
        if entropy is None:
            src.refill(8192, allow_network=allow_network)

        saved = dict(ev.TIER_ELASTICITY)
        try:
            ev.TIER_ELASTICITY.update(self.elasticities)
            tickets, _ = opt.optimize(
                self.popularity, src,
                n_tickets=n_tickets,
                jackpot_eur=jackpot_eur,
                sales=sales,
                tier_prizes=self.tier_prizes,
                n_candidates=n_candidates,
                allow_network=allow_network,
                star_model=self.star_model,
                ev_m1lhao=self.ev_m1lhao,
            )
        finally:
            ev.TIER_ELASTICITY.clear()
            ev.TIER_ELASTICITY.update(saved)

        pairs = [(t.mains, t.stars) for t in tickets]
        sim = portfolio.simulate_portfolio(
            pairs, n_sim=40_000, tier_prizes=self.tier_prizes
        )
        cov = portfolio.coverage_score(pairs)

        return Recommendation(
            tickets=tickets,
            jackpot_eur=jackpot_eur,
            sales=sales,
            ev_total=float(sum(t.ev_liquido for t in tickets)),
            custo=n_tickets * config.TICKET_PRICE_EUR,
            p_nada=float(sim["p_nada"]),
            cobertura=int(cov["numeros_distintos"]),
            popularidade_media=float(np.mean([t.popularity for t in tickets])),
            ev_m1lhao=self.ev_m1lhao,
        )

    def score(self, mains: list[int], stars_: list[int], jackpot_eur: float = 60e6) -> dict:
        """Avalia uma aposta que o utilizador já tem em mente."""
        if self.popularity is None:
            raise RuntimeError("modelo não estimado")
        pm = float(self.popularity.predict_popularity(np.array(mains))[0])
        ps = opt.star_pair_popularity(tuple(stars_), 12, self.star_model)
        pi = pm * ps
        saved = dict(ev.TIER_ELASTICITY)
        try:
            ev.TIER_ELASTICITY.update(self.elasticities)
            r = ev.expected_value(
                jackpot_eur, self.sales_estimate, pi, 12,
                self.tier_prizes, ev_m1lhao=self.ev_m1lhao,
            )
        finally:
            ev.TIER_ELASTICITY.clear()
            ev.TIER_ELASTICITY.update(saved)
        return {
            "numeros": sorted(mains),
            "estrelas": sorted(stars_),
            "popularidade_numeros": round(pm, 3),
            "popularidade_estrelas": round(ps, 3),
            "popularidade_total": round(pi, 3),
            "EV_€": round(r.ev_liquido, 4),
            "retorno_por_euro": round(r.retorno_por_euro, 4),
            "vs_aposta_media_%": round(
                100 * (r.ev_liquido / ev.expected_value(
                    jackpot_eur, self.sales_estimate, 1.0, 12,
                    self.tier_prizes, ev_m1lhao=self.ev_m1lhao).ev_liquido - 1), 1
            ),
        }

    # -- transparência ----------------------------------------------------

    def describe(self) -> str:
        lines = [
            f"EuroMillionsModel v{MODEL_VERSION}",
            f"  estimado em          {self.fitted_at}",
            f"  sorteios             {self.n_draws}",
            f"  quebras de prémios   {self.n_breakdowns}",
            f"  último sorteio       {self.last_draw}",
            f"  vendas estimadas     {self.sales_estimate/1e6:.1f}M por sorteio",
            f"  EV do M1lhão         €{self.ev_m1lhao:.4f} por aposta",
        ]
        if self.star_model is not None:
            lines.append(
                f"  modelo de estrelas   R²={self.star_model.r2:.3f}, "
                f"n={self.star_model.n_obs}"
            )
        return "\n".join(lines)

    def provenance(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(i) for i in PROVENANCE])

    def assumptions(self) -> pd.DataFrame:
        """Só as suposições — é onde o próximo erro vai estar."""
        return self.provenance().query("origem == 'assumido'")

    def to_json(self) -> str:
        return json.dumps(
            {
                "versao": MODEL_VERSION,
                "estimado_em": self.fitted_at,
                "sorteios": self.n_draws,
                "ultimo_sorteio": self.last_draw,
                "ev_m1lhao": self.ev_m1lhao,
                "vendas": self.sales_estimate,
                "elasticidades": self.elasticities,
                "premios": self.tier_prizes,
                "estrelas": (
                    {str(i + 1): self.star_model.star_factor(i + 1)
                     for i in range(12)} if self.star_model else {}
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
