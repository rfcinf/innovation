"""
O equipamento físico do sorteio, e o que se pode (e não pode) inferir dele.

FACTOS APURADOS
---------------
* Local: Paris. Os sorteios são realizados pela Française des Jeux (FDJ)
  em nome das lotarias participantes, às terças e sextas, ~21:05 CET.

* Máquinas: fabricadas pela **Ryo-Catteau** (França), fabricante histórico
  de máquinas de sorteio por gravidade.
    - Números principais (1-50): modelo **Stresa**
    - Estrelas da sorte      : modelo **Pâquerette**

* Tipo: *gravity pick*. Uma câmara com pás rotativas em sentidos opostos
  mistura as bolas; uma porta deslizante liberta-as uma a uma. É o tipo
  considerado mais robusto contra viés sistemático, precisamente porque a
  mistura é mecânica e não depende de fluxo de ar (que é sensível a
  humidade, temperatura e desgaste dos rolamentos).

* Procedimento de segurança (FDJ): ensaios técnicos a partir das 18:15
  para testar máquinas e equipamento; as bolas oficiais estão guardadas em
  cofre; o acesso à sala segura e ao cofre exige a presença simultânea de
  um Comissário de Justiça e de um responsável de sorteio da FDJ; o
  Comissário de Justiça verifica o selo e é o único autorizado a removê-lo.

* Existem múltiplos conjuntos de bolas e mais do que uma máquina, com
  seleção antes de cada sorteio, e as bolas são pesadas e substituídas
  periodicamente.

PORQUE É QUE ISTO IMPORTA — E PORQUE É QUE NÃO CHEGA
-----------------------------------------------------
A única exploração de lotaria fisicamente comprovada da história atacou o
equipamento, não a matemática (Jagger em Monte Carlo, 1873: mapeou o
desgaste de uma roleta durante semanas antes de apostar). Se houvesse uma
brecha do lado dos números do EuroMillions, ela viveria aqui: uma bola com
0,1 g a mais, uma câmara desnivelada, um conjunto que envelheceu mal.

Há, no entanto, um obstáculo de informação que é fatal para a estratégia,
e convém ser explícito quanto a ele: **os dados públicos não identificam
qual a máquina nem qual o conjunto de bolas usado em cada sorteio.** Sem
essa etiqueta, os sorteios de todos os conjuntos ficam misturados no mesmo
agregado. Um viés real de um conjunto específico é diluído pelos restantes
e torna-se estatisticamente invisível — mesmo existindo.

Este módulo faz o que é possível sem essa etiqueta: procura o viés de
forma *latente*, através de segmentações por tempo e por dia da semana
(candidatos naturais a proxy do conjunto usado), e quantifica de forma
honesta o quanto essa diluição nos cega.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .randomness import (
    ball_counts,
    benjamini_hochberg,
    chi_square_uniform,
    star_counts,
)

MACHINES = {
    "principais": {
        "fabricante": "Ryo-Catteau",
        "modelo": "Stresa",
        "tipo": "gravity pick (pás rotativas contra-rotativas)",
        "bolas": 50,
    },
    "estrelas": {
        "fabricante": "Ryo-Catteau",
        "modelo": "Pâquerette",
        "tipo": "gravity pick",
        "bolas": 12,
    },
}

DRAW_LOCATION = "Paris, França — Française des Jeux (FDJ)"
SUPERVISION = "Comissário de Justiça + responsável de sorteio da FDJ; bolas em cofre selado"


def segment_bias_scan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Procura viés em segmentações que possam funcionar como proxy do
    conjunto de bolas: dia da semana, ano, era.

    Se um conjunto de bolas fosse usado preferencialmente às terças (por
    exemplo, por rotação fixa), um viés desse conjunto apareceria como
    diferença entre terças e sextas.
    """
    rows = []

    for dow, sub in df.groupby("dow"):
        if len(sub) < 100:
            continue
        res = chi_square_uniform(ball_counts(sub), f"dia={dow}")
        rows.append({"segmento": f"dia da semana = {dow}", "n": len(sub),
                     "chi2": res.statistic, "p": res.pvalue})

    for year, sub in df.groupby("year"):
        if len(sub) < 80:
            continue
        res = chi_square_uniform(ball_counts(sub), f"ano={year}")
        rows.append({"segmento": f"ano {year}", "n": len(sub),
                     "chi2": res.statistic, "p": res.pvalue})

    for era in config.ERAS:
        sub = df[df["era"] == era.name]
        if len(sub) < 100:
            continue
        res = chi_square_uniform(star_counts(sub, era.star_pool), f"estrelas {era.name}")
        rows.append({"segmento": f"estrelas — era {era.name}", "n": len(sub),
                     "chi2": res.statistic, "p": res.pvalue})

    out = pd.DataFrame(rows)
    if len(out):
        out["p_ajustado"] = benjamini_hochberg(out["p"].to_numpy())
        out["significativo_fdr5"] = out["p_ajustado"] < 0.05
    return out.sort_values("p").reset_index(drop=True)


def weekday_consistency(df: pd.DataFrame, n_sim: int = 20_000, seed: int = 19) -> pd.DataFrame:
    """
    Compara a frequência de cada número às terças vs às sextas.

    Se os dois dias usassem conjuntos de bolas diferentes e um deles fosse
    enviesado, os desvios divergiriam. Se partilhassem um conjunto
    enviesado, os desvios correlacionar-se-iam positivamente.

    NOTA — porque é que aqui há um p-valor simulado e não um rótulo:
    a versão anterior desta função classificava o resultado comparando |r|
    com um limiar inventado (0.15), e classificou r = 0.27 como
    "divergência a investigar". Isso é um gerador de falsos achados: com
    apenas 50 pontos, o desvio-padrão nulo de r é ≈ 0.14, portanto r = 0.27
    é ruído perfeitamente banal. Um limiar arbitrário não sabe disso; a
    distribuição nula sabe.
    """
    sub = df[df["era"] != "E1"]  # E1 só tinha sextas
    tue = sub[sub["dow"] == "Tue"]
    fri = sub[sub["dow"] == "Fri"]
    if len(tue) < 50 or len(fri) < 50:
        return pd.DataFrame()

    pool, picks = config.MAIN_POOL, config.MAIN_PICK
    p0 = picks / pool
    n_t, n_f = len(tue), len(fri)
    dev_t = (ball_counts(tue) - n_t * p0) / (n_t * p0)
    dev_f = (ball_counts(fri) - n_f * p0) / (n_f * p0)
    r = float(np.corrcoef(dev_t, dev_f)[0, 1])

    rng = np.random.default_rng(seed)
    sims = np.empty(n_sim)
    for i in range(n_sim):
        a = np.bincount(
            rng.random((n_t, pool)).argsort(axis=1)[:, :picks].ravel(), minlength=pool
        ).astype(float)
        b = np.bincount(
            rng.random((n_f, pool)).argsort(axis=1)[:, :picks].ravel(), minlength=pool
        ).astype(float)
        sims[i] = np.corrcoef(a - n_t * p0, b - n_f * p0)[0, 1]

    p = float((np.abs(sims) >= abs(r)).mean())
    return pd.DataFrame(
        [
            {
                "sorteios_terca": n_t,
                "sorteios_sexta": n_f,
                "correlacao_desvios": round(r, 4),
                "nulo_dp": round(float(sims.std()), 4),
                "p_simulado": round(p, 4),
                "conclusao": (
                    "compatível com acaso" if p >= 0.05 else "desvio a investigar"
                ),
            }
        ]
    )


def dilution_penalty(n_draws: int, n_ball_sets: int) -> dict:
    """
    Quantifica a cegueira imposta pela ausência de identificação do
    conjunto de bolas.

    Se um só conjunto entre `n_ball_sets` tem um viés de tamanho b, o viés
    observado no agregado é diluído para b/n_ball_sets, e o número de
    sorteios úteis é apenas n_draws/n_ball_sets. O efeito nas necessidades
    de amostra é quadrático: detetar o mesmo viés exige ~n_ball_sets²
    vezes mais sorteios.
    """
    from .randomness import minimum_detectable_bias

    mdb_ideal = minimum_detectable_bias(n_draws)
    mdb_real = minimum_detectable_bias(max(int(n_draws / n_ball_sets), 30))
    return {
        "sorteios_totais": n_draws,
        "conjuntos_assumidos": n_ball_sets,
        "sorteios_por_conjunto": int(n_draws / n_ball_sets),
        "vies_min_detetavel_se_identificado_%": round(mdb_real, 1),
        "vies_min_detetavel_no_agregado_%": round(mdb_ideal * n_ball_sets, 1),
        "penalizacao": (
            "sem etiqueta de conjunto, um viés real teria de ser "
            f"~{n_ball_sets}x maior para ser visível"
        ),
    }


def describe() -> str:
    lines = [
        f"Local do sorteio : {DRAW_LOCATION}",
        f"Supervisão       : {SUPERVISION}",
        "",
        "Máquinas:",
    ]
    for role, spec in MACHINES.items():
        lines.append(
            f"  {role:<12} {spec['fabricante']} {spec['modelo']:<12} "
            f"{spec['tipo']} — {spec['bolas']} bolas"
        )
    return "\n".join(lines)
