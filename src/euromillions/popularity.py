"""
Modelo de popularidade de combinações — o núcleo do sistema.

A IDEIA
-------
Ninguém consegue mudar a probabilidade de acertar. Mas o EuroMillions é um
jogo *pari-mutuel*: o prémio de cada escalão é um bolo fixo dividido pelos
vencedores desse escalão. Logo:

    prémio recebido = bolo / (1 + nº de outras pessoas com a mesma aposta)

A probabilidade de acertar é a mesma para todas as combinações. **O valor
de acertar não é.** Se acertar com uma combinação que mais 30 pessoas
jogaram, recebe 1/31 do bolo. Se acertar com uma que ninguém jogou, recebe
tudo.

Isto não é uma teoria: é aritmética das regras do jogo, e é a única
"brecha" do EuroMillions que sobrevive a análise rigorosa.

COMO SE MEDE
-----------
Ao contrário de quase tudo o que se escreve sobre lotarias, isto é
observável. Os escalões 5+2, 5+1 e 5+0 abrangem *todas* as apostas que
acertaram os 5 números, independentemente das estrelas. A sua soma é,
portanto, uma medição direta de quantas pessoas jogaram exatamente aquela
combinação de 5 números:

    popularidade = vencedores_5_números / (vendas_estimadas / C(50,5))

popularidade = 1.0 significa "combinação tão escolhida como o acaso
mandaria". popularidade = 8 significa que oito vezes mais gente a jogou —
e que, se sair, o prémio se divide por oito vezes mais gente.

Com 1970 sorteios temos 1970 medições independentes desta quantidade,
cada uma emparelhada com a combinação que a produziu. Isso chega para
estimar um modelo de como os humanos escolhem números.

O MODELO
--------
Regressão de Poisson com offset (GLM), estimada por IRLS:

    vencedores_5 ~ Poisson(μ)
    log μ = log(vendas/C(50,5)) + β'x

onde x são características da combinação sorteada. O offset garante que o
modelo explica a *popularidade relativa*, não o volume de vendas.
Sobredispersão é tratada com erros-padrão quasi-Poisson.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config

# ---------------------------------------------------------------------------
# Geometria do boletim
# ---------------------------------------------------------------------------

SLIP_COLS = 10
"""O boletim do EuroMillions dispõe 1..50 numa grelha de 5 linhas × 10
colunas. Muita gente marca linhas, colunas e diagonais — daí as
características geométricas abaixo."""


def slip_row(n: int) -> int:
    return (n - 1) // SLIP_COLS


def slip_col(n: int) -> int:
    return (n - 1) % SLIP_COLS


# ---------------------------------------------------------------------------
# Características de uma combinação
# ---------------------------------------------------------------------------

FEATURE_NAMES: tuple[str, ...] = (
    "n_ate_31",        # viés de datas de nascimento — o efeito dominante
    "n_ate_12",        # meses
    "n_consecutivos",  # pares consecutivos (1,2 / 23,24 ...)
    "soma_norm",       # soma normalizada
    "amplitude_norm",  # (max-min) normalizado
    "n_impares",
    "mesma_coluna",    # nº máx. de números na mesma coluna do boletim
    "mesma_linha",     # nº máx. na mesma linha
    "prog_aritmetica", # espaçamento constante (1-6-11-16-21)
    "n_multiplos_7",
    "tem_7",           # o "número da sorte"
    "espaco_regular",  # desvio-padrão dos intervalos (baixo = padrão visual)
)


def features_main(combo: np.ndarray) -> np.ndarray:
    """Vetor de características de um conjunto de 5 números (ordenado)."""
    c = np.sort(np.asarray(combo, dtype=int))
    diffs = np.diff(c)
    cols = np.array([slip_col(int(x)) for x in c])
    rows = np.array([slip_row(int(x)) for x in c])
    return np.array(
        [
            (c <= 31).sum(),
            (c <= 12).sum(),
            (diffs == 1).sum(),
            c.sum() / 255.0,                       # 255 = soma média
            (c[-1] - c[0]) / 49.0,
            (c % 2 == 1).sum(),
            np.bincount(cols, minlength=SLIP_COLS).max(),
            np.bincount(rows, minlength=5).max(),
            float(len(set(diffs.tolist())) == 1),
            (c % 7 == 0).sum(),
            float(7 in c.tolist()),
            diffs.std(),
        ],
        dtype=float,
    )


def design_matrix(combos: np.ndarray) -> np.ndarray:
    """(n, 1 + p) com coluna de intercepto."""
    X = np.array([features_main(row) for row in combos])
    return np.column_stack([np.ones(len(X)), X])


# ---------------------------------------------------------------------------
# GLM de Poisson por IRLS
# ---------------------------------------------------------------------------

@dataclass
class PoissonGLM:
    """
    Regressão de Poisson com offset, estimada por mínimos quadrados
    iterativamente reponderados (IRLS). Implementada de raiz para não
    depender de statsmodels.
    """

    beta: np.ndarray = field(default_factory=lambda: np.array([]))
    se: np.ndarray = field(default_factory=lambda: np.array([]))
    dispersion: float = 1.0
    names: tuple[str, ...] = ()
    converged: bool = False
    n_obs: int = 0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        offset: np.ndarray,
        names: tuple[str, ...] = (),
        max_iter: int = 100,
        tol: float = 1e-10,
    ) -> "PoissonGLM":
        n, p = X.shape
        beta = np.zeros(p)
        beta[0] = np.log(max(y.mean(), 0.5)) - offset.mean()
        for _ in range(max_iter):
            eta = X @ beta + offset
            eta = np.clip(eta, -30, 30)
            mu = np.exp(eta)
            W = mu
            z = (y - mu) / np.maximum(mu, 1e-9)
            XtWX = X.T @ (X * W[:, None])
            XtWz = X.T @ (W * z)
            try:
                delta = np.linalg.solve(XtWX + 1e-8 * np.eye(p), XtWz)
            except np.linalg.LinAlgError:
                break
            beta = beta + delta
            if np.max(np.abs(delta)) < tol:
                self.converged = True
                break

        eta = np.clip(X @ beta + offset, -30, 30)
        mu = np.exp(eta)
        # Sobredispersão (quasi-Poisson): o comportamento humano é muito
        # mais irregular do que Poisson, ignorá-la inflaciona a confiança.
        pearson = ((y - mu) ** 2 / np.maximum(mu, 1e-9)).sum()
        self.dispersion = float(pearson / max(n - p, 1))
        XtWX = X.T @ (X * mu[:, None])
        cov = np.linalg.pinv(XtWX) * self.dispersion
        self.beta = beta
        self.se = np.sqrt(np.clip(np.diag(cov), 0, None))
        self.names = names or tuple(f"x{i}" for i in range(p))
        self.n_obs = n
        return self

    def summary(self) -> pd.DataFrame:
        from scipy import stats as _st

        z = np.divide(self.beta, self.se, out=np.zeros_like(self.beta), where=self.se > 0)
        return pd.DataFrame(
            {
                "termo": self.names,
                "beta": self.beta,
                "erro_padrao": self.se,
                "z": z,
                "p": 2 * _st.norm.sf(np.abs(z)),
                "efeito_multiplicativo": np.exp(self.beta),
            }
        )

    def predict_log_popularity(self, X: np.ndarray) -> np.ndarray:
        """log da popularidade relativa (offset excluído)."""
        return X @ self.beta

    def predict_popularity(self, combos: np.ndarray) -> np.ndarray:
        """Popularidade relativa prevista: 1.0 = combinação banal."""
        X = design_matrix(np.atleast_2d(combos))
        raw = np.exp(self.predict_log_popularity(X))
        return raw / self._baseline

    _baseline: float = 1.0

    def calibrate(self, reference_combos: np.ndarray) -> "PoissonGLM":
        """
        Normaliza para que a popularidade média sobre o espaço de combinações
        seja 1.0. Sem isto, o intercepto e a escala das características
        misturam-se e os números não são interpretáveis.
        """
        X = design_matrix(reference_combos)
        self._baseline = float(np.exp(self.predict_log_popularity(X)).mean())
        return self


# ---------------------------------------------------------------------------
# Ajuste sobre os dados reais
# ---------------------------------------------------------------------------

def fit_popularity_model(master: pd.DataFrame, seed: int = 3) -> tuple[PoissonGLM, pd.DataFrame]:
    """
    Estima o modelo a partir da tabela mestra (ver dataset.build_master).

    Devolve (modelo, dados_usados).
    """
    d = master.dropna(subset=["winners_main5", "sales_est"]).copy()
    d = d[(d["sales_est"] > 0) & np.isfinite(d["sales_est"])]

    combos = d[["n1", "n2", "n3", "n4", "n5"]].to_numpy(dtype=int)
    y = d["winners_main5"].to_numpy(dtype=float)
    offset = np.log(d["sales_est"].to_numpy(dtype=float) / config.MAIN_COMBINATIONS)

    X = design_matrix(combos)
    # Padronizar as covariáveis (exceto o intercepto) melhora o
    # condicionamento e torna os betas comparáveis entre si.
    mu = X[:, 1:].mean(axis=0)
    sd = X[:, 1:].std(axis=0)
    sd[sd == 0] = 1.0
    Xs = np.column_stack([np.ones(len(X)), (X[:, 1:] - mu) / sd])

    model = PoissonGLM().fit(Xs, y, offset, names=("intercepto",) + FEATURE_NAMES)
    model._center, model._scale = mu, sd  # type: ignore[attr-defined]

    # Reescrever predict para aplicar a mesma padronização.
    def _predict_log(Xraw: np.ndarray, _m=model) -> np.ndarray:
        Xz = np.column_stack(
            [np.ones(len(Xraw)), (Xraw[:, 1:] - _m._center) / _m._scale]  # type: ignore[attr-defined]
        )
        return Xz @ _m.beta

    model.predict_log_popularity = _predict_log  # type: ignore[assignment]

    rng = np.random.default_rng(seed)
    reference = np.array(
        [rng.choice(np.arange(1, config.MAIN_POOL + 1), config.MAIN_PICK, replace=False)
         for _ in range(20_000)]
    )
    model.calibrate(reference)
    return model, d


def observed_popularity_by_feature(master: pd.DataFrame) -> pd.DataFrame:
    """
    Prova visual e não-paramétrica do efeito: popularidade média observada
    em função do nº de números ≤ 31 na combinação sorteada.

    Se o viés de datas de nascimento existe, a popularidade tem de subir
    monotonamente com esta contagem — sem qualquer modelo pelo meio.
    """
    d = master.dropna(subset=["popularity_main5"]).copy()
    combos = d[["n1", "n2", "n3", "n4", "n5"]].to_numpy(dtype=int)
    d["n_ate_31"] = (combos <= 31).sum(axis=1)
    grp = d.groupby("n_ate_31")["popularity_main5"].agg(
        sorteios="size", popularidade_media="mean", mediana="median"
    )
    grp["indice"] = grp["popularidade_media"] / grp["popularidade_media"].min()
    return grp.reset_index()
