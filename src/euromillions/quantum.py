"""
Entropia quântica como motor de seleção.

O QUE ISTO FAZ E O QUE NÃO FAZ
------------------------------
Não faz: aumentar a probabilidade de acertar. Nenhuma fonte de entropia,
por mais exótica, altera p = 1/139.838.160. Qualquer sistema que prometa o
contrário está a mentir, e o utilizador merece que se lhe diga isso em vez
de lhe vender misticismo com vocabulário de física.

Faz, e é real e útil: garante que a nossa escolha é **estatisticamente
independente da psicologia humana**.

Isto não é um detalhe estético — é o que faz o sistema funcionar. A
vantagem inteira (ver ev.py e popularity.py) vem de escolher combinações
que outras pessoas não escolhem. Ora, qualquer processo de escolha com um
humano no circuito está correlacionado com o que os outros humanos fazem:
datas, "números da sorte", padrões visuais no boletim, aversão a
sequências que "não parecem aleatórias". Até um gerador pseudo-aleatório
semeado por uma pessoa herda parte desse enviesamento através do momento
e do modo como é usado.

Uma fonte de entropia física — aqui, flutuações do vácuo quântico medidas
por interferometria óptica — não tem qualquer canal causal que a ligue às
preferências da população de apostadores. É a única forma de obter uma
amostra genuinamente descorrelacionada do comportamento coletivo.

Arquitetura do sistema:

    entropia quântica  →  propõe combinações sem viés humano
    modelo popularidade →  filtra as que a multidão prefere
    motor de EV        →  ordena pelo cheque esperado

A entropia é o motor de busca; a estatística é o filtro. Nenhuma das duas
sozinha serve.

FONTES
------
* ANU QRNG (Australian National University) — vácuo quântico, público.
* LfD / Leibniz Universität Hannover — vácuo quântico, público.
* os.urandom — CSPRNG do sistema operativo, semeado por entropia de
  hardware. É o recurso quando a rede falha, e é criptograficamente sólido.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np
import requests

from . import config

ANU_URL = "https://qrng.anu.edu.au/API/jsonI.php"
LFD_URL = "https://lfdr.de/qrng_api/qrng"


@dataclass
class EntropySource:
    """Reservatório de bytes com origem registada e reabastecimento."""

    name: str = "desconhecida"
    _buffer: bytearray = field(default_factory=bytearray)
    _origin_log: list[str] = field(default_factory=list)

    # -- obtenção ---------------------------------------------------------

    @staticmethod
    def _from_anu(n_bytes: int, timeout: int = 20) -> bytes | None:
        try:
            got = bytearray()
            while len(got) < n_bytes:
                chunk = min(1024, n_bytes - len(got))
                r = requests.get(
                    ANU_URL, params={"length": chunk, "type": "uint8"}, timeout=timeout
                )
                if r.status_code != 200:
                    return None
                payload = r.json()
                if not payload.get("success"):
                    return None
                got.extend(bytes(payload["data"]))
                time.sleep(0.3)
            return bytes(got)
        except Exception:
            return None

    @staticmethod
    def _from_lfd(n_bytes: int, timeout: int = 20) -> bytes | None:
        try:
            got = bytearray()
            while len(got) < n_bytes:
                chunk = min(1024, n_bytes - len(got))
                r = requests.get(
                    LFD_URL, params={"length": chunk, "format": "HEX"}, timeout=timeout
                )
                if r.status_code != 200:
                    return None
                got.extend(bytes.fromhex(r.json()["qrn"]))
            return bytes(got)
        except Exception:
            return None

    def refill(self, n_bytes: int = 2048, allow_network: bool = True) -> str:
        """Enche o reservatório, tentando as fontes quânticas por ordem."""
        data = None
        origin = "os.urandom (CSPRNG de hardware)"
        if allow_network:
            data = self._from_anu(n_bytes)
            if data:
                origin = "ANU QRNG (vácuo quântico, Australian National University)"
            else:
                data = self._from_lfd(n_bytes)
                if data:
                    origin = "LfD QRNG (vácuo quântico, Leibniz Universität Hannover)"
        if data is None:
            data = os.urandom(n_bytes)
        self._buffer.extend(data)
        self.name = origin
        self._origin_log.append(f"{len(data)}B <- {origin}")
        return origin

    # -- consumo ----------------------------------------------------------

    def bytes(self, n: int, allow_network: bool = True) -> bytes:
        if len(self._buffer) < n:
            self.refill(max(2048, n * 4), allow_network=allow_network)
        out = bytes(self._buffer[:n])
        del self._buffer[:n]
        return out

    def uniform_int(self, lo: int, hi: int, allow_network: bool = True) -> int:
        """
        Inteiro uniforme em [lo, hi] sem viés de módulo.

        A rejeição é obrigatória: usar `byte % n` introduz um desvio
        determinístico e seria uma ironia amarga estragar a única
        propriedade que estamos aqui a comprar.
        """
        span = hi - lo + 1
        if span <= 0:
            raise ValueError("intervalo inválido")
        limit = (256 // span) * span
        while True:
            b = self.bytes(1, allow_network=allow_network)[0]
            if b < limit:
                return lo + (b % span)

    def sample_without_replacement(
        self, pool: int, k: int, allow_network: bool = True
    ) -> list[int]:
        """k valores distintos de 1..pool, por amostragem de rejeição."""
        chosen: list[int] = []
        seen: set[int] = set()
        while len(chosen) < k:
            v = self.uniform_int(1, pool, allow_network=allow_network)
            if v not in seen:
                seen.add(v)
                chosen.append(v)
        return sorted(chosen)

    def draw_ticket(self, star_pool: int = 12, allow_network: bool = True) -> tuple[list[int], list[int]]:
        mains = self.sample_without_replacement(
            config.MAIN_POOL, config.MAIN_PICK, allow_network=allow_network
        )
        stars = self.sample_without_replacement(
            star_pool, config.STAR_PICK, allow_network=allow_network
        )
        return mains, stars

    def provenance(self) -> list[str]:
        return list(self._origin_log)


# ---------------------------------------------------------------------------
# Verificação da qualidade da entropia
# ---------------------------------------------------------------------------

def entropy_quality(data: bytes) -> dict:
    """
    Testa a fonte antes de confiar nela.

    Uma fonte "quântica" que devolva lixo (API partida, cache, resposta
    truncada) é pior do que o urandom, e falharia em silêncio. Estes testes
    apanham isso: entropia de Shannon, qui-quadrado sobre os 256 valores e
    proporção de bits a 1.
    """
    from scipy import stats

    arr = np.frombuffer(data, dtype=np.uint8)
    counts = np.bincount(arr, minlength=256).astype(float)
    n = len(arr)
    expected = n / 256
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    p_chi = float(stats.chi2.sf(chi2, 255))

    probs = counts[counts > 0] / n
    shannon = float(-(probs * np.log2(probs)).sum())

    bits = np.unpackbits(arr)
    ones = float(bits.mean())
    z_bits = (ones - 0.5) / (0.5 / np.sqrt(len(bits)))

    return {
        "bytes": n,
        "entropia_shannon_bits_por_byte": round(shannon, 4),
        "maximo_teorico": 8.0,
        "chi2_256": round(chi2, 2),
        "p_chi2": round(p_chi, 4),
        "proporcao_bits_1": round(ones, 5),
        "z_bits": round(float(z_bits), 3),
        "veredicto": (
            "aprovada" if p_chi > 0.01 and abs(z_bits) < 3 and shannon > 7.5
            else "REPROVADA — não usar"
        ),
    }


def get_source(allow_network: bool = True, test_bytes: int = 1024) -> tuple[EntropySource, dict]:
    """Cria uma fonte, enche-a e certifica-a antes de a devolver."""
    src = EntropySource()
    src.refill(max(4096, test_bytes * 2), allow_network=allow_network)
    sample = bytes(src._buffer[:test_bytes])
    report = entropy_quality(sample)
    return src, report
