"""
Recolha do histórico completo de sorteios do EuroMillions.

Duas fases:

  fase 1 (rápida, ~23 pedidos)
      Páginas anuais -> data, números, estrelas, jackpot anunciado.

  fase 2 (lenta, ~1900 pedidos)
      Página de cada sorteio -> nº de vencedores e prémio por escalão.
      Esta fase é a que interessa mesmo: os vencedores por escalão são a
      única janela pública para *como é que as pessoas escolhem números*,
      e é daí que sai a única vantagem real do sistema (ver docs/03).

Ambas as fases são incrementais e idempotentes: reexecutar só vai buscar o
que falta.
"""

from __future__ import annotations

import csv
import datetime as dt
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field

import requests

from . import config

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

_MONTHS = {
    m: i
    for i, m in enumerate(
        [
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
        ],
        start=1,
    )
}


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

class Fetcher:
    """Sessão HTTP com retentativas, backoff exponencial e ritmo educado."""

    def __init__(self, delay: float = 0.4, retries: int = 4, timeout: int = 30):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "en"})
        self.delay = delay
        self.retries = retries
        self.timeout = timeout

    def get(self, url: str) -> str | None:
        wait = 2.0
        for attempt in range(self.retries):
            try:
                resp = self.session.get(url, timeout=self.timeout)
                if resp.status_code == 200:
                    time.sleep(self.delay * random.uniform(0.7, 1.4))
                    return resp.text
                if resp.status_code in (403, 429, 500, 502, 503, 504):
                    time.sleep(wait)
                    wait *= 2
                    continue
                return None  # 404 e afins: não existe, não insistir
            except requests.RequestException:
                time.sleep(wait)
                wait *= 2
        return None


# ---------------------------------------------------------------------------
# Fase 1 — páginas anuais
# ---------------------------------------------------------------------------

@dataclass
class Draw:
    date: dt.date
    mains: list[int]
    stars: list[int]
    jackpot_eur: float | None = None
    jackpot_won: bool | None = None

    def as_row(self) -> dict:
        return {
            "date": self.date.isoformat(),
            "weekday": self.date.strftime("%a"),
            "n1": self.mains[0], "n2": self.mains[1], "n3": self.mains[2],
            "n4": self.mains[3], "n5": self.mains[4],
            "s1": self.stars[0], "s2": self.stars[1],
            "jackpot_eur": "" if self.jackpot_eur is None else f"{self.jackpot_eur:.2f}",
            "jackpot_won": "" if self.jackpot_won is None else int(self.jackpot_won),
        }


_ROW_RE = re.compile(r'<tr class="resultRow".*?</tr>', re.S)
_DATE_RE = re.compile(r'/results/(\d{2})-(\d{2})-(\d{4})')
_BALL_RE = re.compile(r'class="resultBall ball[^"]*">(\d+)</li>')
_STAR_RE = re.compile(r'class="resultBall lucky-star[^"]*">(\d+)</li>')
_JACKPOT_RE = re.compile(r'&euro;([\d,]+(?:\.\d+)?)')


def parse_year_page(html: str) -> list[Draw]:
    draws: list[Draw] = []
    for block in _ROW_RE.findall(html):
        m = _DATE_RE.search(block)
        if not m:
            continue
        day, month, year = (int(g) for g in m.groups())
        mains = [int(x) for x in _BALL_RE.findall(block)]
        stars = [int(x) for x in _STAR_RE.findall(block)]
        if len(mains) != config.MAIN_PICK or len(stars) != config.STAR_PICK:
            continue
        jp = _JACKPOT_RE.search(block)
        jackpot = float(jp.group(1).replace(",", "")) if jp else None
        draws.append(
            Draw(
                date=dt.date(year, month, day),
                mains=sorted(mains),
                stars=sorted(stars),
                jackpot_eur=jackpot,
                jackpot_won="Jackpot Won" in block,
            )
        )
    return draws


def fetch_all_draws(
    start_year: int = 2004,
    end_year: int | None = None,
    fetcher: Fetcher | None = None,
    verbose: bool = True,
) -> list[Draw]:
    fetcher = fetcher or Fetcher()
    end_year = end_year or dt.date.today().year
    all_draws: dict[dt.date, Draw] = {}
    for year in range(start_year, end_year + 1):
        html = fetcher.get(config.YEAR_URL.format(year=year))
        if html is None:
            if verbose:
                print(f"  {year}: falhou", file=sys.stderr)
            continue
        year_draws = parse_year_page(html)
        for d in year_draws:
            all_draws[d.date] = d
        if verbose:
            print(f"  {year}: {len(year_draws)} sorteios", file=sys.stderr)
    return [all_draws[k] for k in sorted(all_draws)]


def write_draws_csv(draws: list[Draw], path: str = config.DRAWS_CSV) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = ["date", "weekday", "n1", "n2", "n3", "n4", "n5", "s1", "s2",
              "jackpot_eur", "jackpot_won"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for d in draws:
            writer.writerow(d.as_row())


# ---------------------------------------------------------------------------
# Fase 2 — quebra de prémios por sorteio
# ---------------------------------------------------------------------------

@dataclass
class Breakdown:
    """Vencedores e prémio por escalão, para um sorteio."""

    date: dt.date
    rows: list[dict] = field(default_factory=list)


# Bloco de Portugal: prémios em euros + vencedores portugueses + total europeu.
_PT_BLOCK_RE = re.compile(r'id="PrizePT".*?</table>', re.S)
_TR_RE = re.compile(r"<tr>.*?</tr>", re.S)
_TIER_RE = re.compile(
    r'<span class="prizeName"><span class="ball">\s*(\d+)\s*</span>'
    r'(?:\s*\+\s*<span class="star">\s*(\d+)\s*</span>)?',
    re.S,
)
_EURO_RE = re.compile(r"&euro;([\d,]+\.\d{2})")
_CELL_RE = re.compile(r'<td[^>]*data-title="([^"]+)"[^>]*>(.*?)</td>', re.S)


def _clean(cell: str) -> str:
    cell = re.sub(r"<[^>]+>", " ", cell)
    cell = cell.replace("&euro;", "€").replace("&nbsp;", " ")
    return re.sub(r"\s+", " ", cell).strip()


def _to_int(text: str) -> int | None:
    m = re.search(r"([\d,]+)\s*$", text)
    if not m:
        return None
    try:
        return int(m.group(1).replace(",", ""))
    except ValueError:
        return None


def _to_eur(text: str) -> float | None:
    m = _EURO_RE.search(text.replace("€", "&euro;"))
    if not m:
        return None
    return float(m.group(1).replace(",", ""))


def parse_draw_page(html: str, date: dt.date) -> Breakdown | None:
    block_m = _PT_BLOCK_RE.search(html)
    if not block_m:
        return None
    block = block_m.group(0)
    out = Breakdown(date=date)
    for tr in _TR_RE.findall(block):
        tier_m = _TIER_RE.search(tr)
        if not tier_m:
            continue
        mains = int(tier_m.group(1))
        stars = int(tier_m.group(2)) if tier_m.group(2) else 0
        cells = {k: _clean(v) for k, v in _CELL_RE.findall(tr)}
        prize = _to_eur(cells.get("Prize Per Winner", ""))
        pt_winners = _to_int(cells.get("Portuguese Winners", ""))
        total_winners = _to_int(cells.get("Total Winners", ""))
        rollover = "Rollover" in tr
        out.rows.append(
            {
                "date": date.isoformat(),
                "tier": f"{mains}+{stars}",
                "prize_eur": "" if prize is None else f"{prize:.2f}",
                "winners_total": "" if total_winners is None else total_winners,
                "winners_pt": "" if pt_winners is None else pt_winners,
                "rollover": int(rollover),
            }
        )
    return out if out.rows else None


def fetch_breakdowns(
    dates: list[dt.date],
    path: str = config.BREAKDOWN_CSV,
    fetcher: Fetcher | None = None,
    verbose: bool = True,
) -> int:
    """Descarrega quebras em falta, escrevendo de forma incremental."""
    fetcher = fetcher or Fetcher()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    done: set[str] = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            done = {row["date"] for row in csv.DictReader(fh)}

    fields = ["date", "tier", "prize_eur", "winners_total", "winners_pt", "rollover"]
    todo = [d for d in dates if d.isoformat() not in done]
    if verbose:
        print(f"  {len(done)} sorteios já em cache, {len(todo)} em falta", file=sys.stderr)

    new_file = not os.path.exists(path)
    written = 0
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if new_file:
            writer.writeheader()
        for i, date in enumerate(todo, 1):
            url = config.DRAW_URL.format(date=date.strftime("%d-%m-%Y"))
            html = fetcher.get(url)
            if html is None:
                continue
            bd = parse_draw_page(html, date)
            if bd is None:
                continue
            writer.writerows(bd.rows)
            fh.flush()
            written += 1
            if verbose and i % 50 == 0:
                print(f"  {i}/{len(todo)} ({date})", file=sys.stderr)
    return written
