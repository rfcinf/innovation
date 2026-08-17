"""Busca de vagas no LinkedIn e extração dos cards de resultado."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from urllib.parse import urlencode

from playwright.sync_api import Page, TimeoutError as PlaywrightTimeout

from .browser import pause
from .config import Query

log = logging.getLogger(__name__)

SEARCH_URL = "https://www.linkedin.com/jobs/search/"

# Códigos do filtro f_E (nível de experiência) do LinkedIn.
EXPERIENCE_CODES = {"junior": "2", "mid": "3", "senior": "4", "staff": "4", "lead": "5"}
DATE_CODES = {"day": "r86400", "week": "r604800", "month": "r2592000"}

# O LinkedIn troca de markup com frequência; cada seletor é tentado em ordem.
CARD_SELECTORS = [
    "div.job-card-container[data-job-id]",
    "li.scaffold-layout__list-item[data-occludable-job-id]",
    "li.jobs-search-results__list-item",
]
TITLE_SELECTORS = [
    "a.job-card-container__link",
    ".job-card-list__title",
    ".job-card-container__link span[aria-hidden='true']",
    "a.job-card-list__title--link",
]
COMPANY_SELECTORS = [
    ".artdeco-entity-lockup__subtitle",
    ".job-card-container__primary-description",
    ".job-card-container__company-name",
]
LOCATION_SELECTORS = [
    ".job-card-container__metadata-wrapper li",
    ".artdeco-entity-lockup__caption",
    ".job-card-container__metadata-item",
]
DESCRIPTION_SELECTORS = [
    "div.jobs-description__content",
    "div.jobs-box__html-content",
    "#job-details",
]


@dataclass
class JobCard:
    job_id: str
    title: str
    company: str
    location: str
    posted: str
    url: str
    description: str = ""


def build_url(query: Query, start: int = 0) -> str:
    params: dict[str, str] = {"keywords": query.keywords, "start": str(start)}
    if query.location:
        params["location"] = query.location
    if query.easy_apply:
        params["f_AL"] = "true"
    if query.remote:
        params["f_WT"] = "2"
    if query.date_posted != "any":
        params["f_TPR"] = DATE_CODES[query.date_posted]
    if query.experience:
        codes = sorted({EXPERIENCE_CODES[level] for level in query.experience})
        params["f_E"] = ",".join(codes)
    return f"{SEARCH_URL}?{urlencode(params)}"


def _first_text(scope, selectors: list[str]) -> str:
    """Texto do primeiro seletor que existir — o markup varia entre layouts."""
    for selector in selectors:
        locator = scope.locator(selector).first
        try:
            if locator.count() and locator.is_visible():
                return " ".join(locator.inner_text().split())
        except PlaywrightTimeout:
            continue
    return ""


def _job_id(card) -> str:
    for attribute in ("data-job-id", "data-occludable-job-id"):
        value = card.get_attribute(attribute)
        if value:
            return value
    link = card.locator("a[href*='/jobs/view/']").first
    if link.count():
        match = re.search(r"/jobs/view/(\d+)", link.get_attribute("href") or "")
        if match:
            return match.group(1)
    return ""


def collect(
    page: Page, query: Query, *, limit: int, delay_range: tuple[float, float]
) -> list[JobCard]:
    """Percorre as páginas de resultado e devolve os cards encontrados."""
    results: list[JobCard] = []
    seen: set[str] = set()
    start = 0

    while len(results) < limit:
        url = build_url(query, start=start)
        log.info("buscando: %s", url)
        page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        pause(delay_range)

        cards = None
        for selector in CARD_SELECTORS:
            locator = page.locator(selector)
            if locator.count():
                cards = locator
                break

        if cards is None or cards.count() == 0:
            log.info("nenhum card na página start=%s; encerrando esta query", start)
            break

        # O LinkedIn renderiza os cards sob demanda: sem rolar, metade fica vazia.
        page.mouse.wheel(0, 4000)
        page.wait_for_timeout(1200)

        page_count = cards.count()
        for index in range(page_count):
            if len(results) >= limit:
                break
            card = cards.nth(index)
            job_id = _job_id(card)
            if not job_id or job_id in seen:
                continue
            seen.add(job_id)

            title = _first_text(card, TITLE_SELECTORS)
            if not title:
                continue
            metadata = card.inner_text()
            posted_match = re.search(
                r"(h[áa]\s+)?(\d+\s*(minutos?|horas?|dias?|semanas?|m[êe]s(es)?|"
                r"minutes?|hours?|days?|weeks?|months?))",
                metadata,
                re.IGNORECASE,
            )
            results.append(
                JobCard(
                    job_id=job_id,
                    title=title,
                    company=_first_text(card, COMPANY_SELECTORS),
                    location=_first_text(card, LOCATION_SELECTORS),
                    posted=posted_match.group(0) if posted_match else "",
                    url=f"https://www.linkedin.com/jobs/view/{job_id}/",
                )
            )

        if page_count < 25:  # última página de resultados
            break
        start += 25

    return results


def fetch_description(page: Page, job: JobCard, delay_range: tuple[float, float]) -> str:
    """Abre a vaga e devolve o texto da descrição (vazio se não carregar)."""
    page.goto(job.url, wait_until="domcontentloaded", timeout=60_000)
    pause(delay_range)

    # "Ver mais" mantém a descrição truncada no DOM até ser clicado.
    for label in ("Ver mais", "See more", "Mostrar mais"):
        button = page.get_by_role("button", name=re.compile(label, re.I)).first
        try:
            if button.count() and button.is_visible():
                button.click(timeout=3_000)
                break
        except PlaywrightTimeout:
            continue

    return _first_text(page, DESCRIPTION_SELECTORS)
