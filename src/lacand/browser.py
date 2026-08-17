"""Sessão de navegador com estado persistido, para não relogar a cada execução."""

from __future__ import annotations

import logging
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from playwright.sync_api import Browser, BrowserContext, Page, sync_playwright

log = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

LOGIN_URL = "https://www.linkedin.com/login"
FEED_URL = "https://www.linkedin.com/feed/"


class Session:
    """Wrapper em torno de um `BrowserContext` autenticado."""

    def __init__(self, context: BrowserContext, state_path: Path) -> None:
        self.context = context
        self.state_path = state_path
        self.page: Page = context.pages[0] if context.pages else context.new_page()

    def save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.context.storage_state(path=str(self.state_path))
        log.info("sessão salva em %s", self.state_path)

    def is_authenticated(self) -> bool:
        """Considera autenticado quando /feed/ não redireciona para o login."""
        self.page.goto(FEED_URL, wait_until="domcontentloaded", timeout=45_000)
        return "/login" not in self.page.url and "/authwall" not in self.page.url


@contextmanager
def launch(
    state_path: Path, *, headless: bool = True, slow_mo: int = 0
) -> Iterator[Session]:
    """Abre um navegador reaproveitando o estado salvo, se existir."""
    with sync_playwright() as playwright:
        browser: Browser = playwright.chromium.launch(
            headless=headless,
            slow_mo=slow_mo,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context(
            user_agent=USER_AGENT,
            viewport={"width": 1440, "height": 900},
            locale="pt-BR",
            storage_state=str(state_path) if state_path.exists() else None,
        )
        session = Session(context, state_path)
        try:
            yield session
        finally:
            context.close()
            browser.close()


def pause(delay_range: tuple[float, float]) -> None:
    """Pausa aleatória entre ações — evita o padrão de tráfego de um bot."""
    low, high = delay_range
    time.sleep(random.uniform(low, high))
