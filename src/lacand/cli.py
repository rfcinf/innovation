"""Interface de linha de comando."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .answers import Resolution  # noqa: F401  (reexport conveniente para scripts)
from .browser import LOGIN_URL, launch, pause
from .config import load_profile, load_search
from .easy_apply import apply_to_job
from .llm import Assistant, JobContext
from .matcher import evaluate
from .search import collect, fetch_description
from .store import Job, Store

app = typer.Typer(
    add_completion=False,
    help="Candidaturas assistidas no LinkedIn: busca, pontua, preenche e (opcionalmente) envia.",
)
console = Console()

CONFIG_DIR = Path("config")
PROFILE_PATH = CONFIG_DIR / "profile.yaml"
SEARCH_PATH = CONFIG_DIR / "search.yaml"
STATE_PATH = Path(".browser-state/linkedin.json")
DB_PATH = Path("data/applications.db")
SCREENSHOT_DIR = Path("screenshots")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    _setup_logging(verbose)


@app.command()
def version() -> None:
    """Mostra a versão."""
    console.print(f"lacand {__version__}")


@app.command()
def init() -> None:
    """Cria config/profile.yaml e config/search.yaml a partir dos exemplos."""
    for example, target in (
        (CONFIG_DIR / "profile.example.yaml", PROFILE_PATH),
        (CONFIG_DIR / "search.example.yaml", SEARCH_PATH),
    ):
        if target.exists():
            console.print(f"[yellow]já existe, mantido:[/] {target}")
            continue
        shutil.copy(example, target)
        console.print(f"[green]criado:[/] {target}")
    console.print("\nEdite os dois arquivos e rode [bold]lacand login[/].")


@app.command()
def login(
    timeout: int = typer.Option(300, help="Segundos para concluir o login manualmente."),
) -> None:
    """Abre o navegador para você logar no LinkedIn e salva a sessão.

    O login é manual de propósito: nenhuma senha é lida, armazenada ou digitada
    por este programa, e o 2FA continua funcionando normalmente.
    """
    with launch(STATE_PATH, headless=False) as session:
        page = session.page
        page.goto(LOGIN_URL, wait_until="domcontentloaded")
        console.print(
            "[bold]Faça login na janela aberta.[/] Aguardando a sessão ser estabelecida…"
        )
        try:
            page.wait_for_url(
                lambda url: "/feed" in url or "/jobs" in url, timeout=timeout * 1000
            )
        except Exception:
            console.print("[red]Login não concluído dentro do tempo limite.[/]")
            raise typer.Exit(1)
        session.save_state()
        console.print(f"[green]Sessão salva em {STATE_PATH}[/]")


@app.command()
def search(
    headless: bool = typer.Option(True, help="Rodar sem interface gráfica."),
    descriptions: bool = typer.Option(
        True, help="Abrir cada vaga para ler a descrição (mais lento, nota melhor)."
    ),
) -> None:
    """Busca vagas, pontua contra o perfil e enfileira as aprovadas."""
    profile = load_profile(PROFILE_PATH)
    config = load_search(SEARCH_PATH)
    delay = config.limits.delay_range

    with Store(DB_PATH) as store, launch(STATE_PATH, headless=headless) as session:
        if not session.is_authenticated():
            console.print("[red]Sessão expirada. Rode `lacand login`.[/]")
            raise typer.Exit(1)

        queued = skipped = 0
        for query in config.queries:
            console.print(f"\n[bold]Query:[/] {query.keywords} — {query.location or 'qualquer local'}")
            cards = collect(
                session.page,
                query,
                limit=config.limits.max_results_per_query,
                delay_range=delay,
            )
            console.print(f"  {len(cards)} vagas encontradas")

            for card in cards:
                if store.has_seen(card.job_id):
                    continue

                description = ""
                if descriptions:
                    description = fetch_description(session.page, card, delay)

                verdict = evaluate(
                    title=card.title,
                    company=card.company,
                    description=description,
                    posted=card.posted,
                    profile=profile,
                    scoring=config.scoring,
                    filters=config.filters,
                )
                store.upsert_job(
                    Job(
                        job_id=card.job_id,
                        title=card.title,
                        company=card.company,
                        url=card.url,
                        location=card.location,
                        posted=card.posted,
                        score=verdict.score,
                        reasons=verdict.reasons,
                        status=verdict.status,
                    )
                )
                if verdict.rejected:
                    skipped += 1
                else:
                    queued += 1
                    console.print(
                        f"  [green]{verdict.score:.2f}[/] {card.title} — {card.company}"
                    )

        console.print(f"\n[bold]{queued} enfileiradas, {skipped} descartadas.[/]")
        console.print("Revise com [bold]lacand list[/] e rode [bold]lacand apply[/].")


@app.command()
def apply(
    submit: bool = typer.Option(
        False,
        "--submit",
        help="ENVIA as candidaturas de verdade. Sem esta flag, apenas preenche e para.",
    ),
    limit: int | None = typer.Option(None, help="Sobrescreve o limite por execução."),
    headless: bool = typer.Option(False, help="Rodar sem interface gráfica."),
    no_llm: bool = typer.Option(False, "--no-llm", help="Não consultar o Claude."),
    yes: bool = typer.Option(False, "--yes", help="Pular a confirmação do --submit."),
) -> None:
    """Preenche o Easy Apply das vagas enfileiradas."""
    profile = load_profile(PROFILE_PATH)
    config = load_search(SEARCH_PATH)
    delay = config.limits.delay_range
    per_run = limit or config.limits.max_applications_per_run

    if submit and not yes:
        console.print(
            "[bold yellow]--submit envia candidaturas reais em seu nome.[/]\n"
            "Automatizar envios contradiz os Termos de Uso do LinkedIn e pode "
            "levar a restrição da conta."
        )
        if not typer.confirm(f"Enviar até {per_run} candidaturas?"):
            raise typer.Exit(0)

    with Store(DB_PATH) as store:
        remaining_today = config.limits.max_applications_per_day - store.applied_today()
        if submit and remaining_today <= 0:
            console.print("[yellow]Limite diário já atingido. Nada a fazer.[/]")
            raise typer.Exit(0)

        budget = min(per_run, remaining_today) if submit else per_run
        jobs = store.queued(limit=budget, min_score=config.scoring.min_score)
        if not jobs:
            console.print("[yellow]Fila vazia. Rode `lacand search` primeiro.[/]")
            raise typer.Exit(0)

        assistant = None if no_llm else Assistant(profile)

        with launch(STATE_PATH, headless=headless) as session:
            if not session.is_authenticated():
                console.print("[red]Sessão expirada. Rode `lacand login`.[/]")
                raise typer.Exit(1)

            for job in jobs:
                console.print(f"\n[bold]{job.title}[/] — {job.company} ({job.score:.2f})")
                context = JobContext(title=job.title, company=job.company)
                try:
                    result = apply_to_job(
                        session.page,
                        job=context,
                        job_url=job.url,
                        profile=profile,
                        store=store,
                        assistant=assistant,
                        delay_range=delay,
                        submit=submit,
                        cover_letter_enabled=config.cover_letter.enabled,
                        cover_letter_words=config.cover_letter.max_words,
                        screenshot_dir=SCREENSHOT_DIR,
                    )
                except Exception as exc:  # uma vaga quebrada não derruba o lote
                    logging.exception("falha ao candidatar-se a %s", job.job_id)
                    store.set_status(job.job_id, "failed", str(exc)[:300])
                    console.print(f"  [red]erro:[/] {exc}")
                    continue

                store.set_status(job.job_id, result.status, result.note)
                color = {"applied": "green", "filled": "cyan"}.get(result.status, "yellow")
                console.print(f"  [{color}]{result.status}[/]: {result.note}")

                for report in result.fields:
                    marker = " " if report.confident and report.answer else "!"
                    console.print(
                        f"    {marker} {report.question[:70]} → "
                        f"{report.answer[:60] or '(vazio)'} [dim]({report.source})[/]"
                    )
                if result.screenshot:
                    console.print(f"    screenshot: {result.screenshot}")

                pause(delay)

        counts = store.counts()
        console.print(f"\n[bold]Resumo:[/] {counts}")
        if not submit:
            console.print(
                "Nada foi enviado. Revise os screenshots e rode com [bold]--submit[/] "
                "para enviar."
            )


@app.command("list")
def list_jobs(
    status: str | None = typer.Option(None, help="queued | filled | applied | skipped | failed"),
    limit: int = typer.Option(30),
) -> None:
    """Lista as vagas registradas."""
    with Store(DB_PATH) as store:
        jobs = store.by_status(status, limit)
        if not jobs:
            console.print("[yellow]Nada encontrado.[/]")
            return

        table = Table(show_lines=False)
        table.add_column("Nota", justify="right")
        table.add_column("Status")
        table.add_column("Cargo", overflow="fold")
        table.add_column("Empresa")
        table.add_column("Motivos", overflow="fold", style="dim")
        for job in jobs:
            table.add_row(
                f"{job.score:.2f}",
                job.status,
                job.title[:50],
                job.company[:28],
                "; ".join(job.reasons)[:60],
            )
        console.print(table)
        console.print(f"\n[bold]Totais:[/] {store.counts()}")


@app.command()
def answers(limit: int = typer.Option(50)) -> None:
    """Mostra as respostas memorizadas — útil para promovê-las ao profile.yaml."""
    with Store(DB_PATH) as store:
        rows = store.all_answers()[:limit]
        if not rows:
            console.print("[yellow]Nenhuma resposta memorizada ainda.[/]")
            return
        table = Table()
        table.add_column("Pergunta", overflow="fold")
        table.add_column("Resposta", overflow="fold")
        table.add_column("Origem")
        for question, answer, source in rows:
            table.add_row(question[:80], answer[:60], source)
        console.print(table)


if __name__ == "__main__":
    app()
