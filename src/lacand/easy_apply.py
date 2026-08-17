"""Preenchimento do formulário de Candidatura Simplificada (Easy Apply)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from playwright.sync_api import Locator, Page, TimeoutError as PlaywrightTimeout

from .answers import Resolution, resolve
from .browser import pause
from .config import Profile
from .llm import Assistant, JobContext
from .store import Store

log = logging.getLogger(__name__)

MAX_STEPS = 12  # trava contra loop infinito se um passo não avançar

MODAL = "div.jobs-easy-apply-modal, div[data-test-modal][role='dialog']"

EASY_APPLY_BUTTON = re.compile(r"(candidatura simplificada|easy apply|candidatar)", re.I)
NEXT_BUTTON = re.compile(r"(avan[çc]ar|pr[óo]xima|continuar|next|continue)", re.I)
REVIEW_BUTTON = re.compile(r"(revisar|review)", re.I)
SUBMIT_BUTTON = re.compile(r"(enviar candidatura|submit application|enviar)", re.I)
DISMISS_BUTTON = re.compile(r"(dispensar|dismiss|fechar|close)", re.I)

COVER_LETTER_HINT = re.compile(
    r"(carta de apresenta|cover letter|por que|why do you|tell us|conte para)", re.I
)


@dataclass
class FieldReport:
    question: str
    answer: str
    source: str
    confident: bool


@dataclass
class ApplyResult:
    status: str  # applied | filled | failed | skipped
    note: str = ""
    steps: int = 0
    fields: list[FieldReport] = field(default_factory=list)
    screenshot: Path | None = None

    @property
    def needs_review(self) -> list[FieldReport]:
        return [f for f in self.fields if not f.confident or not f.answer]


def _label_for(modal: Locator, element: Locator) -> str:
    """Melhor rótulo textual para um campo do formulário."""
    element_id = element.get_attribute("id")
    if element_id:
        # `for` com caracteres especiais quebra o seletor CSS; escape via aspas.
        label = modal.locator(f'label[for="{element_id}"]').first
        if label.count():
            text = " ".join(label.inner_text().split())
            if text:
                return text

    for attribute in ("aria-label", "placeholder", "name"):
        value = element.get_attribute(attribute)
        if value and value.strip():
            return value.strip()

    # Sem rótulo próprio: usa o texto do bloco que contém o campo.
    container = element.locator(
        "xpath=ancestor::*[self::fieldset or "
        "contains(@class,'grouping') or contains(@class,'form-element')][1]"
    ).first
    if container.count():
        text = " ".join(container.inner_text().split())
        if text:
            return text[:300]
    return ""


def _visible(locator: Locator) -> bool:
    try:
        return locator.is_visible()
    except PlaywrightTimeout:
        return False


def _fill_text_inputs(
    modal: Locator, ctx: "_Context", reports: list[FieldReport]
) -> None:
    inputs = modal.locator(
        "input[type='text'], input[type='tel'], input[type='email'], "
        "input[type='number'], textarea"
    )
    for index in range(inputs.count()):
        element = inputs.nth(index)
        if not _visible(element):
            continue
        if (element.input_value() or "").strip():
            continue  # o LinkedIn já preencheu com dados do perfil

        question = _label_for(modal, element)
        if not question:
            continue

        is_textarea = element.evaluate("el => el.tagName.toLowerCase()") == "textarea"
        if is_textarea and ctx.cover_letter_enabled and COVER_LETTER_HINT.search(question):
            answer = ctx.cover_letter()
            resolution = Resolution(answer, "llm")
        else:
            resolution = ctx.resolve(question, multiline=is_textarea)

        if resolution.answer:
            element.fill(resolution.answer)
        reports.append(
            FieldReport(question, resolution.answer, resolution.source, resolution.confident)
        )


def _fill_selects(modal: Locator, ctx: "_Context", reports: list[FieldReport]) -> None:
    selects = modal.locator("select")
    for index in range(selects.count()):
        element = selects.nth(index)
        if not _visible(element):
            continue

        options = [
            " ".join(text.split())
            for text in element.locator("option").all_inner_texts()
        ]
        # A primeira opção costuma ser o placeholder ("Selecione uma opção").
        real = [o for o in options if o and not re.match(r"^(selecione|select)", o, re.I)]
        if not real:
            continue

        current = element.input_value()
        if current and current not in ("", "Select an option", "Selecione uma opção"):
            continue

        question = _label_for(modal, element)
        resolution = ctx.resolve(question, options=real)
        if resolution.answer:
            element.select_option(label=resolution.answer)
        reports.append(
            FieldReport(question, resolution.answer, resolution.source, resolution.confident)
        )


def _fill_radios(modal: Locator, ctx: "_Context", reports: list[FieldReport]) -> None:
    fieldsets = modal.locator("fieldset")
    for index in range(fieldsets.count()):
        group = fieldsets.nth(index)
        radios = group.locator("input[type='radio']")
        count = radios.count()
        if count == 0 or not _visible(group):
            continue
        if group.locator("input[type='radio']:checked").count():
            continue

        legend = group.locator("legend").first
        question = (
            " ".join(legend.inner_text().split())
            if legend.count()
            else _label_for(modal, radios.first)
        )

        options: list[str] = []
        for radio_index in range(count):
            radio = radios.nth(radio_index)
            label = _label_for(group, radio) or radio.get_attribute("value") or ""
            options.append(" ".join(label.split()))

        resolution = ctx.resolve(question, options=[o for o in options if o])
        if resolution.answer in options:
            target = radios.nth(options.index(resolution.answer))
            # O input costuma estar coberto por um label estilizado.
            target.check(force=True)
        reports.append(
            FieldReport(question, resolution.answer, resolution.source, resolution.confident)
        )


def _handle_checkboxes(modal: Locator, reports: list[FieldReport]) -> None:
    """Marca apenas checkboxes obrigatórios; desmarca o 'seguir a empresa'."""
    boxes = modal.locator("input[type='checkbox']")
    for index in range(boxes.count()):
        box = boxes.nth(index)
        if not _visible(box):
            continue
        question = _label_for(modal, box)
        follow = re.search(r"(seguir|follow)", question, re.I)
        required = box.get_attribute("required") is not None or "*" in question

        if follow and box.is_checked():
            box.uncheck(force=True)
            reports.append(FieldReport(question, "desmarcado", "default", True))
        elif required and not box.is_checked():
            box.check(force=True)
            reports.append(FieldReport(question, "marcado", "default", True))


def _upload_resume(modal: Locator, resume: Path | None, reports: list[FieldReport]) -> None:
    if resume is None or not resume.exists():
        return
    uploads = modal.locator("input[type='file']")
    for index in range(uploads.count()):
        element = uploads.nth(index)
        label = _label_for(modal, element).lower()
        if "curr" in label or "resume" in label or "cv" in label or not label:
            element.set_input_files(str(resume))
            reports.append(FieldReport("currículo", resume.name, "profile", True))
            return


class _Context:
    """Estado compartilhado entre os passos do formulário de uma vaga."""

    def __init__(
        self,
        *,
        profile: Profile,
        store: Store,
        job: JobContext,
        assistant: Assistant | None,
        cover_letter_enabled: bool,
        cover_letter_words: int,
    ) -> None:
        self.profile = profile
        self.store = store
        self.job = job
        self.assistant = assistant
        self.cover_letter_enabled = cover_letter_enabled and assistant is not None
        self.cover_letter_words = cover_letter_words
        self._cover_letter: str | None = None

    def resolve(
        self, question: str, *, options: list[str] | None = None, multiline: bool = False
    ) -> Resolution:
        return resolve(
            question,
            profile=self.profile,
            store=self.store,
            job=self.job,
            assistant=self.assistant,
            options=options,
            multiline=multiline,
        )

    def cover_letter(self) -> str:
        """Gera a carta uma vez por vaga e reaproveita entre os passos."""
        if self._cover_letter is None and self.assistant is not None:
            self._cover_letter = self.assistant.cover_letter(
                self.job, max_words=self.cover_letter_words
            )
        return self._cover_letter or ""


def _click(page: Page, pattern: re.Pattern[str]) -> bool:
    button = page.get_by_role("button", name=pattern).first
    try:
        if button.count() and button.is_enabled():
            button.click(timeout=8_000)
            return True
    except PlaywrightTimeout:
        return False
    return False


def apply_to_job(
    page: Page,
    *,
    job: JobContext,
    job_url: str,
    profile: Profile,
    store: Store,
    assistant: Assistant | None,
    delay_range: tuple[float, float],
    submit: bool,
    cover_letter_enabled: bool = True,
    cover_letter_words: int = 180,
    screenshot_dir: Path | None = None,
) -> ApplyResult:
    """Preenche o Easy Apply de uma vaga.

    Com `submit=False` (padrão) o formulário é preenchido até a tela final e o
    envio não acontece: o resultado volta como `filled`, com screenshot, para
    você revisar antes de decidir.
    """
    ctx = _Context(
        profile=profile,
        store=store,
        job=job,
        assistant=assistant,
        cover_letter_enabled=cover_letter_enabled,
        cover_letter_words=cover_letter_words,
    )
    reports: list[FieldReport] = []

    page.goto(job_url, wait_until="domcontentloaded", timeout=60_000)
    pause(delay_range)

    if not _click(page, EASY_APPLY_BUTTON):
        return ApplyResult("skipped", "vaga sem Candidatura Simplificada")

    modal = page.locator(MODAL).first
    try:
        modal.wait_for(state="visible", timeout=15_000)
    except PlaywrightTimeout:
        return ApplyResult("failed", "o modal de candidatura não abriu")

    steps = 0
    while steps < MAX_STEPS:
        steps += 1
        page.wait_for_timeout(900)

        _upload_resume(modal, profile.resume_path, reports)
        _fill_text_inputs(modal, ctx, reports)
        _fill_selects(modal, ctx, reports)
        _fill_radios(modal, ctx, reports)
        _handle_checkboxes(modal, reports)
        pause(delay_range)

        # Botão de envio presente = estamos na última tela.
        submit_button = page.get_by_role("button", name=SUBMIT_BUTTON).first
        if submit_button.count() and submit_button.is_enabled():
            shot: Path | None = None
            if screenshot_dir is not None:
                screenshot_dir.mkdir(parents=True, exist_ok=True)
                shot = screenshot_dir / f"{job_url.rstrip('/').split('/')[-1]}.png"
                page.screenshot(path=str(shot), full_page=False)

            if not submit:
                _click(page, DISMISS_BUTTON)
                # O LinkedIn pergunta se deve salvar como rascunho.
                _click(page, re.compile(r"(descartar|discard)", re.I))
                return ApplyResult(
                    "filled",
                    "formulário preenchido; envio não executado (use --submit)",
                    steps,
                    reports,
                    shot,
                )

            submit_button.click(timeout=10_000)
            page.wait_for_timeout(2_500)
            _click(page, DISMISS_BUTTON)
            return ApplyResult("applied", "candidatura enviada", steps, reports, shot)

        if not (_click(page, REVIEW_BUTTON) or _click(page, NEXT_BUTTON)):
            return ApplyResult(
                "failed",
                f"não foi possível avançar do passo {steps} "
                "(campo obrigatório sem resposta ou formulário não suportado)",
                steps,
                reports,
            )

    return ApplyResult("failed", f"excedeu {MAX_STEPS} passos", steps, reports)
