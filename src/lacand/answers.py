"""Resolução de respostas para perguntas do formulário.

Ordem de consulta, da mais barata e previsível para a mais cara:

1. banco de respostas do profile.yaml (regras que você escreveu)
2. respostas já usadas em candidaturas anteriores (SQLite)
3. heurísticas sobre campos estruturados do perfil (anos de skill, telefone…)
4. Claude

Só o passo 4 custa dinheiro e pode variar entre execuções; por isso vem por último.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from .config import Profile
from .llm import Assistant, JobContext, LLMError
from .store import Store

log = logging.getLogger(__name__)

YES = ("sim", "yes")
NO = ("não", "nao", "no")


@dataclass
class Resolution:
    answer: str
    source: str  # profile | memory | heuristic | llm | default
    confident: bool = True


def _pick_option(value: str, options: list[str]) -> str | None:
    """Casa um valor livre com a opção mais próxima do formulário."""
    lowered = value.strip().lower()
    for option in options:
        if option.strip().lower() == lowered:
            return option
    for option in options:
        if lowered and lowered in option.strip().lower():
            return option
    # Respostas sim/não precisam casar com "Yes"/"No"/"Sim"/"Não" em qualquer idioma.
    if lowered in YES:
        for option in options:
            if option.strip().lower() in YES:
                return option
    if lowered in NO:
        for option in options:
            if option.strip().lower() in NO:
                return option
    return None


def _heuristic(question: str, profile: Profile) -> Resolution | None:
    q = question.lower()
    personal = profile.personal
    auth = profile.work_authorization
    comp = profile.compensation

    if re.search(r"\b(e-?mail|email)\b", q):
        return Resolution(personal.email, "heuristic")
    if re.search(r"(telefone|phone|celular|mobile)", q):
        return Resolution(personal.phone, "heuristic")
    if re.search(r"(nome completo|full name|first and last)", q):
        return Resolution(personal.full_name, "heuristic")
    if re.search(r"(cidade|city|localiza|location)", q):
        return Resolution(f"{personal.city}, {personal.state}".strip(", "), "heuristic")
    if "linkedin" in q and re.search(r"(url|perfil|profile|link)", q):
        return Resolution(personal.linkedin_url, "heuristic")
    if re.search(r"(github|portf[óo]lio|portfolio|site pessoal|website)", q):
        return Resolution(personal.portfolio_url, "heuristic")

    # "Quantos anos de experiência você tem com <skill>?"
    if re.search(r"(quantos anos|how many years|years of experience|anos de experi)", q):
        for skill, years in profile.experience.skills.items():
            if re.search(rf"\b{re.escape(skill)}\b", q):
                return Resolution(str(years), "heuristic")
        return Resolution(str(profile.experience.years_total), "heuristic", confident=False)

    if re.search(r"(sponsor|patroc[íi]nio|work visa|visto de trabalho)", q):
        return Resolution("Sim" if auth.requires_sponsorship else "Não", "heuristic")
    if re.search(r"(autoriza|authoriz|legally.*(work|entitled)|permiss[ãa]o para trabalhar)", q):
        return Resolution("Não" if auth.requires_sponsorship else "Sim", "heuristic")
    if re.search(r"(mudar de cidade|relocat|mudan[çc]a)", q):
        return Resolution("Sim" if auth.willing_to_relocate else "Não", "heuristic")
    if re.search(r"(aviso pr[ée]vio|notice period|dispon[íi]vel para in[íi]cio|start date)", q):
        return Resolution(f"{auth.notice_period_days} dias", "heuristic")

    if re.search(r"(pretens[ãa]o|expectativa salarial|salary expectation|desired salary|remunera)", q):
        if not comp.disclose_expectation:
            return Resolution(
                "Prefiro alinhar a faixa durante o processo.", "heuristic", confident=False
            )
        if comp.expected_monthly:
            return Resolution(str(comp.expected_monthly), "heuristic")
        if comp.expected_annual:
            return Resolution(str(comp.expected_annual), "heuristic")

    for language, level in profile.experience.languages.items():
        if re.search(rf"\b{re.escape(language)}\b", q):
            return Resolution(level, "heuristic")

    return None


def resolve(
    question: str,
    *,
    profile: Profile,
    store: Store,
    job: JobContext,
    assistant: Assistant | None = None,
    options: list[str] | None = None,
    multiline: bool = False,
) -> Resolution:
    """Melhor resposta disponível para a pergunta, com a origem que a produziu."""
    question = " ".join(question.split())

    for candidate, source in (
        (profile.lookup_answer(question), "profile"),
        (store.recall_answer(question), "memory"),
    ):
        if candidate:
            if options:
                matched = _pick_option(candidate, options)
                if matched:
                    return Resolution(matched, source)
                # A resposta guardada não existe neste formulário: siga adiante.
                log.debug("resposta de %s não casa com as opções de %r", source, question)
            else:
                return Resolution(candidate, source)

    guess = _heuristic(question, profile)
    if guess and guess.answer:
        if options:
            matched = _pick_option(guess.answer, options)
            if matched:
                return Resolution(matched, "heuristic", guess.confident)
        else:
            return guess

    if assistant is not None:
        try:
            answer = assistant.answer(
                question, job, options=options, multiline=multiline
            )
            store.remember_answer(question, answer, "llm")
            return Resolution(answer, "llm")
        except LLMError as exc:
            log.warning("Claude não respondeu %r: %s", question, exc)
        except Exception as exc:  # falha de rede, rate limit, etc.
            log.warning("erro ao consultar Claude para %r: %s", question, exc)

    # Último recurso: primeira opção do select, ou vazio para texto livre.
    if options:
        return Resolution(options[0], "default", confident=False)
    return Resolution("", "default", confident=False)
