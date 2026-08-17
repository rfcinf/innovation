"""Pontuação e filtragem de vagas contra o perfil."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .config import Filters, Profile, Scoring

SENIORITY_ORDER = ["junior", "mid", "senior", "staff", "lead"]

# Termos de senioridade como aparecem em títulos de vaga (pt/en).
SENIORITY_HINTS: dict[str, list[str]] = {
    "junior": ["junior", "júnior", "jr", "entry level", "estágio", "trainee"],
    "mid": ["pleno", "mid-level", "mid level", "intermediate"],
    "senior": ["senior", "sênior", "sr", "specialist", "especialista"],
    "staff": ["staff", "principal"],
    "lead": ["lead", "líder", "head", "manager", "gerente", "coordenador"],
}


@dataclass
class Verdict:
    """Resultado da avaliação de uma vaga."""

    score: float
    reasons: list[str]
    rejected: bool = False

    @property
    def status(self) -> str:
        return "skipped" if self.rejected else "queued"


def _contains_any(text: str, terms: list[str]) -> str | None:
    lowered = text.lower()
    for term in terms:
        if term.lower() in lowered:
            return term
    return None


def _detect_seniority(title: str) -> str | None:
    lowered = title.lower()
    # Checa do mais sênior para o menos, para "senior staff engineer" cair em staff.
    for level in reversed(SENIORITY_ORDER):
        for hint in SENIORITY_HINTS[level]:
            if re.search(rf"\b{re.escape(hint)}\b", lowered):
                return level
    return None


def _recency_score(posted: str) -> float:
    """Converte o texto de data do LinkedIn ('2 dias atrás') em 0..1."""
    if not posted:
        return 0.5
    lowered = posted.lower()
    if any(word in lowered for word in ("hora", "hour", "minute", "minuto", "agora")):
        return 1.0
    match = re.search(r"(\d+)\s*(dia|day|semana|week|m[êe]s|month)", lowered)
    if not match:
        return 0.5
    amount, unit = int(match.group(1)), match.group(2)
    days = amount
    if unit.startswith(("semana", "week")):
        days = amount * 7
    elif unit.startswith(("mes", "mês", "month")):
        days = amount * 30
    if days <= 3:
        return 1.0
    if days <= 7:
        return 0.8
    if days <= 14:
        return 0.5
    return 0.2


def evaluate(
    *,
    title: str,
    company: str,
    description: str,
    posted: str,
    profile: Profile,
    scoring: Scoring,
    filters: Filters,
) -> Verdict:
    """Nota de 0 a 1 para a vaga, com as razões que levaram até ela.

    Rejeições (empresa/título vetados) retornam score 0 e `rejected=True`,
    sem passar pelos pesos — não faz sentido pontuar o que já está fora.
    """
    reasons: list[str] = []

    blocked = _contains_any(company, filters.exclude_companies)
    if blocked:
        return Verdict(0.0, [f"empresa vetada: {blocked}"], rejected=True)

    blocked = _contains_any(title, filters.exclude_title_keywords)
    if blocked:
        return Verdict(0.0, [f"título contém termo vetado: {blocked}"], rejected=True)

    if filters.require_title_keywords:
        hit = _contains_any(title, filters.require_title_keywords)
        if not hit:
            return Verdict(
                0.0, ["título não contém nenhum termo obrigatório"], rejected=True
            )
        reasons.append(f"título casa com '{hit}'")
        title_match = 1.0
    else:
        title_match = 0.6

    # Sobreposição entre skills do perfil (mais os termos preferidos) e a descrição.
    haystack = f"{title}\n{description}".lower()
    wanted = set(profile.experience.skills) | {
        k.lower() for k in filters.preferred_keywords
    }
    found = {skill for skill in wanted if skill and skill in haystack}
    skill_overlap = len(found) / len(wanted) if wanted else 0.0
    if found:
        reasons.append("skills: " + ", ".join(sorted(found)[:8]))
    else:
        reasons.append("nenhuma skill do perfil encontrada na descrição")

    # Senioridade: 1.0 no nível exato, 0.6 a um nível de distância, 0.2 além disso.
    detected = _detect_seniority(title)
    if detected is None:
        seniority_match = 0.6
        reasons.append("senioridade não identificada no título")
    else:
        distance = abs(
            SENIORITY_ORDER.index(detected)
            - SENIORITY_ORDER.index(profile.experience.seniority)
        )
        seniority_match = {0: 1.0, 1: 0.6}.get(distance, 0.2)
        reasons.append(f"senioridade da vaga: {detected}")

    recency = _recency_score(posted)

    weights = scoring.weights
    score = (
        title_match * weights.title_match
        + skill_overlap * weights.skill_overlap
        + seniority_match * weights.seniority_match
        + recency * weights.recency
    )
    total_weight = (
        weights.title_match
        + weights.skill_overlap
        + weights.seniority_match
        + weights.recency
    )
    if total_weight > 0:
        score /= total_weight

    score = round(min(1.0, max(0.0, score)), 3)
    if score < scoring.min_score:
        reasons.append(f"nota {score} abaixo do mínimo {scoring.min_score}")
        return Verdict(score, reasons, rejected=True)

    return Verdict(score, reasons, rejected=False)
