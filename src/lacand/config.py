"""Carregamento e validação dos arquivos de configuração."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

Seniority = Literal["junior", "mid", "senior", "staff", "lead"]


class Personal(BaseModel):
    full_name: str
    email: str
    phone: str = ""
    city: str = ""
    state: str = ""
    country: str = ""
    linkedin_url: str = ""
    portfolio_url: str = ""


class WorkAuthorization(BaseModel):
    authorized_in: list[str] = Field(default_factory=list)
    requires_sponsorship: bool = False
    willing_to_relocate: bool = False
    notice_period_days: int = 30


class Education(BaseModel):
    degree: str
    institution: str = ""
    year: int | None = None


class Experience(BaseModel):
    years_total: int = 0
    current_title: str = ""
    current_company: str = ""
    seniority: Seniority = "mid"
    skills: dict[str, int] = Field(default_factory=dict)
    languages: dict[str, str] = Field(default_factory=dict)
    education: list[Education] = Field(default_factory=list)

    @field_validator("skills", mode="before")
    @classmethod
    def _lowercase_skills(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k).lower(): v for k, v in value.items()}
        return value


class Compensation(BaseModel):
    currency: str = "BRL"
    expected_monthly: int | None = None
    expected_annual: int | None = None
    disclose_expectation: bool = True


class Preferences(BaseModel):
    summary: str = ""
    tone: str = "direto e específico"
    language: str = "auto"


class AnswerRule(BaseModel):
    """Resposta fixa para perguntas que casam com uma regex."""

    match: str
    answer: str

    def matches(self, question: str) -> bool:
        return re.search(self.match, question, re.IGNORECASE) is not None


class Profile(BaseModel):
    personal: Personal
    resume_path: Path | None = None
    work_authorization: WorkAuthorization = Field(default_factory=WorkAuthorization)
    experience: Experience = Field(default_factory=Experience)
    compensation: Compensation = Field(default_factory=Compensation)
    preferences: Preferences = Field(default_factory=Preferences)
    answers: list[AnswerRule] = Field(default_factory=list)

    def lookup_answer(self, question: str) -> str | None:
        """Primeira regra do banco de respostas que casa com a pergunta."""
        for rule in self.answers:
            if rule.matches(question):
                return rule.answer
        return None

    def years_with(self, skill: str) -> int | None:
        return self.experience.skills.get(skill.lower())


class Query(BaseModel):
    keywords: str
    location: str = ""
    remote: bool = False
    easy_apply: bool = True
    date_posted: Literal["day", "week", "month", "any"] = "week"
    experience: list[Seniority] = Field(default_factory=list)


class Limits(BaseModel):
    max_applications_per_run: int = 10
    max_applications_per_day: int = 20
    max_results_per_query: int = 50
    delay_range: tuple[float, float] = (4.0, 11.0)


class Weights(BaseModel):
    title_match: float = 0.35
    skill_overlap: float = 0.40
    seniority_match: float = 0.15
    recency: float = 0.10


class Scoring(BaseModel):
    min_score: float = 0.55
    weights: Weights = Field(default_factory=Weights)


class Filters(BaseModel):
    require_title_keywords: list[str] = Field(default_factory=list)
    exclude_title_keywords: list[str] = Field(default_factory=list)
    exclude_companies: list[str] = Field(default_factory=list)
    preferred_keywords: list[str] = Field(default_factory=list)


class CoverLetter(BaseModel):
    enabled: bool = True
    max_words: int = 180


class SearchConfig(BaseModel):
    queries: list[Query]
    limits: Limits = Field(default_factory=Limits)
    scoring: Scoring = Field(default_factory=Scoring)
    filters: Filters = Field(default_factory=Filters)
    cover_letter: CoverLetter = Field(default_factory=CoverLetter)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} não encontrado. Rode `lacand init` e preencha os arquivos em config/."
        )
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_profile(path: Path) -> Profile:
    return Profile.model_validate(_read_yaml(path))


def load_search(path: Path) -> SearchConfig:
    return SearchConfig.model_validate(_read_yaml(path))
