"""Integração com a API da Anthropic para respostas abertas e cartas de apresentação."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import anthropic

from .config import Profile

log = logging.getLogger(__name__)

MODEL = "claude-opus-5"

# Habilita o fallback server-side: se um classificador recusar a requisição,
# a API reexecuta em um modelo alternativo dentro da mesma chamada.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM = """Você preenche formulários de candidatura a vagas em nome de uma pessoa \
real, usando o perfil abaixo como única fonte de fatos.

Regras invioláveis:
- Nunca invente experiência, formação, certificação, número ou data que não esteja \
no perfil. Se o perfil não responde a pergunta, diga isso na resposta.
- Números de anos de experiência vêm do perfil. Não arredonde para cima.
- Responda no mesmo idioma da pergunta.
- Sem preâmbulo. Devolva apenas o conteúdo pedido, no formato pedido.

<perfil>
{profile}
</perfil>"""


class LLMError(RuntimeError):
    pass


@dataclass
class JobContext:
    """Contexto da vaga passado ao modelo."""

    title: str
    company: str
    description: str = ""

    def render(self) -> str:
        description = self.description[:4000]
        return (
            f"Cargo: {self.title}\nEmpresa: {self.company}\n"
            f"Descrição:\n{description or '(não disponível)'}"
        )


def _profile_blob(profile: Profile) -> str:
    """Serializa o perfil como JSON estável — ordem fixa preserva o cache de prompt."""
    data = {
        "nome": profile.personal.full_name,
        "cidade": f"{profile.personal.city}, {profile.personal.country}".strip(", "),
        "cargo_atual": profile.experience.current_title,
        "empresa_atual": profile.experience.current_company,
        "anos_de_experiencia": profile.experience.years_total,
        "senioridade": profile.experience.seniority,
        "skills_anos": profile.experience.skills,
        "idiomas": profile.experience.languages,
        "formacao": [e.model_dump() for e in profile.experience.education],
        "autorizacao_trabalho": profile.work_authorization.model_dump(),
        "resumo": profile.preferences.summary,
        "tom": profile.preferences.tone,
    }
    if profile.compensation.disclose_expectation:
        data["pretensao"] = {
            "moeda": profile.compensation.currency,
            "mensal": profile.compensation.expected_monthly,
            "anual": profile.compensation.expected_annual,
        }
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)


class Assistant:
    """Cliente fino sobre a Messages API, com o perfil fixado no system prompt."""

    def __init__(self, profile: Profile, client: anthropic.Anthropic | None = None) -> None:
        self.profile = profile
        self.client = client or anthropic.Anthropic()
        self._system = [
            {
                "type": "text",
                "text": SYSTEM.format(profile=_profile_blob(profile)),
                # O perfil é idêntico entre chamadas: cacheá-lo derruba o custo
                # das dezenas de perguntas de um mesmo lote de candidaturas.
                "cache_control": {"type": "ephemeral"},
            }
        ]

    # ------------------------------------------------------------ interno

    def _create(self, **kwargs: object) -> anthropic.types.Message:
        """Chama a API com fallback de recusa, degradando se o beta não estiver disponível."""
        try:
            return self.client.beta.messages.create(
                betas=[FALLBACK_BETA], fallbacks="default", **kwargs
            )
        except anthropic.BadRequestError:
            log.debug("fallback server-side indisponível; usando chamada padrão")
            return self.client.messages.create(**kwargs)

    def _text(self, response: anthropic.types.Message) -> str:
        if response.stop_reason == "refusal":
            raise LLMError("o modelo recusou a requisição; responda esta pergunta manualmente")
        for block in response.content:
            if block.type == "text":
                return block.text.strip()
        raise LLMError("resposta sem bloco de texto")

    # ------------------------------------------------------------ público

    def answer(
        self,
        question: str,
        job: JobContext,
        *,
        options: list[str] | None = None,
        multiline: bool = False,
    ) -> str:
        """Responde uma pergunta do formulário.

        Com `options`, a resposta é restrita a um dos valores via structured
        outputs — o formulário só aceita opções existentes, então deixar o
        modelo escrever texto livre ali garantiria uma falha no preenchimento.
        """
        instruction = (
            "Responda à pergunta do formulário abaixo.\n\n"
            f"{job.render()}\n\nPergunta: {question}"
        )
        output_config: dict[str, object] = {"effort": "low"}

        if options:
            instruction += "\n\nEscolha exatamente uma das opções disponíveis."
            output_config["format"] = {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {"choice": {"type": "string", "enum": options}},
                    "required": ["choice"],
                    "additionalProperties": False,
                },
            }
        elif multiline:
            instruction += "\n\nResponda em até 4 frases."
        else:
            instruction += (
                "\n\nResponda de forma curta e literal — apenas o valor pedido "
                "(um número, uma palavra ou uma frase). Sem explicação."
            )

        response = self._create(
            model=MODEL,
            max_tokens=1024,
            system=self._system,
            output_config=output_config,
            messages=[{"role": "user", "content": instruction}],
        )
        text = self._text(response)

        if options:
            choice = json.loads(text)["choice"]
            if choice not in options:
                raise LLMError(f"opção inválida devolvida pelo modelo: {choice!r}")
            return choice
        return text

    def cover_letter(self, job: JobContext, *, max_words: int = 180) -> str:
        instruction = (
            f"Escreva uma carta de apresentação para esta vaga, em no máximo "
            f"{max_words} palavras.\n\n{job.render()}\n\n"
            "Conecte fatos concretos do perfil aos requisitos da vaga. "
            "Sem saudação genérica, sem elogio à empresa, sem repetir o currículo "
            "inteiro. Comece pelo motivo mais forte da candidatura."
        )
        response = self._create(
            model=MODEL,
            max_tokens=2048,
            system=self._system,
            output_config={"effort": "medium"},
            messages=[{"role": "user", "content": instruction}],
        )
        return self._text(response)
