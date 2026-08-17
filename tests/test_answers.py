import pytest

from lacand.answers import resolve
from lacand.llm import JobContext
from lacand.store import Store

JOB = JobContext(title="Engenheiro de Software", company="Contoso")


@pytest.fixture
def store(tmp_path) -> Store:
    with Store(tmp_path / "test.db") as store:
        yield store


def _resolve(question, profile, store, **kwargs):
    return resolve(question, profile=profile, store=store, job=JOB, **kwargs)


def test_banco_de_respostas_do_perfil_tem_prioridade(profile, store):
    result = _resolve("Como você se identifica em termos de diversidade?", profile, store)
    assert result.answer == "Prefiro não responder"
    assert result.source == "profile"


def test_memoria_vence_heuristica(profile, store):
    store.remember_answer("Qual seu telefone?", "+55 11 99999-0000", "llm")
    result = _resolve("Qual seu telefone?", profile, store)
    assert result.answer == "+55 11 99999-0000"
    assert result.source == "memory"


def test_heuristica_de_anos_por_skill(profile, store):
    result = _resolve("Quantos anos de experiência você tem com Python?", profile, store)
    assert result.answer == "8"
    assert result.source == "heuristic"


def test_skill_desconhecida_cai_no_total_com_baixa_confianca(profile, store):
    result = _resolve("How many years of experience do you have with Rust?", profile, store)
    assert result.answer == "8"
    assert result.confident is False


def test_autorizacao_e_patrocinio_sao_opostos(profile, store):
    autorizado = _resolve("Você está autorizado a trabalhar no Brasil?", profile, store)
    patrocinio = _resolve("Você precisa de patrocínio de visto?", profile, store)
    assert autorizado.answer == "Sim"
    assert patrocinio.answer == "Não"


def test_pretensao_salarial(profile, store):
    result = _resolve("Qual sua pretensão salarial?", profile, store)
    assert result.answer == "18000"


def test_resposta_e_mapeada_para_a_opcao_do_formulario(profile, store):
    result = _resolve(
        "Você está autorizado a trabalhar no Brasil?",
        profile,
        store,
        options=["Yes", "No"],
    )
    assert result.answer == "Yes"


def test_sem_llm_e_sem_match_cai_no_default(profile, store):
    result = _resolve("Descreva um projeto do qual você se orgulha.", profile, store)
    assert result.answer == ""
    assert result.source == "default"
    assert result.confident is False


def test_default_com_opcoes_escolhe_a_primeira(profile, store):
    result = _resolve("Pergunta sem correspondência alguma", profile, store, options=["A", "B"])
    assert result.answer == "A"
    assert result.confident is False


def test_llm_e_consultado_e_memorizado(profile, store):
    class FakeAssistant:
        def __init__(self):
            self.calls = 0

        def answer(self, question, job, *, options=None, multiline=False):
            self.calls += 1
            return "Migrei um monolito para Kubernetes."

    assistant = FakeAssistant()
    question = "Descreva um projeto do qual você se orgulha."

    first = _resolve(question, profile, store, assistant=assistant)
    assert first.source == "llm"
    assert assistant.calls == 1

    # Na segunda vez a resposta vem da memória, sem nova chamada de API.
    second = _resolve(question, profile, store, assistant=assistant)
    assert second.source == "memory"
    assert second.answer == first.answer
    assert assistant.calls == 1
