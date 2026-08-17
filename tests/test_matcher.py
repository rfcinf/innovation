from lacand.matcher import _detect_seniority, _recency_score, evaluate


def _evaluate(profile, scoring, filters, **overrides):
    payload = {
        "title": "Engenheiro de Software Sênior",
        "company": "Contoso",
        "description": "Buscamos alguém com Python, Django e AWS. Kubernetes é diferencial.",
        "posted": "há 2 dias",
    }
    payload.update(overrides)
    return evaluate(profile=profile, scoring=scoring, filters=filters, **payload)


def test_boa_vaga_entra_na_fila(profile, scoring, filters):
    verdict = _evaluate(profile, scoring, filters)
    assert not verdict.rejected
    assert verdict.score > 0.7
    assert verdict.status == "queued"


def test_empresa_vetada_e_rejeitada_sem_pontuar(profile, scoring, filters):
    verdict = _evaluate(profile, scoring, filters, company="Acme Tecnologia")
    assert verdict.rejected
    assert verdict.score == 0.0
    assert "empresa vetada" in verdict.reasons[0]


def test_titulo_com_termo_vetado_e_rejeitado(profile, scoring, filters):
    verdict = _evaluate(profile, scoring, filters, title="Estágio em Engenharia de Software")
    assert verdict.rejected
    assert "vetado" in verdict.reasons[0]


def test_titulo_sem_termo_obrigatorio_e_rejeitado(profile, scoring, filters):
    verdict = _evaluate(profile, scoring, filters, title="Analista de Marketing")
    assert verdict.rejected


def test_descricao_sem_skills_derruba_a_nota(profile, scoring, filters):
    rico = _evaluate(profile, scoring, filters)
    pobre = _evaluate(
        profile, scoring, filters, description="Vaga para atuar com tecnologias diversas."
    )
    assert pobre.score < rico.score


def test_senioridade_distante_pontua_menos(profile, scoring, filters):
    senior = _evaluate(profile, scoring, filters, title="Engenheiro de Software Sênior")
    junior = _evaluate(profile, scoring, filters, title="Engenheiro de Software Júnior")
    assert junior.score < senior.score


def test_deteccao_de_senioridade():
    assert _detect_seniority("Senior Staff Engineer") == "staff"
    assert _detect_seniority("Desenvolvedor Pleno") == "mid"
    assert _detect_seniority("Tech Lead") == "lead"
    assert _detect_seniority("Engenheiro de Software") is None


def test_recencia():
    assert _recency_score("há 3 horas") == 1.0
    assert _recency_score("há 2 dias") == 1.0
    assert _recency_score("há 3 semanas") == 0.2
    assert _recency_score("") == 0.5
