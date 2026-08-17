import pytest

from lacand.config import Filters, Profile, Scoring


@pytest.fixture
def profile() -> Profile:
    return Profile.model_validate(
        {
            "personal": {
                "full_name": "Maria Silva",
                "email": "maria@example.com",
                "phone": "+55 11 91234-5678",
                "city": "São Paulo",
                "state": "SP",
                "country": "Brasil",
                "linkedin_url": "https://linkedin.com/in/maria",
            },
            "work_authorization": {
                "authorized_in": ["Brasil"],
                "requires_sponsorship": False,
                "willing_to_relocate": False,
                "notice_period_days": 30,
            },
            "experience": {
                "years_total": 8,
                "seniority": "senior",
                "skills": {"Python": 8, "django": 5, "aws": 4},
                "languages": {"inglês": "avançado"},
            },
            "compensation": {
                "currency": "BRL",
                "expected_monthly": 18000,
                "disclose_expectation": True,
            },
            "answers": [
                {"match": r"(diversidade|diversity)", "answer": "Prefiro não responder"}
            ],
        }
    )


@pytest.fixture
def scoring() -> Scoring:
    return Scoring()


@pytest.fixture
def filters() -> Filters:
    return Filters(
        require_title_keywords=["engenheir", "engineer", "developer"],
        exclude_title_keywords=["estágio", "intern", ".net"],
        exclude_companies=["Acme Tecnologia"],
        preferred_keywords=["python", "kubernetes"],
    )
