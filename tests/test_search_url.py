from urllib.parse import parse_qs, urlparse

from lacand.config import Query
from lacand.search import build_url


def _params(url: str) -> dict[str, list[str]]:
    return parse_qs(urlparse(url).query)


def test_url_com_todos_os_filtros():
    query = Query(
        keywords="Python Developer",
        location="Brasil",
        remote=True,
        easy_apply=True,
        date_posted="week",
        experience=["mid", "senior"],
    )
    params = _params(build_url(query))
    assert params["keywords"] == ["Python Developer"]
    assert params["location"] == ["Brasil"]
    assert params["f_AL"] == ["true"]
    assert params["f_WT"] == ["2"]
    assert params["f_TPR"] == ["r604800"]
    assert params["f_E"] == ["3,4"]


def test_filtros_opcionais_sao_omitidos():
    params = _params(build_url(Query(keywords="Dev", easy_apply=False, date_posted="any")))
    assert "f_AL" not in params
    assert "f_WT" not in params
    assert "f_TPR" not in params
    assert "f_E" not in params


def test_niveis_duplicados_geram_um_unico_codigo():
    # senior e staff mapeiam para o mesmo código do LinkedIn (4).
    params = _params(build_url(Query(keywords="Dev", experience=["senior", "staff"])))
    assert params["f_E"] == ["4"]


def test_paginacao():
    assert _params(build_url(Query(keywords="Dev"), start=50))["start"] == ["50"]
