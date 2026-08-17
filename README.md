# lacand

Sistema para se candidatar a vagas no LinkedIn com o seu perfil: busca as vagas,
pontua cada uma contra o que você sabe fazer, preenche o formulário de
Candidatura Simplificada (Easy Apply) e — se você autorizar — envia.

## Antes de tudo: o risco

Automatizar candidaturas contradiz os [Termos de Uso do LinkedIn][tos] (seção 8.2,
que proíbe bots e acesso automatizado) e pode levar à restrição ou banimento da
sua conta. Isso não é uma formalidade: contas são restringidas por padrão de
tráfego automatizado.

Por isso o padrão aqui é **preencher e parar**. O comando `apply` completa o
formulário até a tela final, tira um screenshot e não clica em enviar. O envio
automático existe atrás da flag `--submit`, com confirmação e teto diário. Ligar
ou não é decisão sua.

Nenhuma senha é lida ou armazenada por este código — o login é manual, em uma
janela de navegador que você controla, e só a sessão resultante fica salva em disco.

[tos]: https://www.linkedin.com/legal/user-agreement

## Instalação

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
playwright install chromium
export ANTHROPIC_API_KEY=sk-ant-...   # opcional; sem ele use `apply --no-llm`
```

## Uso

```bash
lacand init      # cria config/profile.yaml e config/search.yaml
                 # → edite os dois antes de continuar
lacand login     # abre o navegador; você loga (2FA inclusive) e a sessão é salva
lacand search    # busca, pontua e enfileira as vagas aprovadas
lacand list      # revisa a fila antes de gastar candidatura
lacand apply     # preenche os formulários e PARA antes de enviar
lacand apply --submit   # envia de verdade (pede confirmação)
```

Fluxo típico: rodar `search` uma vez por dia, revisar com `list`, rodar `apply`
sem `--submit` nas primeiras vezes para conferir as respostas nos screenshots, e
só então decidir se vale ligar o envio automático.

## Como as respostas são decididas

Toda pergunta de formulário passa por quatro camadas, da mais barata e previsível
para a mais cara:

| Ordem | Origem | O quê |
|---|---|---|
| 1 | `profile` | regras de regex que você escreveu em `profile.yaml` |
| 2 | `memory` | resposta já dada a essa mesma pergunta em candidatura anterior |
| 3 | `heuristic` | campos estruturados do perfil (anos por skill, telefone, autorização, pretensão) |
| 4 | `llm` | Claude, com o perfil como única fonte de fatos |

O Claude só é acionado quando as três primeiras falham — tipicamente em perguntas
abertas ("descreva um projeto…") e cartas de apresentação. O prompt proíbe
inventar experiência, formação ou número que não esteja no perfil. Toda resposta
gerada é memorizada, então a mesma pergunta não custa uma segunda chamada de API.

O `apply` marca com `!` toda resposta de baixa confiança ou vazia. Rode
`lacand answers` para ver o que o modelo respondeu e promova as boas para o
`profile.yaml` como regras fixas — cada uma que você promove é uma variação a
menos entre candidaturas.

Modelo usado: `claude-opus-5`, com o perfil em cache de prompt (o mesmo bloco se
repete em todas as perguntas de um lote) e `effort: low` nas respostas curtas.

## Pontuação das vagas

A nota de 0 a 1 combina quatro sinais, com pesos configuráveis em `search.yaml`:
casamento do título, sobreposição entre as skills do perfil e a descrição,
distância de senioridade e recência da publicação. Empresas e títulos vetados são
descartados antes de qualquer cálculo. Só entra na fila quem passa de `min_score`.

## Limites e proteções

- `max_applications_per_run` — teto por execução do `apply`
- `max_applications_per_day` — teto diário somado entre execuções, contado no banco
- `delay_range` — pausa aleatória entre ações (padrão 4–11s)
- envio exige `--submit` **e** confirmação interativa (pulável com `--yes`)
- toda vaga já vista é registrada: nunca há candidatura duplicada

## Estrutura

```
config/          exemplos de configuração (os arquivos reais são gitignored)
src/lacand/
  config.py      modelos Pydantic dos dois YAMLs
  browser.py     sessão Playwright com estado persistido
  search.py      montagem da URL de busca e extração dos cards
  matcher.py     pontuação e filtros
  answers.py     resolução de respostas nas quatro camadas
  llm.py         cliente Anthropic (respostas abertas e cartas)
  easy_apply.py  preenchimento do formulário multi-etapas
  store.py       SQLite: vagas, status, respostas memorizadas
  cli.py         comandos
tests/           22 testes de pontuação, resolução de respostas e URLs
```

## O que está testado e o que não está

Testado com `pytest`: pontuação e filtros, as quatro camadas de resolução de
respostas (incluindo o mapeamento para as opções de um `select` e o cache que
evita a segunda chamada de API), e a montagem das URLs de busca.

**Não testado contra o LinkedIn real** — não tenho conta para isso. Os seletores
CSS em `search.py` e `easy_apply.py` foram escritos a partir do markup conhecido,
com uma lista de alternativas para cada elemento, mas o LinkedIn troca de layout
com frequência. Espere ajustar seletores na primeira execução: rode
`lacand apply --verbose` com o navegador visível (padrão) e observe onde trava.

Formulários de Easy Apply que saem do padrão (perguntas com autocomplete,
uploads múltiplos, etapas condicionais) podem não completar; o resultado volta
como `failed` com a nota do passo que travou, e a vaga continua disponível para
candidatura manual.
