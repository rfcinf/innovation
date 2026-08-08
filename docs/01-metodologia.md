# Metodologia

## Dados

| | |
|---|---|
| Sorteios | **1970**, de 13-02-2004 (o primeiro) a 07-08-2026 |
| Quebra de prémios | **1936** sorteios (98,3%) |
| Fonte | euro-millions.com — páginas anuais + página de cada sorteio |
| Integridade | 0 problemas: sem datas duplicadas, sem números repetidos, sem estrelas fora do domínio da era |

Os 34 sorteios sem quebra de prémios são anteriores à adesão de Portugal
(8 de outubro de 2004) e não têm bloco de prémios português.

As três eras da matriz do jogo foram declaradas em `config.py` a partir de
fontes documentais e depois **validadas contra os dados**:

| era | período | estrelas | sorteios | estrela máx. observada |
|---|---|---:|---:|---:|
| E1 | 2004-02-13 → 2011-05-06 | 9 | 378 | 9 ✓ |
| E2 | 2011-05-10 → 2016-09-23 | 11 | 562 | 11 ✓ |
| E3 | 2016-09-27 → 2026-08-07 | 12 | 1030 | 12 ✓ |

## Princípios

Três regras governam este projeto. São o que separa uma análise de um
sistema de lotaria à venda.

### 1. Correção para testes múltiplos, sempre

Com 50 números, 12 estrelas, 1225 pares e 71 janelas temporais, aparecem
"anomalias a p < 0,05" às centenas **por construção**. Encontrar padrões é
trivial; encontrar padrões *reais* não é.

Todos os testes deste projeto passam por Benjamini-Hochberg. Sem isso, eu
poderia ter-lhe entregado uma dúzia de "descobertas" — e todas seriam ruído.

### 2. Um teste que não pode falhar não é um teste

O backtest avalia estratégias de números com honestidade suficiente para
elas falharem. E falham.

A validação do modelo de popularidade é feita **fora da amostra**: estimado
até janeiro de 2021, avaliado nos 581 sorteios seguintes, que nunca viu.

### 3. Ausência de prova não é prova de ausência

"Não rejeitámos H₀" não é o mesmo que "não há viés". Por isso o projeto
inclui análise de potência: `minimum_detectable_bias()` diz qual é o menor
viés que este volume de dados conseguiria ver, e portanto que tamanho de
brecha ainda poderia estar escondido.

Com 1970 sorteios: **19,4%**. Só um desvio brutal seria visível.

## Um erro que cometi, e o que ele ensina

A primeira versão do teste de intervalos entre aparições usava
Kolmogorov-Smirnov contra uma distribuição geométrica. Resultado: **47 dos
50 números "significativos"** após correção FDR.

Teria sido a descoberta do projeto. Era um erro: o KS pressupõe uma
distribuição contínua, e os intervalos são inteiros com muitos empates.
Aplicado a dados discretos, produz p-valores inválidos.

Substituído por qui-quadrado com classes agrupadas — o teste correto para
dados discretos. Resultado: **0 de 50**.

Ficou registado em `randomness.gap_analysis()` porque ilustra o mecanismo
exato pelo qual nascem os sistemas de lotaria que prometem padrões: não por
má-fé, mas por um teste mal escolhido que ninguém verificou. A diferença
entre 47/50 e 0/50 foi uma linha de código.

## A bateria de testes

Sobre os números e estrelas (`randomness.py`):

| teste | o que apanharia |
|---|---|
| Qui-quadrado de uniformidade | bola sistematicamente favorecida |
| Binomial por bola + FDR | desvio individual, com controlo de falsas descobertas |
| Autocorrelação (lags 1-10) | memória entre sorteios — a hipótese "quentes/frios" |
| Intervalos (χ² agrupado) | dependência na frequência de reaparição |
| Sequências (Wald-Wolfowitz) | estrutura na paridade |
| Co-ocorrência de 1225 pares | bolas agrupadas por peso ou carregamento |
| Estatísticas de ordem | integridade dos dados |
| KS da soma vs teórica | desvio na distribuição conjunta |
| Entropia + G-test | défice de aleatoriedade global |
| Janelas deslizantes (71) | equipamento a degradar-se num troço |
| Segmentação por dia/ano/era | proxy do conjunto de bolas usado |
| **Persistência fora da amostra** | **o teste decisivo** |

O último merece destaque. Um viés físico é uma propriedade persistente do
equipamento: teria de aparecer nas duas metades do histórico e
correlacionar-se entre elas. Ruído amostral não faz isso, por construção.

Compara-se a correlação entre os vetores de desvio das duas metades com a
distribuição nula obtida por 20.000 simulações de sorteios genuinamente
uniformes. Este teste é imune ao problema de seleção — avalia os 50 números
de uma vez, e valida-se contra a sua própria distribuição nula.

## Reconstrução do volume de vendas

O número de apostas vendidas não é público, mas é recuperável. Para uma
população de apostas:

```
E[vencedores em todos os escalões] = S × P(ganhar alguma coisa)
```

`P(ganhar)` depende apenas da matriz (≈ 1/13 na era atual), logo
`Ŝ = total_vencedores × 13,06`.

O enviesamento humano distorce isto, mas os escalões baixos (2+0, 2+1)
dominam a contagem e são pouco sensíveis à combinação escolhida — acertar
dois números é uma restrição fraca. O erro residual é de poucos por cento
e, para comparar popularidade *entre* sorteios, cancela-se em grande
medida.

Mediana estimada: **24,0 milhões de apostas por sorteio**.

## O modelo de popularidade

GLM de Poisson com offset, estimado por IRLS implementado de raiz
(`popularity.PoissonGLM`):

```
vencedores_5_números ~ Poisson(μ)
log μ = log(vendas / 2.118.760) + β'x
```

O offset garante que o modelo explica a **popularidade relativa** e não o
volume de vendas. A sobredispersão observada é de 3,5 — o comportamento
humano é muito mais irregular do que Poisson — e os erros-padrão são
corrigidos por quasi-verosimilhança. Ignorá-la inflacionaria a confiança
por um fator de ~1,9.

## Reprodutibilidade

```bash
pip install -r requirements.txt
python -m euromillions.cli fetch --breakdown    # ~15 min
python -m euromillions.cli relatorio
pytest tests/ -q
```

Toda a recolha é incremental e idempotente. Todos os resultados citados na
documentação saem destes comandos.
