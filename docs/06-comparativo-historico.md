> ⚠️ **Números corrigidos em [`docs/07-correcoes.md`](07-correcoes.md).**
> O M1lhão estava ausente do motor de EV e as elasticidades por escalão
> eram um palpite. A vantagem real da otimização é **+7,8%**, não os
> valores indicados abaixo. Onde houver conflito, vale o `docs/07`.

# Modelo preditivo vs apostas avulsas — todo o histórico

Confronto direto das quatro formas de jogar contra os **1970 sorteios reais
desde 13 de fevereiro de 2004**, usando os prémios que foram efetivamente
pagos em cada escalão de cada sorteio.

```bash
python -m euromillions.cli comparativo
```

## Correção necessária antes de qualquer número

O backtest anterior cobria apenas a era atual. Estender para todo o
histórico exigiu corrigir uma coisa que teria invalidado a comparação
inteira:

> **O número de estrelas mudou duas vezes: 9 → 11 → 12.**

Gerar um bilhete com a estrela 12 num sorteio de 2005 produz uma aposta que
nunca existiu e que nunca poderia ganhar nos escalões com estrelas. Isso
enviesaria toda a comparação a favor de quem calhasse jogar estrelas
baixas — e a estratégia "datas" joga precisamente estrelas de 1 a 9.

O histórico é agora processado **era a era**, cada uma com a matriz em
vigor. Há um teste (`test_carteiras_respeitam_o_pool_de_estrelas_da_era`)
a impedir que isto volte a partir-se.

## As quatro estratégias

| | como joga |
|---|---|
| **datas** | só números 1-31, estrelas 1-9 — o padrão de aniversários |
| **sobreposta** | 5 bilhetes variando 1 número a partir de uma base |
| **aleatória** | escolha uniforme (equivalente a *quick pick*) |
| **otimizada** | filtros dos dados + baixa popularidade + estrelas 10-12 + cobertura disjunta |

Popularidade média medida de cada uma, passando cada bilhete pelos modelos
estimados:

| estratégia | popularidade |
|---|---:|
| datas | 1,534× |
| sobreposta | 1,020× |
| aleatória | 1,020× |
| **otimizada** | **0,619×** |

## Resultado bruto — 1970 sorteios, 250 carteiras por estratégia

2,46 milhões de apostas por estratégia, €6.156.250 de custo cada:

| estratégia | retorno €/€ | ±95% | P(nada) | ±95% | prémios | maior prémio |
|---|---:|---:|---:|---:|---:|---:|
| datas | 0,1918 | ±0,0168 | 0,6767 | ±0,0018 | 194.015 | €81.968 (5+1) |
| sobreposta | 0,1922 | ±0,0039 | 0,7931 | ±0,0014 | 192.333 | €4.817 (4+2) |
| aleatória | **0,2880** | **±0,1898** | 0,6628 | ±0,0017 | 193.208 | €508.288 (5+1) |
| otimizada | 0,2092 | ±0,0023 | **0,6385** | ±0,0015 | 192.998 | €1.977 (4+2) |

**A estratégia aleatória "ganhou" — e isso não significa nada.**

Repare na margem: ±0,1898. Um único acerto de 5+1 no valor de €508.288
deslocou o retorno total mais do que toda a diferença que queremos medir. O
intervalo de confiança da aleatória cobre todas as outras estratégias.

Ler esta tabela como se mostrasse superioridade seria confundir sorte com
método — exatamente o erro que este projeto existe para não cometer. Se eu
tivesse corrido a simulação com outra semente, teria sido outra estratégia
a "ganhar".

## Prémios por escalão (acumulado)

| escalão | datas | sobreposta | aleatória | otimizada |
|---|---:|---:|---:|---:|
| **5+2 (jackpot)** | **0** | **0** | **0** | **0** |
| 5+1 | 1 | 0 | 1 | 0 |
| 5+0 | 0 | 0 | 2 | 0 |
| 4+2 | 4 | 4 | 7 | 2 |
| 4+1 | 78 | 91 | 85 | 85 |
| 3+2 | 218 | 219 | 192 | 218 |
| 4+0 | 183 | 157 | 188 | 180 |
| 2+2 | 3.072 | 2.923 | 2.975 | 3.059 |
| 3+1 | 3.706 | 3.752 | 3.782 | 3.899 |
| 3+0 | 7.601 | 7.430 | 7.513 | 7.529 |
| 1+2 | 15.915 | 16.167 | 16.033 | 16.246 |
| 2+1 | 54.079 | 53.591 | 53.996 | 54.011 |
| 2+0 | 109.158 | 107.999 | 108.435 | 107.769 |

Duas leituras:

1. **Os escalões frequentes são praticamente idênticos** (diferenças abaixo
   de 1,5%). Nenhuma estratégia acerta mais vezes. Confirmado, agora, em
   quase 10 milhões de apostas.

2. **O jackpot nunca saiu.** Zero vezes, em nenhuma das quatro estratégias,
   ao longo de 22 anos de sorteios simulados 250 vezes cada.

## Comparação analítica — a que consegue mesmo separar

A solução para o ruído de cauda é padrão em simulação: substituir o
resultado realizado pelo seu valor esperado condicional, calculado a partir
das probabilidades exatas de cada escalão. Elimina-se a variância sem
introduzir enviesamento.

Com um jackpot médio de €60M:

| estratégia | popularidade | EV/aposta | retorno €/€ | fatia do jackpot | ganho |
|---|---:|---:|---:|---:|---:|
| datas | 1,534 | €0,6989 | 0,2796 | 87,9% | — |
| sobreposta | 1,020 | €0,7591 | 0,3036 | +8,6% | 91,7% |
| aleatória | 1,020 | €0,7590 | 0,3036 | +8,6% | 91,7% |
| **otimizada** | **0,619** | **€0,8409** | **0,3363** | 94,9% | **+20,3%** |

Aqui a diferença aparece, limpa e sem margem de erro: **+20,3% face a jogar
datas**, e +10,8% face a jogar ao acaso. (Com jackpots maiores a vantagem
cresce — a €111M chega a +36%, porque a partilha pesa mais quanto maior for
o bolo.)

## P(não ganhar nada), era a era

| era | estrelas | sorteios | sobreposta | datas | aleatória | **otimizada** |
|---|---:|---:|---:|---:|---:|---:|
| E1 | 9 | 378 | 0,7859 | 0,6667 | 0,6551 | **0,6337** |
| E2 | 11 | 562 | 0,7964 | 0,6766 | 0,6666 | **0,6363** |
| E3 | 12 | 1030 | 0,7996 | 0,6894 | 0,6717 | **0,6459** |

A vantagem de cobertura é **consistente nas três eras**, com 22 anos e três
matrizes diferentes. Não é um artefacto de um período.

E note-se a coluna "datas": pior do que aleatório em todas as eras. Jogar
aniversários custa duas vezes — divide mais o prémio *e* ganha menos vezes,
porque restringir-se a 1-31 encolhe a cobertura do universo.

## A escala do problema

O número que põe tudo em perspetiva:

```
apostas simuladas por estratégia    2.462.500
jackpots esperados                  0,0176
P(pelo menos um jackpot)            1,75%
apostas para esperar 1 jackpot      139.838.160
anos jogando 5 apostas por sorteio  268.920
```

Simulámos 22 anos de história 250 vezes — o equivalente a 5.500 anos de
jogo — e a probabilidade de alguma dessas vidas ter apanhado o jackpot era
de 1,75%.

**Para *esperar* um jackpot jogando 5 apostas em cada sorteio são precisos
269 mil anos.**

## Conclusão do comparativo

| pergunta | resposta |
|---|---|
| O modelo acerta mais vezes? | **Não.** Diferenças <1,5% nos escalões frequentes, em 10M de apostas. |
| O modelo ganha mais dinheiro no histórico? | **Não é mensurável.** O ruído de cauda é maior do que o efeito. |
| O modelo tem maior valor esperado? | **Sim: +20,3% vs datas**, +10,8% vs aleatório (a €60M). |
| O modelo ganha alguma coisa mais vezes? | **Sim.** P(nada) 0,6385 vs 0,6767 (datas) e 0,7931 (sobreposta). |
| Alguma estratégia dá lucro? | **Não.** A melhor devolve €0,34 por cada €1 apostado. |

O modelo preditivo é melhor do que a aposta avulsa em duas dimensões
reais e mensuráveis, e em nenhuma delas transforma o jogo em investimento.
O retorno da melhor estratégia possível é **€0,34 por euro**.

A honestidade obriga a dizer também isto: uma pessoa que jogue avulso e
tenha sorte ganha muito mais do que uma pessoa que use este sistema e não
tenha. O comparativo bruto acima é a prova — a estratégia aleatória
apanhou €508 mil e ficou em primeiro lugar. Foi sorte, não método, e é por
isso que a coluna ±95% existe.
