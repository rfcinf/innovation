> **Este documento corrige números publicados em `docs/03` a `docs/06`.**
> Onde houver conflito, vale o que está aqui.

# Terceira ronda: três correções, das quais duas reduzem a vantagem

Perguntou onde ainda se podia ajustar o modelo. Ao responder, encontrei um
erro factual no meu próprio motor de valor esperado. Este documento regista
o que estava errado, o que passou a estar certo, e o que isso faz às
conclusões — que ficam **menos favoráveis** ao sistema do que eu tinha
anunciado.

---

## Correção 1 — o M1lhão estava ausente do motor de EV

### O erro

O bilhete português custa €2,50 e **inclui um código M1lhão**: um sorteio
exclusivo de Portugal, prémio de €1.000.000, realizado às sextas-feiras,
com todos os códigos gerados entre sábado e sexta — ou seja, uma semana
inteira de apostas, incluindo as do sorteio de terça.

O meu motor contava os 13 escalões do EuroMillions e ignorava isto por
completo.

### A dimensão

Quota portuguesa estimada pelo escalão 2+0 (vencedores PT / vencedores
totais), período 2025+:

```
quota PT                        8,19%
vendas terça / sexta            19,6M / 26,3M
códigos por sorteio semanal     3.754.609
prémio bruto / líquido          €1.000.000 / €801.000
EV do M1lhão por aposta         €0,2133
```

Estive a **subestimar o retorno em cerca de 21-23%** de todos os valores de
EV publicados.

### A consequência que interessa

O código do M1lhão é **gerado pelo sistema**. Ninguém o escolhe. Não há
combinações populares, não há partilha, não há absolutamente nada a
otimizar.

Por isso, o M1lhão é uma parcela de EV **fixa e igual para toda a gente** —
e **dilui a vantagem percentual** da otimização de combinações. Somar a
mesma constante ao numerador e ao denominador reduz o rácio. A percentagem
menor é a verdadeira.

Nota lateral com valor prático: a quota portuguesa tem descido de forma
sistemática (11,8% em 2019 → 7,9% em 2026). Menos apostas portuguesas
significam menos códigos em circulação, e portanto um M1lhão **mais**
valioso por aposta.

---

## Correção 2 — a tabela de elasticidades era um palpite meu

### O erro

`ev.TIER_ELASTICITY` diz quanto é que a popularidade da nossa combinação
afeta o número de vencedores de cada escalão. Estava assim: 1,0 / 0,7 /
0,4 / 0,2 / 0,1 — **atribuída por mim, por analogia**.

É exatamente o mesmo pecado que corrigi nas estrelas na segunda ronda, e
que não vi na minha própria casa.

### A armadilha na medição

O caminho óbvio — regredir vencedores do escalão na popularidade observada
— não serve. A popularidade observada é calculada dividindo por vendas
estimadas, e os vencedores de qualquer escalão também escalam com as
vendas. O erro da estimativa entra dos dois lados e produz correlação
espúria, inflacionando todas as elasticidades.

A solução foi usar a popularidade **prevista pelo modelo**, que depende só
das características da combinação e não contém vencedores nem vendas:

```
log(W_t) = a_t + b_t · log(π̂) + c_t · log(vencedores_totais) + ε
```

### A validação do método

O escalão 5+0 mede **b = 0,956** quando a teoria exige exatamente 1,0
(acertar os 5 números *é* ter a nossa combinação). O método acerta no caso
em que a resposta é conhecida à partida.

### O resultado

| escalão | escrito à mão | **medido** | z |
|---|---:|---:|---:|
| 5+2 / 5+1 / 5+0 | 1,00 | **1,000** (teoria) | 16,0 no 5+0 |
| 4+0 | 0,70 | **0,601** | 22,1 |
| 4+1 | 0,70 | **0,572** | 13,5 |
| 4+2 | 0,70 | **0,523** | 8,0 |
| 3+0 | 0,40 | **0,258** | 11,0 |
| 3+1 | 0,40 | **0,234** | 5,8 |
| 3+2 | 0,40 | **0,193** | 3,2 |
| 2+0 | 0,20 | **0,009** | 1,1 |
| 2+1 | 0,20 | **0,000** | −0,5 |
| 2+2 | 0,20 | **0,000** | −0,9 |
| 1+2 | 0,10 | **0,000** | −4,1 |

A minha tabela estava **inflacionada em todos os escalões abaixo de 5
números** — nalguns casos por um fator de 2 ou mais.

Um achado inesperado: o escalão 1+2 mede elasticidade **negativa** (−0,24,
p = 5×10⁻⁵). Não é ruído, é composição. Quando a combinação sorteada é
popular, os bilhetes populares tendem a acertar *mais* números, e a massa
desloca-se dos escalões baixos para os altos, esvaziando-os.

Efeito no EV da combinação otimizada: **€0,8409 → €0,8029**, uma
sobrestimativa de 4,5% que desapareceu.

---

## Correção 3 (tentativa falhada) — o estimador sem vendas não resultou

### A hipótese

Na resposta anterior avancei que o modelo das estrelas (ρ = 0,775) supera o
dos números (ρ = 0,411) porque usa uma **razão dentro do mesmo sorteio**,
onde as vendas cancelam, enquanto o dos números divide por vendas
estimadas com erro e deriva.

A proposta era aplicar a mesma lógica aos números:

```
π_rel = 225 · W5 / W4
```

(225 = 5 números × 45 substituições: os vizinhos a distância 1.)

### O teste

Treinei o mesmo modelo de características com cada um dos alvos e avaliei
fora da amostra, contra dois critérios independentes:

| alvo de avaliação | estimador de VENDAS | estimador por RAZÃO |
|---|---:|---:|
| popularidade real (determina o cheque) | **ρ = 0,4071** | ρ = 0,3631 |
| W5 bruto (pessoas com a nossa combinação) | **ρ = 0,3870** | ρ = 0,3313 |

### O veredicto: a minha hipótese estava errada

O estimador por razão é **pior**, nos dois critérios.

A razão é visível na medição direta do viés de datas: o estimador por razão
capta um rácio de 1,054 (p = 4×10⁻³) onde o de vendas capta 1,325
(p = 7×10⁻²⁰). Dividir por W4 cancela o próprio sinal que queremos medir —
**a vizinhança de uma combinação de datas é também ela de datas**.

E isso obriga-me a corrigir o diagnóstico que dei na resposta anterior: a
superioridade do modelo das estrelas **não vem** de ser livre de vendas.
Vem de o domínio ser minúsculo — 12 estrelas, 66 pares, estimáveis
elemento a elemento — enquanto os números vivem num espaço de 2.118.760
combinações que só é acessível através de características agregadas.

Fica como resultado negativo documentado. O estimador baseado em vendas
mantém-se.

---

## O efeito combinado nas conclusões

As duas correções empurram no mesmo sentido: **para baixo**.

### Vantagem da otimização face a jogar datas

| jackpot | antes (publicado) | **agora (correto)** |
|---:|---:|---:|
| €17M | +25,9% | **+7,8%** |
| €60M | +20,3% | **+7,8%** |
| €111M | +17,0% | **+7,9%** |
| €250M | +13,1% | **+7,9%** |

A vantagem real é **+7,8%**, não os +36% que anunciei na segunda ronda.

Repare também que passou a ser **praticamente constante** em todos os
níveis de jackpot. Há uma razão estrutural: em jackpots pequenos os
escalões baixos dominam, e a sua elasticidade é agora quase nula; em
jackpots grandes domina a partilha do jackpot, que dá vantagem, mas o
M1lhão dilui. Os dois efeitos cancelam-se quase exatamente.

### Valor esperado por aposta (€2,50), jackpot €60M

| estratégia | EuroMillions | M1lhão | **total** | retorno |
|---|---:|---:|---:|---:|
| datas | €0,7290 | €0,2133 | €0,9423 | 0,377 €/€ |
| aleatória | €0,7605 | €0,2133 | €0,9738 | 0,390 €/€ |
| **otimizada** | €0,8029 | €0,2133 | **€1,0162** | **0,407 €/€** |

Cerca de 21-23% do valor de um bilhete português vem do M1lhão, onde não há
nada a otimizar.

### O que não mudou

- **Ponto de equilíbrio: continua inatingível**, mesmo com o M1lhão somado
  e no teto de €250M (aí o retorno é 0,819 €/€).
- **Nenhuma estratégia acerta mais vezes.** A probabilidade é 1 em
  139.838.160 e é intocável.
- **P(não ganhar nada) continua reduzível** de 0,80 para 0,646 — a
  alavanca da cobertura não depende de nenhuma destas correções, porque é
  combinatória pura.
- **Os modelos de popularidade continuam validados** fora da amostra
  (ρ = 0,411 nos números, ρ = 0,775 nas estrelas). O que mudou não foi a
  capacidade de prever a partilha, foi **quanto é que essa previsão vale em
  euros**.

---

## O que fica por fazer

Da lista que apresentei, ficaram dois pontos por atacar, e o primeiro é o
mais interessante de todos:

**A mistura de apostas aleatórias.** Uma fração `q` das apostas é gerada por
máquina e não carrega viés humano nenhum. O que observo é
`π_obs = q·1 + (1−q)·π_humano`. Sinal de que isto importa: o meu quintil
mais impopular observa 0,775, **não perto de zero** — há um chão aleatório
a puxar tudo para 1. Se `q` for ~50%, o viés humano real é aproximadamente
o dobro do medido, e o **teto desta vantagem é bastante mais alto do que
qualquer número deste documento**. `q` é estimável, e é a única peça que
diria qual é o limite verdadeiro do sistema.

**Interações no modelo de números.** É aditivo em 12 características
escolhidas à mão, sem interações nem memória de combinações específicas.
Um modelo regularizado levantaria ρ — com risco de sobreajuste que a
disciplina walk-forward teria de travar.
