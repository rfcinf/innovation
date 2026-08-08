# Lacunas, repetições e o mapa do espaço

A pergunta que originou este documento:

> "Quantas vezes na história houve dois sorteios com os mesmos números?
> Sabemos que ao longo de 20 anos não aconteceu, então temos de considerar
> que a chance disso acontecer é menor. São essas lacunas que quero
> abordar, para reduzir o erro."

A pergunta é boa e é testável. A resposta tem duas partes, e a segunda dá
razão à intuição — só que noutro sítio.

```bash
python -m euromillions.cli padroes
```

---

## Parte 1 — a premissa está errada, e de forma verificável

### Aconteceu

A combinação **4 – 30 – 31 – 38 – 42** saiu duas vezes:

| data | dia | estrelas |
|---|---|---|
| 2 de maio de 2014 | sexta | 2 – 11 |
| 31 de agosto de 2018 | sexta | 4 – 6 |

4,3 anos e 452 sorteios de intervalo.

### E era esperado

O erro de cálculo intuitivo é comparar 1970 sorteios com 2.118.760
combinações e concluir "é praticamente impossível". Mas a pergunta não é
"quantas vezes saiu uma combinação-alvo". É **"quantas vezes dois sorteios
quaisquer coincidiram entre si"** — e isso compara todos os pares:

```
pares de sorteios     C(1970, 2) = 1.939.465
prob. de cada par     1 / 2.118.760
repetições esperadas  0,915
repetições observadas 1
```

É o paradoxo dos aniversários. Numa sala de 23 pessoas há 50% de hipótese
de duas fazerem anos no mesmo dia, apesar de haver 365 dias.

### A verificação completa

Se houvesse alguma "resistência a repetir", ela apareceria não só nas
repetições totais mas em **todos os graus de coincidência**. Comparei os
1.939.465 pares de sorteios da história:

| números em comum | observado | esperado | desvio | frequência |
|---:|---:|---:|---:|---|
| 0 | 1.117.956 | 1.118.371 | −0,04% | 1 em 2 |
| 1 | 681.992 | 681.933 | +0,01% | 1 em 3 |
| 2 | 130.268 | 129.892 | +0,29% | 1 em 15 |
| 3 | 9.045 | 9.062 | −0,19% | 1 em 214 |
| 4 | 203 | 206 | −1,44% | 1 em 9.417 |
| **5** | **1** | **0,92** | +9,2% | 1 em 2.118.760 |

**χ² = 1,33 (5 g.l.), p = 0,93.**

É uma das concordâncias mais perfeitas com o acaso puro que se conseguem
obter em dados reais. Em quase dois milhões de comparações, a realidade não
se afasta da teoria em nenhum nível.

Não há resistência a repetir. Não há memória. A máquina não sabe o que já
saiu.

---

## Parte 2 — mas a intuição acerta: há não-uniformidade real, e é enorme

Aqui a ideia de "lacunas" é totalmente correta, e vale a pena ser preciso
sobre ela:

> **As combinações individuais são todas igualmente prováveis. As FAMÍLIAS
> de combinações não são — e nem por sombras.**

Enumerei **as 2.118.760 combinações todas** (não simulei: contei) e calculei
a probabilidade exata de cada família.

Exemplo, quantos números ≤ 31 (o intervalo dos aniversários):

| números ≤31 | combinações | probabilidade | sai 1 vez em | observado | esperado |
|---:|---:|---:|---:|---:|---:|
| 0 | 11.628 | 0,55% | **182 sorteios** | 11 | 10,8 |
| 1 | 120.156 | 5,67% | 18 | 82 | 111,7 |
| 2 | 450.585 | 21,27% | 4,7 | 426 | 418,9 |
| 3 | 768.645 | 36,28% | 2,8 | 750 | 714,7 |
| 4 | 597.835 | 28,22% | 3,5 | 542 | 555,9 |
| 5 | 169.911 | 8,02% | 12,5 | 159 | 158,0 |

Um sorteio sem nenhum número ≤ 31 acontece **1 vez em 182**. Isto é uma
lacuna real e mensurável. A intuição de que "há regiões raras" está certa.

## A distinção que muda tudo

E aqui está o passo em falso que quase todos os sistemas de lotaria dão:

> Uma família rara **não torna o seu bilhete menos provável**.

Se escolher uma combinação sem nenhum número ≤ 31, o seu bilhete tem
probabilidade **1 em 2.118.760**. Exatamente a mesma de qualquer outro
bilhete do planeta.

A aritmética é imediata:

```
P(família "0 números ≤31")  =  11.628 membros / 2.118.760  =  0,55%
P(o SEU bilhete nessa família)  =  1 / 2.118.760
```

A família é rara **exatamente na proporção do número de membros que tem**.
Não há mais nada. Dizer "esta família só sai 1 vez em 182, logo o meu
bilhete é pior" é contar os 11.628 irmãos do bilhete como se fossem
defeitos dele.

Evitar famílias raras não reduz erro nenhum — apenas troca um bilhete de
1 em 2.118.760 por outro bilhete de 1 em 2.118.760.

---

## O varrimento completo: 8 famílias contra a enumeração exata

Se alguma família fosse evitada pela máquina, teria de aparecer aqui.

| família de padrões | χ² | g.l. | p | q (FDR) |
|---|---:|---:|---:|---:|
| números ≤31 | 10,13 | 5 | 0,072 | 0,574 |
| soma dos 5 | 12,08 | 8 | 0,148 | 0,592 |
| quantos ímpares | 5,84 | 5 | 0,323 | 0,779 |
| amplitude (máx−mín) | 7,53 | 8 | 0,481 | 0,779 |
| pares consecutivos | 2,43 | 3 | 0,487 | 0,779 |
| máx. na mesma dezena | 1,99 | 4 | 0,737 | 0,942 |
| números ≤12 | 1,77 | 5 | 0,880 | 0,942 |
| dezenas cobertas | 0,78 | 4 | 0,942 | 0,942 |

**0 de 8 famílias com desvio real.** O espaço de padrões é percorrido
exatamente como a combinatória manda.

---

## "Nunca aconteceu": quando é que isso é informação?

Este é o teste que separa uma lacuna real de uma ilusão de raridade.

Listei **todas** as classes de padrões que ainda não saíram em 22 anos.
São **74**. E depois calculei, para cada uma, quantas vezes deveria ter
saído:

| propriedade | valor nunca visto | sai 1 vez em | esperado em 1970 sorteios |
|---|---:|---:|---:|
| soma | 195 | 811 | 2,43 |
| soma | 198 | 1.028 | 1,92 |
| amplitude | 8 | 1.441 | 1,37 |
| soma | 203 | 1.579 | 1,25 |
| soma | 50 | 1.900 | 1,04 |

Das 74 classes que "nunca aconteceram", **zero** têm ausência informativa
— nenhuma deveria ter saído sequer 3 vezes.

É a regra prática que resume tudo:

> Antes de concluir que algo é improvável por não ter acontecido, calcule
> quantas vezes deveria ter acontecido. Quase sempre a resposta é "menos de
> uma vez" — e nesse caso a ausência não é evidência de nada.

---

## Onde a intuição das lacunas paga mesmo

A ideia de "há regiões do espaço com propriedades diferentes, e devo
escolher onde jogar" **está certa**. Só não é do lado da probabilidade,
porque desse lado o espaço é perfeitamente plano.

É do lado de **quem mais está lá dentro**:

| família | quantas vezes sai | quantas pessoas a jogam |
|---|---|---|
| 5 números ≤31 | 1 em 12,5 | **1,62×** o normal |
| 0-2 números ≤31 | frequente | 0,94× |
| estrela 7 | igual às outras | **1,21×** |
| estrela 12 | igual às outras | 0,72× |

A coluna da esquerda é fixa e não se pode explorar. **A coluna da direita
é onde está tudo o que este projeto conseguiu extrair** — e é onde as
"lacunas" existem mesmo, porque a distribuição das escolhas humanas é
genuinamente irregular, ao contrário da distribuição das bolas.

Reduzir o erro é possível. Não escolhendo onde as bolas caem menos — elas
caem em todo o lado por igual — mas escolhendo onde **as pessoas** estão
menos.
