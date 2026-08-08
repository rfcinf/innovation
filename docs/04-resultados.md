> ⚠️ **Números corrigidos em [`docs/07-correcoes.md`](07-correcoes.md).**
> O M1lhão estava ausente do motor de EV e as elasticidades por escalão
> eram um palpite. A vantagem real da otimização é **+7,8%**, não os
> valores indicados abaixo. Onde houver conflito, vale o `docs/07`.

# Resultados

Todos os valores saem de `python -m euromillions.cli relatorio` sobre 1970
sorteios (13-02-2004 → 07-08-2026).

## 1. Aleatoriedade: nada encontrado

### Testes globais

| teste | estatística | p |
|---|---:|---:|
| Qui-quadrado uniformidade (50 números) | 52,59 | 0,337 |
| Sequências (paridade da soma) | 0,91 | 0,365 |
| Co-ocorrência de 1225 pares | 1253,22 | 0,275 |
| KS soma vs teórica | 0,016 | 0,672 |
| Entropia + G-test | 53,15 | 0,318 |
| Estatísticas de ordem (5) | — | 0,33 – 0,83 |

A entropia empírica é **5,63996 bits** contra um máximo teórico de
**5,64386**. Um défice de 0,0039 bits em 22 anos.

### Por era

| segmento | p |
|---|---:|
| Números — era E1 | 0,881 |
| Números — era E2 | 0,285 |
| Números — era E3 | 0,558 |
| Estrelas — era E1 | 0,716 |
| Estrelas — era E2 | 0,049 |
| Estrelas — era E3 | 0,203 |

O 0,049 das estrelas em E2 é o único valor abaixo de 0,05 — em 20
segmentos testados, é exatamente o que se espera do acaso. Após correção
FDR: q = 0,979. **0 de 20 segmentos** com viés.

### Bolas individuais

| número | saiu | esperado | desvio | p | q (FDR) |
|---:|---:|---:|---:|---:|---:|
| 22 | 154 | 197,0 | −21,8% | 0,00094 | **0,047** |
| 44 | 224 | 197,0 | +13,7% | 0,046 | 0,686 |
| 42 | 223 | 197,0 | +13,2% | 0,055 | 0,686 |

**1 de 50** sobrevive à correção FDR, e por uma margem mínima. Vale a pena
segui-lo, porque o resultado é instrutivo.

### O caso do número 22

Dividido o histórico em quatro troços iguais:

| troço | período | saiu | esperado | desvio | p |
|---:|---|---:|---:|---:|---:|
| 1 | 2004-02 → 2012-06 | 47 | 49,3 | −4,7% | 0,822 |
| 2 | 2012-06 → 2017-03 | 39 | 49,3 | −20,9% | 0,133 |
| 3 | 2017-03 → 2021-11 | 36 | 49,2 | −26,8% | 0,050 |
| 4 | 2021-11 → 2026-08 | 32 | 49,2 | −35,0% | **0,008** |

Um défice que se agrava de forma monótona. É exatamente a assinatura que
um viés físico progressivo produziria — uma bola a ganhar massa, um
desgaste a acumular-se.

E quase de certeza não é nada disso. O número 22 foi selecionado por ser o
mais extremo de 50. Condicionar um teste de tendência ao valor que se
escolheu por ser extremo é uma das formas mais fiáveis de produzir um
resultado falso. O teste seguinte é que decide.

### O teste decisivo

Um viés físico é uma propriedade persistente do equipamento: tem de
aparecer nas duas metades do histórico e correlacionar-se entre elas.

```
Correlação entre os desvios das duas metades:  r = 0,1189
Distribuição nula (20.000 simulações):         média 0,001   dp 0,142
p simulado:                                    0,408
```

Não há persistência. E a versão prática do mesmo teste:

| seleção no treino | desvio no treino | desvio no teste | manteve direção |
|---|---:|---:|---|
| 5 números mais quentes | +16,35% | +1,73% | sim (mas ~zero) |
| 5 números mais frios | −18,58% | +1,32% | **não — inverteu** |

Os desvios evaporam-se. É regressão à média, não sinal.

### Estratégias de números, avaliadas em 1770 sorteios

Cada estratégia escolhe 5 números usando apenas informação anterior:

| estratégia | acertos médios | esperado | z | p |
|---|---:|---:|---:|---:|
| quentes | 0,4881 | 0,5 | −0,78 | 0,434 |
| frios | 0,4780 | 0,5 | −1,45 | 0,147 |
| atrasados | 0,4785 | 0,5 | −1,43 | 0,154 |
| aleatório (controlo) | 0,5147 | 0,5 | +0,95 | 0,341 |

Nenhuma bate o acaso. O controlo aleatório teve o melhor resultado dos
quatro — o que resume tudo.

### Outros varrimentos

- **71 janelas deslizantes** de 200 sorteios: 0 com viés após FDR.
- **20 segmentos** (dia da semana, ano, era): 0 com viés após FDR.
- **Autocorrelação**, lags 1-10: máximo |r| = 0,078; todos os p > 0,37.
- **Intervalos entre aparições** (χ² agrupado): 0 de 50 após FDR.
- **Terças vs sextas**: r = 0,271, p simulado = 0,061 — ruído.

### Potência

| sorteios | viés mínimo detetável (80% potência) |
|---:|---:|
| 1970 (temos) | 19,4% |
| 5.000 | 12,1% |
| 20.000 | 6,0% |
| 100.000 | 2,7% |

Para chegar a 20.000 sorteios ao ritmo de 104/ano: **ano 2199**.

E sem a etiqueta do conjunto de bolas usado em cada sorteio, a penalização
é quadrática: com 3 conjuntos, um viés real teria de ser ~3× maior para ser
visível — 58% numa bola.

**Conclusão:** não há viés detetável, não há memória, e não há persistência.
A porta dos números está fechada por razões estruturais, não por azar
amostral.

## 2. Popularidade: efeito enorme

### Sem modelo

| números ≤31 | sorteios | popularidade média |
|---:|---:|---:|
| 1 | 82 | 0,91 |
| 2 | 418 | 0,94 |
| 3 | 737 | 1,11 |
| 4 | 532 | 1,15 |
| 5 | 157 | **1,62** |

```
≥4 números ≤31 vs ≤2:   rácio 1,32   Mann-Whitney p = 7,1 × 10⁻²⁰
```

### Partilha do jackpot (438 jackpots atribuídos)

| vencedores | sorteios | % |
|---:|---:|---:|
| 1 | 369 | 84,2% |
| 2 | 55 | 12,6% |
| 3 | 8 | 1,8% |
| 4 | 3 | 0,7% |
| 5 | 3 | 0,7% |

**15,8% dos jackpots foram divididos.**

### O modelo (GLM Poisson, n = 1936)

| termo | β | p | efeito |
|---|---:|---:|---:|
| espaco_regular | −0,1297 | <10⁻¹¹ | 0,878 |
| n_consecutivos | −0,0604 | <10⁻⁴ | 0,941 |
| n_ate_12 | +0,0683 | 0,0022 | 1,071 |
| n_ate_31 | +0,0482 | 0,0466 | 1,049 |
| mesma_coluna | +0,0159 | 0,169 | — |
| n_impares | +0,0152 | 0,179 | — |
| soma_norm | −0,0195 | 0,545 | — |
| tem_7 | +0,0043 | 0,724 | — |

Sobredispersão 3,5 (erros-padrão já corrigidos por quasi-verosimilhança).

Os dois efeitos mais fortes são **contra-intuitivos**: números consecutivos
e espaçamento irregular são *sub-jogados*, porque as pessoas acham que "não
parecem aleatórios". O folclore das apostas mandaria evitá-los — o que é
exatamente o erro que os torna valiosos.

### Validação fora da amostra

Estimado em 1355 sorteios até 2021-01-12; avaliado nos 581 seguintes.

```
Spearman ρ = 0,4114     p = 3,8 × 10⁻²⁵
Pearson (log) = 0,4194  p = 3,7 × 10⁻²⁶
```

| quintil previsto | popularidade observada |
|---:|---:|
| 1 | 0,775 |
| 2 | 0,825 |
| 3 | 0,851 |
| 4 | 0,981 |
| 5 | **1,314** |

Monótono, fora da amostra. Rácio extremo **1,70×** → cheque **70% maior**.

## 3. Economia: a brecha não chega

### Imposto do Selo (Portugal)

| bruto | líquido | imposto | taxa efetiva |
|---:|---:|---:|---:|
| €5.000 | €5.000 | €0 | 0,0% |
| €25.000 | €21.000 | €4.000 | 16,0% |
| €1.000.000 | €801.000 | €199.000 | 19,9% |
| €250.000.000 | €200.001.000 | €49.999.000 | 20,0% |

### Valor esperado (vendas 24M/sorteio, aposta €2,50)

| jackpot | π=3,0 (datas) | π=1,0 (média) | π=0,35 (otimizada) | ganho |
|---:|---:|---:|---:|---:|
| €17M | €0,41 | €0,54 | €0,72 | +33,9% |
| €50M | €0,56 | €0,71 | €0,90 | +27,0% |
| €100M | €0,78 | €0,97 | €1,18 | +21,2% |
| €150M | €1,01 | €1,24 | €1,46 | +17,9% |
| €200M | €1,23 | €1,50 | €1,73 | +15,7% |
| €250M (teto) | €1,45 | €1,76 | **€2,01** | +14,2% |

### Ponto de equilíbrio

| estratégia | jackpot necessário |
|---|---|
| combinação popular | **inatingível** |
| combinação média | **inatingível** |
| combinação otimizada | **inatingível** |
| otimizada, sem imposto | **inatingível** |

Nunca existe um momento de valor esperado positivo. Nem no teto. Nem sem
imposto.

## 4. Carteira gerada (exemplo real)

Jackpot €111M, vendas 24M, entropia ANU QRNG (certificada: 7,78 bits/byte,
χ² p = 0,095), 3000 candidatos de 9747 propostas quânticas:

```
 1 38 40 41 45   ★ 10 11   popularidade 0,30x   EV €1,276
 3 36 38 42 45   ★ 10 11   popularidade 0,36x   EV €1,232
15 16 33 34 50   ★ 10 12   popularidade 0,37x   EV €1,227
20 38 39 48 49   ★ 11 12   popularidade 0,37x   EV €1,226
 1 26 39 40 50   ★ 10 11   popularidade 0,43x   EV €1,192
```

(Repare no terceiro: 15-16 e 33-34, dois pares consecutivos. É o modelo a
aplicar o que os dados mostraram, contra a intuição.)

| estratégia | popularidade | EV | fatia do jackpot | ganho |
|---|---:|---:|---:|---:|
| aposta de datas | 1,977 | €0,910 | 84,8% | — |
| aposta média | 1,000 | €1,030 | 91,9% | +13,2% |
| **carteira otimizada** | **0,367** | **€1,228** | **96,9%** | **+34,9%** |

Custo de 5 apostas: €12,50. Valor esperado: €6,15. **Perda esperada:
€6,35.**

## Síntese

| pergunta | resposta | evidência |
|---|---|---|
| Há viés nas bolas? | Não | 12 testes, 0 achados após FDR |
| Os desvios persistem? | Não | r = 0,119, p = 0,408 |
| Estratégias de números funcionam? | Não | 4 estratégias, todas ~0,5 acertos |
| As pessoas escolhem mal? | **Sim, muito** | p = 7 × 10⁻²⁰ |
| Isso é previsível? | **Sim** | ρ = 0,411 fora da amostra |
| Vale dinheiro? | **Sim, +35%** | €0,91 → €1,23 |
| Chega para lucrar? | **Não** | EV máximo €2,01 < €2,50 |
