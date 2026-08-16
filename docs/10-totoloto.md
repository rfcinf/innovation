# Totoloto — estrutura e análise de vieses

```bash
python -m euromillions.cli totoloto
```

**1609 sorteios**, desde 13 de março de 2011 (a data a partir da qual a
Santa Casa disponibiliza estatísticas) até agosto de 2026.

---

## 1. A estrutura, reconstruída e validada

A Santa Casa publica a matriz (5 números em 49 + 1 Nº da Sorte em 13) e diz
que a probabilidade de ganhar um qualquer prémio é "1 em 7". **Não publica a
probabilidade de cada escalão.**

Reconstruí os seis escalões por combinatória e confrontei o resultado com o
número que ela publica:

```
P(algum prémio) calculado = 1 em 6,86
Santa Casa publica        = 1 em 7        ✓
```

Sem este confronto, a estrutura seria apenas uma suposição minha sobre as
regras. Com ele, está validada.

| escalão | combinações | 1 em |
|---|---:|---:|
| 1º — 5 números + Nº da Sorte | 1 | **24.789.492** |
| 2º — 5 números | 1 | 1.906.884 |
| 3º — 4 números | 220 | 8.668 |
| 4º — 3 números | 9.460 | 202 |
| 5º — 2 números | 132.440 | 14 |
| 6º — só Nº da Sorte | — | 13 |

Um detalhe que muda o cálculo: o **Nº da Sorte é um prémio à parte que
acumula** com os escalões de números. Somar as probabilidades daria um valor
errado — tem de se calcular o complementar de "menos de 2 números **e** sem
Nº da Sorte". É o que dá 1 em 6,86.

*(Uma tabela de odds encontrada em fontes secundárias dava "4 acertos = 1 em
211.876". O cálculo direto dá 1 em 8.668. Não a usei.)*

---

## 2. Totoloto vs EuroMilhões

| | Totoloto | EuroMilhões |
|---|---|---|
| odds do 1º prémio | **1 em 24,8M** | 1 em 139,8M |
| preço | **€1,00** | €2,50 |
| **hipóteses de jackpot por €10** | **1 em 2,5M** | 1 em 35M |
| P(algum prémio) | **1 em 6,9** | 1 em 13 |
| jackpot típico | €1M – €19M | €17M – €250M |

Por euro gasto, o Totoloto dá **14 vezes mais hipóteses** de tocar no 1º
prémio. Em troca, o prémio é uma ordem de grandeza menor.

---

## 3. Análise de vieses — 1609 sorteios

### A recolha, e como sei que está certa

A Santa Casa renderiza as estatísticas por JavaScript. As contagens estão
acessíveis noutra fonte, por POST — mas uma extração por expressões
regulares pode facilmente ler a tabela errada e produzir números plausíveis
e falsos.

A verificação que fecha essa porta é aritmética: a soma das contagens dos
números tem de dar **exatamente** 5 × sorteios, e a do Nº da Sorte
**exatamente** 1 × sorteios.

| período | sorteios | Σ números | esperado | Σ Nº da Sorte | esperado |
|---|---:|---:|---:|---:|---:|
| desde 2011-03-16 | 1609 | 8045 | 8045 ✓ | 1609 | 1609 ✓ |
| desde 2022 | 482 | 2410 | 2410 ✓ | 482 | 482 ✓ |
| desde 2023 | 377 | 1885 | 1885 ✓ | 377 | 377 ✓ |
| desde 2024 | 273 | 1365 | 1365 ✓ | 273 | 273 ✓ |
| desde 2025 | 169 | 845 | 845 ✓ | 169 | 169 ✓ |

Batem todas. Há um teste a fixar esta verificação.

### Uniformidade global

| | χ² | g.l. | p | esperado por valor |
|---|---:|---:|---:|---:|
| números 1-49 | 41,00 | 48 | **0,753** | 164,2 |
| Nº da Sorte 1-13 | 11,01 | 12 | **0,528** | 123,8 |

Nada. Ambos perfeitamente compatíveis com sorteio uniforme.

### Valores individuais, com correção de falsas descobertas

Os mais extremos dos 49 números:

| número | saiu | esperado | desvio | p | q (FDR) |
|---:|---:|---:|---:|---:|---:|
| 1 | 133 | 164,2 | −19,0% | 0,009 | 0,462 |
| 12 | 141 | 164,2 | −14,1% | 0,058 | 0,813 |
| 33 | 185 | 164,2 | +12,7% | 0,091 | 0,813 |

E do Nº da Sorte:

| valor | saiu | esperado | desvio | p | q (FDR) |
|---:|---:|---:|---:|---:|---:|
| 11 | 101 | 123,8 | −18,4% | 0,035 | 0,456 |
| 1 | 135 | 123,8 | +9,1% | 0,303 | 0,666 |

**0 de 49** e **0 de 13** sobrevivem à correção. O número 1 sai 19% abaixo
do esperado e o p bruto é 0,009 — mas em 49 testes simultâneos, um valor
desses é o que se espera do acaso. Sem a coluna do FDR, este seria o
"achado" do relatório.

### O teste decisivo — persistência

Um viés físico vive no equipamento e é, por definição, **persistente**: tem
de aparecer em períodos diferentes e correlacionar-se entre eles. Ruído
amostral não faz isso.

Os períodos da fonte são cumulativos; subtraindo-os obtêm-se dois segmentos
que **não partilham um único sorteio**:

| | números 1-49 | Nº da Sorte |
|---|---:|---:|
| 2011-2021 vs 2022-2026 | 1127 vs 482 sorteios | 1127 vs 482 |
| correlação observada | +0,0606 | +0,1349 |
| desvio-padrão do nulo | 0,1439 | 0,2873 |
| **p simulado** | **0,679** | **0,655** |

Zero persistência. Os desvios de um período não dizem nada sobre o
seguinte.

### Potência — que brecha ainda poderia estar escondida

| sorteios | viés mínimo detetável (números) | (Nº da Sorte) |
|---:|---:|---:|
| **1609 (temos)** | **21,3%** | **25,0%** |
| 5.000 | 11,9% | 14,0% |
| 20.000 | 5,9% | 6,9% |

Ao ritmo de 104 sorteios por ano, 20.000 sorteios chegam no ano **2202**.

Isto é o contrapeso honesto ao resultado negativo: não é que não haja viés
nenhum — é que **só um viés superior a 21% seria visível**, e um desvio
dessa magnitude num sorteio supervisionado não é plausível.

---

## 4. Conclusão

| pergunta | resposta |
|---|---|
| Há números "quentes" no Totoloto? | **Não.** χ² p = 0,75; 0 de 49 após FDR |
| Há Nº da Sorte enviesado? | **Não.** χ² p = 0,53; 0 de 13 após FDR |
| Os desvios persistem? | **Não.** r = 0,06, p = 0,68 |
| Poderia haver um viés escondido? | Só se fosse **superior a 21%** |
| O jogo é mais generoso que o EuroMilhões? | **Sim, por euro:** 14× mais hipóteses de jackpot |

O Totoloto comporta-se exatamente como o EuroMilhões nesta dimensão: as
bolas não têm memória, e a porta dos números está fechada por razões
estruturais.

---

## 5. O que falta — e é a parte que vale dinheiro

A vantagem real do sistema do EuroMilhões **não está** na previsão dos
números (que não existe) mas no modelo de popularidade: saber que
combinações as pessoas escolhem, para não ter de dividir o prémio com elas.

Para o Totoloto **esse modelo não está medido**. Exige o número de
vencedores por escalão em cada sorteio — o observável que revela quanta
gente jogou aquela combinação. A Santa Casa não o disponibiliza em formato
acessível.

Enquanto isso não existir, `totoloto.generate()` aplica os princípios
**validados no EuroMilhões** (viés de datas, aversão a consecutivos,
espaçamento) e diz-se explicitamente **transferido, não medido**. O prior do
Nº da Sorte tem um teste que exige que essa limitação continue documentada
— porque já me enganei exatamente assim uma vez: o prior das estrelas errou
por um fator de dois.

Há duas razões para esperar que o efeito seja **maior** no Totoloto:

1. **1-31 cobre 63% dos 49 números** (contra 62% de 50), e o espaço de
   combinações é 10× menor — a concentração das escolhas humanas pesa mais.
2. **Todos os 13 valores do Nº da Sorte são datas plausíveis.** O 13
   acumula duas razões para ser evitado (superstição e estar acima dos 12
   meses), pelo que deverá ser o menos jogado de todos — mas isto é
   dedução, não medição.
