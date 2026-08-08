# Segunda ronda: mais brechas, e backtests preditivos

Depois da primeira análise, o pedido foi claro: continuar a procurar
brechas, reduzir ao máximo as possibilidades negativas, amplificar as de
ganho, e fazer backtests preditivos. Esta ronda encontrou **duas brechas
novas**, fechou **três hipóteses** com resultado negativo, e corrigiu
**dois erros meus**.

---

## Brecha nova nº 1 — as estrelas eram um palpite, e agora são uma medição

### O que estava mal

A primeira versão tratava as estrelas com uma regra escrita à mão: "as
baixas parecem datas, devem ser mais jogadas". Direção certa, magnitude
errada por um fator de dois, e errada em concreto na estrela 1.

Isto era a parte mais fraca do sistema — e era evitável, porque a
popularidade das estrelas **é observável**.

### O identificador

Entre as apostas que acertaram exatamente `k` números, a repartição por
estrelas deveria seguir, sob escolha uniforme:

```
P(2 estrelas) : P(0 estrelas)  =  1 : C(pool−2, 2)
```

(pool 12 → 1:45; pool 11 → 1:36; pool 9 → 1:21 — a normalização tem de
mudar com a era, e não mudar isto contaminaria as três eras num só número.)

O desvio dessa razão mede diretamente quanta gente escolheu aquele par:

```
π_estrelas ≈ C(pool−2, 2) · W(k,2) / W(k,0)
```

Com k = 2, 3, 4 obtêm-se ~3 observações por sorteio, **independentes do
lado dos números**. São 3090 observações na era atual para estimar 12
parâmetros.

### O resultado

R² = 0,589. Teste conjunto χ² = 4386 (11 g.l.), **p ≈ 0**.

| estrela | popularidade | | estrela | popularidade |
|---:|---:|---|---:|---:|
| **7** | **1,213** | | 4 | 1,024 |
| 5 | 1,138 | | 1 | 0,924 |
| 3 | 1,137 | | 11 | 0,844 |
| 8 | 1,109 | | 10 | 0,819 |
| 2 | 1,084 | | **12** | **0,718** |
| 9 | 1,083 | | | |

O 7 — o "número da sorte" — é a estrela mais sobre-jogada, como a
psicologia previa. E a estrela 1, que o meu palpite dava como popular, é
das menos jogadas.

Pares extremos:

| melhores (impopulares) | | piores (populares) | |
|---|---:|---|---:|
| 10-12 | 0,588 | 5-7 | 1,380 |
| 11-12 | 0,606 | 3-7 | 1,379 |
| 1-12 | 0,664 | 7-8 | 1,345 |

**Rácio 2,35×.** Jogar 5-7 em vez de 10-12 significa 135% mais gente com
quem dividir, com exatamente a mesma probabilidade de sair.

### Validação fora da amostra

Estimado até 2023-08-25, avaliado nos 927 registos seguintes:

```
Spearman ρ = 0,7747     p = 2,5 × 10⁻¹⁸⁶
Pearson (log) = 0,7462  p = 1,3 × 10⁻¹⁶⁵
```

| quartil previsto | popularidade observada |
|---:|---:|
| 1 | 0,788 |
| 2 | 0,908 |
| 3 | 1,097 |
| 4 | 1,265 |

Monótono. **ρ = 0,77 é quase o dobro do sinal do modelo dos números
(0,41)** — as estrelas são muito mais previsíveis, porque o domínio é
pequeno e a psicologia é mais concentrada.

---

## Brecha nova nº 2 — reduzir a probabilidade de não ganhar nada

Esta é a resposta direta a "reduzir as possibilidades negativas ao máximo",
e é independente de tudo o resto.

Uma aposta isolada tem P(algum prémio) ≈ 7,7%. Para N apostas, o resultado
depende de **como elas se relacionam entre si**:

* Bilhetes que partilham números **falham em conjunto** (correlação
  positiva).
* Bilhetes disjuntos cobrem mais universo: quando um falha, o outro tem
  mais hipóteses.

Simulação com 150.000 sorteios, 5 apostas (€12,50):

| carteira | nºs distintos | P(nada) | P(algum prémio) | redução |
|---|---:|---:|---:|---:|
| sobreposta (varia 1 nº) | 9 | 0,8099 ±0,002 | 19,0% | — |
| aleatória | 18 | 0,6906 ±0,002 | 30,9% | 14,7% |
| **disjunta (25 nºs)** | 25 | **0,6415** ±0,002 | **35,9%** | **20,8%** |

De 19,0% para 35,9% de hipótese de ganhar alguma coisa, **pelo mesmo
preço**. Referência teórica se as apostas fossem independentes: 0,6696 — a
carteira disjunta bate-a, porque a cobertura cria correlação negativa.

**Aviso que faz parte do resultado:** isto **não** aumenta o valor
esperado. O EV de N apostas é sempre N × EV(1), qualquer que seja a
sobreposição — a esperança é linear. O que muda é a distribuição: menos
variância, prémios pequenos mais frequentes. Quem quiser maximizar a
hipótese de tocar no jackpot deve ignorar esta alavanca.

---

## Backtest preditivo walk-forward

1030 sorteios reais (2016-09-27 → 2026-08-07) × 400 carteiras
independentes por estratégia = 2,06 milhões de apostas simuladas, avaliadas
contra os **prémios históricos efetivamente pagos** em cada escalão de cada
sorteio.

| estratégia | P(nada) | ±95% | taxa de acerto | ±95% | retorno €/€ | ±95% |
|---|---:|---:|---:|---:|---:|---:|
| datas | 0,6880 | 0,0018 | 0,07673 | 0,00036 | 0,174 | 0,009 |
| sobreposta | 0,7988 | 0,0014 | 0,07709 | 0,00057 | 0,179 | 0,002 |
| aleatória | 0,6683 | 0,0019 | 0,07715 | 0,00037 | 0,260 | 0,123 |
| **otimizada** | **0,6461** | 0,0015 | 0,07656 | 0,00033 | 0,205 | 0,026 |

*(taxa de acerto teórica: 0,07708)*

Três leituras:

1. **A taxa de acerto é a mesma para todas** — todas dentro da margem da
   teórica. Nenhuma estratégia altera a probabilidade de ganhar. Isto é
   agora um facto medido em 2 milhões de apostas contra sorteios reais, não
   uma afirmação de princípio.

2. **P(nada) difere e os intervalos não se sobrepõem.** A carteira
   otimizada sai de mãos vazias 0,6461 das vezes contra 0,7988 da carteira
   sobreposta — o erro mais comum na prática.

3. **Jogar datas custa duas vezes.** P(nada) = 0,688, pior do que aleatório
   (0,668), porque restringir-se a 1–31 encolhe a cobertura. Além de
   dividir mais o prémio, ganha-se menos vezes.

### O que este backtest NÃO consegue medir — e porquê

O retorno €/€ tem margens enormes (a "aleatória" dá 0,260 ±0,123) e as
estratégias são indistinguíveis nessa coluna. Isso é esperado e tem uma
razão de fundo:

> O prémio histórico de cada escalão é um número fixo, **já dividido pelos
> vencedores que realmente existiram**. O histórico não pode revelar quanto
> teríamos recebido com uma combinação diferente.

A vantagem de popularidade vive quase toda nos escalões altos, que num
backtest de 1030 sorteios nunca são atingidos. Ela está validada noutro
sítio e de outra forma: em `popularity_out_of_sample()` e em
`stars.validate_out_of_sample()`, que preveem, fora da amostra, quantas
pessoas partilham cada combinação.

Confundir as duas coisas levaria a concluir — erradamente — que a vantagem
não existe, só porque este teste em particular não a consegue ver.

---

## Hipóteses testadas e fechadas

### Jackpots grandes atraem apostadores mais previsíveis?

Seria muito útil: bastaria jogar só quando o jackpot está alto. **Não se
confirma.** O rácio datas/não-datas até desce ligeiramente à medida que as
vendas sobem:

| vendas | mediana | rácio datas/não-datas |
|---|---:|---:|
| baixo | 18,4M | 1,387 |
| médio | 24,0M | 1,347 |
| alto | 35,0M | 1,293 |

Spearman(vendas, popularidade) = −0,005, p = 0,82. Não há sinal.
**Não há vantagem em esperar por jackpots grandes** para explorar o viés.
(Há vantagem em jogar jackpots grandes por outra razão — o EV sobe com o
prémio — mas isso já estava no motor de EV.)

### O efeito está a desaparecer?

Esta era a ameaça séria: se a adoção de apostas aleatórias estivesse a
crescer, a brecha fechar-se-ia. **Não está.**

| período | sorteios | rácio | p |
|---|---:|---:|---:|
| 2004-2010 | 326 | 1,372 | 2,8 × 10⁻⁷ |
| 2011-2015 | 503 | 1,247 | 7,0 × 10⁻⁶ |
| 2016-2020 | 522 | 1,301 | 7,7 × 10⁻⁸ |
| **2021-2026** | **585** | **1,303** | **1,6 × 10⁻⁸** |

Estável há 22 anos, e ainda altamente significativo no período mais
recente. A brecha está aberta hoje.

### O perfil de apostador muda entre terça e sexta?

Não. Rácio 1,319 (terça) vs 1,331 (sexta). Nada a explorar.

---

## Dois erros meus, corrigidos

### 1. Barras de erro demasiado estreitas no backtest

A primeira versão calculava o intervalo de confiança de P(nada) como se os
309.000 pares sorteio×carteira fossem independentes. **Não são**: os 1030
sorteios enfrentados por uma mesma carteira partilham os mesmos números. A
unidade independente é a carteira, e havia apenas 300.

Corrigido para erros-padrão agrupados por réplica. As margens corretas são
várias vezes maiores — e as conclusões sobrevivem, o que só se sabe depois
de as calcular bem.

### 2. O otimizador tinha as duas alavancas a lutar uma contra a outra

Ao ordenar os candidatos só por valor esperado, a carteira ficava com
**19 números distintos** e P(nada) = 0,687 — **pior do que jogar ao acaso**.

A razão: as combinações impopulares concentram-se nos números altos (>31),
por isso os melhores bilhetes individuais repetem os mesmos números entre
si. EV ótimo por bilhete, carteira péssima.

A correção é hierárquica: restringir primeiro ao quantil mais impopular (é
aí que está o valor, e dentro dele o EV varia pouco), e só depois maximizar
a cobertura. Resultado: 25 números distintos, P(nada) = 0,6461, e a
vantagem de popularidade praticamente intacta (+35,8% contra +38,0%).

Custou 2 pontos percentuais de vantagem no cheque para ganhar 4 pontos
percentuais na probabilidade de ganhar alguma coisa. É uma troca boa, e é
uma escolha explícita — `maximize_coverage=False` reverte-a.

---

## O estado atual do sistema

Carteira gerada com jackpot de €111M e vendas de 24M:

```
 4 34 40 41 50   ★ 10 12   popularidade 0.35x
16 17 22 48 50   ★ 10 12   popularidade 0.37x
 2 38 40 41 49   ★  8 12   popularidade 0.41x
```

| estratégia | popularidade | EV | fatia do jackpot | ganho |
|---|---:|---:|---:|---:|
| aposta de datas | 2,351 | €0,878 | 82,3% | — |
| aposta média | 1,000 | €1,030 | 91,9% | +17,3% |
| **otimizada** | **0,429** | **€1,193** | **96,4%** | **+35,8%** |

Cobertura: 25/50 números. P(nada) = 0,6461 contra 0,6696 se fossem
independentes.

**E continua a não chegar.** O valor esperado de 5 apostas é €5,98 contra
€12,50 de custo. A brecha vale +36% num jogo que paga 40%. Somar as duas
alavancas melhora muito a experiência de jogar — e não transforma um jogo
de soma negativa num investimento.

Isso não é uma limitação deste sistema. É o EuroMillions.
