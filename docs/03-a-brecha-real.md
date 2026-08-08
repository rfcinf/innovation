> ⚠️ **Números corrigidos em [`docs/07-correcoes.md`](07-correcoes.md).**
> O M1lhão estava ausente do motor de EV e as elasticidades por escalão
> eram um palpite. A vantagem real da otimização é **+7,8%**, não os
> valores indicados abaixo. Onde houver conflito, vale o `docs/07`.

# A brecha

Foi-me pedido que usasse todos os fatores, negativos e positivos, para
encontrar uma brecha que nunca tivesse sido encontrada. Encontrei uma
brecha. Não está onde se costuma procurar, e é por isso que sobrevive.

## Onde a brecha não está

Toda a gente procura no mesmo sítio: nos números. Quentes, frios,
atrasados, ciclos, somas, geometria do boletim. Este projeto testou tudo
isso com 1970 sorteios e um aparato estatístico sério, e o resultado é
inequívoco (`docs/04-resultados.md`).

A razão profunda não é a "falácia do jogador" — é uma questão de
**conservação de informação**. Para que o passado ajudasse a prever o
futuro, teria de existir um canal físico a transportar informação de um
sorteio para o seguinte. As bolas voltam ao cofre. A máquina é
reinicializada. O único canal possível seria o desgaste do equipamento — e
esse foi investigado a fundo em `docs/02-maquinas-e-bolas.md`, com
resultado negativo e com a razão estrutural pela qual permanecerá negativo.

Não é pessimismo. É a leitura correta de onde a informação existe e onde
não existe.

## Onde a brecha está

O EuroMillions tem duas variáveis, e quase toda a gente só olha para uma.

```
        valor da aposta  =  P(acertar)  ×  E[cheque | acertar]
                            ^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^^
                             blindada          EXPOSTA
```

`P(acertar)` é 1 em 139.838.160 e não se move. Nunca se moveu, para
ninguém, em 22 anos.

`E[cheque | acertar]` **move-se**, e move-se muito, porque o EuroMillions é
um jogo *pari-mutuel*: o prémio de cada escalão é um bolo fixo dividido
pelos vencedores desse escalão.

> Acertar nos números certos que 40 outras pessoas também escolheram vale
> um quadragésimo de acertar nos números certos que mais ninguém escolheu.
>
> **A probabilidade dos dois eventos é exatamente a mesma.**

Isto não é uma teoria sobre o comportamento das bolas. É aritmética das
regras do jogo. E, ao contrário de tudo o resto neste domínio, é
**mensurável e verificável**.

## Porque é que ninguém explora isto

Porque parece que não há dados. A intuição diz que não podemos saber que
números as outras pessoas jogaram — a Santa Casa não publica isso.

A intuição está errada, e é aqui que está a descoberta operacional deste
projeto:

> Os escalões **5+2, 5+1 e 5+0** abrangem *todas* as apostas que acertaram
> os cinco números, seja qual for a combinação de estrelas. A sua soma é,
> portanto, uma contagem direta e exata de quantas pessoas jogaram
> **aquela combinação concreta de cinco números**.

Isto é publicado. Sorteio a sorteio. Desde 2004.

Cada sorteio é, então, uma experiência natural: o universo escolheu uma
combinação ao acaso, e o mercado dos apostadores revelou quantas pessoas a
tinham escolhido. Com 1936 sorteios com quebra de prémios recolhida, temos
**1936 medições independentes do comportamento coletivo de escolha**.

Normalizando pelo volume de vendas estimado:

```
popularidade = vencedores_de_5_números / (vendas ÷ 2.118.760)
```

`popularidade = 1` significa "combinação tão jogada como o acaso mandaria".
`popularidade = 1.6` significa que 60% mais gente a escolheu — e que o
cheque, se sair, se divide por 60% mais gente.

## A medição

Sem qualquer modelo pelo meio, apenas agrupando os 1936 sorteios pelo
número de bolas ≤ 31 (as compatíveis com dias de aniversário):

| números ≤31 na combinação | sorteios | popularidade média |
|---:|---:|---:|
| 1 | 82 | 0,91 |
| 2 | 418 | 0,94 |
| 3 | 737 | 1,11 |
| 4 | 532 | 1,15 |
| 5 | 157 | **1,62** |

Monótono. E o teste formal:

```
≥4 números ≤31  vs  ≤2 números ≤31
rácio das médias        1,32
Mann-Whitney            p = 7,1 × 10⁻²⁰
```

Um p-valor de 10⁻²⁰ não é uma sugestão. É um dos efeitos mais sólidos que
se conseguem medir em dados de comportamento humano.

E a partilha não é hipotética. Nos 438 sorteios em que o jackpot saiu:

| vencedores | sorteios | % |
|---:|---:|---:|
| 1 | 369 | 84,2% |
| 2 | 55 | 12,6% |
| 3 | 8 | 1,8% |
| 4 | 3 | 0,7% |
| 5 | 3 | 0,7% |

**15,8% dos jackpots foram divididos.** Cada uma dessas divisões foi, em
parte, uma consequência de as pessoas escolherem os mesmos números.

## O modelo, e a sua validação

Um GLM de Poisson com offset (`popularity.py`) estima o efeito de doze
características da combinação. Estimado em 1355 sorteios até janeiro de
2021, e avaliado nos 581 sorteios seguintes, **que nunca viu**:

```
Spearman ρ = 0,411      p = 3,8 × 10⁻²⁵
Pearson (log) r = 0,419  p = 3,7 × 10⁻²⁶
```

Popularidade **observada**, por quintil de popularidade **prevista**:

| quintil previsto | popularidade observada |
|---:|---:|
| 1 (mais impopular) | 0,775 |
| 2 | 0,825 |
| 3 | 0,851 |
| 4 | 0,981 |
| 5 (mais popular) | **1,314** |

Perfeitamente monótono, fora da amostra. O modelo sabe genuinamente prever
com quantas pessoas vamos ter de dividir o prémio. A diferença entre os
extremos é **1,7×** — um cheque **70% maior** para a mesma probabilidade de
acertar.

## O que os dados corrigiram na minha intuição

Escrevi a primeira versão dos filtros com o folclore habitual das apostas:
proibir números consecutivos, controlar a soma. O modelo desmentiu-me, e a
correção ficou no código:

- **`n_consecutivos`: β = −0,060, p < 0,0001.** Combinações com pares
  consecutivos são *menos* populares. As pessoas acham que 33-34 "não
  parece aleatório" e evitam-no — o que os torna sub-jogados e portanto
  **bons**. Proibi-los, como manda a intuição, seria deitar fora as
  melhores combinações.

- **`espaco_regular`: β = −0,130, p < 10⁻¹¹.** O preditor mais forte de
  todos. As pessoas preferem combinações visualmente regulares no boletim.
  Espaçamento irregular é impopular, logo valioso.

- **`soma_norm`: p = 0,55.** Sem significância nenhuma. O filtro da soma
  era superstição e foi removido.

Os testes em `tests/test_core.py` fixam estas correções para que a intuição
não volte a entrar pela porta das traseiras.

## O papel da entropia quântica

Foi pedido que se usasse imprevisibilidade quântica. Ela tem aqui uma
função real, e não decorativa — mas convém ser exato sobre qual é.

A entropia quântica **não altera** `P(acertar)`. Nada altera.

O que ela faz é garantir que a nossa escolha é **estatisticamente
independente da psicologia humana**. Toda a vantagem descrita acima vem de
escolher combinações que os outros não escolhem. Ora, qualquer processo de
escolha com um humano no circuito está correlacionado com o que os outros
humanos fazem — é exatamente esse o efeito que medimos com p = 10⁻²⁰.

Flutuações do vácuo quântico não têm nenhum canal causal que as ligue às
preferências da população de apostadores. É a forma mais limpa de obter
propostas descorrelacionadas do comportamento coletivo.

```
entropia quântica  →  propõe combinações sem viés humano
modelo estatístico →  filtra as que a multidão prefere
motor de EV        →  ordena pelo cheque esperado
```

Há ainda uma segunda função, defensiva. Se este método se tornasse popular,
deixaria de funcionar — combinações "impopulares" passariam a ser jogadas
por toda a gente que corresse o mesmo código. A aleatoriedade quântica da
proposta protege contra isso: dois utilizadores com sementes diferentes
obtêm bilhetes diferentes, ambos com popularidade igualmente baixa.

## O tamanho honesto da brecha

Aqui está a parte que nenhum sistema de lotaria à venda lhe dirá, e que é o
resultado mais importante deste projeto.

A brecha é real, é mensurável, e **não chega**.

Com vendas de 24M de apostas por sorteio e prémios medianos observados
desde 2020:

| jackpot | EV combinação popular | EV média | EV otimizada | ganho |
|---:|---:|---:|---:|---:|
| €17M | €0,41 | €0,54 | €0,72 | +34% |
| €100M | €0,78 | €0,97 | €1,18 | +21% |
| €250M (teto) | €1,45 | €1,76 | **€2,01** | +14% |

A aposta custa **€2,50**.

Mesmo no teto absoluto de €250M, mesmo com a combinação mais impopular que
o sistema consegue construir, o valor esperado é **€2,01**. O ponto de
equilíbrio é **inatingível** — e continua inatingível mesmo removendo o
Imposto do Selo do cálculo.

Três forças fecham a porta, e é instrutivo ver como se combinam:

1. **A margem estrutural.** Só ~50% das vendas volta como prémios. Começa-se
   com 50% de desvantagem, antes de tudo o resto.
2. **O teto de €250M.** A acumulação não pode crescer indefinidamente até
   compensar. Acima do teto, o excedente escorre para os escalões
   inferiores.
3. **O Imposto do Selo.** 20% sobre tudo o que exceda €5.000 — uma taxa
   efetiva de 19,9% num prémio de €1M. Portugal é um dos três países que
   tributa o EuroMillions, e o mais rígido.

A brecha vale +14% a +35%. A desvantagem estrutural é de −60% a −80%. A
brecha é real e é insuficiente, e essas duas coisas são verdadeiras ao
mesmo tempo.

## O que isto significa na prática

Não existe jogo perfeito no sentido de um jogo lucrativo. Isso não é uma
limitação deste sistema — é uma propriedade do EuroMillions, e o sistema
serve precisamente para o demonstrar em vez de o assumir.

Existe, isso sim, um **jogo ótimo**: dado que se vai jogar, há uma forma de
jogar que vale mais 35% do que a forma como a maioria das pessoas joga,
pelo mesmo preço e com a mesma probabilidade de acertar.

É a diferença entre transformar €2,50 em €3,00 (impossível) e transformar
uma perda esperada de €1,59 numa perda esperada de €1,27 — com um prémio
substancialmente maior no caso improvável de acertar.

É a única vantagem que existe. Este sistema entrega-a, inteira, e não
finge que é maior do que é.
