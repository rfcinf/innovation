# A noite inteira, não o valor esperado

```bash
python -m euromillions.cli simulacao --jackpot 111e6 --tamanhos 1,2,5,10,20
```

O valor esperado é o número que menos informa sobre uma noite de sorteio.
Diz quanto se ganharia *em média* se a mesma noite se repetisse um milhão
de vezes. Mas a noite acontece **uma vez**, e a média cai num ponto onde a
distribuição quase nunca aterra.

Com 5 apostas otimizadas e um jackpot de €111M, o valor esperado é ~€6,70.
A probabilidade de ganhar exatamente €6,70 é **zero**. O que acontece de
facto é: €0 em 64% das noites, entre €3 e €10 em quase todo o resto, e uma
cauda que vale dezenas de milhões com probabilidade de 1 em 28 milhões.

Dizer "o valor esperado é €6,70" é verdade e é inútil. Este documento
descreve o que o substitui.

---

## As três partes, e porquê separá-las

Uma única simulação de Monte Carlo não consegue cobrir a distribuição
toda, e fingir que consegue é o erro que este módulo evita.

### 1. O corpo — por simulação

`simulate_night()` sorteia centenas de milhares de noites e mede o que
acontece. É aqui que a noite realmente aterra: P(não ganhar nada), os
percentis do ganho, P(recuperar o custo).

Porquê simular em vez de calcular? Porque **os bilhetes partilham o mesmo
sorteio** e estão portanto correlacionados. A fórmula de independência,
`P(nada) = (1−p)^n`, dá 0,6696 para 5 apostas. A realidade de uma carteira
com números disjuntos é 0,6425 — melhor, porque bilhetes que não repetem
números não podem falhar todos da mesma maneira. Essa diferença de 2,7
pontos percentuais é grátis, e uma fórmula de independência não a vê.

A simulação é vetorizada por multiplicação de matrizes booleanas: constrói
a matriz de pertença dos sorteios (n_sim × 50) e multiplica pela dos
bilhetes (n_bilhetes × 50). O produto dá diretamente o número de acertos
de cada bilhete em cada noite, sem um único ciclo de Python.

### 2. A cauda — por combinatória exata

Os escalões de 5 acertos têm probabilidade de 1 em milhões. Uma simulação
de 400 mil noites **nunca os vê** — e se por acaso visse um, esse único
acerto deslocaria a média simulada em vários euros e seria lido como
"diferença entre carteiras" quando é ruído puro.

Por isso a cauda não se simula: calcula-se. O número esperado de acertos
num escalão é `n × p`, exato por linearidade da esperança, **sem supor
nada sobre a correlação entre bilhetes**.

### Um detalhe que torna `n × p` exato, e não aproximação

Para escalões de **3 ou mais acertos**, dois bilhetes com números
disjuntos **não podem ganhar os dois na mesma noite**: seriam precisos 6
números sorteados distintos e só saem 5.

Logo, nesses escalões, `P(pelo menos um bilhete acerta) = n × p`
exatamente — e não `1 − (1−p)^n`. A coluna `exclusivo` da tabela marca
quais os escalões em que isto se aplica.

### 3. O cheque — o que a cauda paga mesmo

O jackpot anunciado não é o que se recebe. Duas coisas acontecem entre um
e outro:

```
  anunciado        €111.000.000
  ÷ (1 + K)        K ~ Poisson(λ),  λ = vendas × p_jackpot × popularidade
  − Imposto Selo   20% sobre o que exceder €5.000
  = cheque
```

**A ordem importa.** O imposto incide sobre o que *cada vencedor* recebe,
não sobre o prémio bruto. Aplicá-lo antes de dividir sobrestimaria a carga
fiscal — e há um teste que fixa esta ordem, precisamente porque é o género
de detalhe que se inverte sem dar por isso.

`jackpot_check()` calcula `E[líquido(J / (1+K))]` somando sobre a
distribuição de Poisson, em vez de aplicar o líquido ao valor médio. Com
`λ = 0,073` (a carteira otimizada, que é impopular), P(ficar sozinho) é
**0,93** — e é aí que está toda a diferença entre esta carteira e jogar
datas, onde λ é ~20 vezes maior.

---

## O trade-off que não se pode otimizar nos dois sentidos

Esta é a descoberta que a simulação trouxe e que o valor esperado escondia
por completo.

Cobrir muitos números distintos (**dispersar**) e fazer os bilhetes
partilharem quase todos os números (**concentrar**) são objetivos em
oposição direta. Medido sobre 400 mil noites, 5 apostas, jackpot €111M:

| | dispersa (25 nºs) | concentrada (9 nºs) |
|---|---:|---:|
| popularidade | 0,456 | **0,309** |
| P(não ganhar nada) | **0,6446** | 0,8302 |
| P(recuperar os €12,50) | 0,0142 | **0,0540** |
| P(ganhar ≥ €25) | 0,0012 | **0,0212** |
| P(ganhar ≥ €100) | 0,00014 | **0,00042** |
| percentil 95 | €8,04 | **€16,08** |
| percentil 99 | €14,18 | **€31,26** |
| cheque se acertar | €85,5M | **€86,5M** |
| P(sozinho \| acertar) | 0,928 | **0,949** |
| valor esperado | €6,40 | €6,44 |

O mecanismo é a correlação entre bilhetes. Se os números partilhados
saírem, **vários bilhetes ganham ao mesmo tempo** e os prémios somam-se;
se não saírem, nenhum ganha. Dispersar faz o contrário: espalha as
hipóteses, o que reduz as noites a zero mas também torna improvável um
pagamento concentrado.

O valor esperado é praticamente igual nas duas — €6,40 contra €6,44 — e é
exatamente por isso que é o número que menos informa. **A esperança é
linear e não vê correlação nenhuma.** Toda a diferença entre as duas
carteiras vive em momentos da distribuição que o valor esperado não
regista.

Note-se que a carteira concentrada perde **só** em `P(não ganhar nada)`.
Ganha em popularidade, nos percentis altos, no cheque e na probabilidade
de ficar sozinha — porque é construída a partir do melhor bilhete e dos
seus vizinhos mais impopulares, em vez de sacrificar qualidade para cobrir
o tabuleiro.

Não há resposta estatística para qual é melhor. Há uma preferência, e
agora está quantificada.

---

## O que a comparação entre orçamentos mostra

A tabela do `sweep` compara carteiras de 1, 2, 5, 10 e 20 apostas na mesma
noite. A coluna a ler primeiro é **`€_por_€`**: mantém-se praticamente
constante. **Nenhum orçamento torna o jogo favorável** — a esperança é
linear, e gastar o dobro devolve o dobro do mesmo retorno negativo.

O que muda com o orçamento é a **forma** da distribuição:

- `P(nada)` cai — mais números distintos cobertos, menos noites a zero;
- `1_em_jackpot` cai proporcionalmente ao número de apostas;
- `cheque_€M` **também cai ligeiramente** — e esta é a tensão real: à
  medida que a carteira cresce, esgotam-se as combinações verdadeiramente
  impopulares e a popularidade média sobe, o que aumenta λ e reduz o
  cheque no caso de acertar.

Ou seja: mais apostas compram mais bilhetes na rifa e um prémio um pouco
mais partilhado. Não há orçamento "ótimo" no sentido de maximizar retorno
— há um que enquadra a distribuição que se prefere.

---

## O que isto não é

Não melhora a probabilidade de acertar. A probabilidade do 1º prémio por
bilhete é 1 em 139.838.160, seja qual for a combinação, e este módulo não
lhe toca. O que faz é mostrar a distribuição inteira em vez de um único
número — para que a decisão seja tomada sobre o que vai mesmo acontecer, e
não sobre uma média que nunca acontece.
