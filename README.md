# Sistema estatístico para o EuroMillions

Análise completa dos **1970 sorteios** do EuroMillions desde o primeiro
(13 de fevereiro de 2004) até 7 de agosto de 2026, com deteção de viés
físico, modelação do comportamento de escolha humana, motor de valor
esperado com fiscalidade portuguesa, e geração de bilhetes a partir de
entropia quântica.

---

## O resumo em quatro linhas

1. **Não é possível prever os números.** 12 famílias de testes, correção
   para testes múltiplos, análise de potência, e um backtest de 2 milhões
   de apostas contra sorteios reais. Zero achados. Não por preconceito: por
   medição.
2. **É possível prever com quantas pessoas vai dividir o prémio.** Nos
   números (ρ = 0,411, p = 3,8 × 10⁻²⁵) e, muito mais forte, nas estrelas
   (ρ = 0,775, p = 2,5 × 10⁻¹⁸⁶). Ambos validados fora da amostra.
3. **É possível reduzir a probabilidade de não ganhar nada** — de 0,80 para
   0,646, sem gastar mais um cêntimo.
4. **Tudo somado vale +7,8% por aposta — e não chega para lucrar.** O
   retorno da melhor estratégia possível é €0,41 por cada €1 apostado.

Os quatro resultados são verdadeiros ao mesmo tempo. O sistema entrega os
três primeiros por inteiro e não finge sobre o quarto.

> A vantagem chegou a ser anunciada como +36%. Estava errada: faltava o
> M1lhão no motor de EV e as elasticidades por escalão eram um palpite meu.
> Corrigido em [`docs/07-correcoes.md`](docs/07-correcoes.md), com a
> vantagem a cair para +7,8%.

### As três alavancas

| alavanca | o que faz | ganho medido | validação |
|---|---|---:|---|
| popularidade dos números | cheque maior | ver abaixo | ρ = 0,411 fora da amostra |
| popularidade das estrelas | cheque maior | par 2,35× | ρ = 0,775 fora da amostra |
| cobertura da carteira | ganhar mais vezes | P(nada) 0,80 → 0,646 | 1030 sorteios reais |

Combinadas: **+7,8% no valor esperado** e **P(algum prémio) de 19% para
36%**, ao mesmo preço e com a mesma probabilidade de jackpot.

Cerca de 21% do valor de um bilhete português vem do **M1lhão**, onde o
código é gerado pelo sistema e não há nada a otimizar — é o que faz a
vantagem percentual encolher.

### Modelo vs apostas avulsas, 1970 sorteios reais

Confronto com todo o histórico desde 2004 — 2,46 milhões de apostas por
estratégia ([`docs/06`](docs/06-comparativo-historico.md)):

| | acerta mais? | EV (jackpot €60M, com M1lhão) | P(nada) |
|---|---|---:|---:|
| datas | não | €0,942 | 0,677 |
| aleatória | não | €0,974 | 0,663 |
| **otimizada** | não | **€1,016 (+7,8%)** | **0,639** |

O jackpot **nunca saiu** — em nenhuma estratégia, em 22 anos simulados 250
vezes. Jogando 5 apostas por sorteio, esperar um jackpot leva **269 mil
anos**.

### E as "lacunas"? Já saiu a mesma combinação duas vezes?

Sim: **4-30-31-38-42**, a 02-05-2014 e a 31-08-2018 — quando o esperado
eram 0,92 repetições. Comparando os 1.939.465 pares de sorteios da
história em todos os graus de coincidência: **χ² = 1,33, p = 0,93**.
Concordância quase perfeita com o acaso puro
([`docs/08`](docs/08-lacunas-e-padroes.md)).

Das 74 classes de padrões que "nunca aconteceram", **zero** têm ausência
informativa — nenhuma deveria ter saído sequer 3 vezes.

### Modelo de produção e auditor

`modelo` consolida os cinco componentes estimados e declara a origem de
cada número que usa — oficial, medido ou assumido. Restam três suposições,
todas assinaladas.

`auditoria` verifica o sistema em seis áreas (dados, modelos, deriva,
suposições, afirmações publicadas, lacunas), revalida tudo fora da amostra
e sai com código 1 se houver problemas críticos. Estado atual: **aprovado
com reservas, 5 avisos** ([`docs/09`](docs/09-modelo-e-auditor.md)).

---

## A ideia

O EuroMillions tem duas variáveis, e quase toda a gente só ataca uma:

```
     valor da aposta  =  P(acertar)  ×  E[cheque | acertar]
                         ^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^^
                          blindada          EXPOSTA
```

`P(acertar)` = 1 em 139.838.160. Não se move. Não se moveu para ninguém em
22 anos, e a razão é estrutural, não circunstancial.

`E[cheque | acertar]` **move-se**. O EuroMillions é *pari-mutuel*: o prémio
é um bolo dividido pelos vencedores. Acertar nos números que outras 40
pessoas escolheram vale 1/40 de acertar nos números que mais ninguém
escolheu — **com exatamente a mesma probabilidade**.

A descoberta operacional que torna isto explorável: os escalões 5+2, 5+1 e
5+0, somados, contam **exatamente** quantas pessoas jogaram aquela
combinação de cinco números. É publicado, sorteio a sorteio, desde 2004.
São 1936 medições diretas do comportamento coletivo de escolha.

Detalhes em [`docs/03-a-brecha-real.md`](docs/03-a-brecha-real.md).

---

## Utilização

```bash
pip install -r requirements.txt
export PYTHONPATH=src

python -m euromillions.cli fetch --breakdown   # recolha (~15 min)
python -m euromillions.cli relatorio           # corre tudo

python -m euromillions.cli modelo              # modelo consolidado + recomendação
python -m euromillions.cli simulacao           # distribuição completa de uma noite
python -m euromillions.cli auditoria           # auditar o sistema e caçar lacunas

python -m euromillions.cli aleatoriedade       # bateria de testes
python -m euromillions.cli maquinas            # equipamento e viés por segmento
python -m euromillions.cli popularidade        # modelo de escolha humana (números)
python -m euromillions.cli estrelas            # popularidade medida das estrelas
python -m euromillions.cli carteira            # reduzir P(não ganhar nada)
python -m euromillions.cli padroes             # repetições e o mapa do espaço
python -m euromillions.cli m1lhao              # a parcela portuguesa do EV
python -m euromillions.cli elasticidade        # elasticidades medidas
python -m euromillions.cli comparativo         # modelo vs avulso, todo o histórico
python -m euromillions.cli valor               # EV e ponto de equilíbrio
python -m euromillions.cli backtest            # walk-forward + fora da amostra
python -m euromillions.cli jogar --jackpot 111e6 --bilhetes 5
```

Exemplo de saída de `jogar`:

```
 4 34 40 41 50   ★ 10 12   popularidade 0.35x   EV €1.237
16 17 22 48 50   ★ 10 12   popularidade 0.37x   EV €1.224
 2 38 40 41 49   ★  8 12   popularidade 0.41x   EV €1.204

                 estratégia  popularidade   EV_€   ganho_vs_datas
aposta de datas (todos ≤31)         2.351  0.878              0%
   aposta média / aleatória         1.000  1.030          +17.3%
         carteira otimizada         0.429  1.193          +35.8%

  números distintos cobertos : 25 de 50 (50.0%)
  P(não ganhar nada)         : 0.6461
```

---

## Estrutura

```
src/euromillions/
  config.py        matriz, eras, 13 escalões, Imposto do Selo
  fetch.py         recolha incremental (sorteios + quebra de prémios)
  dataset.py       normalização, validação, reconstrução de vendas
  randomness.py    12 famílias de testes + FDR + potência
  machines.py      equipamento, viés por segmento, custo da diluição
  popularity.py    GLM de Poisson por IRLS — popularidade dos números
  stars.py         popularidade medida das estrelas (R²=0,59, 3090 obs.)
  m1lhao.py        a parcela portuguesa do EV (~21% do valor do bilhete)
  elasticity.py    elasticidades por escalão, medidas e não arbitradas
  patterns.py      enumeração das 2.118.760 combinações e teste de padrões
  model.py         modelo consolidado, com proveniência declarada de cada input
  audit.py         auditor: revalida, deteta deriva e caça lacunas
  portfolio.py     cobertura: minimizar P(não ganhar nada)
  ev.py            valor esperado, partilha pari-mutuel, fiscalidade
  quantum.py       QRNG (ANU / LfD) com certificação da fonte
  optimizer.py     filtros ditados pelos dados + EV e cobertura em conjunto
  backtest.py      walk-forward sobre sorteios reais + validação fora da amostra
  simulacao.py     distribuição de uma noite: corpo simulado + cauda exata
  cli.py           interface

docs/
  01-metodologia.md      dados, princípios, um erro que cometi
  02-maquinas-e-bolas.md Ryo-Catteau Stresa e Pâquerette
  03-a-brecha-real.md    onde está a brecha e qual é o seu tamanho
  04-resultados.md       todos os números
  05-segunda-ronda.md    estrelas, carteira, backtest walk-forward
  06-comparativo-historico.md  modelo vs apostas avulsas, 1970 sorteios
  07-correcoes.md        M1lhão, elasticidades medidas, uma hipótese falhada
  08-lacunas-e-padroes.md  repetições, coincidências, o mapa exato do espaço
  09-modelo-e-auditor.md   o modelo de produção e o sistema que o verifica
  10-totoloto.md           Totoloto: estrutura validada e análise de vieses
  11-simulacao.md          a noite inteira, em vez do valor esperado
```

`pytest tests/ -q` → 85 testes. As probabilidades oficiais
(1 em 139.838.160) são calculadas de raiz e verificadas.

---

## Sobre o rigor

Foi-me pedido que ignorasse a "falácia do jogador" e que procurasse padrões
onde não se encontram. Fiz as duas coisas, no sentido que as torna úteis:
não descartei nada por preconceito, procurei em sítios onde ninguém procura
(comportamento dos apostadores, não comportamento das bolas), e **usei os
fatores negativos com os positivos** — a partilha, o teto de €250M e o
imposto entram no mesmo cálculo que a acumulação do jackpot, porque é do
cruzamento deles que sai a resposta.

O que não fiz foi baixar a fasquia da prova. Três decisões que valeu a pena
tomar:

- **Correção para testes múltiplos em tudo.** Com 50 números, 1225 pares e
  71 janelas, "anomalias a p < 0,05" aparecem às centenas por construção.
  Sem controlo de FDR, eu poderia ter-lhe entregado uma dúzia de
  descobertas — todas ruído.

- **Corrigi um erro meu em vez de o publicar.** A primeira versão do teste
  de intervalos usava Kolmogorov-Smirnov e dava **47 de 50 números
  significativos**. Teria sido a descoberta do projeto. Era um teste
  contínuo aplicado a dados discretos. Com o teste correto: **0 de 50**. O
  episódio ficou documentado em `randomness.gap_analysis()` porque é o
  mecanismo exato pelo qual nascem os sistemas de lotaria que prometem
  padrões.

- **Deixei os dados corrigirem a minha intuição.** Escrevi filtros que
  proibiam números consecutivos, como manda o folclore. O modelo mostrou
  que consecutivos são *sub-jogados* (β < 0, p < 0,0001) e portanto
  valiosos. Os filtros foram reescritos ao contrário, e há testes a impedir
  que a intuição volte a entrar.

- **Corrigi barras de erro que me favoreciam.** O backtest tratava 309.000
  pares sorteio×carteira como independentes, quando só havia 300 carteiras
  independentes. As margens corretas são várias vezes maiores. As
  conclusões sobreviveram — mas isso só se sabe depois de as calcular bem.

- **Corrigi um erro factual que me favorecia, ao ser questionado.** Faltava
  o M1lhão no motor de EV (~21% do valor de um bilhete português) e as
  elasticidades por escalão eram um palpite inflacionado. Ao corrigir
  ambos, a vantagem anunciada caiu de +36% para +7,8%. Testei também uma
  hipótese minha para melhorar o modelo — falhou, e está documentada como
  resultado negativo em `docs/07`.

- **Apanhei o otimizador a sabotar-se.** Ordenar candidatos só por valor
  esperado dava carteiras com 19 números distintos e P(nada) *pior do que
  jogar ao acaso*: as combinações impopulares concentram-se nos números
  altos, e os melhores bilhetes repetiam-se entre si. As duas alavancas
  estão agora otimizadas em conjunto, hierarquicamente.

O número 22 saiu 21,8% abaixo do esperado, com um défice que se agrava
monotonamente ao longo de quatro décadas de sorteios. Parece uma bola
gasta. Não é: é o mais extremo de 50 números, e o teste de persistência
fora da amostra (r = 0,119, p = 0,408) mostra que o sinal não sobrevive.
Um sistema menos honesto tinha-lhe vendido isso.

---

## Aviso

Este projeto não aumenta — nem pode aumentar — a probabilidade de ganhar,
que é 1 em 139.838.160 por aposta. O que faz é maximizar o prémio no caso
improvável de acertar, e quantificar com honestidade o custo de jogar.

O valor esperado de uma aposta é **sempre negativo**, em todos os cenários
testados — incluindo o jackpot no teto de €250M, incluindo o cálculo sem
imposto, e incluindo já a parcela do M1lhão. Jogue apenas o que estiver disposto a perder.

Em caso de dependência do jogo, em Portugal: **SICAD — Linha Vida
1414** (gratuita, todos os dias).
