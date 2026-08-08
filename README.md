# Sistema estatístico para o EuroMillions

Análise completa dos **1970 sorteios** do EuroMillions desde o primeiro
(13 de fevereiro de 2004) até 7 de agosto de 2026, com deteção de viés
físico, modelação do comportamento de escolha humana, motor de valor
esperado com fiscalidade portuguesa, e geração de bilhetes a partir de
entropia quântica.

---

## O resumo em três linhas

1. **Não é possível prever os números.** Testámo-lo a sério — 12 famílias
   de testes, correção para testes múltiplos, análise de potência. Zero
   achados. Não por preconceito: por medição.
2. **É possível prever com quantas pessoas vai dividir o prémio.** Isto é
   novo, é mensurável, e valida-se fora da amostra com ρ = 0,411 e
   p = 3,8 × 10⁻²⁵.
3. **Isso vale +35% por aposta — e mesmo assim não chega para lucrar.** O
   valor esperado máximo é €2,01 numa aposta de €2,50.

Os três resultados são verdadeiros ao mesmo tempo. O sistema entrega o
segundo por inteiro e não finge sobre o terceiro.

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

python -m euromillions.cli aleatoriedade       # bateria de testes
python -m euromillions.cli maquinas            # equipamento e viés por segmento
python -m euromillions.cli popularidade        # modelo de escolha humana
python -m euromillions.cli valor               # EV e ponto de equilíbrio
python -m euromillions.cli backtest            # validação fora da amostra
python -m euromillions.cli jogar --jackpot 111e6 --bilhetes 5
```

Exemplo de saída de `jogar`:

```
 1 38 40 41 45   ★ 10 11   popularidade 0.30x   EV €1.276
 3 36 38 42 45   ★ 10 11   popularidade 0.36x   EV €1.232
15 16 33 34 50   ★ 10 12   popularidade 0.37x   EV €1.227

                 estratégia  popularidade   EV_€   ganho_vs_datas
aposta de datas (todos ≤31)         1.977  0.910              0%
   aposta média / aleatória         1.000  1.030          +13.2%
         carteira otimizada         0.367  1.228          +34.9%
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
  popularity.py    GLM de Poisson por IRLS — o núcleo
  ev.py            valor esperado, partilha pari-mutuel, fiscalidade
  quantum.py       QRNG (ANU / LfD) com certificação da fonte
  optimizer.py     filtros ditados pelos dados + seleção por EV
  backtest.py      avaliação honesta, fora da amostra
  cli.py           interface

docs/
  01-metodologia.md      dados, princípios, um erro que cometi
  02-maquinas-e-bolas.md Ryo-Catteau Stresa e Pâquerette
  03-a-brecha-real.md    onde está a brecha e qual é o seu tamanho
  04-resultados.md       todos os números
```

`pytest tests/ -q` → 34 testes. As probabilidades oficiais
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
testados, incluindo o jackpot no teto de €250M e incluindo o cálculo sem
imposto. Jogue apenas o que estiver disposto a perder.

Em caso de dependência do jogo, em Portugal: **SICAD — Linha Vida
1414** (gratuita, todos os dias).
