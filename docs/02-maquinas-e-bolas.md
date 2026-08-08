# O equipamento físico

## O que está apurado

| Item | Facto |
|---|---|
| Local | Paris, França — sorteio conduzido pela Française des Jeux (FDJ) |
| Horário | Terças e sextas, ~21:05 CET |
| Máquina dos números | **Ryo-Catteau Stresa** |
| Máquina das estrelas | **Ryo-Catteau Pâquerette** |
| Tipo | *Gravity pick* — câmara com pás contra-rotativas, porta deslizante |
| Custódia | Bolas em cofre selado |
| Supervisão | Comissário de Justiça + responsável de sorteio da FDJ, presença simultânea obrigatória |
| Ensaios | Testes técnicos das máquinas a partir das 18:15, antes de cada sorteio |

O `gravity pick` é o tipo mecanicamente mais estável. Ao contrário das
máquinas de mistura por ar — sensíveis a humidade, temperatura e desgaste
de rolamentos, e historicamente as responsáveis pelos casos documentados de
viés — a mistura aqui é feita por contacto mecânico direto.

## Porque é que isto é o sítio certo para procurar

A única exploração de lotaria fisicamente comprovada da história não previu
o futuro: mediu equipamento. Joseph Jagger, Monte Carlo, 1873, contratou
seis pessoas para registarem os resultados de seis roletas durante semanas.
Uma delas tinha um desvio mecânico. Ele apostou nesse desvio e ganhou.

Isso funcionou porque um objeto físico tem propriedades persistentes. Uma
bola com massa ligeiramente diferente, uma câmara desnivelada, um conjunto
que envelheceu mal — qualquer destas coisas produziria um desvio **estável
ao longo do tempo**, e portanto detetável e explorável.

É por isso que a bateria de testes deste projeto não se limita ao
qui-quadrado agregado. Procura o viés onde ele apareceria de facto:
segmentado por período, por dia da semana, e em janelas deslizantes que
apanhariam um equipamento a degradar-se durante um troço do histórico.

## O obstáculo que mata a estratégia

Há um problema de informação que é decisivo, e convém dizê-lo sem rodeios:

> **Os dados públicos não identificam qual a máquina nem qual o conjunto de
> bolas usado em cada sorteio.**

Existem múltiplos conjuntos de bolas, escolhidos antes de cada sorteio, e as
bolas são pesadas e substituídas periodicamente. Sem a etiqueta do conjunto,
todos os sorteios ficam misturados no mesmo agregado.

O efeito é quantificável, e o módulo `machines.dilution_penalty()` fá-lo. Se
um conjunto entre `k` tem um viés de tamanho `b`:

- o viés observado no agregado dilui-se para `b/k`;
- os sorteios úteis por conjunto caem para `n/k`.

Como a sensibilidade de um teste estatístico escala com `√n`, a penalização
combinada é **quadrática**: detetar o mesmo viés exige cerca de `k²` vezes
mais sorteios.

Com 3 conjuntos, precisaríamos de ~9× mais histórico do que temos. Ao ritmo
de 104 sorteios por ano, isso são séculos.

## A conclusão empírica

Corridos todos os testes sobre 1970 sorteios (ver `docs/01-metodologia.md`
e a saída de `cli.py aleatoriedade`), **nenhum viés físico foi detetado**, e
— mais importante — **nenhum desvio observado persiste fora da amostra**.

O teste decisivo compara os desvios da primeira metade do histórico com os
da segunda. Um viés físico é uma propriedade do equipamento: teria de
aparecer nas duas e correlacionar-se entre elas. Ruído amostral não faz
isso.

```
correlação observada  r = 0.119
p simulado                0.408   (20 000 réplicas de sorteios uniformes)
```

Não há sinal. E a análise de potência diz-nos qual é o tamanho da brecha que
ainda poderia estar escondida: com 1970 sorteios, só detetaríamos um viés
superior a **19,4%** numa bola. Um desvio dessa magnitude num equipamento
inspecionado por um Comissário de Justiça antes de cada sorteio não é
plausível.

Ou seja: não é que não tenhamos procurado bem. É que a combinação de
equipamento robusto, supervisão notarial, rotação de conjuntos não
identificados e volume limitado de sorteios fecha esta porta de forma
estrutural, não acidental.
