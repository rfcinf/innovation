# O modelo consolidado e o auditor

```bash
python -m euromillions.cli modelo      # a recomendação
python -m euromillions.cli auditoria   # a verificação
```

---

## Parte 1 — o modelo

### O que "melhor segundo o histórico" quer dizer

A expressão admite duas leituras, e uma delas está morta:

| leitura | estado |
|---|---|
| "jogar os números que mais saíram" | **não funciona** — 12 famílias de testes, 2 milhões de apostas simuladas, zero achados |
| "usar o histórico para saber o que as outras pessoas escolhem, e evitar" | **funciona** — validado fora da amostra |

`model.py` implementa a segunda. Junta cinco componentes, todos estimados
a partir de 1970 sorteios e 1936 quebras de prémios:

| componente | como foi obtido | validação |
|---|---|---|
| popularidade dos números | GLM Poisson, 12 características | ρ = 0,411 fora da amostra |
| popularidade das estrelas | 12 parâmetros, 3090 observações | ρ = 0,775 fora da amostra |
| elasticidades por escalão | regressão com π previsto | 5+0 mede 0,956 vs teoria 1,0 |
| M1lhão | quota PT medida no escalão 2+0 | aritmética direta |
| cobertura da carteira | combinatória exata | simulação a 150k sorteios |

### Proveniência declarada

Todo o número que o modelo usa tem a origem registada em `PROVENANCE`, numa
de três categorias: **oficial**, **medido** ou **assumido**.

Isto não é burocracia. Dos cinco erros deste projeto, quatro eram números
escritos à mão que ninguém tinha ido medir — o *prior* das estrelas, a
tabela de elasticidades, os filtros de consecutivos, as barras de erro.
Declarar a origem transforma cada suposição num item de uma lista que o
auditor percorre.

Restam **três suposições**, e estão todas assinaladas:

| suposição | porque é frágil |
|---|---|
| vendas por sorteio | estimador circular: vencedores ÷ P(prémio) |
| fração de apostas aleatórias | não estimada — comprime a popularidade medida |
| preços dos escalões baixos | mediana desde 2020; variam com vendas e acumulação |

### O que produz

```
 nº        números estrelas  popularidade   EV_€
  1 01 31 34 44 45    10 12         0.378 1.0694
  2 10 37 38 41 48    11 12         0.385 1.0673
  3 24 25 40 49 50    01 12         0.464 1.0455
  4 14 22 32 33 46    10 11         0.582 1.0222
  5 02 35 36 42 44    04 12         0.413 1.0589

  popularidade média : 0.444x
  cobertura          : 24 de 50 números
  P(não ganhar nada) : 0.6450
  custo              : €12.50
  valor esperado     : €5.26   (perda esperada: €7.24)
```

E avalia apostas que já tenha em mente:

```bash
python -m euromillions.cli modelo --avaliar "7,13,21,30,42,3,7"
```

```
popularidade_numeros     1.312
popularidade_estrelas    1.379
popularidade_total       1.809
vs_aposta_media_%        -4.7
```

Aquela aposta — números de aniversário com as estrelas 3 e 7 — é jogada
por **1,8× mais gente** do que a média, e vale menos 4,7% do que uma aposta
qualquer, com exatamente a mesma probabilidade de acertar.

---

## Parte 2 — o auditor

### Porque existe

Cometi cinco erros neste projeto:

1. um teste de Kolmogorov-Smirnov contínuo aplicado a dados discretos, que
   "descobriu" desvios em 47 dos 50 números;
2. filtros que proibiam números consecutivos por folclore, quando os dados
   mostram que são sub-jogados;
3. barras de erro que tratavam 309.000 observações correlacionadas como
   independentes;
4. um otimizador que produzia carteiras com P(nada) **pior** do que jogar
   ao acaso;
5. o M1lhão em falta no motor de EV — 21% do valor de um bilhete português.

**Todos me favoreciam.** Todos faziam o sistema parecer melhor do que era.
E nenhum foi detetado por olhar para o resultado — foram detetados por
verificar o processo.

O auditor automatiza essa verificação.

### As seis áreas

**A. Dados** — integridade, coerência das eras, frescura, continuidade da
série, cobertura das quebras de prémios.

**B. Modelos** — reestima e revalida **fora da amostra**. Não confia nos
valores publicados: recalcula-os. Inclui um auto-teste do método de
elasticidades, que exige que o escalão 5+0 meça 1,0 (a resposta é conhecida
por teoria) — se o método se partir, é aqui e em mais lado nenhum que se
saberia.

**C. Deriva** — a vantagem ainda está viva? Testa o viés de datas nos
últimos 400 sorteios e a estabilidade das estrelas entre as duas metades do
histórico. Se a brecha fechar, o auditor reprova.

**D. Suposições** — percorre a lista de proveniência e sinaliza tudo o que
esteja marcado como "assumido". Verifica ativamente a mais perigosa: a
deriva do estimador de vendas.

**E. Afirmações** — recalcula os números publicados no README e compara com
o que o código produz hoje, com tolerâncias declaradas. A documentação e o
código afastam-se com o tempo, e o resultado é uma página que promete um
valor que o programa já não dá.

**F. Lacunas** — varrimento ativo à procura de sinal novo: bolas, intervalos,
janelas temporais, famílias de padrões, coincidências entre sorteios,
persistência fora da amostra. **Tudo com correção de falsas descobertas** —
sem isso, testar dezenas de hipóteses garante "achados" por construção, e o
auditor passaria a ser uma fábrica de superstições em vez de um travão.

### O resultado da execução atual

```
VEREDICTO: APROVADO COM RESERVAS — 5 avisos
```

Tudo o que passou:

| área | verificação | resultado |
|---|---|---|
| dados | integridade | 1970 sorteios, sem anomalias |
| dados | frescura | último sorteio há 1 dia |
| dados | cobertura | 1936 de 1970 (98,3%) |
| modelos | números fora da amostra | ρ = 0,411, p = 3,8×10⁻²⁵ |
| modelos | estrelas fora da amostra | ρ = 0,775, p = 2,5×10⁻¹⁸⁶ |
| modelos | auto-teste das elasticidades | 5+0 mede 0,956 (teoria: 1,0) |
| deriva | viés de datas ativo | rácio 1,273, p = 4,0×10⁻⁶ |
| deriva | estabilidade das estrelas | correlação entre metades 0,981 |
| afirmações | as 5 do README | todas dentro da tolerância |

Avisos em aberto — as três suposições declaradas, a deriva do estimador de
vendas (ρ = −0,317 entre tempo e popularidade global, não explicada), e o
número 22, que sobrevive à correção FDR por margem mínima e não persiste
fora da amostra.

### O auditor apanhou um erro do auditor

Na primeira execução, a verificação de continuidade sinalizou **377
intervalos anómalos**. Era falso: até maio de 2011 havia um único sorteio
por semana, à sexta, e intervalos de 7 dias eram o normal. A verificação
assumia dois sorteios semanais em todo o histórico.

Corrigido para verificar era a era, com o limite certo em cada uma.
Resultado: **0 anomalias**.

Vale a pena registar porque é o mesmo tipo de erro que o auditor existe
para apanhar — transformar uma regra do jogo numa avaria dos dados — e
desta vez o erro era dele. Um auditor que não possa ser auditado é apenas
mais uma fonte de confiança injustificada.

---

## Como usar em conjunto

```bash
# antes de confiar numa recomendação
python -m euromillions.cli auditoria

# se aprovar, gerar
python -m euromillions.cli modelo --jackpot 90e6 --bilhetes 5
```

O comando `auditoria` termina com código de saída 1 se houver problemas
críticos, para poder correr em automatismos. Sempre que houver sorteios
novos, o ciclo é: `fetch` → `auditoria` → `modelo`.
