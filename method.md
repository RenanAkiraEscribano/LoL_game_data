# Guia Metodológico: Machine Learning em Séries Temporais
## Aplicado à Predição de Resultados em League of Legends

---

## Sumário

1. [Séries Temporais](#1-séries-temporais)
2. [Data Leakage](#2-data-leakage)
3. [Split de Dados](#3-split-de-dados)
4. [Repetição de Experimentos (30x)](#4-repetição-de-experimentos-30x)
5. [Os Três Métodos Abordados](#5-os-três-métodos-abordados)
6. [Comparativo Final](#6-comparativo-final)

---

## 1. Séries Temporais

### O que é uma série temporal?

Uma **série temporal** é uma sequência de observações coletadas em instantes de tempo sucessivos e ordenados. O elemento fundamental que a diferencia de dados tabulares comuns é a **dependência temporal**: o valor de hoje carrega informação sobre o valor de amanhã.

```
Dado tabular comum (sem ordem):
  partida_A → [gold=5000, kills=3, ...]   ← não importa a ordem
  partida_B → [gold=4200, kills=1, ...]

Série temporal (ordem importa):
  partida_A, t=1 → [gold=500,  kills=0, ...]
  partida_A, t=2 → [gold=1100, kills=1, ...]
  partida_A, t=3 → [gold=1800, kills=1, ...]
  ...
```

### Séries temporais no contexto do LoL

Cada partida de LoL é naturalmente uma série temporal. A cada minuto, um **snapshot** é coletado com o estado completo do jogo: ouro acumulado, abates, torres destruídas, objetivos capturados, visão no mapa, etc.

```
match_id  | t_min | blue_gold | red_gold | blue_kills | dragon_diff | ... | y_blue_win
----------|-------|-----------|----------|------------|-------------|-----|----------
AAA-001   |   1   |   1.250   |  1.300   |     0      |      0      |     |     1
AAA-001   |   2   |   2.600   |  2.450   |     1      |      0      |     |     1
AAA-001   |   3   |   3.900   |  3.100   |     1      |      1      |     |     1
...
AAA-001   |  28   |  58.000   | 41.000   |    18      |      3      |     |     1
```

> **Observação importante:** o rótulo `y_blue_win` é o **mesmo para todos os minutos** de uma partida — ele representa o resultado final, não o estado atual.

### Por que a ordem importa?

Em dados temporais, embaralhar as linhas destrói a informação de progressão. Um modelo que não respeita a ordem temporal pode aprender padrões impossíveis na prática — por exemplo, usar o estado do minuto 25 para "prever" o resultado com base no minuto 5.

---

## 2. Data Leakage

### Definição

**Data leakage** (ou vazamento de dados) ocorre quando o modelo tem acesso durante o treinamento a informações que **não estariam disponíveis no momento da predição real**. É o erro mais crítico em projetos de Machine Learning e frequentemente passa despercebido.

> Um modelo com leakage aprende a "trapacear" — ele parece funcionar bem nos experimentos, mas falha completamente em produção.

### Tipos de leakage no contexto do LoL

#### Leakage Temporal — usar dados do futuro

Este é o leakage mais intuitivo em séries temporais. Se o objetivo é prever o resultado no minuto T, o modelo não pode ver nenhuma informação gerada após o minuto T.

```
❌ ERRADO — usar gameDuration_s no treino:
   Uma partida que durou 20 minutos provavelmente terminou com uma rendição.
   O modelo aprende esse padrão e "prevê" o passado usando o resultado.

✅ CORRETO — remover gameDuration_s:
   A duração da partida só é conhecida ao final — nunca está disponível no minuto T.
```

Outros exemplos de features que causam leakage temporal:

| Feature | Problema |
|---|---|
| `gameDuration_s` | Revela indiretamente o resultado (partidas curtas = rendição) |
| `patch` | Pode introduzir viés se partidas de patches diferentes estiverem misturadas |
| Qualquer dado coletado após T | Informação do futuro |

#### Leakage de Partidas — mesma partida em treino e teste

Este é o leakage mais comum e mais danoso em dados agrupados. Se os dados são divididos linha por linha (e não por partida), a mesma partida pode ter alguns minutos no treino e outros no teste.

```
❌ ERRADO — split aleatório por linha:
   Treino: [AAA t=1, AAA t=2, AAA t=5, BBB t=1, ...]
   Teste:  [AAA t=3, AAA t=4, BBB t=3, ...]
   → O modelo viu a partida AAA durante o treino e "lembra" dela no teste.

✅ CORRETO — split por match_id:
   Treino: [AAA t=1..28, CCC t=1..22, ...]
   Teste:  [BBB t=1..31, DDD t=1..19, ...]
   → O modelo nunca viu BBB ou DDD durante o treino.
```

#### Leakage de Pré-processamento — fit nos dados completos

Mesmo que o split seja feito corretamente, ainda existe risco de leakage se técnicas de normalização ou encoding forem aplicadas **antes** da divisão.

```python
# ❌ ERRADO — StandardScaler vê o teste antes do treino
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_full)   # usa média/std do dataset inteiro
X_train, X_test = split(X_scaled)

# ✅ CORRETO — fit apenas no treino
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit somente aqui
X_test_scaled  = scaler.transform(X_test)         # apenas aplica
```

#### Leakage de Early Stopping — teste usado durante o treino

Em modelos com parada antecipada (early stopping), o conjunto de teste **não pode** ser o critério de parada. Se for, o modelo está sendo selecionado com base no desempenho no teste — que deveria ser "virgem".

```python
# ❌ ERRADO — early stopping no teste
model.fit(X_train, eval_set=test_pool, use_best_model=True)
# O modelo para no ponto onde o TESTE foi melhor → teste contamina o treino

# ✅ CORRETO — early stopping na validação
model.fit(X_train, eval_set=val_pool, use_best_model=True)
# Teste só é tocado uma única vez, na avaliação final
```

---

## 3. Split de Dados

### Os três conjuntos

Um experimento bem conduzido exige a divisão dos dados em três conjuntos com papéis distintos e sem sobreposição:

```
Dataset completo
│
├── Treino (~75%)      → O modelo aprende aqui
├── Validação (~15%)   → Early stopping e ajuste de hiperparâmetros
└── Teste (~10%)       → Avaliação final, tocado UMA única vez
```

> A regra de ouro: **o conjunto de teste deve ser tratado como se não existisse durante todo o desenvolvimento**. Ele só é aberto quando o modelo está completamente finalizado.

### Por que três conjuntos e não dois?

| Situação | Problema |
|---|---|
| Usar o teste para early stopping | O modelo é selecionado para aquele teste específico |
| Usar o teste para escolher hiperparâmetros | Você está otimizando para o teste — as métricas ficam infladas |
| Reportar a métrica da validação como resultado final | A validação foi usada para decisões — não é uma estimativa honesta |

### Split estratificado

Em problemas de classificação binária, é importante que a proporção de classes (vitória/derrota) seja mantida em todos os splits. O **StratifiedShuffleSplit** garante isso.

```
Dataset: 60% blue_win, 40% red_win

Treino:   60% blue_win, 40% red_win  ✅
Val:      60% blue_win, 40% red_win  ✅
Teste:    60% blue_win, 40% red_win  ✅
```

### Split cronológico vs. aleatório

Para dados com componente temporal forte (como patches do LoL), o split aleatório pode introduzir um viés sutil: o modelo treina em partidas de um mês e testa em partidas de um mês anterior, vendo o "futuro" do meta.

```
Split aleatório (risco em dados com drift temporal):
  Treino: jan, mar, mai, ago, out...
  Teste:  fev, abr, jun, set, nov...  ← mistura de qualquer época

Split cronológico (mais realista para produção):
  Treino: jan → set
  Val:    out
  Teste:  nov → dez  ← sempre avalia no futuro
```

---

## 4. Repetição de Experimentos (30x)

### Por que repetir?

Um único experimento com uma única semente aleatória (`random_state`) pode ser enganoso. O resultado observado pode ser um **acidente estatístico** — o split gerado naquela semente pode favorecer (ou prejudicar) o modelo de forma não representativa.

> Imagine lançar uma moeda 10 vezes e obter 8 caras. Isso não significa que a moeda tem 80% de chance de cara — é variação aleatória amostral. O mesmo acontece com o split de dados.

### O protocolo de 30 repetições

O número 30 não é arbitrário. Pelo **Teorema Central do Limite**, com 30 ou mais amostras independentes, a distribuição das médias se aproxima suficientemente da distribuição normal, tornando os intervalos de confiança e testes estatísticos confiáveis.

```python
SEEDS = list(range(30))  # 30 sementes diferentes
resultados = []

for seed in SEEDS:
    # Novo split a cada iteração
    df_train, df_val, df_test = split(data, random_state=seed)

    modelo = treinar(df_train, df_val)
    acc = avaliar(modelo, df_test)
    resultados.append(acc)

# O que reportar:
media   = np.mean(resultados)
desvio  = np.std(resultados)
ic_95   = (media - 1.96 * desvio / np.sqrt(30),
           media + 1.96 * desvio / np.sqrt(30))

print(f"Accuracy: {media:.4f} ± {desvio:.4f}")
print(f"IC 95%: [{ic_95[0]:.4f}, {ic_95[1]:.4f}]")
```

### O que reportar vs. o que não reportar

| ❌ Não reporte | ✅ Reporte |
|---|---|
| A melhor accuracy das 30 rodadas | A média das 30 rodadas |
| O resultado de uma única semente | Média ± desvio padrão |
| Métricas sem intervalo de confiança | Intervalo de confiança 95% |

### Comparando modelos com as 30 repetições

Com as distribuições de resultados, é possível aplicar testes estatísticos para verificar se a diferença entre dois modelos é real ou apenas variação aleatória.

```python
from scipy import stats

# Wilcoxon: preferível ao t-test para métricas de ML (não assume normalidade)
stat, p_value = stats.wilcoxon(resultados_modelo_A, resultados_modelo_B)

if p_value < 0.05:
    print("Diferença estatisticamente significativa (p < 0.05)")
else:
    print("Diferença NÃO significativa — pode ser variação aleatória")
```

---

## 5. Os Três Métodos Abordados

### Visão geral

Todos os três métodos compartilham o mesmo objetivo — prever o resultado da partida — mas diferem em **como representam a informação temporal**.

```
Dado bruto: sequência de snapshots por partida
  [t=1: {...}] → [t=2: {...}] → ... → [t=T: {...}]

Método 1 — Snapshot Único:    usa apenas o estado em t=T
Método 2 — Vetor Achatado:    concatena todos os estados de t=1 até t=T
Método 3 — GRU:               processa a sequência preservando sua ordem
```

---

### Método 1: Snapshot Único (CatBoost)

#### Como funciona

Descarta toda a trajetória histórica e usa **apenas o estado do jogo no minuto T**. A partida inteira é representada por um único vetor de features.

```
Partida AAA:
  t=1: [gold=500,  kills=0, dragon=0, ...]
  t=2: [gold=1100, kills=1, dragon=0, ...]
  ...
  t=10: [gold=9800, kills=5, dragon=1, ...]  ← único vetor usado

Entrada do modelo: [gold=9800, kills=5, dragon=1, ...]
Saída:             P(blue_win) = 0.73
```

#### Vantagens

- Implementação simples e rápida
- Modelos como CatBoost lidam nativamente com features categóricas
- Boa interpretabilidade via importância de features
- Boa baseline — frequentemente já apresenta resultados sólidos

#### Desvantagens

- Ignora completamente a trajetória: não sabe se o time azul estava perdendo e virou o jogo, ou se sempre esteve à frente
- Perde informação de ritmo, aceleração e momentum
- Dois jogos com o mesmo estado no minuto 10 mas trajetórias opostas recebem a mesma entrada

#### Cuidados com leakage

- Remover `gameDuration_s`, `patch`, `t_min` e identificadores
- Split por `match_id` (não por linha)
- Early stopping na validação, não no teste

---

### Método 2: Vetor Achatado (CatBoost)

#### Como funciona

Preserva toda a trajetória de t=1 até t=T, mas a **concatena em um único vetor longo**. Cada feature recebe um sufixo de tempo (`__t1`, `__t2`, ..., `__tT`), criando uma representação flat.

```
Partida AAA com T=3 e 2 features (gold, kills):
  t=1: gold=500,  kills=0
  t=2: gold=1100, kills=1
  t=3: gold=1800, kills=1

Vetor achatado:
  [gold__t1=500, kills__t1=0, gold__t2=1100, kills__t2=1, gold__t3=1800, kills__t3=1]

Dimensão: T × n_features = 3 × 2 = 6 colunas por partida
```

#### Tratamento de minutos ausentes

Se uma partida não tem dados para todos os minutos de 1 a T, aplica-se **forward-fill**: o último valor disponível é propagado para os minutos ausentes.

```
Dados reais:   t=1: [500, 0]   t=2: ausente   t=3: [1800, 1]
Após reindex:  t=1: [500, 0]   t=2: [500, 0]  t=3: [1800, 1]
                                      ↑ cópia de t=1
```

#### Filtro de cobertura mínima

Uma partida com apenas 1 minuto real e T=10 teria 90% do vetor preenchido com o mesmo valor repetido — sem informação real. Por isso, aplica-se um critério de cobertura mínima:

```python
MIN_COVERAGE = 0.5  # pelo menos 50% de dados reais
min_real_minutes = int(T * MIN_COVERAGE)  # para T=10: mínimo 5 minutos reais
```

#### Vantagens

- Captura a trajetória completa e o ritmo de progressão
- Ainda usa modelos tabulares robustos (CatBoost, XGBoost)
- O modelo pode aprender padrões como "virada no minuto 8"
- Cada feature em cada minuto tem importância calculável individualmente

#### Desvantagens

- Dimensão cresce linearmente com T (T=30 e 50 features = 1.500 colunas)
- Não captura dependências sequenciais de forma explícita — o modelo trata cada coluna independentemente
- Partidas de durações diferentes exigem padding

#### Comparação com Método 1

| | Snapshot Único | Vetor Achatado |
|---|---|---|
| Informação temporal | ❌ Nenhuma | ✅ Toda a trajetória |
| Dimensão de entrada | `n_features` | `T × n_features` |
| Custo computacional | Baixo | Moderado |
| Risco de overfitting | Baixo | Maior (mais colunas) |

---

### Método 3: GRU (Gated Recurrent Unit)

#### Contexto: redes recorrentes

As redes recorrentes foram projetadas especificamente para dados sequenciais. Ao contrário dos métodos anteriores, o GRU **processa cada timestep em ordem**, mantendo um estado interno (memória) que acumula informação dos passos anteriores.

```
Entrada: sequência de vetores (T, n_features)

t=1: [500, 0, ...]  →  GRU  →  estado_1
t=2: [1100, 1, ...] →  GRU  →  estado_2   (usa estado_1)
t=3: [1800, 1, ...] →  GRU  →  estado_3   (usa estado_2)
...
t=T: [9800, 5, ...] →  GRU  →  estado_T   →  Classificador → P(vitória)
```

#### Por que GRU e não LSTM?

O GRU é uma versão simplificada do LSTM com menos parâmetros. Para sequências curtas (10–30 timesteps) e datasets de tamanho moderado (típico de partidas de Challenger), o GRU oferece desempenho equivalente ou superior ao LSTM com treinamento mais rápido e menor risco de overfitting.

#### Por que não Transformer?

O Transformer utiliza mecanismo de atenção global — cada timestep pode "ver" todos os outros diretamente. Isso é vantajoso para sequências longas (50+ timesteps) e datasets grandes (10k+ exemplos). Para o contexto deste projeto, o Transformer seria over-engineered e exigiria mais dados para generalizar bem.

#### Bidirecionalidade e leakage conceitual

Um GRU bidirecional lê a sequência tanto da esquerda para a direita quanto da direita para a esquerda. Em séries temporais causais, isso é um **leakage conceitual**: o modelo usaria informações do minuto 15 para "prever" o estado do minuto 5.

```python
# ❌ ERRADO — bidirecional em série temporal causal
gru = nn.GRU(bidirectional=True)   # lê minutos futuros → leakage conceitual

# ✅ CORRETO — apenas causal
gru = nn.GRU(bidirectional=False)  # cada minuto só vê o passado
```

#### Encoder — fit apenas no treino

O GRU exige que todas as features sejam numéricas. Features categóricas (nome do campeão, posição) precisam ser encodadas. O `LabelEncoder` e o `StandardScaler` devem ser ajustados **exclusivamente** nos dados de treino.

```python
encoder = FeatureEncoder(cat_cols, num_cols)
encoder.fit(train_records)          # ← somente treino

X_train = encoder.transform(...)    # fit + transform
X_val   = encoder.transform(...)    # apenas transform
X_test  = encoder.transform(...)    # apenas transform
```

#### Early stopping correto

```python
# Salva o melhor modelo pela val_loss
if val_loss < best_val_loss:
    best_state = model.state_dict()   # salva checkpoint

# Ao final, restaura o melhor checkpoint
model.load_state_dict(best_state)

# Só então avalia no teste virgem
avaliar(model, test_loader)
```

#### Vantagens

- Captura dependências temporais de forma explícita e eficiente
- A memória do GRU permite modelar padrões como "aceleração de ouro nas últimas 3 rodadas"
- Mais expressivo que o vetor achatado para padrões sequenciais complexos

#### Desvantagens

- Mais difícil de implementar e depurar
- Requer mais dados para generalizar bem
- Menos interpretável (caixa preta)
- Treinamento mais lento e sensível a hiperparâmetros

---

## 6. Comparativo Final

### Tabela resumo

| Critério | Snapshot (M1) | Vetor Achatado (M2) | GRU (M3) |
|---|---|---|---|
| **Representa trajetória** | ❌ | ✅ | ✅ |
| **Ordem temporal explícita** | ❌ | ❌ | ✅ |
| **Complexidade de implementação** | Baixa | Baixa | Alta |
| **Custo computacional** | Baixo | Moderado | Alto |
| **Interpretabilidade** | Alta | Alta | Baixa |
| **Performance com poucos dados** | ✅ | ✅ | ⚠️ |
| **Lida com features categóricas** | Nativo | Nativo | Requer encoding |
| **Risco de leakage temporal** | Moderado | Moderado | Alto (bidirecional) |

### Caminho metodológico recomendado

```
Etapa 1 — Baseline
  └── Snapshot Único com CatBoost no minuto T
      └── Métrica: AUC-ROC e F1 por valor de T (5, 10, 15, 20 min)

Etapa 2 — Trajetória com modelo tabular
  └── Vetor Achatado com CatBoost
      └── Compara com baseline: o histórico adiciona valor?

Etapa 3 — Modelagem sequencial
  └── GRU com sequência completa
      └── Compara com Etapa 2: a ordem explícita adiciona valor?

Em todas as etapas:
  ✅ Split por match_id (treino / val / teste)
  ✅ Early stopping apenas na validação
  ✅ Encoder fit apenas no treino
  ✅ 30 repetições com sementes diferentes
  ✅ Reportar média ± desvio padrão
  ✅ Teste estatístico entre métodos (Wilcoxon)
```

### Checklist anti-leakage

```
[ ] Split feito por match_id (nunca por linha)
[ ] gameDuration_s removido das features
[ ] patch removido das features
[ ] t_min removido das features
[ ] StandardScaler/LabelEncoder fit apenas no treino
[ ] Early stopping usando val_pool, não test_pool
[ ] GRU com bidirectional=False
[ ] Minutos após T completamente descartados
[ ] Teste tocado somente na avaliação final
[ ] 30 repetições reportando média ± desvio padrão
```

---

*Documento produzido como material de apoio para o projeto de predição de resultados em League of Legends utilizando dados da API oficial da Riot Games.*
