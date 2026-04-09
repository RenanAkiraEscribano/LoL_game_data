# League of Legends Match Prediction Pipeline

Este projeto implementa um pipeline completo para **coleta de dados da
Riot API**, **construção de dataset temporal de partidas** e
**treinamento de modelos de machine learning** para predição do
resultado das partidas.

O fluxo do projeto segue quatro etapas principais:

1.  Coleta dos dados da Riot API
2.  Conversão e filtragem dos dados
3.  Análise exploratória do dataset
4.  Treinamento e avaliação de modelos de Machine Learning

------------------------------------------------------------------------

# Estrutura do Projeto

    .
    ├── dataExtractor.py
    ├── dataFilter.py
    ├── analises.py
    ├── mlTeste.py
    ├── dataset_ts_csv/
    ├── matches/
    ├── timelines/
    └── seen_matches.json

------------------------------------------------------------------------

# Requisitos

O projeto foi desenvolvido em **Python 3.9+**.

## Bibliotecas necessárias

Instale as dependências com:

``` bash
pip install pandas requests catboost scikit-learn
```

Bibliotecas utilizadas:

-   pandas
-   requests
-   catboost
-   scikit-learn
-   json
-   glob
-   datetime
-   os
-   time
-   collections

As últimas bibliotecas fazem parte da **biblioteca padrão do Python**.

------------------------------------------------------------------------

# Configuração da Riot API

Para coletar dados é necessário possuir uma **API Key da Riot Games**.

1.  Crie uma conta em:

https://developer.riotgames.com/

2.  Gere sua API Key.

3.  No arquivo **dataExtractor.py** substitua:

``` python
API_KEY = "SUA_API_KEY"
```

------------------------------------------------------------------------

# Pipeline de Execução

A execução do projeto deve seguir a seguinte ordem.

## 1️⃣ Coleta de dados -- dataExtractor.py

### Função

Realiza a **coleta de partidas da Riot API**.

### O que o script faz

-   Obtém jogadores da liga **Challenger Solo/Duo**
-   Extrai o **PUUID** dos jogadores
-   Busca **match IDs**
-   Baixa:
    -   detalhes das partidas
    -   timeline completa das partidas
-   Salva os dados em **JSON**
-   Evita downloads duplicados usando `seen_matches.json`

### Execução

``` bash
python dataExtractor.py
```

### Saídas geradas

    matches/
    timelines/
    seen_matches.json

------------------------------------------------------------------------

# 2️⃣ Conversão para Dataset -- dataFilter.py

### Função

Transforma os **JSONs das partidas em datasets CSV estruturados**.

### O que o script faz

-   Lê arquivos de partidas e timelines
-   Filtra partidas da fila **Ranked Solo/Duo (420)**
-   Percorre os **frames da timeline**
-   Extrai variáveis relevantes de jogo
-   Constrói registros por **minuto/frame**
-   Gera **um CSV por partida**
-   Evita reprocessar partidas já convertidas

### Execução

``` bash
python dataFilter.py
```

### Saída

    dataset_ts_csv/

Cada arquivo CSV representa uma **partida estruturada como série
temporal**.

------------------------------------------------------------------------

# 3️⃣ Análise Exploratória -- analises.py

### Função

Realizar **análise exploratória do dataset**.

### O que o script analisa

-   número total de partidas
-   número de colunas
-   balanceamento das classes
-   distribuição por patch
-   campeões mais utilizados
-   duração das partidas
-   número de frames por partida
-   quantidade de kills por partida

### Execução

``` bash
python analises.py
```

Essa etapa ajuda a **entender a estrutura do dataset antes da
modelagem**.

------------------------------------------------------------------------

# 4️⃣ Treinamento do Modelo -- mlTeste.py

### Função

Executar um **teste de Machine Learning para prever o vencedor da
partida**.

### Etapas realizadas

1.  Carrega todos os CSVs do dataset
2.  Remove partidas muito curtas (`t_min <= 3`) Situações de rendição devido a ociosidade
3.  Seleciona um **snapshot temporal por partida**
4.  Divide dados em **treino e teste**
5.  Identifica variáveis:
    -   numéricas
    -   categóricas
6.  Treina um modelo **CatBoostClassifier**
7.  Avalia o modelo com:

-   Accuracy
-   F1-score
-   Matriz de confusão
-   Classification report

### Execução

``` bash
python mlTeste.py
```

------------------------------------------------------------------------

# Fluxo Completo

Execute os scripts na seguinte ordem:

``` bash
python dataExtractor.py
python dataFilter.py
python analises.py
python mlTeste.py
```

Fluxo do pipeline:

    Riot API
       ↓
    dataExtractor.py
       ↓
    JSON de partidas e timelines
       ↓
    dataFilter.py
       ↓
    Dataset temporal CSV
       ↓
    analises.py
       ↓
    Análise exploratória
       ↓
    mlTeste.py
       ↓
    Modelo de Machine Learning

------------------------------------------------------------------------

# Observações

-   A Riot API possui **limites de requisição**, portanto a coleta pode
    levar tempo.
-   As partidas são estruturadas como **séries temporais minuto a
    minuto**.
-   O modelo atual utiliza **CatBoost**, mas outros modelos podem ser
    testados.
-   O pipeline pode ser expandido para:
    -   previsão minuto a minuto
    -   modelos sequenciais (LSTM, Transformers)
    -   engenharia de atributos temporal

------------------------------------------------------------------------

# Licença

Uso acadêmico e educacional.
