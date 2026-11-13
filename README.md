# Intenção de compra em loja virtual (Online Shoppers Purchasing Intention)

Projeto da disciplina **Introdução ao Aprendizado de Máquina** · IFSP Campinas · Tecnologia em Análise e Desenvolvimento de Sistemas.

**Integrantes:** Melissa Batista Junqueira · Rafaela Laryssa Mello · Sophia Ferreira Boonen

## Problema

Prever se uma sessão de navegação em uma loja virtual termina em compra (`purchase`), a partir de informações de
navegação, origem do tráfego, dispositivo e período. É um problema de **classificação binária** com classes desbalanceadas
(cerca de 15,5% de compras).

## Base de dados

- **Fonte:** [UCI Machine Learning Repository — Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
  (Sakar & Kastro, 2018), licença CC BY 4.0. DOI: https://doi.org/10.24432/C5F88Q
- Cada linha é uma **sessão**. 17 atributos preditores e o alvo `purchase`.
- Divisão fornecida pelo professor e preservada: `online_shoppers_train.csv` (9.864 sessões) e
  `online_shoppers_test.csv` (2.466 sessões).

## Metodologia

1. **Preparação:** correção de tipos, remoção de 98 duplicatas exatas no treino (o teste é mantido como fornecido), verificação de valores inválidos e inconsistências. Não há dados faltantes. Os outliers foram mantidos (justificativa no notebook) e tratados com `log1p` no modelo linear.
2. **Análise exploratória:** 10 perguntas respondidas com análise e visualização (gráficos de barras, linha, boxplot, histograma e mapa de calor), com testes estatísticos quando cabível.
3. **Modelagem:**
   - Técnicas: `DummyClassifier` (referência), **Regressão Logística (baseline)**, **Random Forest** e **HistGradientBoosting**.
   - Transformações dentro de `Pipeline` (sem vazamento): `log1p` + padronização para o modelo linear e *one-hot* para categóricas.
   - Balanceamento por `class_weight`, sem reamostragem.
   - Validação: `StratifiedKFold` (5 dobras) no treino. Busca de hiperparâmetros com `RandomizedSearchCV` (PR-AUC).
   - Limiar de decisão escolhido com predições *out-of-fold* no treino (maximiza o F1).
   - **Dois cenários**, por causa do risco de vazamento do `page_value`: **A** (com `page_value`) e **B** (sem `page_value`).
   - Métricas: PR-AUC (principal), ROC-AUC, precisão, revocação, F1 e acurácia balanceada.

## Principais resultados (conjunto de teste)

| Cenário | Modelo | PR-AUC | ROC-AUC | Precisão | Revocação | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A (com `page_value`) | Regressão Logística (baseline) | 0,656 | 0,924 | 0,588 | 0,775 | 0,669 |
| A (com `page_value`) | **Random Forest** | **0,762** | 0,935 | 0,639 | 0,757 | 0,693 |
| A (com `page_value`) | HistGradientBoosting | 0,748 | 0,936 | 0,642 | 0,738 | 0,687 |
| B (sem `page_value`) | Regressão Logística (baseline) | 0,333 | 0,763 | 0,285 | 0,694 | 0,404 |
| B (sem `page_value`) | **Random Forest** | **0,381** | 0,782 | 0,322 | 0,618 | 0,423 |
| B (sem `page_value`) | HistGradientBoosting | 0,369 | 0,788 | 0,312 | 0,623 | 0,416 |

A prevalência de compras no teste é 0,155 (é a PR-AUC de um modelo sem poder preditivo).

- O `page_value` responde por boa parte do desempenho (PR-AUC cai de 0,76 para 0,38 sem ele) e é potencialmente vazado.
- Random Forest e HistGradientBoosting empatam tecnicamente e superam o baseline linear.

## Como executar

1. Instale as dependências: `pip install -r requirements.txt`
2. Mantenha os arquivos todos na mesma pasta, lado a lado:
   ```
   .
   ├── Projeto_Shoppers.ipynb
   ├── README.md
   ├── requirements.txt
   ├── online_shoppers_train.csv
   └── online_shoppers_test.csv
   ```
3. Abra o notebook e use **Run All**. A busca de hiperparâmetros leva alguns minutos.

O notebook lê os CSVs por caminho relativo, da pasta onde ele está. Se existir uma subpasta `data/`, ele também a aceita.
No Google Colab, se os CSVs não estiverem ao lado do notebook, ele monta o Google Drive e procura em
`Colab Notebooks/Online_Shoppers`. Todas as etapas aleatórias usam `random_state=42`.
