# 🚢 Titanic: Machine Learning from Disaster (Desafio 03)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Pipeline-orange.svg)](https://scikit-learn.org/)
[![CRISP-DM](https://img.shields.io/badge/Methodology-CRISP--DM-green.svg)](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining)
[![Kaggle](https://img.shields.io/badge/Kaggle-Titanic%20Challenge-20BEFF.svg)](https://www.kaggle.com/c/titanic)

Este repositório contém a solução completa para o **Desafio 03 – Titanic: Machine Learning from Disaster**, desenvolvido pela equipe **InsurMinds (I2A2)**. O projeto combina a metodologia **CRISP-DM**, análise de dados de saúde/riscos sob a perspectiva da teoria de seguros e desenvolvimento de modelos preditivos com prevenção de vazamento de dados (*Data Leakage*).

---

## 📋 Estrutura do Repositório

```text
.
├── Desafios_InsurMinds/
│   └── titanic_solucao_desafio3.ipynb   # Notebook principal (Storytelling em 10 Seções)
├── dados/
│   ├── train.csv                         # Dataset de Treino oficial
│   └── test.csv                          # Dataset de Teste oficial
├── PLANO ESTRATÉGICO DO DESAFIO3.pdf     # Plano de Ação Estratégico (InsurMinds)
├── submissao_titanic.csv                 # Arquivo de predições finais para o Kaggle
├── .gitignore
└── README.md
```

---

## 🎯 Filosofia de Trabalho & Metodologia

Alinhado ao plano estratégico, o desenvolvimento seguiu uma **investigação científica rigorosa**:
1. **Prevenção de Data Leakage:** Uso de `Pipeline` e `ColumnTransformer` do Scikit-Learn.
2. **Avaliação Robusta:** Validação Cruzada Estratificada (`StratifiedKFold`, $k=5$).
3. **Pergunta Tripla:** Cada etapa responde: *O que estamos fazendo? Por que estamos fazendo? Como sabemos que melhorou o modelo?*

---

## 🧪 Hipóteses Investigadas ($H1$ - $H8$)

| Hipótese | Fator Analisado | Resultado |
|---|---|---|
| **H1** | **Sexo:** Mulheres tiveram prioridade na evacuação. | **Confirmada** (~74% de sobrevivência para mulheres vs ~19% homens) |
| **H2** | **Idade:** Crianças (<12 anos) tiveram taxas de sobrevivência superiores. | **Confirmada** (~58% sobrevivência em crianças) |
| **H3** | **Classe (Pclass):** 1ª Classe obteve maior sobrevivência. | **Confirmada** (63% na 1ª Classe vs 24% na 3ª) |
| **H4** | **Família:** Famílias pequenas (2-4 pessoas) se organizaram melhor. | **Confirmada** (pico de sobrevivência >55%) |
| **H5** | **IsAlone:** Passageiros sozinhos sobreviveram menos. | **Confirmada** (~30% sós vs ~51% acompanhados) |
| **H6** | **Fare:** Tarifas maiores correlacionam com sobrevivência. | **Confirmada** (Spearman = 0.32) |
| **H7** | **Title:** Títulos sociais contidos no nome indicam status/idade. | **Confirmada** (`Mrs`/`Miss` >70%, `Mr` ~15%) |
| **H8** | **Cabin/Deck:** Registro de cabine indica status elevado. | **Confirmada** (~67% com cabine vs ~30% sem) |

---

## 🤖 Desempenho dos Modelos (5-Fold Stratified CV)

| Algoritmo | Acurácia Média | Precisão Média | Recall Médio | F1-Score Médio |
|---|---|---|---|---|
| **Regressão Logística (Baseline)** | 0.8328 ± 0.0132 | 0.7897 | 0.7688 | 0.7787 |
| **Árvore de Decisão** | 0.8215 ± 0.0299 | 0.8016 | 0.7133 | 0.7537 |
| 🏆 **Random Forest (Otimizado)** | **0.8384 ± 0.0137** | **0.8199** | **0.7425** | **0.7785** |

---

## 🚀 Como Executar

```bash
# 1. Clonar o repositório
git clone https://github.com/edcarloscardoso/Titanic_Machine_Learning_from_Disaster.git
cd Titanic_Machine_Learning_from_Disaster

# 2. Criar ambiente virtual e instalar dependências
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy matplotlib seaborn scikit-learn

# 3. Abrir o Jupyter Notebook
jupyter notebook Desafios_InsurMinds/titanic_solucao_desafio3.ipynb
```
