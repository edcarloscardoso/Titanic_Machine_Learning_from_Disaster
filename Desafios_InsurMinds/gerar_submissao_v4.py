"""
TITANIC V4 — Baseado na V1 (que foi a melhor: 0.78468)
======================================================

DIAGNÓSTICO DAS VERSÕES ANTERIORES:
  V1: Random Forest simples, CV 83.84% → Kaggle 0.78468 ✅ MELHOR
  V2: Voting Ensemble + features extras → Kaggle 0.76555 ❌ Overfitting
  V3: XGBoost tuned (GridSearch 2880 fits) → Kaggle 0.76315 ❌ Mais overfitting

LIÇÃO APRENDIDA:
  - Com apenas 891 amostras, SIMPLICIDADE VENCE.
  - Cada feature nova ou hiperparâmetro extra é chance de overfitting.
  - V1 teve o MENOR CV (83.84%) mas o MELHOR Kaggle — isso prova que
    CV alto neste dataset = overfitting.
  - XGBoost e Ensemble são muito poderosos e memorizam facilmente 891 linhas.

ESTRATÉGIA V4 — "V1 refinada, não revolucionada":
  1. MESMO modelo: Random Forest (igual V1)
  2. MESMAS features que V1, com 2 melhorias cirúrgicas:
     a) Imputação de Age pela MEDIANA DO TÍTULO (em vez de mediana global)
        → Master=3.5, Miss=21, Mr=30, Mrs=35, Rare=48.5
     b) Embarked: preencher NaN com 'S' (moda) antes do pipeline
  3. MENOS overfitting: n_estimators=200, max_depth=5 (mais raso que V1=6)
  4. NENHUM GridSearchCV (causa overfitting ao CV com dados pequenos)
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

SEED = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# 1. CARREGAR DADOS
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("TITANIC V4 — V1 refinada (simplicidade vence)")
print("=" * 60)
print("\n[1/5] Carregando dados...")

train_df = pd.read_csv(os.path.join(ROOT_DIR, 'dados', 'train.csv'))
test_df  = pd.read_csv(os.path.join(ROOT_DIR, 'dados', 'test.csv'))
print(f"     Treino: {train_df.shape} | Teste: {test_df.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING V4
#    Idêntico ao V1, com 2 melhorias cirúrgicas
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/5] Feature Engineering V4 (= V1 + imputação Age por Título)...")

# Medianas de Age por Title calculadas NO TREINO (aplicadas a ambos)
AGE_MEDIANS_BY_TITLE = {'Master': 3.5, 'Miss': 21.0, 'Mr': 30.0, 'Mrs': 35.0, 'Rare': 48.5}

def feature_engineering_v4(df):
    """
    Engenharia de features V4 — idêntica à V1, com 2 melhorias:
    1. Age imputada pela mediana do Title (em vez de global)
    2. Embarked preenchido com 'S' antes do pipeline
    """
    df = df.copy()

    # === FEATURES IDÊNTICAS À V1 ===

    # 1. Tamanho da Família (H4)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # 2. Flag 'IsAlone' (H5)
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # 3. Título Social (H7) — MESMA lógica V1
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    title_mapping = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
        'Mlle': 'Miss', 'Mme': 'Mrs', 'Don': 'Rare', 'Lady': 'Rare',
        'Countess': 'Rare', 'Jonkheer': 'Rare', 'Sir': 'Rare',
        'Capt': 'Rare', 'Ms': 'Miss'
    }
    df['Title'] = df['Title'].map(title_mapping).fillna('Rare')

    # 4. Cabine Conhecida (H8)
    df['CabinKnown'] = df['Cabin'].notnull().astype(int)

    # 5. Convés (Deck)
    df['Deck'] = df['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')

    # 6. Tamanho do Grupo por Bilhete — MESMA lógica V1 (per-dataset)
    ticket_counts = df['Ticket'].value_counts()
    df['TicketGroupSize'] = df['Ticket'].map(ticket_counts)

    # === MELHORIA 1: Age imputada pela mediana do Title ===
    # (V1 deixava pro Pipeline imputar com mediana global de ~28 anos)
    # (V4 imputa Master=3.5, Miss=21, Mr=30, Mrs=35, Rare=48.5)
    for title, median_age in AGE_MEDIANS_BY_TITLE.items():
        mask = (df['Age'].isnull()) & (df['Title'] == title)
        df.loc[mask, 'Age'] = median_age

    # 7. Faixas Etárias (AgeGroup) — MESMA lógica V1
    df['AgeGroup'] = pd.cut(
        df['Age'],
        bins=[-1, 12, 18, 60, 120],
        labels=['Child', 'Teen', 'Adult', 'Senior']
    )
    df['AgeGroup'] = df['AgeGroup'].astype(str).replace('nan', 'Unknown')

    # 8. Faixas de Tarifa (FareGroup) — MESMA lógica V1 (per-dataset qcut)
    df['FareGroup'] = pd.qcut(
        df['Fare'].fillna(df['Fare'].median()),
        q=4,
        labels=['Low', 'Medium', 'High', 'VeryHigh']
    )
    df['FareGroup'] = df['FareGroup'].astype(str)

    # === MELHORIA 2: Embarked preenchido com 'S' ===
    df['Embarked'] = df['Embarked'].fillna('S')

    return df


# Aplicar (mesma abordagem V1: cada dataset independente)
train_fe = feature_engineering_v4(train_df)
test_fe  = feature_engineering_v4(test_df)

print(f"     Age NaN restantes treino: {train_fe['Age'].isnull().sum()}")
print(f"     Age NaN restantes teste:  {test_fe['Age'].isnull().sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. SEPARAR X, y, X_test
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/5] Preparando features (mesmas da V1)...")

X_train  = train_fe.drop(['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
y_train  = train_fe['Survived']
X_test   = test_fe.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
test_ids = test_df['PassengerId']

# MESMAS features que V1
numeric_features = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize', 'TicketGroupSize']
categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'IsAlone', 'CabinKnown', 'Deck', 'AgeGroup', 'FareGroup']

print(f"     Numéricas  ({len(numeric_features)}): {numeric_features}")
print(f"     Categóricas({len(categorical_features)}): {categorical_features}")
print(f"     X_train: {X_train.shape} | X_test: {X_test.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. PIPELINE + MODELO
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/5] Construindo Pipeline e avaliando modelos...")

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot',  OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# Testar vários RFs com parâmetros conservadores (sem GridSearchCV pesado)
rf_configs = {
    'V1 original (n=100, d=6, s=5)': RandomForestClassifier(
        random_state=SEED, n_estimators=100, max_depth=6, min_samples_split=5
    ),
    'V4a (n=200, d=5, s=8, l=3)': RandomForestClassifier(
        random_state=SEED, n_estimators=200, max_depth=5,
        min_samples_split=8, min_samples_leaf=3, max_features='sqrt'
    ),
    'V4b (n=300, d=5, s=10, l=4)': RandomForestClassifier(
        random_state=SEED, n_estimators=300, max_depth=5,
        min_samples_split=10, min_samples_leaf=4, max_features='sqrt'
    ),
    'V4c (n=500, d=4, s=10, l=5)': RandomForestClassifier(
        random_state=SEED, n_estimators=500, max_depth=4,
        min_samples_split=10, min_samples_leaf=5, max_features='sqrt'
    ),
}

print(f"\n{'Modelo':<40} | {'Acurácia':>10} | {'Std':>8}")
print("-" * 65)

best_score  = 0
best_name   = ''
best_model  = None

for name, model in rf_configs.items():
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='accuracy')
    mu, std = scores.mean(), scores.std()
    marker = ' <<<' if mu > best_score else ''
    print(f"{name:<40} | {mu:>10.4f} | {std:>8.4f}{marker}")
    if mu > best_score:
        best_score = mu
        best_name  = name
        best_model = model

print(f"\n>>> Melhor: {best_name} (CV={best_score:.4f})")

# ─────────────────────────────────────────────────────────────────────────────
# 5. TREINAR E GERAR SUBMISSÃO
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/5] Treinando melhor modelo e gerando submissão...")

final_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier',   best_model)
])
final_pipe.fit(X_train, y_train)

test_predictions = final_pipe.predict(X_test)

sobreviventes = int(test_predictions.sum())
total         = len(test_predictions)
print(f"     Não sobreviveram (0): {total - sobreviventes} ({(total-sobreviventes)/total*100:.1f}%)")
print(f"     Sobreviveram    (1): {sobreviventes} ({sobreviventes/total*100:.1f}%)")

submission = pd.DataFrame({'PassengerId': test_ids, 'Survived': test_predictions})
output_path = os.path.join(ROOT_DIR, 'submissao_titanic_v4.csv')
submission.to_csv(output_path, index=False)

# Comparar com V1
v1 = pd.read_csv(os.path.join(ROOT_DIR, 'submissao_titanic.csv'))
diff = (v1['Survived'] != submission['Survived']).sum()

print(f"\n     Arquivo salvo: {output_path}")
print(f"     Diferença vs V1: {diff} passageiros mudaram de previsão")

print("\n" + "=" * 60)
print("RESUMO FINAL V4")
print("=" * 60)
print(f"  V1: RF (n=100,d=6)      | CV 83.84%  → Kaggle 0.78468")
print(f"  V2: Voting Ensemble     | CV 84.62%  → Kaggle 0.76555")
print(f"  V3: XGBoost GridSearch  | CV 85.52%  → Kaggle 0.76315")
print(f"  V4: RF refinado + Age   | CV {best_score:.4f}  → Kaggle ???")
print("=" * 60)
print(f"\n  Mudanças V4 vs V1:")
print(f"    1. Age imputada por Title (3.5/21/30/35/48.5) em vez de global")
print(f"    2. Embarked NaN preenchido com 'S' antes do pipeline")
print(f"    3. RF com max_depth mais raso para reduzir overfitting")
print(f"    4. SEM GridSearchCV (que overfitou nos dados pequenos)")
