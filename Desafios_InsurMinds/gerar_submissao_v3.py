"""
TITANIC V3 — XGBoost Puro + Features Limpas + Zero Leakage
------------------------------------------------------------
Lições aprendidas da V2 (score 0.76555):
  - FareGroup com pd.qcut em treino e teste separados = bins diferentes = leakage
  - TicketPrefix com top_prefixes calculados por dataset = inconsistência
  - Voting Ensemble com LR fraco puxou as predições para baixo
  - Excesso de features ruidosas prejudicou a generalização

Estratégia V3:
  - XGBoost puro (foi o melhor individual: 84.62% CV na V2)
  - Apenas features testadas e confiáveis
  - Todas as transformações que dependem de estatística são calculadas
    NO TREINO e aplicadas ao teste (zero leakage)
  - GridSearchCV para tuning fino do XGBoost
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
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from xgboost import XGBClassifier

SEED = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# 1. CARREGAR DADOS
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("TITANIC V3 — XGBoost + Features Limpas")
print("=" * 60)
print("\n[1/6] Carregando dados...")

train_df = pd.read_csv(os.path.join(ROOT_DIR, 'dados', 'train.csv'))
test_df  = pd.read_csv(os.path.join(ROOT_DIR, 'dados', 'test.csv'))
print(f"     Treino: {train_df.shape} | Teste: {test_df.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING V3 — LIMPO E SEM LEAKAGE
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/6] Feature Engineering V3 (sem leakage)...")

def extract_title(df):
    """Extrai e normaliza o título social do nome."""
    df = df.copy()
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    rare_titles = {
        'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
        'Don': 'Rare', 'Lady': 'Rare', 'Countess': 'Rare', 'Jonkheer': 'Rare',
        'Sir': 'Rare', 'Capt': 'Rare',
        'Mlle': 'Miss', 'Mme': 'Mrs', 'Ms': 'Miss'
    }
    df['Title'] = df['Title'].replace(rare_titles)
    df['Title'] = df['Title'].where(df['Title'].isin(['Mr', 'Miss', 'Mrs', 'Master', 'Rare']), 'Rare')
    return df

def feature_engineering_v3(df):
    """
    Engenharia de features V3.
    Apenas transformações DETERMINÍSTICAS — nenhum cálculo estatístico
    que precise ser ajustado no treino e aplicado ao teste.
    Estatísticas (imputação, escala) ficam dentro do Pipeline Scikit-Learn.
    """
    df = df.copy()

    # 1. Título Social (determinístico — apenas mapeamento de texto)
    df = extract_title(df)

    # 2. Tamanho da Família
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # 3. IsAlone
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # 4. FamilyCategory (determinístico — regra fixa)
    def family_cat(n):
        if n == 1:   return 0  # Solo
        if n <= 3:   return 1  # Pequena
        if n <= 6:   return 2  # Média
        return 3                # Grande
    df['FamilyCategory'] = df['FamilySize'].apply(family_cat)

    # 5. CabinKnown (determinístico)
    df['CabinKnown'] = df['Cabin'].notnull().astype(int)

    # 6. Deck numérico (determinístico — mapeamento fixo)
    deck_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8}
    df['DeckNum'] = df['Cabin'].apply(
        lambda s: deck_map.get(s[0], 0) if pd.notnull(s) else 0
    )

    # 7. IsChild (determinístico — limiar fixo)
    #    Importante: usamos Age diretamente, o Pipeline fará imputação
    #    Aqui usamos fillna(99) só para a flag, sem contaminar Age original
    df['IsChild'] = (df['Age'].fillna(99) < 12).astype(int)

    # 8. IsMother (determinístico — regra fixa)
    df['IsMother'] = (
        (df['Sex'] == 'female') &
        (df['Parch'] > 0) &
        (df['Age'].fillna(0) > 18)
    ).astype(int)

    # 9. TitleNum — encoding ordinal confiável
    title_num = {'Master': 0, 'Miss': 1, 'Mrs': 2, 'Mr': 3, 'Rare': 4}
    df['TitleNum'] = df['Title'].map(title_num).fillna(4).astype(int)

    # 10. FarePerPerson — dividido pelo SibSp+Parch+1 (sem TicketGroupSize!)
    #     TicketGroupSize causou leakage na V2; FamilySize é mais seguro
    df['FarePerPerson'] = df['Fare'].fillna(df['Fare'].median()) / df['FamilySize']

    return df

train_fe = feature_engineering_v3(train_df)
test_fe  = feature_engineering_v3(test_df)

orig_cols = set(train_df.columns)
new_cols  = [c for c in train_fe.columns if c not in orig_cols]
print(f"     Novas features ({len(new_cols)}): {new_cols}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. PREPARAR X_train, y_train, X_test
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/6] Preparando features...")

DROP_COLS = ['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin']

X_train  = train_fe.drop([c for c in DROP_COLS if c in train_fe.columns], axis=1)
y_train  = train_fe['Survived']
X_test   = test_fe.drop([c for c in DROP_COLS if c in test_fe.columns and c != 'Survived'], axis=1)
test_ids = test_df['PassengerId']

# Features numéricas: Age e Fare entram raw — o Pipeline cuida da imputação
numeric_features = [
    'Age', 'Fare', 'SibSp', 'Parch',
    'FamilySize', 'FamilyCategory', 'IsAlone',
    'CabinKnown', 'DeckNum', 'IsChild', 'IsMother',
    'TitleNum', 'FarePerPerson'
]

# Features categóricas: apenas as estáveis e com baixa cardinalidade
categorical_features = [
    'Pclass', 'Sex', 'Embarked', 'Title'
]

numeric_features     = [f for f in numeric_features     if f in X_train.columns]
categorical_features = [f for f in categorical_features if f in X_train.columns]

print(f"     Numéricas  ({len(numeric_features)}): {numeric_features}")
print(f"     Categóricas({len(categorical_features)}): {categorical_features}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. PIPELINE — PREPROCESSOR
# ─────────────────────────────────────────────────────────────────────────────
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot',  OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'
)

# ─────────────────────────────────────────────────────────────────────────────
# 5. GRIDSEARCHCV — TUNING DO XGBOOST
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/6] Buscando melhores hiperparâmetros (GridSearchCV)...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

xgb_base = XGBClassifier(
    random_state=SEED,
    eval_metric='logloss',
    verbosity=0,
    use_label_encoder=False
)

param_grid = {
    'classifier__n_estimators':     [200, 300, 400],
    'classifier__max_depth':        [3, 4],
    'classifier__learning_rate':    [0.03, 0.05, 0.08],
    'classifier__subsample':        [0.75, 0.85],
    'classifier__colsample_bytree': [0.75, 0.85],
    'classifier__reg_alpha':        [0.0, 0.1],
    'classifier__reg_lambda':       [1.0, 1.5],
    'classifier__min_child_weight': [1, 3],
}

pipe_for_gs = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier',   xgb_base)
])

grid_search = GridSearchCV(
    pipe_for_gs,
    param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    refit=True
)

grid_search.fit(X_train, y_train)

print(f"\n     Melhor score CV: {grid_search.best_score_:.4f}")
print(f"     Melhores params: {grid_search.best_params_}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. GERAR ARQUIVO DE SUBMISSÃO
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/6] Gerando predições com o melhor modelo...")

best_pipeline    = grid_search.best_estimator_
test_predictions = best_pipeline.predict(X_test)

sobreviventes = test_predictions.sum()
total         = len(test_predictions)
print(f"     Não sobreviveram (0): {total - sobreviventes} ({(total-sobreviventes)/total*100:.1f}%)")
print(f"     Sobreviveram    (1): {sobreviventes} ({sobreviventes/total*100:.1f}%)")

submission = pd.DataFrame({'PassengerId': test_ids, 'Survived': test_predictions})
output_path = os.path.join(ROOT_DIR, 'submissao_titanic_v3.csv')
submission.to_csv(output_path, index=False)

print(f"\n[6/6] Arquivo salvo: {output_path}")
print(f"      Total de registros: {len(submission)}")

print("\n" + "=" * 60)
print("RESUMO FINAL")
print("=" * 60)
print(f"  Score V1 Kaggle:    0.78468  (Random Forest)")
print(f"  Score V2 Kaggle:    0.76555  (Voting Ensemble — piorou)")
print(f"  CV Score V3 local: {grid_search.best_score_:.5f}  (XGBoost tuned)")
print(f"  Arquivo:            submissao_titanic_v3.csv")
print("=" * 60)

print("\nPrimeiras 10 linhas da submissão:")
print(submission.head(10).to_string(index=False))
