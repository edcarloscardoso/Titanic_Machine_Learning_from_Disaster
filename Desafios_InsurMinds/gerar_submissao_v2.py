"""
Script Python equivalente ao notebook V2 — executa todas as células
e gera submissao_titanic_v2.csv na raiz do projeto.
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

SEED = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)  # raiz do projeto

# ── 1. Carregar dados ─────────────────────────────────────────────────────────
print("[1/5] Carregando dados...")
train_df = pd.read_csv(os.path.join(ROOT_DIR, 'dados', 'train.csv'))
test_df  = pd.read_csv(os.path.join(ROOT_DIR, 'dados', 'test.csv'))
print(f"     Treino: {train_df.shape} | Teste: {test_df.shape}")

# ── 2. Feature Engineering V2 ────────────────────────────────────────────────
print("[2/5] Executando Feature Engineering V2...")

def feature_engineering_v2(df, ticket_counts_ref=None):
    df = df.copy()
    df['FamilySize']    = df['SibSp'] + df['Parch'] + 1
    df['IsAlone']       = (df['FamilySize'] == 1).astype(int)
    df['Title']         = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    title_mapping = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
        'Mlle': 'Miss', 'Mme': 'Mrs', 'Don': 'Rare', 'Lady': 'Rare',
        'Countess': 'Rare', 'Jonkheer': 'Rare', 'Sir': 'Rare',
        'Capt': 'Rare', 'Ms': 'Miss'
    }
    df['Title']         = df['Title'].map(title_mapping).fillna('Rare')
    df['CabinKnown']    = df['Cabin'].notnull().astype(int)
    df['Deck']          = df['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')
    deck_map            = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'M': 0, 'T': 8}
    df['DeckNum']       = df['Deck'].map(deck_map).fillna(0).astype(int)
    if ticket_counts_ref is None:
        ticket_counts_ref = df['Ticket'].value_counts()
    df['TicketGroupSize'] = df['Ticket'].map(ticket_counts_ref).fillna(1).astype(int)
    df['AgeGroup']      = pd.cut(df['Age'], bins=[-1, 5, 12, 18, 35, 60, 120],
                                 labels=['Baby', 'Child', 'Teen', 'YoungAdult', 'Adult', 'Senior'])
    df['AgeGroup']      = df['AgeGroup'].astype(str).replace('nan', 'Unknown')
    df['FareGroup']     = pd.qcut(df['Fare'].fillna(df['Fare'].median()), q=4,
                                  labels=['Low', 'Medium', 'High', 'VeryHigh'])
    df['FareGroup']     = df['FareGroup'].astype(str)
    df['FarePerPerson'] = df['Fare'].fillna(df['Fare'].median()) / df['TicketGroupSize']

    def extract_ticket_prefix(ticket):
        parts = ticket.split()
        if len(parts) > 1:
            return parts[0].replace('.', '').replace('/', '').upper()[:4]
        return 'NUM'
    df['TicketPrefix']  = df['Ticket'].apply(extract_ticket_prefix)
    top_prefixes        = df['TicketPrefix'].value_counts().nlargest(10).index
    df['TicketPrefix']  = df['TicketPrefix'].where(df['TicketPrefix'].isin(top_prefixes), 'OTHER')

    def family_category(size):
        if size == 1: return 'Solo'
        if size <= 3: return 'Small'
        if size <= 6: return 'Medium'
        return 'Large'
    df['FamilyCategory'] = df['FamilySize'].apply(family_category)
    df['Age_filled']     = df['Age'].fillna(df['Age'].median())
    df['IsMother']       = ((df['Sex'] == 'female') & (df['Parch'] > 0) & (df['Age_filled'] > 18)).astype(int)
    df['IsChild']        = (df['Age_filled'] < 12).astype(int)
    df.drop(columns=['Age_filled'], inplace=True)
    title_num            = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}
    df['TitleNum']       = df['Title'].map(title_num).fillna(4)
    return df, ticket_counts_ref

train_fe, ticket_counts = feature_engineering_v2(train_df)
test_fe,  _             = feature_engineering_v2(test_df, ticket_counts_ref=ticket_counts)

orig_cols = set(train_df.columns)
new_cols  = [c for c in train_fe.columns if c not in orig_cols]
print(f"     Novas features ({len(new_cols)}): {new_cols}")

# ── 3. Preparar X, y, X_test ─────────────────────────────────────────────────
DROP_COLS        = ['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin']
numeric_features = [
    'Age', 'Fare', 'SibSp', 'Parch',
    'FamilySize', 'TicketGroupSize', 'FarePerPerson',
    'DeckNum', 'TitleNum', 'IsAlone', 'CabinKnown', 'IsMother', 'IsChild'
]
categorical_features = [
    'Pclass', 'Sex', 'Embarked', 'Title', 'Deck',
    'AgeGroup', 'FareGroup', 'FamilyCategory', 'TicketPrefix'
]

X_train  = train_fe.drop([c for c in DROP_COLS if c in train_fe.columns], axis=1)
y_train  = train_fe['Survived']
X_test   = test_fe.drop([c for c in DROP_COLS if c in test_fe.columns and c != 'Survived'], axis=1)
test_ids = test_df['PassengerId']

numeric_features     = [f for f in numeric_features     if f in X_train.columns]
categorical_features = [f for f in categorical_features if f in X_train.columns]

# ── 4. Preprocessor ──────────────────────────────────────────────────────────
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'
)

# ── 5. Validação Cruzada dos Modelos ─────────────────────────────────────────
print("[3/5] Avaliando modelos com 5-Fold CV...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

lr_model   = LogisticRegression(random_state=SEED, max_iter=1000, C=0.1)
rf_model   = RandomForestClassifier(
    random_state=SEED, n_estimators=200, max_depth=6,
    min_samples_split=10, min_samples_leaf=4, max_features='sqrt'
)
xgb_model  = XGBClassifier(
    random_state=SEED, n_estimators=300, max_depth=3,
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, eval_metric='logloss', verbosity=0
)
lgbm_model = LGBMClassifier(
    random_state=SEED, n_estimators=300, max_depth=4,
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, num_leaves=15, verbose=-1
)
voting_model = VotingClassifier(
    estimators=[('lr', lr_model), ('rf', rf_model), ('xgb', xgb_model), ('lgbm', lgbm_model)],
    voting='soft'
)

models_to_eval = {
    'Regressao Logistica': lr_model,
    'Random Forest V1':    rf_model,
    'XGBoost':             xgb_model,
    'LightGBM':            lgbm_model,
    'Voting Ensemble':     voting_model
}

print(f"\n{'Modelo':<30} | {'Acuracia':>10} | {'Std':>8}")
print("-" * 55)
best_score = 0
best_name  = ''
for name, model in models_to_eval.items():
    pipe   = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='accuracy')
    mu, std = scores.mean(), scores.std()
    print(f"{name:<30} | {mu:>10.4f} | {std:>8.4f}")
    if mu > best_score:
        best_score = mu
        best_name  = name

print(f"\n>>> Melhor modelo CV: {best_name} ({best_score:.4f})")
print(f">>> Score V1 Kaggle: 0.78468  |  Meta V2: 0.80000+")

# ── 6. Treinamento Final (Voting Ensemble) ───────────────────────────────────
print("\n[4/5] Treinando Voting Ensemble no conjunto completo...")

final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier',   VotingClassifier(
        estimators=[
            ('lr',   LogisticRegression(random_state=SEED, max_iter=1000, C=0.1)),
            ('rf',   RandomForestClassifier(
                random_state=SEED, n_estimators=200, max_depth=6,
                min_samples_split=10, min_samples_leaf=4, max_features='sqrt'
            )),
            ('xgb',  XGBClassifier(
                random_state=SEED, n_estimators=300, max_depth=3,
                learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0, eval_metric='logloss', verbosity=0
            )),
            ('lgbm', LGBMClassifier(
                random_state=SEED, n_estimators=300, max_depth=4,
                learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0, num_leaves=15, verbose=-1
            ))
        ],
        voting='soft'
    ))
])
final_pipeline.fit(X_train, y_train)

# ── 7. Gerar Submissão ───────────────────────────────────────────────────────
print("[5/5] Gerando arquivo de submissao...")
test_predictions = final_pipeline.predict(X_test)

sobreviventes = test_predictions.sum()
total         = len(test_predictions)
print(f"     Nao sobreviveram (0): {total - sobreviventes} ({(total-sobreviventes)/total*100:.1f}%)")
print(f"     Sobreviveram    (1): {sobreviventes} ({sobreviventes/total*100:.1f}%)")

submission = pd.DataFrame({'PassengerId': test_ids, 'Survived': test_predictions})
output_path = os.path.join(ROOT_DIR, 'submissao_titanic_v2.csv')
submission.to_csv(output_path, index=False)

print(f"\n✅ Arquivo salvo: {output_path}")
print(f"   Total de registros: {len(submission)}")
print("\nPrimeiras 5 linhas:")
print(submission.head().to_string(index=False))
