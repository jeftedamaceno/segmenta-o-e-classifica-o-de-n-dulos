import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV  
from sklearn.preprocessing import RobustScaler # 🌟 Mudado para tratar dados vazios/medianos
from sklearn.metrics import (
    accuracy_score, classification_report, roc_curve, auc, 
    confusion_matrix, confusion_matrix, f1_score
)

# Modelos Escolhidos
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# =====================================================
# 1. CARREGAMENTO E TRATAMENTO DEFENSIVO
# =====================================================
ARQUIVO_CSV = "sem_duplicates.csv"
PASTA_OUTPUT = "visualizacoes_diagnosticas"

os.makedirs(PASTA_OUTPUT, exist_ok=True)

print("🔄 Carregando dataset e aplicando Engenharia Defensiva...")
df = pd.read_csv(ARQUIVO_CSV)

# Atributos compostos para blindagem
if 'Volume_Ideal_GT_Consolidado_mm3' in df.columns:
    df['Indice_Calcificacao_Por_Tamanho'] = df['Proporcao_Calcificacao_IA'] / (df['Volume_Ideal_GT_Consolidado_mm3'] + 1e-5)
else:
    df['Indice_Calcificacao_Por_Tamanho'] = df['Proporcao_Calcificacao_IA'] / (df['HU_Std_IA'] + 1e-5)

if 'HU_Variance_IA' in df.columns and 'Haralick_Contraste' in df.columns:
    df['Caos_Tecidual_Combinado'] = df['HU_Variance_IA'] * df['Haralick_Contraste']

colunas_vazamento = ['ID_Nodulo', 'ID_Nodulo_Unico', 'Volume_Ideal_GT_mm3', 'Volume_Ideal_GT_Consolidado_mm3', 'Diferenca_Absoluta_Volume_mm3', 'Dice_Score', 'IoU_Score']
colunas_remover = [col for col in colunas_vazamento if col in df.columns]

X = df.drop(columns=colunas_remover + ['Malignidade_Real_Target', 'Classe'], errors='ignore')
y = df['Malignidade_Real_Target']

# Substitui nulos pela mediana (essencial para não quebrar o treino)
X = X.fillna(X.median())

# =====================================================
# 2. DIVISÃO E ESCALONAMENTO ROBUSTO
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# RobustScaler lida muito melhor com features zeradas ou ruidosas vindas do teste em lote
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Salva o scaler junto para usar no script do teste em lote!
joblib.dump(scaler, "scaler_robusto.joblib")

# =====================================================
# 3. TUNING DE HIPERPARÂMETROS COM FOCO EM RECALL/F1
# =====================================================
print("\n⚙️ Sintonizando Random Forest com penalização equilibrada...")
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [5, 8, 12],
    'min_samples_split': [3, 5],
    'max_features': ['sqrt']
}

grid_rf = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight='balanced_subsample'), 
    param_grid_rf, 
    cv=5, 
    scoring='f1', # Força o acerto da classe minoritária/complicada
    n_jobs=-1
)
grid_rf.fit(X_train_scaled, y_train)

# Esteira de modelos configurada para combater o viés
modelos = {
    "Random Forest Otimizado": grid_rf.best_estimator_,
    "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42, class_weight='balanced'),
    "Regressão Logística": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

resultados_metricas = {}

print("\n🚀 Treinando e aplicando limiar adaptativo de decisão...")
for nome, modelo in modelos.items():
    if nome == "Gradient Boosting":
        pesos = y_train.map({0: 1.0, 1: len(y_train[y_train==0])/len(y_train[y_train==1])})
        modelo.fit(X_train_scaled, y_train, sample_weight=pesos)
    else:
        modelo.fit(X_train_scaled, y_train)
        
    # Captura a probabilidade contínua em vez da classe direta dura (0 ou 1)
    y_prob = modelo.predict_proba(X_test_scaled)[:, 1]
    
    # 🎯 CORREÇÃO DO VIÉS CLÍNICO: Se a probabilidade for maior que 35%, assume maligno.
    # Isso impede que o modelo esconda os 42 malignos sob o rótulo de benigno por medo de errar.
    LIMIAR_CLINICO = 0.35 
    y_pred_adaptado = (y_prob >= LIMIAR_CLINICO).astype(int)
    
    acc = accuracy_score(y_test, y_pred_adaptado)
    f1 = f1_score(y_test, y_pred_adaptado)
    
    resultados_metricas[nome] = {"Acurácia": acc, "F1-Score": f1}
    print(f" ✅ {nome} calibrado com limiar de {LIMIAR_CLINICO*100}%!")

# Salva o modelo campeão corrigido
joblib.dump(modelos["Random Forest Otimizado"], "modelo_malignidade_treinado.joblib")
print("\n💾 Modelo com limiar ajustado e scaler robusto foram salvos com sucesso!")

# Exibe tabela resumo
df_metricas = pd.DataFrame(resultados_metricas).T
print("\n" + "="*60)
print("🏆 MODELOS CORRIGIDOS E REBALANCED")
print("="*60)
print(df_metricas.to_string())
print("="*60)