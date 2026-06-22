import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV  
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, roc_curve, auc, 
    confusion_matrix, ConfusionMatrixDisplay, f1_score
)

# Modelos Escolhidos
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# =====================================================
# 1. CARREGAMENTO E ENGENHARIA DE ATRIBUTOS AVANÇADA
# =====================================================
ARQUIVO_CSV = "sem_duplicates.csv"
PASTA_OUTPUT = "visualizacoes_diagnosticas"

# Cria a pasta de saída para os gráficos se ela não existir
os.makedirs(PASTA_OUTPUT, exist_ok=True)

print("🔄 Carregando dataset e aplicando Engenharia de Atributos Avançada...")
df = pd.read_csv(ARQUIVO_CSV)

# Atributo Composto 1: Calcificação proporcional ao tamanho
if 'Volume_Ideal_GT_Consolidado_mm3' in df.columns:
    df['Indice_Calcificacao_Por_Tamanho'] = df['Proporcao_Calcificacao_IA'] / (df['Volume_Ideal_GT_Consolidado_mm3'] + 1e-5)
else:
    df['Indice_Calcificacao_Por_Tamanho'] = df['Proporcao_Calcificacao_IA'] / (df['HU_Std_IA'] + 1e-5)

# Recursos de Interação Não Linear para facilitar o corte das árvores
if 'HU_Variance_IA' in df.columns and 'Haralick_Contraste' in df.columns:
    df['Caos_Tecidual_Combinado'] = df['HU_Variance_IA'] * df['Haralick_Contraste']
if 'Haralick_Homogeneidade' in df.columns and 'LBP_Energia_Textura' in df.columns:
    df['Estabilidade_Textural_Combinada'] = df['Haralick_Homogeneidade'] * df['LBP_Energia_Textura']

# 2. LIMPEZA DE COLUNAS
colunas_vazamento = [
    'ID_Nodulo', 'ID_Nodulo_Unico',
    'Volume_Ideal_GT_mm3', 'Volume_Ideal_GT_Consolidado_mm3',
    'Diferenca_Absoluta_Volume_mm3', 
    'Dice_Score', 'IoU_Score'
]
colunas_remover = [col for col in colunas_vazamento if col in df.columns]

X = df.drop(columns=colunas_remover + ['Malignidade_Real_Target', 'Classe'], errors='ignore')
y = df['Malignidade_Real_Target']

print(f"-> Atributos preditores expandidos para o treino ({X.shape[1]}): {list(X.columns)}")

X = X.fillna(X.median())

# =====================================================
# 3. DIVISÃO EM TREINO E TESTE + NORMALIZAÇÃO
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Salva também o scaler robusto para alinhar com o script de inferência do professor
joblib.dump(scaler, "scaler_robusto.joblib")

# =====================================================
# 4. TUNING DE HIPERPARÂMETROS COM GRIDSEARCHCV
# =====================================================
print("\n⚙️ Buscando as melhores combinações de parâmetros (GridSearchCV)...")

param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt', 'log2']
}

grid_rf = GridSearchCV(
    RandomForestClassifier(random_state=42), 
    param_grid_rf, 
    cv=5, 
    scoring='roc_auc', 
    n_jobs=-1
)
grid_rf.fit(X_train_scaled, y_train)
print(f" -> Melhor configuração para Random Forest: {grid_rf.best_params_}")

# Definindo a esteira de modelos com o campeão otimizado
modelos = {
    "Random Forest Otimizado": grid_rf.best_estimator_,
    "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42),
    "Regressão Logística": LogisticRegression(max_iter=1000, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

resultados_metricas = {}
Dicionario_Roc = {} # Armazena taxas para plotagem posterior

print("\n🚀 Treinando os modelos finais...")
for nome, modelo in modelos.items():
    modelo.fit(X_train_scaled, y_train)
    y_pred = modelo.predict(X_test_scaled)
    y_prob = modelo.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    resultados_metricas[nome] = {"Acurácia": acc, "F1-Score": f1, "AUC": roc_auc}
    Dicionario_Roc[nome] = (fpr, tpr, roc_auc)
    print(f" ✅ {nome} treinado!")

# =====================================================
# 5. SALVANDO O MODELO OTIMIZADO PARA O PROFESSOR
# =====================================================
melhor_modelo_nome = "Random Forest Otimizado"
print(f"\n💾 Salvando modelo de alta precisão...")
joblib.dump(modelos[melhor_modelo_nome], "modelo_malignidade_treinado.joblib")
print(" -> Arquivo binário '.joblib' atualizado com os parâmetros ideais!")

# =====================================================
# 6. TABELA COMPARATIVA FINAL
# =====================================================
df_metricas = pd.DataFrame(resultados_metricas).T
print("\n" + "="*60)
print("🏆 TABELA RESUMO DOS MODELOS EVOLUÍDOS")
print("="*60)
print(df_metricas.to_string())
print("="*60)

# =====================================================
# 7. GERAÇÃO DA CURVA ROC EVOLUÍDA
# =====================================================
print(f"\n🎨 Gerando gráfico comparativo das Curvas ROC...")
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8.5, 6.5))

cores_modelos = {
    "Random Forest Otimizado": "#2ecc71",
    "SVM (RBF)": "#3498db",
    "Regressão Logística": "#9b59b6",
    "Gradient Boosting": "#e74c3c"
}

for nome, (fpr, tpr, roc_auc) in Dicionario_Roc.items():
    plt.plot(fpr, tpr, lw=2.5, color=cores_modelos[nome],
             label=f'{nome} (AUC = {roc_auc:.4f})')

plt.plot([0, 1], [0, 1], color='#7f8c8d', lw=1.5, linestyle='--')
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('Taxa de Falsos Positivos (1 - Especificidade)', fontsize=11, fontweight='bold', labelpad=10)
plt.ylabel('Taxa de Verdadeiros Positivos (Sensibilidade)', fontsize=11, fontweight='bold', labelpad=10)
plt.title('Comparação de Modelos Evoluídos - Curva ROC', fontsize=13, weight='bold', pad=15)
plt.legend(loc="lower right", frameon=True, facecolor='white', edgecolor='#bdc3c7', fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

caminho_roc = os.path.join(PASTA_OUTPUT, "curva_roc_modelos_evoluidos.png")
plt.savefig(caminho_roc, dpi=300)
plt.close()
print(f" -> [OK] Curva ROC salva em: '{caminho_roc}'")

# =====================================================
# 8. 📊 MATRIZ DE CONFUSÃO DO MODELO CAMPEÃO (NOVO BLOCO)
# =====================================================
print(f"\n📊 Gerando Matriz de Confusão para o {melhor_modelo_nome}...")

# Predição do conjunto de teste usando o modelo campeão
modelo_campeao = modelos[melhor_modelo_nome]
y_pred_campeao = modelo_campeao.predict(X_test_scaled)

# Cálculo matemático da matriz
cm = confusion_matrix(y_test, y_pred_campeao)

# Configuração visual refinada com Seaborn
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Greens', # Paleta combinando com a cor do Random Forest no gráfico ROC
    linewidths=1.5, 
    cbar=False,
    annot_kws={"fontsize": 13, "weight": "bold"},
    xticklabels=['Benigno (0)', 'Maligno (1)'],
    yticklabels=['Benigno (0)', 'Maligno (1)']
)

plt.xlabel('Classe Predita pela IA', fontsize=11, fontweight='bold', labelpad=10)
plt.ylabel('Classe Real (Gabarito Médico)', fontsize=11, fontweight='bold', labelpad=10)
plt.title(f'Matriz de Confusão Final\n({melhor_modelo_nome})', fontsize=12, weight='bold', pad=15)
plt.tight_layout()

caminho_cm = os.path.join(PASTA_OUTPUT, "matriz_confusao_modelo_final.png")
plt.savefig(caminho_cm, dpi=300)
plt.close()

print(f" -> [OK] Matriz de Confusão salva com sucesso em: '{caminho_cm}'")
print("\n📋 Distribuição Numérica da Matriz (Conjunto de Teste):")
print(f"    - Verdadeiros Benignos (VN): {cm[0][0]}")
print(f"    - Falsos Malignos (FP):     {cm[0][1]}")
print(f"    - Falsos Benignos (FN):     {cm[1][0]}")
print(f"    - Verdadeiros Malignos (VP): {cm[1][1]}")
print("="*60)
print("🏁 Execução concluída! Todos os insumos visuais e arquivos binários estão prontos.")
print("="*60)