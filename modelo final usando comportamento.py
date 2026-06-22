
# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import joblib

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import (
#     accuracy_score, classification_report, roc_curve, auc, 
#     confusion_matrix, ConfusionMatrixDisplay, f1_score
# )

# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression

# # =====================================================
# # 1. CARREGAMENTO E ENGENHARIA DE ATRIBUTOS
# # =====================================================
# ARQUIVO_CSV = "dataset_radiomica_segmentado_ia_unico.csv"

# print("🔄 Carregando dataset e aplicando Engenharia de Atributos...")
# df = pd.read_csv(ARQUIVO_CSV)

# # 🌟 ENGENHARIA DE ATRIBUTOS: Criando o indicador de "Calcificação Relativa ao Tamanho"
# # Como o volume real foi dropado, estimamos a "Massa Geométrica da IA" usando a variação 
# # absoluta de erro combinada ao volume do GT, ou criamos um proxy aproximado baseado nas texturas.
# # Usaremos o Volume Consolidado antes de dropar a coluna para gerar o recurso inteligente da IA:
# if 'Volume_Ideal_GT_Consolidado_mm3' in df.columns:
#     # Evita divisão por zero adicionando um epsilon estável (1e-5)
#     df['Indice_Calcificacao_Por_Tamanho'] = df['Proporcao_Calcificacao_IA'] / (df['Volume_Ideal_GT_Consolidado_mm3'] + 1e-5)
# else:
#     # Proxy alternativo caso use apenas dados puros de intensidade da IA
#     df['Indice_Calcificacao_Por_Tamanho'] = df['Proporcao_Calcificacao_IA'] / (df['HU_Std_IA'] + 1e-5)

# # 2. LIMPEZA DE COLUNAS (Evitando vazamento de dados do Ground Truth)
# colunas_vazamento = [
#     'ID_Nodulo', 'ID_Nodulo_Unico',
#     'Volume_Ideal_GT_mm3', 'Volume_Ideal_GT_Consolidado_mm3',
#     'Diferenca_Absoluta_Volume_mm3', 
#     'Dice_Score', 'IoU_Score'
# ]
# colunas_remover = [col for col in colunas_vazamento if col in df.columns]

# # Separando Atributos (X) e Alvo (y)
# # Mantemos nossa nova coluna 'Indice_Calcificacao_Por_Tamanho' dentro de X!
# X = df.drop(columns=colunas_remover + ['Malignidade_Real_Target', 'Classe'], errors='ignore')
# y = df['Malignidade_Real_Target']

# print(f"-> Total de instâncias únicas: {len(df)}")
# print(f"-> Novo atributo adicionado: 'Indice_Calcificacao_Por_Tamanho'")
# print(f"-> Total de atributos preditores atuais: {X.shape[1]}")

# # Trata nulos usando a mediana
# X = X.fillna(X.median())

# # =====================================================
# # 3. VALIDAÇÃO MATEMÁTICA DA SUA NOVA HIPÓTESE
# # =====================================================
# print("\n💡 AVALIAÇÃO DA SUA HIPÓTESE DA CALCIFICAÇÃO RELATIVA:")
# mediana_novo_indicador = df['Indice_Calcificacao_Por_Tamanho'].median()

# chance_benigno_geral = (df['Malignidade_Real_Target'] == 0).mean()
# # Filtra os nódulos com alto índice de calcificação em relação ao tamanho
# sub_calcificados_pequenos = df[df['Indice_Calcificacao_Por_Tamanho'] > mediana_novo_indicador]
# chance_benigno_hipotese = (sub_calcificados_pequenos['Malignidade_Real_Target'] == 0).mean()

# print(f"   - Chance de um nódulo qualquer do dataset ser BENIGNO: {chance_benigno_geral*100:.1f}%")
# print(f"   - Se o nódulo tiver ALTA CALCIFICAÇÃO e MENOR TAMANHO, a chance de ser BENIGNO vai para: {chance_benigno_hipotese*100:.1f}%")

# # Matriz de Correlação incluindo o novo recurso
# colunas_analise = ['HU_Variance_IA', 'Proporcao_Calcificacao_IA', 'Indice_Calcificacao_Por_Tamanho', 'Haralick_Contraste']
# colunas_presentes = [c for c in colunas_analise if c in X.columns]
# matriz_corr = df[colunas_presentes + ['Malignidade_Real_Target']].corr(method='spearman')

# plt.figure(figsize=(7, 5))
# sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
# plt.title('Impacto do Novo Atributo Combinado no Target')
# plt.tight_layout()
# plt.show()

# # =====================================================
# # 4. DIVISÃO EM TREINO E TESTE + NORMALIZAÇÃO
# # =====================================================
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.20, random_state=42, stratify=y
# )

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # =====================================================
# # 5. TREINAMENTO E VALIDAÇÃO DOS MODELOS RETREINADOS
# # =====================================================
# modelos = {
#     "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
#     "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42),
#     "Regressão Logística": LogisticRegression(max_iter=1000, random_state=42),
#     "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
# }

# resultados_metricas = {}
# print("\n🚀 Retreinando modelos preditivos com o comportamento integrado...")
# for nome, modelo in modelos.items():
#     modelo.fit(X_train_scaled, y_train)
#     y_pred = modelo.predict(X_test_scaled)
#     y_prob = modelo.predict_proba(X_test_scaled)[:, 1]
    
#     acc = accuracy_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#     fpr, tpr, _ = roc_curve(y_test, y_prob)
#     roc_auc = auc(fpr, tpr)
    
#     resultados_metricas[nome] = {"Acurácia": acc, "F1-Score": f1, "AUC": roc_auc}

# # =====================================================
# # 6. EXPORTANDO O MODELO ATUALIZADO PARA O PROFESSOR
# # =====================================================
# melhor_modelo_nome = "Random Forest"
# print(f"\n💾 Salvando modelo robusto atualizado em disco...")
# joblib.dump(modelos[melhor_modelo_nome], "modelo_malignidade_treinado.joblib")
# print(" -> Arquivo 'modelo_malignidade_treinado.joblib' atualizado com sucesso!")

# # =====================================================
# # 7. TABELA COMPARATIVA FINAL
# # =====================================================
# df_metricas = pd.DataFrame(resultados_metricas).T
# print("\n" + "="*60)
# print("🏆 TABELA RESUMO DOS MODELOS (COM NOVO ATRIBUTO COMPORTAMENTAL)")
# print("="*60)
# print(df_metricas.to_string())
# print("="*60)
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

# Atributo Composto 1: Calcificação proporcional ao tamanho (Sua hipótese)
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
# 7. GERAÇÃO DA CURVA ROC EVOLUÍDA (NOVO BLOCO)
# =====================================================
print(f"\n🎨 Gerando gráfico comparativo das Curvas ROC...")
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8.5, 6.5))

# Define cores estéticas para cada modelo
cores_modelos = {
    "Random Forest Otimizado": "#2ecc71",
    "SVM (RBF)": "#3498db",
    "Regressão Logística": "#9b59b6",
    "Gradient Boosting": "#e74c3c"
}

for nome, (fpr, tpr, roc_auc) in Dicionario_Roc.items():
    plt.plot(fpr, tpr, lw=2.5, color=cores_modelos[nome],
             label=f'{nome} (AUC = {roc_auc:.4f})')

# Linha de referência diagonal (classificador aleatório)
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

print(f" -> [OK] Curva ROC salva com sucesso em: '{caminho_roc}'")
print("="*60)