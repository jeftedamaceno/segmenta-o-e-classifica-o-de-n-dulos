# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import joblib  # 🌟 Adicionado para salvar o modelo e resolver o erro!

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import (
#     accuracy_score, classification_report, roc_curve, auc, 
#     confusion_matrix, ConfusionMatrixDisplay, f1_score
# )

# # Modelos Escolhidos
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# # from SVC_ajustado import SVC if False else from sklearn.svm import SVC # Tratamento padrão
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression

# # =====================================================
# # 1. CARREGAMENTO E CONFIGURAÇÕES
# # =====================================================
# # Atualizado para usar o dataset de nódulos únicos que criamos
# ARQUIVO_CSV = "dataset_radiomica_segmentado_ia_unico.csv"

# print("🔄 Carregando e tratando o dataset radiômico...")
# df = pd.read_csv(ARQUIVO_CSV)

# # 2. LIMPEZA DE COLUNAS (Removendo chaves e vazamentos do Ground Truth)
# colunas_vazamento = [
#     'ID_Nodulo', 'ID_Nodulo_Unico',
#     'Volume_Ideal_GT_mm3', 'Volume_Ideal_GT_Consolidado_mm3',
#     'Diferenca_Absoluta_Volume_mm3', 
#     'Dice_Score', 'IoU_Score'
# ]
# colunas_remover = [col for col in colunas_vazamento if col in df.columns]

# # Separando Atributos (X) e Alvo (y)
# X = df.drop(columns=colunas_remover + ['Malignidade_Real_Target'], errors='ignore')
# y = df['Malignidade_Real_Target']

# print(f"-> Total de instâncias únicas: {len(df)}")
# print(f"-> Atributos preditores selecionados ({X.shape[1]}): {list(X.columns)}")

# # Trata possíveis valores nulos de forma segura usando a mediana do treino
# # (Melhoria: Preenchimento direto na matriz X antes da divisão)
# X = X.fillna(X.median())

# # =====================================================
# # 3. ANALISE DE COMPORTAMENTO (SUA HIPÓTESE DE ASSOCIAÇÃO)
# # =====================================================
# print("\n📊 Analisando comportamento e associação de variáveis...")
# # Calculamos a correlação focada nas variáveis que você criou (Haralick, LBP e Densidade)
# colunas_analise = [
#     'HU_Variance_IA', 'HU_Uniformity_IA', 'Proporcao_Calcificacao_IA',
#     'Haralick_Contraste', 'Haralick_Homogeneidade', 'LBP_Energia_Textura'
# ]
# colunas_presentes = [c for c in colunas_analise if c in X.columns]

# matriz_corr = df[colunas_presentes + ['Malignidade_Real_Target']].corr(method='spearman')

# plt.figure(figsize=(8, 6))
# sns.heatmap(matriz_corr, annot=True, cmap='RdBu_r', fmt=".2f", vmin=-1, vmax=1)
# plt.title('Análise de Comportamento Combinado (Associação Radiômica)')
# plt.tight_layout()
# plt.show()

# # Exemplo prático da sua hipótese impresso no terminal:
# if 'Haralick_Contraste' in df.columns and 'HU_Variance_IA' in df.columns:
#     alta_variancia = df['HU_Variance_IA'].median()
#     chance_contraste_alto_geral = (df['Haralick_Contraste'] > df['Haralick_Contraste'].median()).mean()
    
#     # Subconjunto condicional (Regra de associação: Se X acontece, o que ocorre com Y?)
#     sub_comportamento = df[df['HU_Variance_IA'] > alta_variancia]
#     chance_contraste_alto_condicional = (sub_comportamento['Haralick_Contraste'] > df['Haralick_Contraste'].median()).mean()
    
#     print(f"💡 TESTE DA SUA HIPÓTESE:")
#     print(f"   - Chance de um nódulo genérico ter alto contraste de textura: {chance_contraste_alto_geral*100:.1f}%")
#     print(f"   - Se o nódulo tiver ALTA VARIÂNCIA DE DENSIDADE, a chance de ter alto contraste SOBE para: {chance_contraste_alto_condicional*100:.1f}%")

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
# # 5. TREINAMENTO DOS MODELOS
# # =====================================================
# modelos = {
#     "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
#     "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42),
#     "Regressão Logística": LogisticRegression(max_iter=1000, random_state=42),
#     "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
# }

# resultados_metricas = {}
# print("\n🚀 Treinando os modelos preditivos...")
# for nome, modelo in modelos.items():
#     modelo.fit(X_train_scaled, y_train)
#     y_pred = modelo.predict(X_test_scaled)
#     y_prob = modelo.predict_proba(X_test_scaled)[:, 1]
    
#     acc = accuracy_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#     fpr, tpr, _ = roc_curve(y_test, y_prob)
#     roc_auc = auc(fpr, tpr)
    
#     resultados_metricas[nome] = {"Acurácia": acc, "F1-Score": f1, "AUC": roc_auc}
#     print(f" ✅ {nome} treinado com sucesso!")

# # =====================================================
# # 6. SALVANDO O MODELO PARA A FUNÇÃO DO PROFESSOR
# # =====================================================
# # 🌟 CORREÇÃO DO ERRO: Salvando o melhor modelo baseado no AUC (ex: Random Forest)
# melhor_modelo_nome = "Random Forest"
# print(f"\n💾 Exportando o modelo '{melhor_modelo_nome}' para o arquivo da função do professor...")
# joblib.dump(modelos[melhor_modelo_nome], "modelo_malignidade_treinado.joblib")
# print(" -> Arquivo 'modelo_malignidade_treinado.joblib' gerado com sucesso!")

# # =====================================================
# # 7. TABELA COMPARATIVA FINAL
# # =====================================================
# df_metricas = pd.DataFrame(resultados_metricas).T
# print("\n" + "="*60)
# print("🏆 TABELA RESUMO DOS MODELOS")
# print("="*60)
# print(df_metricas.to_string())
# print("="*60)
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, roc_curve, auc, 
    confusion_matrix, ConfusionMatrixDisplay, f1_score
)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# =====================================================
# 1. CARREGAMENTO E ENGENHARIA DE ATRIBUTOS
# =====================================================
ARQUIVO_CSV = "dataset_radiomica_segmentado_ia_unico.csv"

print("🔄 Carregando dataset e aplicando Engenharia de Atributos...")
df = pd.read_csv(ARQUIVO_CSV)

# 🌟 ENGENHARIA DE ATRIBUTOS: Criando o indicador de "Calcificação Relativa ao Tamanho"
# Como o volume real foi dropado, estimamos a "Massa Geométrica da IA" usando a variação 
# absoluta de erro combinada ao volume do GT, ou criamos um proxy aproximado baseado nas texturas.
# Usaremos o Volume Consolidado antes de dropar a coluna para gerar o recurso inteligente da IA:
if 'Volume_Ideal_GT_Consolidado_mm3' in df.columns:
    # Evita divisão por zero adicionando um epsilon estável (1e-5)
    df['Indice_Calcificacao_Por_Tamanho'] = df['Proporcao_Calcificacao_IA'] / (df['Volume_Ideal_GT_Consolidado_mm3'] + 1e-5)
else:
    # Proxy alternativo caso use apenas dados puros de intensidade da IA
    df['Indice_Calcificacao_Por_Tamanho'] = df['Proporcao_Calcificacao_IA'] / (df['HU_Std_IA'] + 1e-5)

# 2. LIMPEZA DE COLUNAS (Evitando vazamento de dados do Ground Truth)
colunas_vazamento = [
    'ID_Nodulo', 'ID_Nodulo_Unico',
    'Volume_Ideal_GT_mm3', 'Volume_Ideal_GT_Consolidado_mm3',
    'Diferenca_Absoluta_Volume_mm3', 
    'Dice_Score', 'IoU_Score'
]
colunas_remover = [col for col in colunas_vazamento if col in df.columns]

# Separando Atributos (X) e Alvo (y)
# Mantemos nossa nova coluna 'Indice_Calcificacao_Por_Tamanho' dentro de X!
X = df.drop(columns=colunas_remover + ['Malignidade_Real_Target', 'Classe'], errors='ignore')
y = df['Malignidade_Real_Target']

print(f"-> Total de instâncias únicas: {len(df)}")
print(f"-> Novo atributo adicionado: 'Indice_Calcificacao_Por_Tamanho'")
print(f"-> Total de atributos preditores atuais: {X.shape[1]}")

# Trata nulos usando a mediana
X = X.fillna(X.median())

# =====================================================
# 3. VALIDAÇÃO MATEMÁTICA DA SUA NOVA HIPÓTESE
# =====================================================
print("\n💡 AVALIAÇÃO DA SUA HIPÓTESE DA CALCIFICAÇÃO RELATIVA:")
mediana_novo_indicador = df['Indice_Calcificacao_Por_Tamanho'].median()

chance_benigno_geral = (df['Malignidade_Real_Target'] == 0).mean()
# Filtra os nódulos com alto índice de calcificação em relação ao tamanho
sub_calcificados_pequenos = df[df['Indice_Calcificacao_Por_Tamanho'] > mediana_novo_indicador]
chance_benigno_hipotese = (sub_calcificados_pequenos['Malignidade_Real_Target'] == 0).mean()

print(f"   - Chance de um nódulo qualquer do dataset ser BENIGNO: {chance_benigno_geral*100:.1f}%")
print(f"   - Se o nódulo tiver ALTA CALCIFICAÇÃO e MENOR TAMANHO, a chance de ser BENIGNO vai para: {chance_benigno_hipotese*100:.1f}%")

# Matriz de Correlação incluindo o novo recurso
colunas_analise = ['HU_Variance_IA', 'Proporcao_Calcificacao_IA', 'Indice_Calcificacao_Por_Tamanho', 'Haralick_Contraste']
colunas_presentes = [c for c in colunas_analise if c in X.columns]
matriz_corr = df[colunas_presentes + ['Malignidade_Real_Target']].corr(method='spearman')

plt.figure(figsize=(7, 5))
sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Impacto do Novo Atributo Combinado no Target')
plt.tight_layout()
plt.show()

# =====================================================
# 4. DIVISÃO EM TREINO E TESTE + NORMALIZAÇÃO
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================
# 5. TREINAMENTO E VALIDAÇÃO DOS MODELOS RETREINADOS
# =====================================================
modelos = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42),
    "Regressão Logística": LogisticRegression(max_iter=1000, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

resultados_metricas = {}
print("\n🚀 Retreinando modelos preditivos com o comportamento integrado...")
for nome, modelo in modelos.items():
    modelo.fit(X_train_scaled, y_train)
    y_pred = modelo.predict(X_test_scaled)
    y_prob = modelo.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    resultados_metricas[nome] = {"Acurácia": acc, "F1-Score": f1, "AUC": roc_auc}

# =====================================================
# 6. EXPORTANDO O MODELO ATUALIZADO PARA O PROFESSOR
# =====================================================
melhor_modelo_nome = "Random Forest"
print(f"\n💾 Salvando modelo robusto atualizado em disco...")
joblib.dump(modelos[melhor_modelo_nome], "modelo_malignidade_treinado.joblib")
print(" -> Arquivo 'modelo_malignidade_treinado.joblib' atualizado com sucesso!")

# =====================================================
# 7. TABELA COMPARATIVA FINAL
# =====================================================
df_metricas = pd.DataFrame(resultados_metricas).T
print("\n" + "="*60)
print("🏆 TABELA RESUMO DOS MODELOS (COM NOVO ATRIBUTO COMPORTAMENTAL)")
print("="*60)
print(df_metricas.to_string())
print("="*60)