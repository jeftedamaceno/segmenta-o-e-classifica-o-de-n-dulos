import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc

# Modelos
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# =====================================================
# 1. CONFIGURAÇÕES DE PASTAS E CARREGAMENTO
# =====================================================
ARQUIVO_CSV = "dataset_radiomica_segmentado_ia.csv"
PASTA_OUTPUT = "visualizacoes_diagnosticas"

# Cria a pasta de saída se ela não existir
os.makedirs(PASTA_OUTPUT, exist_ok=True)

print("🔄 Carregando e tratando o dataset...")
df = pd.read_csv(ARQUIVO_CSV)

# Identifica as colunas de validação e IDs para remoção do treino
colunas_vazamento = [
    'ID_Nodulo', 'Volume_Ideal_GT_mm3', 'Diferenca_Absoluta_Volume_mm3', 
    'Dice_Score', 'IoU_Score'
]
colunas_remover = [col for col in colunas_vazamento if col in df.columns]

# Variáveis preditoras (X) e alvo (y)
X = df.drop(columns=colunas_remover + ['Malignidade_Real_Target'], errors='ignore')
y = df['Malignidade_Real_Target']

# Trata possíveis NaNs por segurança
X = X.fillna(X.mean())

# Divisão e Escalonamento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Configuração global de estilo para os gráficos ficarem elegantes
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13})

# =====================================================
# 2. GERAÇÃO E SALVAMENTO DE VISUALIZAÇÕES GERAIS
# =====================================================
print(f"📁 Salvando gráficos na pasta: '{PASTA_OUTPUT}'...")

# --- GRAFICO 1: Matriz de Correlação (Spearman) ---
plt.figure(figsize=(11, 9))
# Spearman captura melhor relações não-lineares comuns em radiômica
corr_matrix = df.drop(columns=['ID_Nodulo'], errors='ignore').corr(method='spearman')
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="Spectral", 
            vmin=-1, vmax=1, cbar_kws={"shrink": .8}, annot_kws={"size": 8})
plt.title("Matriz de Correlação de Spearman (Atributos IA + Métricas)", weight='bold', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(PASTA_OUTPUT, "1_matriz_correlacao.png"), dpi=300)
plt.close()
print(" -> [OK] 1_matriz_correlacao.png")

# --- GRAFICO 2: Análise de Componentes Principais (PCA 2D) ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
var_explicada = pca.explained_variance_ratio_

plt.figure(figsize=(9, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette={0: '#2ecc71', 1: '#e74c3c'}, 
                alpha=0.8, s=70, edgecolor='black', style=y)
plt.xlabel(f"Componente Principal 1 ({var_explicada[0]*100:.1f}% da Var.)")
plt.ylabel(f"Componente Principal 2 ({var_explicada[1]*100:.1f}% da Var.)")
plt.title("Projeção PCA 2D: Distribuição Espacial dos Nódulos", weight='bold', pad=15)
plt.legend(title="Target", labels=["Benigno (0)", "Maligno (1)"])
plt.tight_layout()
plt.savefig(os.path.join(PASTA_OUTPUT, "2_pca_2d.png"), dpi=300)
plt.close()
print(" -> [OK] 2_pca_2d.png")

# --- GRAFICO 3: Comportamento da Calcificação vs Variável Alvo ---
# --- GRAFICO 3: Comportamento da Calcificação vs Variável Alvo (CORRIGIDO) ---
plt.figure(figsize=(8, 5))
df_plot_calc = df.copy()
df_plot_calc['IC_Porcentagem'] = df_plot_calc['Proporcao_Calcificacao_IA'] * 100

# Convertemos para string explicitamente para bater com as chaves da paleta
df_plot_calc['Malignidade_Real_Target'] = df_plot_calc['Malignidade_Real_Target'].astype(str)

paleta_corrigida = {'0': '#3498db', '1': '#e67e22'}

sns.violinplot(
    x='Malignidade_Real_Target', 
    y='IC_Porcentagem', 
    data=df_plot_calc, 
    hue='Malignidade_Real_Target', # Define o hue explicitamente
    palette=paleta_corrigida, 
    inner="quartile",
    legend=False # Remove legenda redundante
)
sns.stripplot(
    x='Malignidade_Real_Target', 
    y='IC_Porcentagem', 
    data=df_plot_calc, 
    color='black', 
    alpha=0.3, 
    size=4, 
    jitter=0.2
)
plt.xticks([0, 1], ['Benigno (0)', 'Maligno (1)'])
plt.xlabel("Malignidade Real (Padrão Ouro)")
plt.ylabel("Índice de Calcificação (%)")
plt.title("Distribuição do Índice de Calcificação por Classe de Tumor", weight='bold', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(PASTA_OUTPUT, "3_calcificacao_vs_target.png"), dpi=300)
plt.close()
print(" -> [OK] 3_calcificacao_vs_target.png")
# --- GRAFICO 4: Dispersão de Variáveis Principais (Volume vs Média HU) ---
plt.figure(figsize=(9, 6))
sns.scatterplot(data=df, x='Volume_IA_mm3', y='HU_Mean_IA', hue='Malignidade_Real_Target',
                palette={0: '#2980b9', 1: '#c0392b'}, alpha=0.8, s=65, edgecolor='w')
plt.xlabel("Volume Segmentado pela IA (mm³)")
plt.ylabel("Densidade Média Interna (Unidades Hounsfield - HU)")
plt.title("Dispersão Clínica: Volume vs Atenuação Média", weight='bold', pad=15)
plt.legend(title="Malignidade")
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(PASTA_OUTPUT, "4_dispersao_volume_vs_hu.png"), dpi=300)
plt.close()
print(" -> [OK] 4_dispersao_volume_vs_hu.png")

# =====================================================
# 3. TREINAMENTO E MODELAGEM COM GRÁFICOS COMPLEMENTARES
# =====================================================
modelos = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42),
    "Regressão Logística": LogisticRegression(max_iter=1000, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

plt.figure(figsize=(8, 6))

for nome, modelo in modelos.items():
    modelo.fit(X_train_scaled, y_train)
    y_prob = modelo.predict_proba(X_test_scaled)[:, 1]
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{nome} (AUC = {roc_auc:.2f})')

# --- GRAFICO 5: Curvas ROC ---
plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Comparação de Modelos - Curva ROC', weight='bold', pad=15)
plt.legend(loc="lower right")
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(PASTA_OUTPUT, "5_curvas_roc_modelos.png"), dpi=300)
plt.close()
print(" -> [OK] 5_curvas_roc_modelos.png")

# --- GRAFICO 6: Importância de Recursos (Random Forest - CORRIGIDO) ---
importancias = modelos["Random Forest"].feature_importances_
df_importancia = pd.DataFrame({
    'Atributo': X.columns,
    'Importancia': importancias
}).sort_values(by='Importancia', ascending=False)

plt.figure(figsize=(10, 6))
# Definindo hue='Atributo' e legend=False para silenciar o FutureWarning do Seaborn
sns.barplot(x='Importancia', y='Atributo', data=df_importancia, hue='Atributo', palette='plasma', legend=False)
plt.xlabel('Grau de Importância Relativa')
plt.ylabel('Atributo Radiômico')
plt.title('Atributos Mais Determinantes (Random Forest)', weight='bold', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(PASTA_OUTPUT, "6_importancia_atributos.png"), dpi=300)
plt.close()
print(" -> [OK] 6_importancia_atributos.png")

# =====================================================
# CONFIGURAÇÕES DE PASTAS E CARREGAMENTO
# =====================================================
ARQUIVO_CSV = "dataset_radiomica_segmentado_ia.csv"
PASTA_OUTPUT = "visualizacoes_diagnosticas"


# Garante que as colunas necessárias de validação existam no arquivo
colunas_validacao = ['Volume_Ideal_GT_mm3', 'Volume_IA_mm3', 'Dice_Score', 'IoU_Score', 'Diferenca_Absoluta_Volume_mm3']
for col in colunas_validacao:
    if col not in df.columns:
        raise ValueError(f"A coluna essencial '{col}' não foi encontrada no seu CSV.")

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13})

# =====================================================
# 7. GRÁFICO DE BLAND-ALTMAN (CONCORDÂNCIA DE VOLUME)
# =====================================================
plt.figure(figsize=(9, 6))
vol_gt = df['Volume_Ideal_GT_mm3'].values
vol_ia = df['Volume_IA_mm3'].values

# Cálculo das variáveis do Bland-Altman
medias = (vol_gt + vol_ia) / 2
diferencas = vol_ia - vol_gt  # IA - Ouro
media_diff = np.mean(diferencas)
std_diff = np.std(diferencas)

# Limites de concordância de 95% (± 1.96 Desvios Padrão)
limite_superior = media_diff + (1.96 * std_diff)
limite_inferior = media_diff - (1.96 * std_diff)

sns.scatterplot(x=medias, y=diferencas, color='#2c3e50', alpha=0.7, s=60, edgecolor='black')
plt.axhline(media_diff, color='red', linestyle='-', linewidth=2, label=f'Viés Médio ({media_diff:.1f} mm³)')
plt.axhline(limite_superior, color='red', linestyle='--', linewidth=1.5, label=f'Limite Sup (+1.96 SD: {limite_superior:.1f})')
plt.axhline(limite_inferior, color='red', linestyle='--', linewidth=1.5, label=f'Limite Inf (-1.96 SD: {limite_inferior:.1f})')

plt.xlabel("Média dos Volumes (Padrão Ouro + IA) / 2 [mm³]")
plt.ylabel("Diferença dos Volumes (IA - Padrão Ouro) [mm³]")
plt.title("Gráfico de Bland-Altman: Concordância Volumétrica", weight='bold', pad=15)
plt.legend(loc="upper right", frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(PASTA_OUTPUT, "7_bland_altman_volume.png"), dpi=300)
plt.close()
print(" -> [OK] 7_bland_altman_volume.png")

# =====================================================
# 2. DISTRIBUIÇÃO DOS SCORES DE SOBREPOSIÇÃO (DICE E IOU - CORRIGIDO)
# =====================================================
plt.figure(figsize=(8, 6))

df_scores = df[['Dice_Score', 'IoU_Score']].melt(var_name='Métrica', value_name='Score')
df_scores['Métrica'] = df_scores['Métrica'].map({'Dice_Score': 'Dice Coefficient', 'IoU_Score': 'Jaccard Index (IoU)'})

paleta_scores = {'Dice Coefficient': '#9b59b6', 'Jaccard Index (IoU)': '#1abc9c'}

# Corrigido passando hue e desativando a legenda redundante
sns.violinplot(
    x='Métrica', 
    y='Score', 
    data=df_scores, 
    hue='Métrica', 
    palette=paleta_scores, 
    inner=None, 
    linewidth=1.5,
    legend=False
)

# Corrigido o argumento de 'whiskproproprops' para 'whiskerprops' (palavra correta do matplotlib)
sns.boxplot(
    x='Métrica', 
    y='Score', 
    data=df_scores, 
    width=0.15, 
    color='white', 
    boxprops=dict(alpha=0.6), 
    whiskerprops=dict(color='black'), 
    medianprops=dict(color='red', linewidth=2)
)

plt.ylim([-0.05, 1.05])
plt.xlabel("Métricas de Validação Espacial")
plt.ylabel("Valor do Coeficiente (0 a 1)")
plt.title("Distribuição e Densidade de Acerto da Segmentação IA", weight='bold', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(PASTA_OUTPUT, "8_distribuicao_dice_iou.png"), dpi=300)
plt.close()
print(" -> [OK] 8_distribuicao_dice_iou.png")

# =====================================================
# 3. HISTOGRAMA DO ERRO ABSOLUTO DE VOLUME
# =====================================================
plt.figure(figsize=(9, 5))
sns.histplot(df['Diferenca_Absoluta_Volume_mm3'], bins=25, kde=True, color='#e74c3c', alpha=0.6, edgecolor='black')

median_error = df['Diferenca_Absoluta_Volume_mm3'].median()
mean_error = df['Diferenca_Absoluta_Volume_mm3'].mean()

plt.axvline(median_error, color='blue', linestyle='--', linewidth=2, label=f'Mediana do Erro: {median_error:.1f} mm³')
plt.axvline(mean_error, color='purple', linestyle=':', linewidth=2, label=f'Média do Erro: {mean_error:.1f} mm³')

plt.xlabel("Diferença Absoluta Volumétrica ( |Ideal - IA| ) [mm³]")
plt.ylabel("Frequência de Nódulos")
plt.title("Análise do Desvio Volumétrico Absoluto", weight='bold', pad=15)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PASTA_OUTPUT, "9_erro_absoluto_volume.png"), dpi=300)
plt.close()
print(" -> [OK] 9_erro_absoluto_volume.png")

# =====================================================
# RESUMO NO TERMINAL
# =====================================================
print("\n" + "="*60)
print("📈 MÉTRICAS GERAIS DE QUALIDADE DA SEGMENTAÇÃO")
print("="*60)
print(f"-> Média Geral do Coeficiente Dice : {df['Dice_Score'].mean():.4f}")
print(f"-> Mediana Geral do Coeficiente Dice: {df['Dice_Score'].median():.4f}")
print(f"-> Média Geral do Jaccard (IoU)   : {df['IoU_Score'].mean():.4f}")
print(f"-> Erro Médio de Volume            : {mean_error:.2f} mm³")
print(f"-> Viés de Medição (Bland-Altman)  : {media_diff:.2f} mm³")
print(f"Todas as figuras de validação foram salvas em '{PASTA_OUTPUT}'")
print("="*60)