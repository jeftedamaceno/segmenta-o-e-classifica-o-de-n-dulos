import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# 1. PREPARAÇÃO DO AMBIENTE E DADOS
# =====================================================
ARQUIVO_CSV = "dataset_final.csv"
PASTA_SAIDA = "visualizacoes_diagnosticas"

# Cria a pasta requisitada se ela não existir
os.makedirs(PASTA_SAIDA, exist_ok=True)

print("🔄 Carregando dataset limpo de nódulos únicos...")
df = pd.read_csv(ARQUIVO_CSV)

# Mapeia o Target numérico para nomes amigáveis nos gráficos
df['Classe'] = df['Malignidade_Real_Target'].map({0: 'Benigno', 1: 'Maligno'})

# Volume do Ground Truth unificado para representar o tamanho real da estrutura
coluna_tamanho = 'Volume_Ideal_GT_Consolidado_mm3'

# 5 Variáveis adicionais de textura que você implementou
variaveis_textura = [
    'Haralick_Contraste', 
    'Haralick_Homogeneidade', 
    'Haralick_Energia', 
    'Haralick_Correlacao', 
    'LBP_Energia_Textura'
]

print("📊 Gerando estudos diagnósticos visuais...")

# =====================================================
# ESTUDO 1: A INTERAÇÃO CALCIFICAÇÃO VS TAMANHO (VOLUME)
# =====================================================
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df, 
    x=coluna_tamanho, 
    y='Proporcao_Calcificacao_IA', 
    hue='Classe', 
    style='Classe',
    palette={'Benigno': '#2ecc71', 'Maligno': '#e74c3c'},
    alpha=0.8, 
    s=100
)
plt.title('Associação Clínica: Volume do Nódulo vs. Índice de Calcificação', fontsize=12, pad=15)
plt.xlabel('Volume do Nódulo (mm³)', fontsize=10)
plt.ylabel('Proporção de Calcificação Detectada pela IA', fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
caminho_grafico1 = os.path.join(PASTA_SAIDA, '1_interacao_volume_calcificacao.png')
plt.savefig(caminho_grafico1, dpi=300)
plt.close()
print(f" -> Salvo: {caminho_grafico1}")

# =====================================================
# ESTUDO 2: DISTRIBUIÇÃO DAS 5 VARIÁVEIS DE TEXTURA
# =====================================================
# 🌟 CORRIGIDO: Adicionado hue='Classe' e legend=False para sumir com o aviso do Seaborn!
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.ravel()

for idx, var in enumerate(variaveis_textura):
    if var in df.columns:
        sns.boxplot(
            data=df, 
            x='Classe', 
            y=var, 
            hue='Classe',
            ax=axes[idx],
            palette={'Benigno': '#a8e6cf', 'Maligno': '#ff8b94'},
            width=0.5,
            legend=False
        )
        sns.stripplot(data=df, x='Classe', y=var, ax=axes[idx], color='black', alpha=0.3, size=4)
        axes[idx].set_title(f'Distribuição de {var}', fontsize=11)
        axes[idx].set_xlabel('')
        axes[idx].set_ylabel('Valor Métrico')

# Remove o último eixo que sobrou do grid 2x3
fig.delaxes(axes[-1])

plt.suptitle('Comportamento das 5 Variáveis de Textura Adicionais por Classe', fontsize=14, weight='bold', y=0.98)
plt.tight_layout()
caminho_grafico2 = os.path.join(PASTA_SAIDA, '2_distribuicao_texturas_adicionais.png')
plt.savefig(caminho_grafico2, dpi=300)
plt.close()
print(f" -> Salvo: {caminho_grafico2}")

# =====================================================
# ESTUDO 3: CORRELAÇÃO DE SEPARAÇÃO DA TEXTURA VS TARGET
# =====================================================
# 🌟 CORRIGIDO: Alterado o método para 'spearman', que aceita perfeitamente variáveis binárias (Target)
# cruzadas com variáveis contínuas (texturas) sem quebrar o pandas.
colunas_corr = [coluna_tamanho, 'Proporcao_Calcificacao_IA'] + variaveis_textura + ['Malignidade_Real_Target']
matriz_focada = df[colunas_corr].corr(method='spearman')

plt.figure(figsize=(8, 6))
corr_target = matriz_focada[['Malignidade_Real_Target']].sort_values(by='Malignidade_Real_Target', ascending=False)
sns.heatmap(corr_target, annot=True, cmap='coolwarm', fmt=".3f", vmin=-1, vmax=1, cbar=False)
plt.title('Força de Correlação Linear Direta com o Target (Malignidade)', fontsize=12, pad=15)
plt.tight_layout()
caminho_grafico3 = os.path.join(PASTA_SAIDA, '3_correlacao_direta_target.png')
plt.savefig(caminho_grafico3, dpi=300)
plt.close()
print(f" -> Salvo: {caminho_grafico3}")

# =====================================================
# ESTUDO 4: MAPA DE COMPORTAMENTO MULTIVARIADO (PAIRPLOT)
# =====================================================
colunas_par = [coluna_tamanho, 'Haralick_Contraste', 'LBP_Energia_Textura', 'Classe']
g = sns.pairplot(
    df[colunas_par], 
    hue='Classe', 
    palette={'Benigno': '#2ecc71', 'Maligno': '#e74c3c'},
    diag_kind='kde',
    plot_kws={'alpha': 0.6, 's': 50}
)
g.fig.suptitle('Fronteiras de Decisão: Tamanho Anatômico Combinado com Microtexturas', y=1.02, fontsize=14, weight='bold')
caminho_grafico4 = os.path.join(PASTA_SAIDA, '4_mapa_multivariado_fronteiras.png')
g.savefig(caminho_grafico4, dpi=300)
plt.close()
print(f" -> Salvo: {caminho_grafico4}")

print("\n" + "="*60)
print(f"🎉 ANÁLISE CONCLUÍDA! Verifique a pasta '{PASTA_SAIDA}' no seu computador.")
print("="*60)