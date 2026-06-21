import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# CONFIGURAÇÕES DE PASTAS E CARREGAMENTO
# =====================================================
ARQUIVO_CSV = "dataset_radiomica_segmentado_ia.csv"
PASTA_OUTPUT = "visualizacoes_diagnosticas"

os.makedirs(PASTA_OUTPUT, exist_ok=True)

print("🔄 Carregando dados para cálculo do erro percentual...")
df = pd.read_csv(ARQUIVO_CSV)

# Configuração de estilo
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13})

# =====================================================
# CÁLCULO DO ERRO PERCENTUAL EM RELAÇÃO AO PADRÃO OURO
# =====================================================
# Evita divisão por zero caso haja algum GT zerado por ruído
vol_gt = df['Volume_Ideal_GT_mm3'].values
vol_ia = df['Volume_IA_mm3'].values

df['Erro_Percentual_Relativo'] = (np.abs(vol_ia - vol_gt) / (vol_gt + 1e-8)) * 100

# Medidas estatísticas para os destaques no gráfico
mediana_erro_pct = df['Erro_Percentual_Relativo'].median()
media_erro_pct = df['Erro_Percentual_Relativo'].mean()

# --- GRAFICO 10: Histograma do Erro Percentual Relativo ---
plt.figure(figsize=(9, 5.5))

# Criamos o histograma com linha de densidade suavizada (KDE)
sns.histplot(df['Erro_Percentual_Relativo'], bins=20, kde=True, color='#27ae60', alpha=0.6, edgecolor='black')

# Adiciona linhas de referência para interpretação rápida
plt.axvline(mediana_erro_pct, color='blue', linestyle='--', linewidth=2, 
            label=f'Mediana do Erro: {mediana_erro_pct:.1f}%')
plt.axvline(media_erro_pct, color='purple', linestyle=':', linewidth=2, 
            label=f'Média do Erro: {media_erro_pct:.1f}%')

plt.xlabel("Erro Volumétrico Relativo ao Padrão Ouro (%)")
plt.ylabel("Frequência (Quantidade de Nódulos)")
plt.title("Proporção Real do Erro de Segmentação da IA", weight='bold', pad=15)
plt.legend(loc="upper right")
plt.tight_layout()

plt.savefig(os.path.join(PASTA_OUTPUT, "10_erro_percentual_volume.png"), dpi=300)
plt.close()

print(" -> [OK] 10_erro_percentual_volume.png")

# =====================================================
# IMPRESSÃO AUDITORIA DETALHADA NO TERMINAL
# =====================================================
print("\n" + "="*60)
print("📊 ANÁLISE RELATIVA DO ERRO DA IA")
print("="*60)
print(f"-> Em metade dos casos (Mediana), a IA errou menos de: {mediana_erro_pct:.2f}%")
print(f"-> Erro médio global ponderado: {media_erro_pct:.2f}%")

# Faixas de tolerância clínica para apresentar no texto do trabalho
ate_10_pct = np.sum(df['Erro_Percentual_Relativo'] <= 10) / len(df) * 100
ate_25_pct = np.sum(df['Erro_Percentual_Relativo'] <= 25) / len(df) * 100

print(f"-> Nódulos com precisão cirúrgica (Erro <= 10%): {ate_10_pct:.1f}% do dataset")
print(f"-> Nódulos com erro clinicamente aceitável (Erro <= 25%): {ate_25_pct:.1f}% do dataset")
print("="*60)