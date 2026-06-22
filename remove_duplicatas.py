import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist

ARQUIVO_INPUT = "dataset_final.csv"
ARQUIVO_OUTPUT = "sem_duplicates_inteligente.csv"

print("🔄 Carregando dataset para análise de diferenciação...")
df = pd.read_csv(ARQUIVO_INPUT)
total_inicial = len(df)

# Colunas numéricas que definem as características do nódulo
colunas_assinatura = [
    'HU_Mean_IA', 'HU_Std_IA', 'HU_Variance_IA', 'HU_Skewness_IA', 
    'HU_Kurtosis_IA', 'HU_90thPercentile_IA', 'HU_Entropy_IA', 'HU_Uniformity_IA',
    'Haralick_Contraste', 'Haralick_Homogeneidade', 'Haralick_Energia', 
    'Haralick_Correlacao', 'LBP_Energia_Textura'
]
colunas_presentes = [c for c in colunas_assinatura if c in df.columns]

# --- 1. CÁLCULO DE DIFERENCIAÇÃO INICIAL ---
print("🧮 Calculando a variabilidade das linhas no dataset original (isso pode demorar um pouco)...")
# Normalização rápida para a distância não ser distorcida pela escala de HU
df_norm_inicial = (df[colunas_presentes] - df[colunas_presentes].mean()) / df[colunas_presentes].std()
# Amostragem de até 1000 linhas para o cálculo de distância não estourar a memória
amostra_inicial = df_norm_inicial.sample(min(1000, len(df_norm_inicial)), random_state=42)
distancia_media_inicial = np.mean(pdist(amostra_inicial, metric='euclidean'))

# --- 2. PURIFICAÇÃO INTELIGENTE MANTENDO O ID ORIGINAL ---
print("\n🧹 Iniciando remoção inteligente (Mantendo a fatia com maior Dice_Score por ID)...")

# Ordena pelo Dice_Score (maiores primeiro) para garantir que a melhor fatia fique no topo
if 'Dice_Score' in df.columns:
    df_filtrado = df.sort_values(by='Dice_Score', ascending=False)
else:
    df_filtrado = df

# Remove mantendo apenas 1 linha por ID_Nodulo_Unico. 
# O ID original é preservado intacto.
df_purificado = df_filtrado.drop_duplicates(subset=['ID_Nodulo_Unico'], keep='first')

# Reorganiza o dataset pela ordem crescente dos IDs originais
df_purificado = df_purificado.sort_values(by='ID_Nodulo_Unico').reset_index(drop=True)

# --- 3. CÁLCULO DE DIFERENCIAÇÃO FINAL ---
print("🧮 Calculando a variabilidade no dataset purificado...")
df_norm_final = (df_purificado[colunas_presentes] - df_purificado[colunas_presentes].mean()) / df_purificado[colunas_presentes].std()
amostra_final = df_norm_final.sample(min(1000, len(df_norm_final)), random_state=42)
distancia_media_final = np.mean(pdist(amostra_final, metric='euclidean'))

# --- 4. RELATÓRIO DE IMPACTO ---
ganho_diferenciacao = ((distancia_media_final - distancia_media_inicial) / distancia_media_inicial) * 100

print("\n" + "="*60)
print("🏁 RELATÓRIO FINAL DE SANEAMENTO E DIFERENCIAÇÃO")
print("="*60)
print(f"-> Total de Linhas Originais     : {total_inicial}")
print(f"-> Total de Linhas Purificadas   : {len(df_purificado)}")
print(f"-> Linhas Redundantes Removidas  : {total_inicial - len(df_purificado)}")
print(f"-> IDs Duplicados Restantes      : {df_purificado['ID_Nodulo_Unico'].duplicated().sum()}")
print("-"*60)
print(f"-> Distância Média Original      : {distancia_media_inicial:.4f} (Métrica de distinção)")
print(f"-> Distância Média Pós-Limpeza   : {distancia_media_final:.4f}")
print(f"📈 GANHO DE DIFERENCIAÇÃO REAL  : {ganho_diferenciacao:+.2f}%")
print("="*60)

# Salva o arquivo final
df_purificado.to_csv(ARQUIVO_OUTPUT, index=False)
print(f"💾 Dataset definitivo salvo com sucesso em: '{ARQUIVO_OUTPUT}'")
print("="*60)