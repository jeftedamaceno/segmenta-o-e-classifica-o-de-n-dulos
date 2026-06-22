import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
import pydicom
import matplotlib.pyplot as plt
from skimage.draw import polygon

# =====================================================
# CONFIGURAÇÕES DE CAMINHO (Ajuste se necessário)
# =====================================================
CSV_PATH = "dados_segmentacao_nodulos_completo.csv"
RAIZ_DICOM = r"C:\dataset nodulos\lidc_idri\lidc_idri"
NODULO_ALVO_RELATORIO = "MI014_19564"  # Caso volumoso ideal identificado no CSV

# =====================================================
# RECONSTRUÇÃO DO PADRÃO OURO
# =====================================================
def reconstruir_padrao_ouro(shape, df_nodulo, mapa_sop):
    mask = np.zeros(shape, dtype=np.uint8)
    mapa_sop_limpo = {str(k).strip().rstrip('.0'): v for k, v in mapa_sop.items()}

    for _, row in df_nodulo.iterrows():
        try:
            sop_uid = str(row["sop_uid_fat_dicom"]).strip().rstrip('.0')
            if sop_uid not in mapa_sop_limpo:
                continue
            z = mapa_sop_limpo[sop_uid]
            separador = ";" if ";" in str(row["contorno_x"]) else ","
            xs = list(map(int, map(float, str(row["contorno_x"]).split(separador))))
            ys = list(map(int, map(float, str(row["contorno_y"]).split(separador))))

            rr, cc = polygon(ys, xs, shape=(shape[1], shape[2]))
            mask[z, rr, cc] = 1
        except:
            continue
    return mask

def obter_seed(df_nodulo, mapa_sop):
    mapa_sop_limpo = {str(k).strip().rstrip('.0'): v for k, v in mapa_sop.items()}
    linha = df_nodulo.iloc[len(df_nodulo)//2]
    sop = str(linha["sop_uid_fat_dicom"]).strip().rstrip('.0')
    z = mapa_sop_limpo[sop]
    x = int(float(linha["centro_x"]))
    y = int(float(linha["centro_y"]))
    return (z, y, x)

# =====================================================
# FUNÇÃO DE SEGMENTAÇÃO ORIGINAL (MÉTODO PROPOSTO)
# =====================================================
def segmentar_regiao_com_limite_distancia(volume, seed, spacing, dist_max_mm=20.0):
    z_s, y_s, x_s = seed
    lista_seed = [(int(x_s), int(y_s), int(z_s))]
    valor_seed_direto = float(volume[z_s, y_s, x_s])
    
    # 1. Construção da Máscara Restritiva (Esfera)
    sz, sy, sx = volume.shape
    grid_z, grid_y, grid_x = np.ogrid[:sz, :sy, :sx]
    dist_real_mm = np.sqrt(
        ((grid_z - z_s) * spacing[0]) ** 2 +
        ((grid_y - y_s) * spacing[1]) ** 2 +
        ((grid_x - x_s) * spacing[2]) ** 2
    )
    mascara_distancia = (dist_real_mm <= dist_max_mm).astype(np.uint8)

    # 2. Multi-Limiarização Dinâmica
    if valor_seed_direto < -500:
        limiar_inferior_adaptativo = max(valor_seed_direto - 100, -850)
    else:
        limiar_inferior_adaptativo = -620

    limiar_tecido_mole = (volume >= limiar_inferior_adaptativo) & (volume <= 100)
    limiar_calcificacao = (volume > 100)
    mascara_combinada = (limiar_tecido_mole | limiar_calcificacao) & (mascara_distancia == 1)
    
    # 3. Filtros Morfológicos no SimpleITK
    img_sitk_bruta = sitk.GetImageFromArray(mascara_combinada.astype(np.uint8))
    img_erodida = sitk.BinaryMorphologicalOpening(img_sitk_bruta, [1, 1, 1])
    
    img_conectada = sitk.ConnectedThreshold(
        img_erodida, seedList=lista_seed, lower=1, upper=1, replaceValue=1
    )
    
    img_dilatada = sitk.BinaryMorphologicalClosing(img_conectada, [2, 2, 2])
    img_final = sitk.BinaryFillhole(img_dilatada)
    
    return sitk.GetArrayFromImage(img_final)

# =====================================================
# MONTAGEM VISUAL DA LINHA DO TEMPO (PIPELINE SEQUENCIAL)
# =====================================================
def gerar_linha_do_tempo_segmentacao(volume, mask_gt, mask_seg, seed, spacing, nod_id, dist_max_mm=20.0):
    z, y_s, x_s = seed
    v_min, v_max = -1000, 400
    sz, sy, sx = volume.shape
    grid_y, grid_x = np.ogrid[:sy, :sx]
    
    # Etapas reconstruídas na fatia Z para exibição cronológica
    dist_real_mm_fatia = np.sqrt(
        ((grid_y - y_s) * spacing[1]) ** 2 +
        ((grid_x - x_s) * spacing[2]) ** 2
    )
    mascara_distancia = (dist_real_mm_fatia <= dist_max_mm).astype(np.uint8)
    
    valor_seed_direto = float(volume[z, y_s, x_s])
    limiar_inferior = max(valor_seed_direto - 100, -850) if valor_seed_direto < -500 else -620
    mascara_limiar = ((volume[z] >= limiar_inferior) & (volume[z] <= 100)) | (volume[z] > 100)
    mascara_combinada = mascara_limiar & (mascara_distancia == 1)

    # Configuração da figura (2 linhas x 3 colunas)
    fig, axs = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"LINHA DO TEMPO DE PROCESSAMENTO - CASO SELECIONADO: {nod_id}\n(Nódulo Volumoso na Fatia Z = {z})", 
                 fontsize=14, fontweight='bold', color='#2c3e50', y=0.98)

    # 1. Tomografia Original
    axs[0, 0].imshow(volume[z], cmap='gray', vmin=v_min, vmax=v_max)
    axs[0, 0].scatter(x_s, y_s, c='#e74c3c', s=60, edgecolors='white', zorder=5, label=f'Semente ({int(valor_seed_direto)} HU)')
    axs[0, 0].set_title("1. Imagem Médica Bruta DICOM", fontsize=11, fontweight='bold')
    axs[0, 0].legend(loc='lower left')
    axs[0, 0].axis('off')

    # 2. Esfera Restritiva
    axs[0, 1].imshow(volume[z], cmap='gray', vmin=v_min, vmax=v_max)
    axs[0, 1].imshow(mascara_distancia, cmap='jet', alpha=0.25)
    circle = plt.Circle((x_s, y_s), dist_max_mm / spacing[2], color='#2ecc71', fill=False, linestyle='--', linewidth=1.5, label=f'Janela {dist_max_mm}mm')
    axs[0, 1].add_patch(circle)
    axs[0, 1].set_title("2. Aplicação de ROI de Distância", fontsize=11, fontweight='bold')
    axs[0, 1].legend(loc='lower left')
    axs[0, 1].axis('off')

    # 3. Limiarização
    axs[0, 2].imshow(mascara_limiar, cmap='bone')
    axs[0, 2].set_title(f"3. Filtro Adaptativo (>{limiar_inferior} HU)", fontsize=11, fontweight='bold')
    axs[0, 2].axis('off')

    # 4. Interseção das Restrições
    axs[1, 0].imshow(mascara_combinada, cmap='magma')
    axs[1, 0].set_title("4. Interseção Lógica Combinada", fontsize=11, fontweight='bold')
    axs[1, 0].axis('off')

    # 5. Máscara Pós-Filtros SimpleITK
    axs[1, 1].imshow(mask_seg[z], cmap='gray')
    axs[1, 1].set_title("5. Máscara IA Final (Connected + Morphology)", fontsize=11, fontweight='bold')
    axs[1, 1].axis('off')

    # 6. Sobreposição Final vs Padrão Ouro
    axs[1, 2].imshow(volume[z], cmap='gray', vmin=v_min, vmax=v_max)
    if np.sum(mask_gt[z]) > 0:
        axs[1, 2].imshow(np.ma.masked_where(mask_gt[z] == 0, mask_gt[z]), cmap='Blues', alpha=0.6)
    if np.sum(mask_seg[z]) > 0:
        axs[1, 2].imshow(np.ma.masked_where(mask_seg[z] == 0, mask_seg[z]), cmap='autumn', alpha=0.4)
    axs[1, 2].set_title("6. Validação (Azul: Real | Vermelho: IA)", fontsize=11, fontweight='bold')
    axs[1, 2].axis('off')

    # Ajusta o zoom focando confortavelmente na região anatômica do nódulo
    for row in axs:
        for ax in row:
            ax.set_xlim(max(0, x_s - 55), min(sx, x_s + 55))
            ax.set_ylim(min(sy, y_s + 55), max(0, y_s - 55))

    plt.tight_layout()
    plt.show()

# =====================================================
# ALGORITMO DE BUSCA INTELIGENTE PELO MELHOR NÓDULO
# =====================================================
print("🔍 Analisando banco de dados para buscar o nódulo ideal para o relatório...")
df = pd.read_csv(CSV_PATH)

caso_resolvido = False

# Passo 1: Tenta filtrar prioritariamente o nódulo ótimo detectado na análise estatística
df_filtrado_alvo = df[df["nodulo_original_id"] == NODULO_ALVO_RELATORIO]
if df_filtrado_alvo.empty:
    print(f"⚠️ Identificador {NODULO_ALVO_RELATORIO} indisponível. Buscando os maiores nódulos alternativos...")
    # Ordena de forma decrescente para pegar os maiores contornos cadastrados no banco
    df["tamanho_estimado"] = df["contorno_x"].str.split(";").str.len()
    df_ordenado = df.sort_values(by="tamanho_estimado", ascending=False)
else:
    df_ordenado = df_filtrado_alvo

# Passo 2: Varre as pastas físicas para correlacionar o DICOM correspondente
for _, linha_nodulo in df_ordenado.dropna(subset=["paciente_serie_uid"]).iterrows():
    if caso_resolvido:
        break
        
    uid_alvo = str(linha_nodulo["paciente_serie_uid"]).strip()
    nodulo_id = linha_nodulo["nodulo_original_id"]

    for pasta_atual, _, arquivos in os.walk(RAIZ_DICOM):
        dicoms = [f for f in arquivos if f.endswith(".dcm") or f.startswith("1.")]
        if len(dicoms) == 0:
            continue
            
        try:
            amostra = os.path.join(pasta_atual, dicoms[0])
            ds = pydicom.dcmread(amostra, stop_before_pixels=True)
            uid_pasta = str(ds.SeriesInstanceUID).strip()
            
            if uid_alvo in uid_pasta or uid_pasta in uid_alvo:
                print(f"\n🚀 Caso ideal localizado na pasta: {pasta_atual}")
                print(f"📦 Extraindo dados do Nódulo Altamente Visível: {nodulo_id}")
                
                # Leitura da série tridimensional completa
                reader = sitk.ImageSeriesReader()
                arquivos_serie = reader.GetGDCMSeriesFileNames(pasta_atual)
                reader.SetFileNames(arquivos_serie)
                volume_sitk = reader.Execute()
                
                volume = sitk.GetArrayFromImage(volume_sitk)
                spacing = volume_sitk.GetSpacing()[::-1]

                # Mapeamento exato de UIDs de fatia
                mapa_sop = {}
                for idx, arq in enumerate(arquivos_serie):
                    ds_fatia = pydicom.dcmread(arq, stop_before_pixels=True)
                    mapa_sop[str(ds_fatia.SOPInstanceUID).strip()] = idx

                df_nodulo = df[df["nodulo_original_id"] == nodulo_id]
                mask_gt = reconstruir_padrao_ouro(volume.shape, df_nodulo, mapa_sop)
                seed = obter_seed(df_nodulo, mapa_sop)
                
                # Processamento e montagem da imagem cronológica
                mask_seg = segmentar_regiao_com_limite_distancia(volume, seed, spacing)
                
                print("🎨 Renderizando a linha do tempo selecionada com destaque anatômico...")
                gerar_linha_do_tempo_segmentacao(volume, mask_gt, mask_seg, seed, spacing, nodulo_id)
                
                caso_resolvido = True
                print("\n🏁 PROCESSO CONCLUÍDO! Imagem pronta para captura de tela (print) para o seu relatório.")
                break
                
        except Exception as e:
            continue