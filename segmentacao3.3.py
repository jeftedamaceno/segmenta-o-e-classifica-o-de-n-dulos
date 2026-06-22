import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
import pydicom
import matplotlib.pyplot as plt

from skimage.draw import polygon
from skimage import measure
from scipy.spatial.distance import directed_hausdorff
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# =====================================================
# CONFIGURAÇÕES
# =====================================================
CSV_PATH = "dados_segmentacao_nodulos_completo.csv"
RAIZ_DICOM = r"C:\dataset nodulos\lidc_idri\lidc_idri"
NUM_EXAMES = 6
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

def gerar_analise_estatistica_sementes(volume, mask_seg, seed, spacing, caso_id):
    """
    Gera gráficos e relatórios estatísticos sobre a semente e a segmentação.
    
    volume: array 3D da tomografia (HU)
    mask_seg: array 3D da segmentação gerada pela IA
    seed: tupla (z, y, x) da semente
    spacing: espaçamento dos voxels (z, y, x)
    caso_id: string ou int para identificar o gráfico
    """
    z_s, y_s, x_s = seed
    
    # 1. ANÁLISE DA SEMENTE
    valor_semente = float(volume[z_s, y_s, x_s])
    
    # Extrai uma vizinhança 3D pequena (cubo 5x5x5) ao redor da semente para caracterização
    z_min, z_max = max(0, z_s - 2), min(volume.shape[0], z_s + 3)
    y_min, y_max = max(0, y_s - 2), min(volume.shape[1], y_s + 3)
    x_min, x_max = max(0, x_s - 2), min(volume.shape[2], x_s + 3)
    vizinhanca = volume[z_min:z_max, y_min:y_max, x_min:x_max]
    
    media_vizinhanca = float(np.mean(vizinhanca))
    max_vizinhanca = float(np.max(vizinhanca))
    min_vizinhanca = float(np.min(vizinhanca))
    
    # 2. CARACTERIZAÇÃO DA SEGMENTAÇÃO E PRIORIZAÇÃO DE CALCIFICAÇÃO
    valores_segmentados = volume[mask_seg == 1]
    
    if len(valores_segmentados) > 0:
        # Definição clínica: HU > 100 geralmente indica estruturas densas/calcificadas em pulmão
        pixels_calcificados = np.sum(valores_segmentados > 100)
        pixels_tec_mole = np.sum((valores_segmentados >= -620) & (valores_segmentados <= 100))
        total_pixels = len(valores_segmentados)
        
        indice_calcificacao = (pixels_calcificados / total_pixels) * 100
        media_nodulo_hu = np.mean(valores_segmentados)
    else:
        pixels_calcificados = 0
        pixels_tec_mole = 0
        indice_calcificacao = 0.0
        media_nodulo_hu = 0.0

    # PRINT DO RELATÓRIO TEXTUAL NO TERMINAL
    print("\n" + "-"*50)
    print(f"📊 RELATÓRIO ESTATÍSTICO DA SEMENTE - CASO {caso_id}")
    print("-"*50)
    print(f"-> Coordenadas da Semente (Z, Y, X): ({z_s}, {y_s}, {x_s})")
    print(f"-> Valor exato na Semente          : {valor_semente:.1f} HU")
    print(f"-> Média da Vizinhança da Semente  : {media_vizinhanca:.1f} HU (Máx: {max_vizinhanca:.1f} | Mín: {min_vizinhanca:.1f})")
    print(f"-> Tipo de semente estimada       : {'Calcificada/Densa' if valor_semente > 100 else 'Tecido Mole/Vidro Fosco'}")
    print("\nPROPRIEDADES DA SEGMENTAÇÃO:")
    print(f"-> Média de Atenuação do Nódulo    : {media_nodulo_hu:.1f} HU")
    print(f"-> Voxel de Tecido Mole detectados : {pixels_tec_mole} voxels")
    print(f"-> Voxels Calcificados detectados  : {pixels_calcificados} voxels")
    print(f"-> ÍNDICE DE CALCIFICAÇÃO (IC)     : {indice_calcificacao:.2f}%")
    print(f"-> Classificação sugerida          : {'Altamente provável BENIGNO (Calcificado)' if indice_calcificacao > 10.0 else 'Padrão Suspeito (Não Calcificado)'}")
    print("-"*50)

    # 3. GERAÇÃO DO PAINEL DE VISUALIZAÇÃO GRÁFICA
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f"Análise Diagnóstica da Semente e Densidade do Nódulo (Caso {caso_id})", fontsize=14, fontweight='bold')

    # Gráfico 1: Perfil de Intensidade da Semente vs Vizinhança
    categorias = ['Mínimo local', 'Média Vizinhança', 'Ponto da Semente', 'Máximo local']
    valores_barra = [min_vizinhanca, media_vizinhanca, valor_semente, max_vizinhanca]
    cores_barra = ['#34495e', '#3498db', '#e74c3c', '#2ecc71']
    
    axs[0].bar(categorias, valores_barra, color=cores_barra, edgecolor='black', width=0.6)
    axs[0].axhline(100, color='purple', linestyle='--', linewidth=1.5, label='Limiar de Calcificação (100 HU)')
    axs[0].set_ylabel('Intensidade (Unidades Hounsfield - HU)', fontsize=11)
    axs[0].set_title('Perfil de Tecido na Região da Semente', fontsize=12, fontweight='bold')
    axs[0].grid(True, linestyle=':', alpha=0.6)
    axs[0].legend()

    # Gráfico 2: Histograma de Frequência de HU dentro do Nódulo Segmentado
    if len(valores_segmentados) > 0:
        axs[1].hist(valores_segmentados, bins=40, color='#9b59b6', alpha=0.7, edgecolor='black', label='Densidade do Nódulo')
        axs[1].axvline(100, color='purple', linestyle='--', linewidth=2, label=f'Cálcio (IC = {indice_calcificacao:.1f}%)')
        axs[1].set_xlabel('Unidades Hounsfield (HU)', fontsize=11)
        axs[1].set_ylabel('Frequência (Quantidade de Voxels)', fontsize=11)
        axs[1].set_title('Distribuição de Densidade da Segmentação', fontsize=12, fontweight='bold')
        axs[1].grid(True, linestyle=':', alpha=0.6)
        axs[1].legend()
    else:
        axs[1].text(0.5, 0.5, "Segmentação vazia\nSem dados para o histograma", 
                    ha='center', va='center', color='red', fontsize=12)

    plt.tight_layout()
    plt.show()
# =====================================================
# MÉTRICAS E COMPARAÇÃO DE VOLUME
# =====================================================
def calcular_volumes(mask_gt, mask_seg, spacing):
    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
    vol_gt = np.sum(mask_gt) * voxel_volume_mm3
    vol_seg = np.sum(mask_seg) * voxel_volume_mm3
    return vol_gt, vol_seg

def dice_score(gt, pred):
    inter = np.sum(gt * pred)
    return (2 * inter) / (np.sum(gt) + np.sum(pred) + 1e-8)

def iou_score(gt, pred):
    inter = np.sum(gt * pred)
    union = np.sum(gt) + np.sum(pred) - inter
    return inter / (union + 1e-8)

def hausdorff_distance(mask1, mask2):
    pts1 = np.argwhere(mask1 > 0)
    pts2 = np.argwhere(mask2 > 0)
    if len(pts1) == 0 or len(pts2) == 0:
        return np.nan
    return max(directed_hausdorff(pts1, pts2)[0], directed_hausdorff(pts2, pts1)[0])

# =====================================================
# RECONSTRUÇÃO PADRÃO OURO
# =====================================================
def reconstruir_padrao_ouro(shape, df_nodulo, mapa_sop):
    mask = np.zeros(shape, dtype=np.uint8)
    contagem_fatias_desenhadas = 0
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
            contagem_fatias_desenhadas += 1
        except:
            continue

    print(f" -> Sucesso: {contagem_fatias_desenhadas}/{len(df_nodulo)} fatias mapeadas tridimensionalmente.")
    return mask

# =====================================================
# SEED
# =====================================================
def obter_seed(df_nodulo, mapa_sop):
    mapa_sop_limpo = {str(k).strip().rstrip('.0'): v for k, v in mapa_sop.items()}
    linha = df_nodulo.iloc[len(df_nodulo)//2]
    sop = str(linha["sop_uid_fat_dicom"]).strip().rstrip('.0')
    z = mapa_sop_limpo[sop]
    x = int(float(linha["centro_x"]))
    y = int(float(linha["centro_y"]))
    return (z, y, x)

# =====================================================
# SEGMENTAÇÃO MORFOLÓGICA REFINADA (MÉTODO PROPOTO)
# =====================================================
# def segmentar_regiao_com_limite_distancia(volume, seed, spacing, dist_max_mm=20.0):
#     z_s, y_s, x_s = seed
#     lista_seed = [(int(x_s), int(y_s), int(z_s))]
    
#     # 1. CONSTRUÇÃO DA MÁSCARA RESTRITIVA (ESFERA TRIDIMENSIONAL)
#     sz, sy, sx = volume.shape
#     grid_z, grid_y, grid_x = np.ogrid[:sz, :sy, :sx]
#     dist_real_mm = np.sqrt(
#         ((grid_z - z_s) * spacing[0]) ** 2 +
#         ((grid_y - y_s) * spacing[1]) ** 2 +
#         ((grid_x - x_s) * spacing[2]) ** 2
#     )
#     # Geramos a ROI circular/esférica de proteção
#     mascara_distancia = (dist_real_mm <= dist_max_mm).astype(np.uint8)

#     # 2. MULTI-LIMIARIZAÇÃO DINÂMICA (Nódulo Tecido Mole + Nódulo Calcificado)
#     # Isola o tecido mole (tipicamente entre -600 e 40 HU para nódulos não calcificados)
#     limiar_tecido_mole = (volume >= -620) & (volume <= 100)
#     # Isola o complexo de cálcio/estruturas densas internas (> 100 HU até ossos/cálcio puro)
#     limiar_calcificacao = (volume > 100)
    
#     # União lógica (OR) restrita apenas à nossa esfera protetora
#     mascara_combinada = (limiar_tecido_mole | limiar_calcificacao) & (mascara_distancia == 1)
    
#     # Convertemos para imagem nativa SimpleITK (formato unsigned char necessário para morfologia)
#     img_sitk_bruta = sitk.GetImageFromArray(mascara_combinada.astype(np.uint8))

#     # 3. ELIMINAÇÃO DE "NÃO-NÓDULOS" (Erosão para desconectar vasos adjacentes)
#     # Raio [1, 1, 1] cria um kernel estruturante 3D pequeno para cortar pontes finas de vazamento
#     img_erodida = sitk.BinaryMorphologicalOpening(img_sitk_bruta, [1, 1, 1])

#     # 4. FILTRO DE CONECTIVIDADE CRÍTICO (Isola o componente que toca na Semente)
#     # Esse filtro analisa todas as ilhas binárias 3D dentro da esfera e apaga 
#     # automaticamente estruturas flutuantes que não pertençam ao nódulo principal clicado.
#     img_conectada = sitk.ConnectedThreshold(
#         img_erodida, 
#         seedList=lista_seed,
#         lower=1, 
#         upper=1, 
#         replaceValue=1
#     )

#     # 5. RESTAURAÇÃO DE BORDAS E PREENCHIMENTO DE BURACOS (Dilatação + Hole Filling)
#     # Recupera o volume original perdido na erosão periférica
#     img_dilatada = sitk.BinaryMorphologicalClosing(img_conectada, [2, 2, 2])
    
#     # Fecha lacunas ou vazios gerados por transições abruptas de densidade interna (como o core de cálcio)
#     img_final = sitk.BinaryFillhole(img_dilatada)
    
#     # Conversão final para array numpy
#     mask_seg = sitk.GetArrayFromImage(img_final)
    
#     # Extração opcional de métrica diagnóstica para uso futuro (Índice de Calcificação)
#     if np.sum(mask_seg) > 0:
#         pixels_calcificados = np.sum((volume > 100) & (mask_seg == 1))
#         indice_calcificacao = (pixels_calcificados / np.sum(mask_seg)) * 100
#         if indice_calcificacao > 5.0:
#             print(f" -> Alerta Clínico: Nódulo apresenta {indice_calcificacao:.1f}% de densidade cálcica (Alta chance de Benignidade).")

#     return mask_seg

# =====================================================
# ALTERAÇÃO DENTRO DA SUA FUNÇÃO DE SEGMENTAÇÃO
# =====================================================
def segmentar_regiao_com_limite_distancia(volume, seed, spacing, dist_max_mm=20.0):
    z_s, y_s, x_s = seed
    lista_seed = [(int(x_s), int(y_s), int(z_s))]
    
    # 1. Obter o valor da semente para adaptação dinâmica
    valor_seed_direto = float(volume[z_s, y_s, x_s])
    
    # 2. Construção da Máscara Restritiva (Esfera)
    sz, sy, sx = volume.shape
    grid_z, grid_y, grid_x = np.ogrid[:sz, :sy, :sx]
    dist_real_mm = np.sqrt(
        ((grid_z - z_s) * spacing[0]) ** 2 +
        ((grid_y - y_s) * spacing[1]) ** 2 +
        ((grid_x - x_s) * spacing[2]) ** 2
    )
    mascara_distancia = (dist_real_mm <= dist_max_mm).astype(np.uint8)

    # 3. MULTI-LIMIARIZAÇÃO DINÂMICA (CORRIGIDA)
    # Se a semente for muito escura (vidro fosco), baixamos o limite inferior para dar margem
    if valor_seed_direto < -500:
        limiar_inferior_adaptativo = max(valor_seed_direto - 100, -850)
    else:
        limiar_inferior_adaptativo = -620 # Padrão para nódulos sólidos

    # Aplicamos o limiar adaptativo calculado na hora
    limiar_tecido_mole = (volume >= limiar_inferior_adaptativo) & (volume <= 100)
    limiar_calcificacao = (volume > 100)
    
    # União lógica (OR) restrita à esfera
    mascara_combinada = (limiar_tecido_mole | limiar_calcificacao) & (mascara_distancia == 1)
    
    # --- O restante do seu código de morfologia continua exatamente igual ---
    img_sitk_bruta = sitk.GetImageFromArray(mascara_combinada.astype(np.uint8))
    img_erodida = sitk.BinaryMorphologicalOpening(img_sitk_bruta, [1, 1, 1])
    
    img_conectada = sitk.ConnectedThreshold(
        img_erodida, seedList=lista_seed, lower=1, upper=1, replaceValue=1
    )
    
    img_dilatada = sitk.BinaryMorphologicalClosing(img_conectada, [2, 2, 2])
    img_final = sitk.BinaryFillhole(img_dilatada)
    
    return sitk.GetArrayFromImage(img_final)
# =====================================================
# PLOTS E VISUALIZAÇÃO
# =====================================================
def comparar_3d_focado(mask_gt, mask_seg, seed):
    fig = plt.figure(figsize=(14, 7))
    pontos_gt = np.argwhere(mask_gt > 0)
    
    if len(pontos_gt) > 0:
        z_min, y_min, x_min = pontos_gt.min(axis=0) - 5
        z_max, y_max, x_max = pontos_gt.max(axis=0) + 5
    else:
        z, y, x = seed
        z_min, z_max = max(0, z - 15), min(mask_gt.shape[0], z + 15)
        y_min, y_max = max(0, y - 40), min(mask_gt.shape[1], y + 40)
        x_min, x_max = max(0, x - 40), min(mask_gt.shape[2], x + 40)

    def plotar_mascara(ax, mascara, cor, titulo):
        pontos = np.argwhere(mascara > 0)
        
        if len(pontos) == 0:
            ax.text((x_min+x_max)/2, (y_min+y_max)/2, (z_min+z_max)/2, 
                    "IA Zerou:\nSem dados nesta regiao", color='red', ha='center', va='center')
        else:
            sucesso_mesh = False
            try:
                sub_vol = mascara[max(0, z_min):z_max, max(0, y_min):y_max, max(0, x_min):x_max]
                if np.sum(sub_vol) > 0:
                    verts, faces, _, _ = measure.marching_cubes(sub_vol, level=0.5)
                    if len(verts) > 0:
                        verts[:, 0] += max(0, z_min)
                        verts[:, 1] += y_min
                        verts[:, 2] += x_min
                        
                        verts_cartesianos = np.zeros_like(verts)
                        verts_cartesianos[:, 0] = verts[:, 2]  
                        verts_cartesianos[:, 1] = verts[:, 1]  
                        verts_cartesianos[:, 2] = verts[:, 0]  
                        
                        mesh = Poly3DCollection(verts_cartesianos[faces], alpha=0.6, facecolor=cor, edgecolor='none')
                        ax.add_collection3d(mesh)
                        sucesso_mesh = True
            except:
                pass 
            
            if not sucesso_mesh:
                ax.scatter(pontos[:, 2], pontos[:, 1], pontos[:, 0], c=cor, s=25, alpha=0.6, marker='s')
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.set_title(titulo)
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=25, azim=45)

    ax1 = fig.add_subplot(121, projection='3d')
    plotar_mascara(ax1, mask_gt, '#1f77b4', "Padrao Ouro Reconstruido")

    ax2 = fig.add_subplot(122, projection='3d')
    plotar_mascara(ax2, mask_seg, '#d62728', "Segmentacao Local IA")

    plt.tight_layout()
    plt.show()

def mostrar_resultado(volume, mask_gt, mask_seg, seed):
    z = seed[0]
    fig, axs = plt.subplots(2, 2, figsize=(12,10))
    v_min, v_max = -1000, 400
    
    axs[0,0].imshow(volume[z], cmap='gray', vmin=v_min, vmax=v_max)
    axs[0,0].scatter(seed[2], seed[1], c='red', s=50, label='Seed')
    axs[0,0].set_title(f"Tomografia Original (Fatia z={z})")
    axs[0,0].legend()

    axs[0,1].imshow(mask_gt[z], cmap='gray', vmin=0, vmax=1)
    axs[0,1].set_title("Padrao Ouro (Contorno Poligonal)")

    axs[1,0].imshow(mask_seg[z], cmap='gray', vmin=0, vmax=1)
    axs[1,0].set_title("Segmentacao Avançada Combinada")

    axs[1,1].imshow(volume[z], cmap='gray', vmin=v_min, vmax=v_max)
    if np.sum(mask_gt[z]) > 0:
        axs[1,1].imshow(np.ma.masked_where(mask_gt[z] == 0, mask_gt[z]), cmap='Blues', alpha=0.6)
    if np.sum(mask_seg[z]) > 0:
        axs[1,1].imshow(np.ma.masked_where(mask_seg[z] == 0, mask_seg[z]), cmap='autumn', alpha=0.5)
    axs[1,1].set_title("Sobreposicao (Azul: Real | Amarelo/Vermelho: IA)")

    plt.tight_layout()
    plt.show()

# =====================================================
# LOOP PRINCIPAL
# =====================================================
print("Localizando exames...")
df = pd.read_csv(CSV_PATH)

if "contorno_x" in df.columns:
    df['tamanho_estimado'] = df['contorno_x'].astype(str).str.len()
    df = df.sort_values(by='tamanho_estimado', ascending=False)

exames_disponiveis = {}
for pasta_atual, _, arquivos in os.walk(RAIZ_DICOM):
    dicoms = [f for f in arquivos if f.endswith(".dcm") or f.startswith("1.")]
    if len(dicoms) == 0:
        continue
    try:
        amostra = os.path.join(pasta_atual, dicoms[0])
        ds = pydicom.dcmread(amostra, stop_before_pixels=True)
        uid = str(ds.SeriesInstanceUID).strip()
        match = df[df["paciente_serie_uid"].str.contains(uid, na=False, case=False)]
        if not match.empty:
            prioridade_uid = match["paciente_serie_uid"].iloc[0]
            if prioridade_uid not in exames_disponiveis:
                exames_disponiveis[prioridade_uid] = pasta_atual
    except:
        pass

uids = list(exames_disponiveis.keys())[:NUM_EXAMES]
print(f"{len(uids)} exames de nodulos priorizados encontrados.")

for i, uid in enumerate(uids):
    print("\n" + "="*70)
    print(f"PROCESSANDO CASO {i+1}/{len(uids)}")
    print("="*70)

    pasta = exames_disponiveis[uid]
    reader = sitk.ImageSeriesReader()
    arquivos = reader.GetGDCMSeriesFileNames(pasta)
    reader.SetFileNames(arquivos)
    volume_sitk = reader.Execute()
    volume = sitk.GetArrayFromImage(volume_sitk)
    spacing = volume_sitk.GetSpacing()[::-1] 

    mapa_sop = {}
    for idx, arq in enumerate(arquivos):
        try:
            ds = pydicom.dcmread(arq, stop_before_pixels=True)
            mapa_sop[str(ds.SOPInstanceUID).strip()] = idx
        except:
            pass

    df_exame = df[df["paciente_serie_uid"] == uid]
    nodulo = df_exame["nodulo_original_id"].dropna().iloc[0]
    df_nodulo = df_exame[df_exame["nodulo_original_id"] == nodulo]

    print(f"Nodulo Selecionado ID: {nodulo}")
    mask_gt = reconstruir_padrao_ouro(volume.shape, df_nodulo, mapa_sop)
    
    try:
        seed = obter_seed(df_nodulo, mapa_sop)
        mask_seg = segmentar_regiao_com_limite_distancia(volume, seed, spacing, dist_max_mm=20.0)
        
        dice = dice_score(mask_gt, mask_seg)
        iou = iou_score(mask_gt, mask_seg)
        hd = hausdorff_distance(mask_gt, mask_seg)
        vol_gt, vol_seg = calcular_volumes(mask_gt, mask_seg, spacing)

        print(f"  Resultado Dice Coefficient : {dice:.4f}")
        print(f"  Resultado Jaccard (IoU)   : {iou:.4f}")
        print(f"  Distancia de Hausdorff     : {hd:.2f} voxels")
        print(f"  Volume Padrao Ouro         : {vol_gt:.2f} mm3")
        print(f"  Volume Segmentado (IA)     : {vol_seg:.2f} mm3")

        # --- NOVA LINHA ADICIONADA AQUI ---
        # Executa a análise profunda das sementes e o comportamento da segmentação
        gerar_analise_estatistica_sementes(volume, mask_seg, seed, spacing, caso_id=i+1)
        # ----------------------------------

        mostrar_resultado(volume, mask_gt, mask_seg, seed)
        comparar_3d_focado(mask_gt, mask_seg, seed)
            
    except Exception as e:
        print(f"Falha no processamento deste caso: {e}")

print("\nPIPELINE PROCESSADO COMPLETAMENTE!")