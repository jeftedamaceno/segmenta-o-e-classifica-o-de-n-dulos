import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
import pydicom
import scipy.stats as stats
from skimage.draw import polygon
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

# =====================================================
# CONFIGURAÇÕES DE CAMINHO E PARÂMETROS
# =====================================================
CSV_PATH = "dados_segmentacao_nodulos_completo.csv"
RAIZ_DICOM = r"C:\dataset nodulos\lidc_idri\lidc_idri"
OUTPUT_CSV = "dataset_radiomica_segmentado_ia_unico.csv"
LIMITE_PACIENTES = 50  # Executa para os primeiros 50 pacientes físicos encontrados

# =====================================================
# RECONSTRUÇÃO PADRÃO OURO CONSOLIDADA (MÉDIA DOS MÉDICOS)
# =====================================================
def reconstruir_padrao_ouro_consolidado(shape, df_nodulo, mapa_sop):
    """
    Cria uma máscara de validação combinando o desenho de todos os médicos 
    que anotaram este mesmo nódulo físico (interseção/voto majoritário).
    """
    mask_acumulada = np.zeros(shape, dtype=np.float32)
    mapa_sop_limpo = {str(k).strip().rstrip('.0'): v for k, v in mapa_sop.items()}
    
    # Conta quantos médicos de fato desenharam contornos válidos
    medicos_validos = 0
    
    # Agrupa por médico (cada leitura única costuma ter um ID ou índice de linha)
    # Como o df_nodulo contém linhas de vários médicos, iteramos linha por linha
    for _, row in df_nodulo.iterrows():
        try:
            sop_uid = str(row["sop_uid_fat_dicom"]).strip().rstrip('.0')
            if sop_uid not in mapa_sop_limpo:
                continue
            z = mapa_sop_limpo[sop_uid]
            separador = ";" if ";" in str(row["contorno_x"]) else ","
            xs = list(map(int, map(float, str(row["contorno_x"]).split(separador))))
            ys = list(map(int, map(float, str(row["contorno_y"]).split(separador))))

            mask_medico = np.zeros((shape[1], shape[2]), dtype=np.uint8)
            rr, cc = polygon(ys, xs, shape=(shape[1], shape[2]))
            mask_medico[rr, cc] = 1
            
            mask_acumulada[z, :, :] += mask_medico
            medicos_validos += 1
        except:
            continue
            
    # Retorna uma máscara binária onde pelo menos um médico desenhou
    # (Padrão Ouro consolidado)
    return (mask_acumulada > 0).astype(np.uint8)

def obter_seed_media(df_nodulo, mapa_sop):
    """
    Calcula o centro geométrico médio absoluto usando as coordenadas
    de todos os médicos combinadas para encontrar o 'coração' do nódulo real.
    """
    mapa_sop_limpo = {str(k).strip().rstrip('.0'): v for k, v in mapa_sop.items()}
    
    zs, ys, xs = [], [], []
    for _, linha in df_nodulo.iterrows():
        try:
            sop = str(linha["sop_uid_fat_dicom"]).strip().rstrip('.0')
            if sop in mapa_sop_limpo:
                zs.append(mapa_sop_limpo[sop])
                xs.append(float(linha["centro_x"]))
                ys.append(float(linha["centro_y"]))
        except:
            continue
            
    # Retorna o centro médio arredondado
    z_medio = int(np.round(np.mean(zs))) if zs else 0
    y_medio = int(np.round(np.mean(ys))) if ys else 0
    x_medio = int(np.round(np.mean(xs))) if xs else 0
    return (z_medio, y_medio, x_medio)

# =====================================================
# SEGMENTAÇÃO MORFOLÓGICA ALTAMENTE OTIMIZADA (CROP ROI)
# =====================================================
def segmentar_regiao_com_limite_distancia_otimizado(volume, seed, spacing, dist_max_mm=20.0):
    z_s, y_s, x_s = seed
    sz, sy, sx = volume.shape
    
    pad_z = int(np.ceil(dist_max_mm / spacing[0])) + 2
    pad_y = int(np.ceil(dist_max_mm / spacing[1])) + 2
    pad_x = int(np.ceil(dist_max_mm / spacing[2])) + 2
    
    z_min, z_max = max(0, z_s - pad_z), min(sz, z_s + pad_z)
    y_min, y_max = max(0, y_s - pad_y), min(sy, y_s + pad_y)
    x_min, x_max = max(0, x_s - pad_x), min(sx, x_s + pad_x)
    
    sub_volume = volume[z_min:z_max, y_min:y_max, x_min:x_max]
    z_s_local, y_s_local, x_s_local = z_s - z_min, y_s - y_min, x_s - x_min
    lista_seed_local = [(int(x_s_local), int(y_s_local), int(z_s_local))]
    
    gz, gy, gx = np.ogrid[:sub_volume.shape[0], :sub_volume.shape[1], :sub_volume.shape[2]]
    dist_real_mm = np.sqrt(
        ((gz - z_s_local) * spacing[0]) ** 2 +
        ((gy - y_s_local) * spacing[1]) ** 2 +
        ((gx - x_s_local) * spacing[2]) ** 2
    )
    mascara_distancia = (dist_real_mm <= dist_max_mm).astype(np.uint8)

    valor_seed_direto = float(sub_volume[z_s_local, y_s_local, x_s_local])
    if valor_seed_direto < -500:
        limiar_inferior_adaptativo = max(valor_seed_direto - 100, -850)
    else:
        limiar_inferior_adaptativo = -620

    limiar_tecido_mole = (sub_volume >= limiar_inferior_adaptativo) & (sub_volume <= 100)
    limiar_calcificacao = (sub_volume > 100)
    mascara_combinada = (limiar_tecido_mole | limiar_calcificacao) & (mascara_distancia == 1)
    
    img_sitk_bruta = sitk.GetImageFromArray(mascara_combinada.astype(np.uint8))
    img_erodida = sitk.BinaryMorphologicalOpening(img_sitk_bruta, [1, 1, 1])
    
    img_conectada = sitk.ConnectedThreshold(
        img_erodida, seedList=lista_seed_local, lower=1, upper=1, replaceValue=1
    )
    
    img_dilatada = sitk.BinaryMorphologicalClosing(img_conectada, [2, 2, 2])
    img_final = sitk.BinaryFillhole(img_dilatada)
    sub_mask_seg = sitk.GetArrayFromImage(img_final)
    
    mask_seg_global = np.zeros(volume.shape, dtype=np.uint8)
    mask_seg_global[z_min:z_max, y_min:y_max, x_min:x_max] = sub_mask_seg
    return mask_seg_global

# =====================================================
# EXTRATOR DE ATRIBUTOS PARA NÓDULOS ÚNICOS
# =====================================================
def extrair_atributos_e_validar(volume, mask_seg, mask_gt, spacing, id_nodulo, classe_malignidade_real):
    img_sitk = sitk.GetImageFromArray(mask_seg.astype(np.uint8))
    img_erodida = sitk.BinaryErode(img_sitk, [2, 2, 2]) 
    mask_nucleo = sitk.GetArrayFromImage(img_erodida)
    
    if np.sum(mask_nucleo) == 0:
        mask_nucleo = mask_seg
        
    pixels_ia_nucleo = volume[mask_nucleo > 0]
    
    if len(pixels_ia_nucleo) == 0:
        return None
        
    try:
        id_limpo = int(str(id_nodulo).upper().replace("NODULE", "").strip())
    except:
        id_limpo = id_nodulo

    features = {"ID_Nodulo_Unico": id_limpo, "Malignidade_Real_Target": classe_malignidade_real}
    
    # Atributos Clínicos de Intensidade no Núcleo
    features["HU_Mean_IA"] = float(np.mean(pixels_ia_nucleo))
    features["HU_Std_IA"] = float(np.std(pixels_ia_nucleo))
    features["HU_Variance_IA"] = float(np.var(pixels_ia_nucleo))
    features["HU_Skewness_IA"] = float(stats.skew(pixels_ia_nucleo)) if len(pixels_ia_nucleo) > 2 else 0.0
    features["HU_Kurtosis_IA"] = float(stats.kurtosis(pixels_ia_nucleo)) if len(pixels_ia_nucleo) > 2 else 0.0
    features["HU_90thPercentile_IA"] = float(np.percentile(pixels_ia_nucleo, 90))
    
    counts, _ = np.histogram(pixels_ia_nucleo, bins=32, density=True)
    probs = counts / (np.sum(counts) + 1e-8)
    probs = probs[probs > 0]
    features["HU_Entropy_IA"] = float(-np.sum(probs * np.log2(probs)))
    features["HU_Uniformity_IA"] = float(np.sum(probs ** 2))
    
    voxels_calcificados = np.sum(pixels_ia_nucleo > 100)
    indice_calcificacao = float(voxels_calcificados / len(pixels_ia_nucleo))
    features["Proporcao_Calcificacao_IA"] = indice_calcificacao
    features["Sugerido_Benigno_Por_Calcificacao"] = 1 if indice_calcificacao > 0.10 else 0

    # Extração das 5 Características de Textura (Haralick + LBP)
    fatias_com_nodulo = np.any(mask_nucleo > 0, axis=(1, 2))
    indices_z = np.where(fatias_com_nodulo)[0]
    
    lista_contraste, lista_homogeneidade, lista_energia, lista_correlacao, lista_lbp = [], [], [], [], []

    for z in indices_z:
        fatia_vol = volume[z, :, :]
        fatia_mask = mask_nucleo[z, :, :]
        
        coords = np.argwhere(fatia_mask > 0)
        if len(coords) == 0:
            continue
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0) + 1
        
        roi_vol = fatia_vol[max(0, y_min-1):y_max+1, max(0, x_min-1):x_max+1]
        roi_mask = fatia_mask[max(0, y_min-1):y_max+1, max(0, x_min-1):x_max+1]
        
        roi_normalizada = np.clip(roi_vol, -1000, 400)
        faixa_valores = np.ptp(roi_normalizada)
        roi_normalizada = ((roi_normalizada - roi_normalizada.min()) / (faixa_valores + 1e-8) * 63).astype(np.uint8)
        roi_processar = np.where(roi_mask > 0, roi_normalizada, 0)

        glcm = graycomatrix(roi_processar, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=64, symmetric=True, normed=True)
        glcm[0, :, :, :] = 0
        glcm[:, 0, :, :] = 0
        if glcm.sum() > 0:
            glcm = glcm / glcm.sum()

        lista_contraste.append(graycoprops(glcm, 'contrast')[0, 0])
        lista_homogeneidade.append(graycoprops(glcm, 'homogeneity')[0, 0])
        lista_energia.append(graycoprops(glcm, 'energy')[0, 0])
        lista_correlacao.append(graycoprops(glcm, 'correlation')[0, 0])

        lbp = local_binary_pattern(roi_processar, P=8, R=1, method='uniform')
        valores_lbp = lbp[roi_mask > 0]
        if len(valores_lbp) > 0:
            hist_lbp, _ = np.histogram(valores_lbp, bins=10, density=True)
            lista_lbp.append(np.sum(hist_lbp ** 2))

    features["Haralick_Contraste"] = float(np.mean(lista_contraste)) if lista_contraste else 0.0
    features["Haralick_Homogeneidade"] = float(np.mean(lista_homogeneidade)) if lista_homogeneidade else 0.0
    features["Haralick_Energia"] = float(np.mean(lista_energia)) if lista_energia else 0.0
    features["Haralick_Correlacao"] = float(np.mean(lista_correlacao)) if lista_correlacao else 0.0
    features["LBP_Energia_Textura"] = float(np.mean(lista_lbp)) if lista_lbp else 0.0

    # Colunas de Validação Mapeadas contra o Consolidado dos Médicos
    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
    vol_seg_mm3 = np.sum(mask_seg) * voxel_volume_mm3
    vol_gt_mm3 = np.sum(mask_gt) * voxel_volume_mm3
    features["Volume_Ideal_GT_Consolidado_mm3"] = vol_gt_mm3
    features["Diferenca_Absoluta_Volume_mm3"] = abs(vol_gt_mm3 - vol_seg_mm3)
    
    inter = np.sum(mask_gt * mask_seg)
    union = np.sum(mask_gt) + np.sum(mask_seg) - inter
    features["Dice_Score"] = (2 * inter) / (np.sum(mask_gt) + np.sum(mask_seg) + 1e-8)
    features["IoU_Score"] = inter / (union + 1e-8)

    return features

# =====================================================
# PIPELINE EXECUTÁVEL UNIFICADO (NÓDULO ÚNICO)
# =====================================================
if __name__ == "__main__":
    print("Step 1: Cruzando tabelas clínicas e localizando arquivos DICOM...")
    df = pd.read_csv(CSV_PATH)
    coluna_malignidade = [c for c in df.columns if "malign" in c.lower()][0]

    uids_no_csv = set(df["paciente_serie_uid"].dropna().str.strip().unique())
    exames_disponiveis = {}
    
    for pasta_atual, _, arquivos in os.walk(RAIZ_DICOM):
        dicoms = [f for f in arquivos if f.lower().endswith(".dcm") or f.startswith("1.")]
        if len(dicoms) == 0:
            continue
        try:
            ds = pydicom.dcmread(os.path.join(pasta_atual, dicoms[0]), stop_before_pixels=True)
            uid_dicom = str(ds.SeriesInstanceUID).strip()
            if uid_dicom in uids_no_csv and uid_dicom not in exames_disponiveis:
                exames_disponiveis[uid_dicom] = pasta_atual
        except:
            continue

    uids_finais = list(exames_disponiveis.keys())[:LIMITE_PACIENTES]
    print(f" -> Encontrados {len(exames_disponiveis)} exames em disco. Processando {len(uids_finais)} exames ativos.")

    dataset_radiomico_ia = []

    print("\n🚀 INICIANDO EXTRAÇÃO EXCLUSIVA DE NÓDULOS FÍSICOS ÚNICOS...")
    for idx, uid in enumerate(uids_finais):
        print(f"[{idx+1:02d}/{len(uids_finais)}] Paciente Série UID: ...{uid[-12:]}")
        pasta = exames_disponiveis[uid]
        
        try:
            reader = sitk.ImageSeriesReader()
            arquivos = reader.GetGDCMSeriesFileNames(pasta)
            if len(arquivos) < 2:
                continue
            reader.SetFileNames(arquivos)
            volume_sitk = reader.Execute()
            volume = sitk.GetArrayFromImage(volume_sitk)
            spacing = volume_sitk.GetSpacing()[::-1]

            mapa_sop = {}
            for f_idx, arq in enumerate(arquivos):
                try:
                    ds = pydicom.dcmread(arq, stop_before_pixels=True)
                    mapa_sop[str(ds.SOPInstanceUID).strip()] = f_idx
                except:
                    pass

            df_exame = df[df["paciente_serie_uid"] == uid]
            
            # 🌟 MUDANÇA CRÍTICA AQUI: O groupby une todas as linhas de todos os médicos 
            # referentes àquele mesmo 'nodulo_original_id'. O loop agora roda 1 VEZ por nódulo.
            nodulos_no_exame = df_exame.groupby("nodulo_original_id")
            
            for nodulo_id, df_nodulo in nodulos_no_exame:
                # Malignidade média combinada dos médicos
                nota_media = df_nodulo[coluna_malignidade].dropna().mean()
                if pd.isna(nota_media):
                    continue
                classe_real = 0 if nota_media < 3 else 1
                
                # Consolida as máscaras de todos os médicos em um Ground Truth único
                mask_gt = reconstruir_padrao_ouro_consolidado(volume.shape, df_nodulo, mapa_sop)
                
                # Gera uma semente única (média) baseada em todas as leituras
                seed = obter_seed_media(df_nodulo, mapa_sop)
                
                # Executa a IA apenas 1 vez para este objeto anatômico
                mask_seg = segmentar_regiao_com_limite_distancia_otimizado(volume, seed, spacing)
                
                atributos = extrair_atributos_e_validar(volume, mask_seg, mask_gt, spacing, nodulo_id, classe_real)
                
                if atributos:
                    # Remove linhas onde a máscara foi apagada pela erosão (erros críticos)
                    if atributos["Haralick_Contraste"] == 0.0 and atributos["Dice_Score"] < 0.1:
                        continue
                        
                    dataset_radiomico_ia.append(atributos)
                    print(f"   └─ Nódulo Anatômico Único [{nodulo_id}] -> Segmentado! Dice Consolidado: {atributos['Dice_Score']:.3f}")
                        
        except Exception as e:
            print(f"   [FALHA] Erro crítico ao processar o paciente {uid}: {e}")
            continue

    # Salvamento final sem redundâncias
    if len(dataset_radiomico_ia) > 0:
        df_resultado = pd.DataFrame(dataset_radiomico_ia)
        df_resultado.to_csv(OUTPUT_CSV, index=False)
        print("\n" + "="*70)
        print(f"💾 PIPELINE CONCLUÍDO COM SUCESSO!\nDataset limpo (Nódulos Únicos) salvo em: {OUTPUT_CSV}")
        print(f"Total de linhas (nódulos reais): {len(df_resultado)}")
        print(f"Média Geral do coeficiente Dice Consolidado: {df_resultado['Dice_Score'].mean():.4f}")
        print("="*70)
    else:
        print("\n❌ Nenhuma amostra pôde ser processada. Verifique os caminhos das pastas.")