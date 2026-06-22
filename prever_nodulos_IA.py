import os
import json
import joblib  
import numpy as np
import pandas as pd
import SimpleITK as sitk
import scipy.stats as stats
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

# =====================================================
# FUNÇÕES AUXILIARES DE EXTRAÇÃO DE CARACTERÍSTICAS
# (Alinhadas perfeitamente com a extração do Dataset)
# =====================================================

def _segmentar_regiao_com_limite_distancia_otimizado(volume, seed, spacing, dist_max_mm=20.0):
    """Modelo de segmentação otimizado idêntico ao gerador de CSVs."""
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


def _extrair_atributos_radiomicos(volume, mask_seg, spacing):
    """Extrai as colunas na ordem exata esperada pelo RobustScaler e pelo classificador final."""
    img_sitk = sitk.GetImageFromArray(mask_seg.astype(np.uint8))
    mask_nucleo = sitk.GetArrayFromImage(sitk.BinaryErode(img_sitk, [2, 2, 2]))
    if np.sum(mask_nucleo) == 0:
        mask_nucleo = mask_seg
        
    pixels = volume[mask_nucleo > 0]
    if len(pixels) == 0:
        return None

    nvoxels = np.sum(mask_seg)
    volume_ia_mm3 = float(nvoxels * (spacing[0] * spacing[1] * spacing[2]))

    res = {}
    res['HU_Mean_IA'] = float(np.mean(pixels))
    res['HU_Std_IA'] = float(np.std(pixels))
    res['HU_Variance_IA'] = float(np.var(pixels))
    res['HU_Skewness_IA'] = float(stats.skew(pixels)) if len(pixels) > 2 else 0.0
    res['HU_Kurtosis_IA'] = float(stats.kurtosis(pixels)) if len(pixels) > 2 else 0.0
    res['HU_90thPercentile_IA'] = float(np.percentile(pixels, 90))
    
    counts, _ = np.histogram(pixels, bins=32, density=True)
    probs = counts / (np.sum(counts) + 1e-8)
    probs = probs[probs > 0]
    res['HU_Entropy_IA'] = float(-np.sum(probs * np.log2(probs)))
    res['HU_Uniformity_IA'] = float(np.sum(probs ** 2))
    
    indice_calcificacao = float(np.sum(pixels > 100) / len(pixels))
    res['Proporcao_Calcificacao_IA'] = indice_calcificacao
    res['Sugerido_Benigno_Por_Calcificacao'] = 1.0 if indice_calcificacao > 0.10 else 0.0

    fatias = np.where(np.any(mask_nucleo > 0, axis=(1, 2)))[0]
    l_cont, l_homo, l_ener, l_corr, l_lbp = [], [], [], [], []

    for z in fatias:
        coords = np.argwhere(mask_nucleo[z, :, :] > 0)
        if len(coords) == 0: continue
        y_m, x_m = coords.min(axis=0)
        y_M, x_M = coords.max(axis=0) + 1
        
        roi_vol = volume[z, max(0, y_m-1):y_M+1, max(0, x_m-1):x_M+1]
        roi_mask = mask_nucleo[z, max(0, y_m-1):y_M+1, max(0, x_m-1):x_M+1]
        
        roi_norm = np.clip(roi_vol, -1000, 400)
        roi_norm = ((roi_norm - roi_norm.min()) / (np.ptp(roi_norm) + 1e-8) * 63).astype(np.uint8)

        glcm = graycomatrix(roi_norm, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=64, symmetric=True, normed=True)
        glcm[0,:,:,:], glcm[:,0,:,:] = 0, 0
        if glcm.sum() > 0: glcm /= glcm.sum()

        l_cont.append(graycoprops(glcm, 'contrast')[0, 0])
        l_homo.append(graycoprops(glcm, 'homogeneity')[0, 0])
        l_ener.append(graycoprops(glcm, 'energy')[0, 0])
        l_corr.append(graycoprops(glcm, 'correlation')[0, 0])

        valores_lbp = local_binary_pattern(roi_norm, P=8, R=1, method='uniform')[roi_mask > 0]
        if len(valores_lbp) > 0:
            hist_lbp, _ = np.histogram(valores_lbp, bins=10, density=True)
            l_lbp.append(np.sum(hist_lbp ** 2))

    res['Haralick_Contraste'] = float(np.mean(l_cont)) if l_cont else 0.0
    res['Haralick_Homogeneidade'] = float(np.mean(l_homo)) if l_homo else 0.0
    res['Haralick_Energia'] = float(np.mean(l_ener)) if l_ener else 0.0
    res['Haralick_Correlacao'] = float(np.mean(l_corr)) if l_corr else 0.0
    res['LBP_Energia_Textura'] = float(np.mean(l_lbp)) if l_lbp else 0.0

    res['Indice_Calcificacao_Por_Tamanho'] = res['Proporcao_Calcificacao_IA'] / (volume_ia_mm3 + 1e-5)
    res['Caos_Tecidual_Combinado'] = res['HU_Variance_IA'] * res['Haralick_Contraste']

    df_features = pd.DataFrame([res])
    
    # 🎯 CONFIGURAÇÃO EXATA DAS 18 COLUNAS EXIGIDAS NO SEU FIT DE TREINO
    colunas_modelo = [
        'HU_Mean_IA', 'HU_Std_IA', 'HU_Variance_IA', 'HU_Skewness_IA', 'HU_Kurtosis_IA',
        'HU_90thPercentile_IA', 'HU_Entropy_IA', 'HU_Uniformity_IA', 'Proporcao_Calcificacao_IA',
        'Sugerido_Benigno_Por_Calcificacao', 'Haralick_Contraste', 'Haralick_Homogeneidade',
        'Haralick_Energia', 'Haralick_Correlacao', 'LBP_Energia_Textura',
        'Indice_Calcificacao_Por_Tamanho', 'Caos_Tecidual_Combinado'
    ]
    
    # Garante apenas as colunas corretas e na ordem correta
    return df_features[colunas_modelo]

# =====================================================
# INTERFACE PRINCIPAL PREDITIVA
# =====================================================

def classificar_nodulos(exame_tc: str, lista_coordenadas: list) -> dict:
    """Roda inferências manuais sincronizadas e robustas para o professor."""
    caminho_modelo = "modelo_malignidade_treinado.joblib"
    caminho_scaler = "scaler_robusto.joblib"
    
    if not os.path.exists(caminho_modelo):
        raise FileNotFoundError(f"O arquivo do modelo '{caminho_modelo}' precisa estar nesta pasta.")
    if not os.path.exists(caminho_scaler):
        raise FileNotFoundError(f"O arquivo do scaler '{caminho_scaler}' precisa estar nesta pasta.")
    
    modelo = joblib.load(caminho_modelo)
    scaler = joblib.load(caminho_scaler)
    
    reader = sitk.ImageSeriesReader()
    arquivos_dicom = reader.GetGDCMSeriesFileNames(exame_tc)
    if len(arquivos_dicom) == 0:
        raise ValueError(f"Nenhum arquivo DICOM em: {exame_tc}")
        
    reader.SetFileNames(arquivos_dicom)
    volume_sitk = reader.Execute()
    volume = sitk.GetArrayFromImage(volume_sitk)
    spacing = volume_sitk.GetSpacing()[::-1] 

    id_exame = os.path.basename(os.path.normpath(exame_tc))
    resultado_json = {"exame": id_exame, "nodulos": []}

    for nodulo_req in lista_coordenadas:
        n_id = nodulo_req["id"]
        x_base, y_base, z_base = int(nodulo_req["x"]), int(nodulo_req["y"]), int(nodulo_req["z"])
        
        mask_seg = None
        df_atributos = None
        
        # Micro-varredura defensiva contra erros geográficos ou inversão de eixos
        vizinhança_desvios = [
            (0, 0, 0),    
            (0, 0, -5), (0, 0, 5),   
            (0, -5, 0), (0, 5, 0),   
            (-1, 0, 0), (1, 0, 0),   
            (-2, 0, 0), (2, 0, 0)    
        ]
        
        for dz, dy, dx in vizinhança_desvios:
            try:
                seed_tentativa = (z_base + dz, y_base + dy, x_base + dx)
                
                if (0 <= seed_tentativa[0] < volume.shape[0] and
                    0 <= seed_tentativa[1] < volume.shape[1] and
                    0 <= seed_tentativa[2] < volume.shape[2]):
                    
                    # Chamando o novo segmentador otimizado
                    mask_seg_try = _segmentar_regiao_com_limite_distancia_otimizado(volume, seed_tentativa, spacing)
                    df_attr_try = _extrair_atributos_radiomicos(volume, mask_seg_try, spacing)
                    
                    if df_attr_try is not None:
                        mask_seg = mask_seg_try
                        df_atributos = df_attr_try
                        break 
            except Exception:
                continue

        try:
            if df_atributos is not None:
                # Transforma utilizando os nomes corretos para evitar alertas na tela
                vetor_scaled = scaler.transform(df_atributos)
                
                probabilidade_maligno = float(modelo.predict_proba(vetor_scaled)[0][1])
                
                # Ponto de corte clínico calibrado para compensar desvios de textura (0.35)
                classe_predita = "maligno" if probabilidade_maligno >= 0.35 else "benigno"
                confianca = probabilidade_maligno if classe_predita == "maligno" else (1.0 - probabilidade_maligno)
            else:
                classe_predita = "indeterminado (erro de segmentação anatômica)"
                confianca = 0.0
                
        except Exception as e:
            classe_predita = f"erro na inferência matemática: {str(e)}"
            confianca = 0.0

        dados_nodulo = {
            "id": n_id,
            "coordenada": {"x": x_base, "y": y_base, "z": z_base},
            "classe_predita": classe_predita,
            "probabilidade": round(confianca, 2)
        }
        resultado_json["nodulos"].append(dados_nodulo)

    return resultado_json


# =====================================================
# 🖥️ PAINEL EXCLUSIVO PARA TESTE MANUAL DO PROFESSOR
# =====================================================
if __name__ == "__main__":
    print("\n" + "="*85)
    print("▶️ INTERFACE INTERATIVA - PAINEL DE AVALIAÇÃO DE NÓDULOS (USO DO PROFESSOR)")
    print("="*85)
    
    caminho_demonstracao = r"C:\dataset nodulos\lidc_idri\lidc_idri\LIDC-IDRI-0001\30178\03192"
    
    print("Instruções: Insira os dados solicitados ou aperte ENTER para usar o caso padrão.")
    print("-" * 85)
    
    input_caminho = input(f"Caminho da pasta DICOM [{caminho_demonstracao}]: ").strip()
    if not input_caminho:
        input_caminho = caminho_demonstracao
        
    if os.path.exists(input_caminho):
        try:
            input_x = input("Digite a coordenada X do Nódulo [ex: 367]: ").strip()
            input_y = input("Digite a coordenada Y do Nódulo [ex: 345]: ").strip()
            input_z = input("Digite a coordenada Z do Nódulo [ex: 44]: ").strip()
            
            val_x = int(input_x) if input_x else 367
            val_y = int(input_y) if input_y else 345
            val_z = int(input_z) if input_z else 44
            
            coordenadas_manuais = [{
                "id": "Nodulo_Analise_Manual_Professor",
                "x": val_x, "y": val_y, "z": val_z
            }]
            
            print(f"\n🔍 Buscando parênquima usando modelo otimizado: X={val_x}, Y={val_y}, Z={val_z}")
            resultado_manual = classificar_nodulos(exame_tc=input_caminho, lista_coordenadas=coordenadas_manuais)
            
            print("\n📊 RESULTADO DO LAUDO DE IA EM TEMPO REAL:")
            print(json.dumps(resultado_manual, indent=2, ensure_ascii=False))
            
        except ValueError:
            print("❌ Erro: As coordenadas inseridas precisam ser números inteiros válidos.")
        except Exception as e:
            print(f"❌ Erro crítico na execução do laudo: {e}")
    else:
        print(f"❌ Caminho inválido ou inacessível no HD local: {input_caminho}")

    print("\n" + "="*85)
    print("🏁 Pronto para a apresentação! Módulo interativo finalizado.")
    print("="*85)