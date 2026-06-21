import os
import json
import joblib  # Para carregar o modelo de Machine Learning treinado
import numpy as np
import SimpleITK as sitk
import pydicom
import scipy.stats as stats
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

# =====================================================
# FUNÇÕES AUXILIARES DE EXTRAÇÃO DE CARACTERÍSTICAS
# =====================================================

def _segmentar_regiao_com_limite_distancia(volume, seed, spacing, dist_max_mm=20.0):
    """Função interna para segmentar o nódulo a partir da coordenada informada pelo professor."""
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
    dist_real_mm = np.sqrt(((gz - z_s_local) * spacing[0])**2 + ((gy - y_s_local) * spacing[1])**2 + ((gx - x_s_local) * spacing[2])**2)
    mascara_distancia = (dist_real_mm <= dist_max_mm).astype(np.uint8)

    valor_seed_direto = float(sub_volume[z_s_local, y_s_local, x_s_local])
    limiar_inferior = max(valor_seed_direto - 100, -850) if valor_seed_direto < -500 else -620

    mascara_combinada = ((sub_volume >= limiar_inferior) & (sub_volume <= 100) | (sub_volume > 100)) & (mascara_distancia == 1)
    
    img_sitk_bruta = sitk.GetImageFromArray(mascara_combinada.astype(np.uint8))
    img_erodida = sitk.BinaryMorphologicalOpening(img_sitk_bruta, [1, 1, 1])
    img_conectada = sitk.ConnectedThreshold(img_erodida, seedList=lista_seed_local, lower=1, upper=1, replaceValue=1)
    img_dilatada = sitk.BinaryMorphologicalClosing(img_conectada, [2, 2, 2])
    img_final = sitk.BinaryFillhole(img_dilatada)
    
    mask_seg_global = np.zeros(volume.shape, dtype=np.uint8)
    mask_seg_global[z_min:z_max, y_min:y_max, x_min:x_max] = sitk.GetArrayFromImage(img_final)
    return mask_seg_global


def _extrair_atributos_radiomicos(volume, mask_seg, spacing):
    """Extrai exatamente as mesmas variáveis que foram usadas para treinar o modelo."""
    img_sitk = sitk.GetImageFromArray(mask_seg.astype(np.uint8))
    mask_nucleo = sitk.GetArrayFromImage(sitk.BinaryErode(img_sitk, [2, 2, 2]))
    if np.sum(mask_nucleo) == 0:
        mask_nucleo = mask_seg
        
    pixels = volume[mask_nucleo > 0]
    if len(pixels) == 0:
        return None

    features = [
        float(np.mean(pixels)),               # HU_Mean_IA
        float(np.std(pixels)),                # HU_Std_IA
        float(np.var(pixels)),                # HU_Variance_IA
        float(stats.skew(pixels)) if len(pixels) > 2 else 0.0,      # HU_Skewness_IA
        float(stats.kurtosis(pixels)) if len(pixels) > 2 else 0.0,  # HU_Kurtosis_IA
        float(np.percentile(pixels, 90))      # HU_90thPercentile_IA
    ]
    
    counts, _ = np.histogram(pixels, bins=32, density=True)
    probs = counts / (np.sum(counts) + 1e-8)
    probs = probs[probs > 0]
    features.append(float(-np.sum(probs * np.log2(probs))))  # HU_Entropy_IA
    features.append(float(np.sum(probs ** 2)))               # HU_Uniformity_IA
    
    indice_calcificacao = float(np.sum(pixels > 100) / len(pixels))
    features.append(indice_calcificacao)                     # Proporcao_Calcificacao_IA
    features.append(1.0 if indice_calcificacao > 0.10 else 0.0) # Sugerido_Benigno_Por_Calcificacao

    # Texturas (Haralick + LBP)
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
        roi_proc = np.where(roi_mask > 0, roi_norm, 0)

        glcm = graycomatrix(roi_proc, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=64, symmetric=True, normed=True)
        glcm[0,:,:,:], glcm[:,0,:,:] = 0, 0
        if glcm.sum() > 0: glcm /= glcm.sum()

        l_cont.append(graycoprops(glcm, 'contrast')[0, 0])
        l_homo.append(graycoprops(glcm, 'homogeneity')[0, 0])
        l_ener.append(graycoprops(glcm, 'energy')[0, 0])
        l_corr.append(graycoprops(glcm, 'correlation')[0, 0])

        valores_lbp = local_binary_pattern(roi_proc, P=8, R=1, method='uniform')[roi_mask > 0]
        if len(valores_lbp) > 0:
            hist_lbp, _ = np.histogram(valores_lbp, bins=10, density=True)
            l_lbp.append(np.sum(hist_lbp ** 2))

    features.extend([
        float(np.mean(l_cont)) if l_cont else 0.0,
        float(np.mean(l_homo)) if l_homo else 0.0,
        float(np.mean(l_ener)) if l_ener else 0.0,
        float(np.mean(l_corr)) if l_corr else 0.0,
        float(np.mean(l_lbp)) if l_lbp else 0.0
    ])
    
    return np.array(features).reshape(1, -1)

# =====================================================
# INTERFACE EXIGIDA PELO PROFESSOR (ENTRADA E SAÍDA)
# =====================================================

def classificar_nodulos(exame_tc: str, lista_coordenadas: list) -> dict:
    """
    Função principal exigida pelo professor.
    
    Parâmetros:
    - exame_tc: String com o caminho da pasta contendo os arquivos DICOM do exame.
    - lista_coordenadas: Lista de dicionários contendo {'id', 'x', 'y', 'z'}
    
    Retorno:
    - Dicionário Python estruturado pronto para conversão em JSON.
    """
    # 1. Carrega o modelo de IA que você treinou anteriormente e salvou no GitHub
    caminho_modelo = "modelo_malignidade_treinado.joblib"
    if not os.path.exists(caminho_modelo):
        raise FileNotFoundError(f"O arquivo do modelo '{caminho_modelo}' precisa estar na mesma pasta.")
    
    modelo = joblib.load(caminho_modelo)
    
    # 2. Carrega o volume tridimensional DICOM usando o SimpleITK
    reader = sitk.ImageSeriesReader()
    arquivos_dicom = reader.GetGDCMSeriesFileNames(exame_tc)
    if len(arquivos_dicom) == 0:
        raise ValueError(f"Nenhum arquivo DICOM válido foi encontrado na pasta: {exame_tc}")
        
    reader.SetFileNames(arquivos_dicom)
    volume_sitk = reader.Execute()
    volume = sitk.GetArrayFromImage(volume_sitk)
    spacing = volume_sitk.GetSpacing()[::-1] # Ajusta ordem para Z, Y, X

    # Identificador do exame baseado no nome da pasta
    id_exame = os.path.basename(os.path.normpath(exame_tc))
    
    resultado_json = {
        "exame": id_exame,
        "nodulos": []
    }

    # 3. Processa cada nódulo solicitado pelo professor
    for nodulo_req in lista_coordenadas:
        n_id = nodulo_req["id"]
        # Convertendo as coordenadas passadas para inteiros
        seed = (int(nodulo_req["z"]), int(nodulo_req["y"]), int(nodulo_req["x"]))
        
        try:
            # Segmenta o nódulo dinamicamente a partir do ponto dado
            mask_seg = _segmentar_regiao_com_limite_distancia(volume, seed, spacing)
            
            # Extrai o vetor de atributos (as 15 colunas que criamos)
            vetor_atributos = _extrair_atributos_radiomicos(volume, mask_seg, spacing)
            
            if vetor_atributos is not None:
                # Faz a predição probabilística usando o modelo carregado
                probabilidade_maligno = float(modelo.predict_proba(vetor_atributos)[0][1])
                
                classe_predita = "maligno" if probabilidade_maligno >= 0.50 else "benigno"
                confianca = probabilidade_maligno if classe_predita == "maligno" else (1.0 - probabilidade_maligno)
            else:
                # Caso ocorra falha crítica na extração dos pixels
                classe_predita = "indeterminado (erro de segmentação)"
                confianca = 0.0
                
        except Exception as e:
            classe_predita = f"erro no processamento: {str(e)}"
            confianca = 0.0

        # Monta a estrutura exata do nódulo pedida no enunciado
        dados_nodulo = {
            "id": n_id,
            "coordenada": {"x": nodulo_req["x"], "y": nodulo_req["y"], "z": nodulo_req["z"]},
            "classe_predita": classe_predita,
            "probabilidade": round(confianca, 2)
        }
        resultado_json["nodulos"].append(dados_nodulo)

    return resultado_json


# =====================================================
# EXEMPLO DE USO EM AMBIENTE DE TESTE
# =====================================================
if __name__ == "__main__":
    # Teste simulado local idêntico ao que o professor executará:
    caminho_teste = r"C:\dataset nodulos\lidc_idri\lidc_idri\Pasta_De_Um_Exame_Exemplo"
    coordenadas_teste = [
        {"id": "nodulo_1", "x": 120, "y": 85, "z": 42},
        {"id": "nodulo_2", "x": 210, "y": 130, "z": 55}
    ]
    
    print("Executando teste da função...")
    try:
        resultado = classificar_nodulos(exame_tc=caminho_teste, lista_coordenadas=coordenadas_teste)
        # Exibe a saída formatada lindamente em JSON
        print(json.dumps(resultado, indent=2))
    except Exception as e:
        print(f"Nota: Altere os caminhos do bloco de teste para rodar localmente. Erro esperado: {e}")