import os
import json
import re
import numpy as np
import pandas as pd
import SimpleITK as sitk
from prever_nodulos_IA import classificar_nodulos  # Importando o arquivo de inferência corrigido

# ==============================================================================
# CONFIGURAÇÕES DE DIRETÓRIOS E ARQUIVOS (AMOSTRAGEM CONTROLADA)
# ==============================================================================
ARQUIVO_METADADOS = "dados_segmentacao_nodulos_completo.csv"
RAIZ_EXAMES = r"C:\dataset nodulos\lidc_idri\lidc_idri"
LIMITE_AMOSTRAS = 150  # 🎯 Restrição de amostragem de nódulos

def executar_validacao_em_lote():
    print("=" * 85)
    print(f"🚀 PIPELINE DE AUDITORIA ({LIMITE_AMOSTRAS} NÓDULOS) - VERIFICAÇÃO DE DISTRIBUIÇÃO DA IA")
    print("=" * 85)

    if not os.path.exists(ARQUIVO_METADADOS):
        raise FileNotFoundError(f"Não foi possível localizar o arquivo de metadados: '{ARQUIVO_METADADOS}'")
    
    if not os.path.exists(RAIZ_EXAMES):
        raise FileNotFoundError(f"A pasta raiz de exames não existe no HD: '{RAIZ_EXAMES}'")

    # 1. Mapeia pastas físicas no HD
    pastas_no_hd = [d for d in os.listdir(RAIZ_EXAMES) if os.path.isdir(os.path.join(RAIZ_EXAMES, d))]

    # 2. Carrega metadados do Padrão Ouro
    df_meta = pd.read_csv(ARQUIVO_METADADOS)
    
    col_x = 'centro_x'
    col_y = 'centro_y'
    col_z = 'coord_z'
    col_target = 'malignancy'
    col_uid = 'paciente_serie_uid'
    col_xml = 'caminho_xml_origem'

    df_meta = df_meta.dropna(subset=[col_target])

    # 3. Filtra e limita rigorosamente para as amostras válidas no HD local
    linhas_filtradas = []
    for idx, linha in df_meta.iterrows():
        caminho_xml = str(linha[col_xml])
        numeros_no_caminho = re.findall(r'\d+', caminho_xml)
        
        if not numeros_no_caminho:
            continue
            
        match_encontrado = False
        pasta_detectada = None
        
        for num in numeros_no_caminho:
            num_normalizado = num.zfill(4)
            for pasta in pastas_no_hd:
                if num in pasta or num_normalizado in pasta:
                    match_encontrado = True
                    pasta_detectada = pasta
                    break
            if match_encontrado:
                break
                
        if match_encontrado:
            linha_copia = linha.copy()
            linha_copia['pasta_fisica_detectada'] = pasta_detectada
            linhas_filtradas.append(linha_copia)
            if len(linhas_filtradas) >= LIMITE_AMOSTRAS:
                break
                
    # ✨ CORREÇÃO DO BUG: Removido o typo 'lignes_filtradas' que quebrava a execução
    if not linhas_filtradas:
        print("❌ Nenhuma correspondência encontrada entre o CSV e os exames locais.")
        return
        
    df_local = pd.DataFrame(linhas_filtradas)
    print(f"✅ Amostragem definida: {len(df_local)} nódulos prontos para avaliação.")
    print("-" * 85)

    # 4. Inicialização de contadores estatísticos e de distribuição da IA
    estatisticas = {
        "Total_Nodulos_Avaliados": 0,
        "Verdadeiros_Malignos_Acertos": 0,    
        "Verdadeiros_Benignos_Acertos": 0,    
        "Falsos_Malignos_Erros": 0,           
        "Falsos_Benignos_Erros": 0,           
        "Casos_Moderados_Ignorados": 0,       
        "Falhas_Segmentacao_Indeterminados": 0, 
        "Erros_Leitura_DICOM": 0              
    }
    
    distribuicao_ia = {
        "Previsoes_Maligno": 0,
        "Previsoes_Benigno": 0
    }
    
    detalhes_previsoes = []
    exames_agrupados = df_local.groupby(col_uid)

    # 5. Loop de Processamento Tridimensional Otimizado
    for uid_serie, linhas_nodulos in exames_agrupados:
        pasta_paciente = linhas_nodulos.iloc[0]['pasta_fisica_detectada']
        pasta_busca = os.path.join(RAIZ_EXAMES, pasta_paciente)
        
        pastas_com_dicom = [os.path.dirname(os.path.join(dp, f[0])) for dp, dn, f in os.walk(pasta_busca) if len(f) > 0 and f[0].lower().endswith('.dcm')]
        
        if not pastas_com_dicom:
            estatisticas["Erros_Leitura_DICOM"] += len(linhas_nodulos)
            continue
            
        caminho_dicom_final = pastas_com_dicom[0]
        
        try:
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(caminho_dicom_final)
            reader.SetFileNames(dicom_names)
            imagem_volume = reader.Execute()
            origem_z = imagem_volume.GetOrigin()[2]       
            espacamento_z = imagem_volume.GetSpacing()[2] 
            profundidade_maxima = imagem_volume.GetSize()[2]
        except Exception:
            origem_z = 0
            espacamento_z = 1.0
            profundidade_maxima = 300

        lista_coordenadas_exame = []
        mapeamento_real = {}
        
        for idx, linha in linhas_nodulos.iterrows():
            nodulo_id = f"Nodulo_Linha_{idx}"
            z_fisico = float(linha[col_z])
            
            if espacamento_z != 0:
                z_fatia = int(round(abs(z_fisico - origem_z) / espacamento_z))
            else:
                z_fatia = int(abs(z_fisico))
                
            if z_fatia >= profundidade_maxima or z_fatia < 0:
                z_fatia = max(0, min(20, profundidade_maxima - 1)) 
                
            lista_coordenadas_exame.append({
                "id": nodulo_id,
                "x": int(float(linha[col_x])),
                "y": int(float(linha[col_y])),
                "z": z_fatia
            })
            
            nota_malignancy = int(float(linha[col_target]))
            if nota_malignancy >= 4:
                mapeamento_real[nodulo_id] = "maligno"
            elif nota_malignancy <= 2:
                mapeamento_real[nodulo_id] = "benigno"
            else:
                mapeamento_real[nodulo_id] = "moderado"

        try:
            resultado_ia = classificar_nodulos(exame_tc=caminho_dicom_final, lista_coordenadas=lista_coordenadas_exame)
            
            for nodulo_predito in resultado_ia["nodulos"]:
                n_id = nodulo_predito["id"]
                classe_ia = str(nodulo_predito["classe_predita"]).lower()
                classe_real = mapeamento_real[n_id]
                
                if classe_real == "moderado":
                    estatisticas["Casos_Moderados_Ignorados"] += 1
                    continue
                    
                estatisticas["Total_Nodulos_Avaliados"] += 1
                
                # Registra o que a IA escolheu (desde que não tenha falhado a segmentação anatômica)
                if "maligno" in classe_ia:
                    distribuicao_ia["Previsoes_Maligno"] += 1
                elif "benigno" in classe_ia:
                    distribuicao_ia["Previsoes_Benigno"] += 1
                
                # Sincronização com as strings de erro retornadas pelo novo laudo interativo
                if "indeterminado" in classe_ia or "erro" in classe_ia:
                    estatisticas["Falhas_Segmentacao_Indeterminados"] += 1
                    status_resultado = "Falha de Segmentação Anatômica"
                elif "maligno" in classe_ia and classe_real == "maligno":
                    estatisticas["Verdadeiros_Malignos_Acertos"] += 1
                    status_resultado = "Acerto (Verdadeiro Maligno)"
                elif "benigno" in classe_ia and classe_real == "benigno":
                    estatisticas["Verdadeiros_Benignos_Acertos"] += 1
                    status_resultado = "Acerto (Verdadeiro Benigno)"
                elif "maligno" in classe_ia and classe_real == "benigno":
                    estatisticas["Falsos_Malignos_Erros"] += 1
                    status_resultado = "Erro (Falso Maligno)"
                elif "benigno" in classe_ia and classe_real == "maligno":
                    estatisticas["Falsos_Benignos_Erros"] += 1
                    status_resultado = "Erro (Falso Benigno)"
                    
                detalhes_previsoes.append({
                    "paciente": pasta_paciente,
                    "coordenadas": nodulo_predito["coordenada"],
                    "gabarito_medico": classe_real,
                    "ia_previsao": classe_ia,
                    "status": status_resultado,
                    "probabilidade_confianca": nodulo_predito["probabilidade"]
                })
                
            print(f"✅ [PROCESSADO] Paciente {pasta_paciente} analisado via modelo de texturas.")
            
        except Exception as e:
            print(f"❌ Falha no pipeline do paciente {pasta_paciente}: {e}")
            estatisticas["Erros_Leitura_DICOM"] += len(lista_coordenadas_exame)

    # ==============================================================================
    # 📊 PAINEL DE RELATÓRIO COM AUDITORIA DE VIÉS DA IA
    # ==============================================================================
    print("\n" + "=" * 85)
    print("🔍 PAINEL DE AUDITORIA DE VIÉS (DISTRIBUIÇÃO DE PREVISÕES DA IA)")
    print("=" * 85)
    total_classificados = (distribuicao_ia['Previsoes_Maligno'] + distribuicao_ia['Previsoes_Benigno'])
    
    print(f"📢 QUANTIDADE QUE A IA CLASSIFICOU COMO MALIGNO: {distribuicao_ia['Previsoes_Maligno']}")
    print(f"📢 QUANTIDADE QUE A IA CLASSIFICOU COMO BENIGNO: {distribuicao_ia['Previsoes_Benigno']}")
    print("-" * 85)
    
    if total_classificados > 0:
        if distribuicao_ia["Previsoes_Maligno"] == total_classificados or distribuicao_ia["Previsoes_Benigno"] == total_classificados:
            print("⚠️ ALERTAR PROFESSOR: O modelo está VIESADO. Está prevendo 100% dos casos numa classe só.")
        else:
            print("⚖️ ASSINATURA SAUDÁVEL: O classificador está distribuindo previsões de forma harmônica entre ambas as classes.")
    else:
        print("⚠️ Atenção: Nenhum nódulo foi classificado devido a falhas geográficas nas coordenadas Z.")

    print("\n" + "=" * 85)
    print(f"📊 MATRIZ DE DESEMPENHO COMPLETA")
    print("=" * 85)
    total_acertos = estatisticas["Verdadeiros_Malignos_Acertos"] + estatisticas["Verdadeiros_Benignos_Acertos"]
    taxa_acerto = (total_acertos / total_classificados * 100) if total_classificados > 0 else 0.0

    print(f"📌 Total de Nódulos Processados: {estatisticas['Total_Nodulos_Avaliados'] + estatisticas['Casos_Moderados_Ignorados']}")
    print(f"📌 Nódulos Efetivamente Classificados pela IA: {total_classificados}")
    print(f"📌 Falhas de Região/Segmentação (Indeterminados): {estatisticas['Falhas_Segmentacao_Indeterminados']}")
    print("-" * 85)
    print(f" -> [ACERTO] Verdadeiros Malignos : {estatisticas['Verdadeiros_Malignos_Acertos']}")
    print(f" -> [ACERTO] Verdadeiros Benignos : {estatisticas['Verdadeiros_Benignos_Acertos']}")
    print(f" -> [ERRO]   Falsos Malignos      : {estatisticas['Falsos_Malignos_Erros']}")
    print(f" -> [ERRO]   Falsos Benignos      : {estatisticas['Falsos_Benignos_Erros']}")
    print("-" * 85)
    print(f"🎯 ACURÁCIA LÍQUIDA DO MODELO: {taxa_acerto:.2f}%")
    print("=" * 85)

if __name__ == "__main__":
    executar_validacao_em_lote()