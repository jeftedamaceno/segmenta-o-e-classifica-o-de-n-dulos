import os
import scipy.stats as stats
import pandas as pd
import numpy as np
import SimpleITK as sitk
import pydicom
import matplotlib.pyplot as plt

CSV_PATH = "dados_segmentacao_nodulos_completo.csv"
RAIZ_DICOM = r"C:\dataset nodulos\lidc_idri\lidc_idri"

if __name__ == "__main__":
    print("Mapeando exames fisicos no diretorio...")
    exames_no_hd = {}
    
    for pasta_atual, _, arquivos in os.walk(RAIZ_DICOM):
        dicoms = [f for f in arquivos if f.lower().endswith(".dcm") or f.startswith("1.")]
        if len(dicoms) == 0:
            continue
        try:
            amostra = os.path.join(pasta_atual, dicoms[0])
            ds = pydicom.dcmread(amostra, stop_before_pixels=True)
            uid_real_dicom = str(ds.SeriesInstanceUID).strip()
            exames_no_hd[uid_real_dicom] = pasta_atual
        except:
            continue

    df_completo = pd.read_csv(CSV_PATH)
    df = df_completo[df_completo["paciente_serie_uid"].isin(exames_no_hd.keys())].copy()
    
    uids_finais = df["paciente_serie_uid"].unique()
    
    coluna_malignidade = None
    for col in df.columns:
        if "malign" in col.lower():
            coluna_malignidade = col
            break
            
    volumes_reais = []
    coordenadas_x = []
    coordenadas_y = []
    coordenadas_z = []
    valores_hu_sementes = []
    notas_malignidade = []

    print("Extraindo propriedades geometricas e radiologicas...")
    for uid in uids_finais:
        pasta = exames_no_hd[uid]
        try:
            reader = sitk.ImageSeriesReader()
            arquivos = reader.GetGDCMSeriesFileNames(pasta)
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
            nodulos = df_exame.groupby("nodulo_original_id")

            for nodulo_id, df_nodulo in nodulos:
                nota_media = df_nodulo[coluna_malignidade].dropna().mean()
                if not pd.isna(nota_media):
                    notas_malignidade.append(round(nota_media))

                mapa_sop_limpo = {str(k).strip().rstrip('.0'): v for k, v in mapa_sop.items()}
                fatias_validas = []
                
                for _, row in df_nodulo.iterrows():
                    sop_uid = str(row["sop_uid_fat_dicom"]).strip().rstrip('.0')
                    if sop_uid in mapa_sop_limpo:
                        fatias_validas.append(row)
                        
                if len(fatias_validas) == 0:
                    continue

                df_fatias = pd.DataFrame(fatias_validas)
                cx = df_fatias["centro_x"].mean()
                cy = df_fatias["centro_y"].mean()
                cz = df_fatias["coord_z"].mean()
                
                coordenadas_x.append(cx)
                coordenadas_y.append(cy)
                coordenadas_z.append(cz)

                z_indices = [mapa_sop_limpo[str(r["sop_uid_fat_dicom"]).strip().rstrip('.0')] for r in fatias_validas]
                z_centro_idx = int(np.mean(z_indices))
                y_centro_idx = int(cy)
                x_centro_idx = int(cx)

                try:
                    pixel_hu = volume[z_centro_idx, y_centro_idx, x_centro_idx]
                    valores_hu_sementes.append(float(pixel_hu))
                except:
                    pass

                voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
                total_pixels_contornos = 0
                for _, row in df_nodulo.iterrows():
                    separador = ";" if ";" in str(row["contorno_x"]) else ","
                    total_pixels_contornos += len(str(row["contorno_x"]).split(separador))
                
                volumes_reais.append(total_pixels_contornos * voxel_volume_mm3)

        except:
            continue

    fig = plt.figure(figsize=(16, 12))
    
    ax1 = fig.add_subplot(2, 2, 1)
    if notas_malignidade:
        df_malign = pd.Series(notas_malignidade).value_counts().sort_index()
        ax1.bar(df_malign.index, df_malign.values, color="steelblue", edgecolor="black")
        ax1.set_title("Distribuicao das Notas de Malignidade")
        ax1.set_xlabel("Nota de Malignidade")
        ax1.set_ylabel("Quantidade de Nodulos")
        ax1.set_xticks(range(1, 6))
        ax1.grid(True, linestyle=":", alpha=0.6)

    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    if coordenadas_x:
        sc = ax2.scatter(coordenadas_x, coordenadas_y, coordenadas_z, c=coordenadas_z, cmap="jet", alpha=0.8, edgecolors="w")
        ax2.set_title("Concentracao Espacial dos Nodulos no Pulmao")
        ax2.set_xlabel("Eixo X (Anatomico)")
        ax2.set_ylabel("Eixo Y (Anatomico)")
        ax2.set_zlabel("Eixo Z (Altura Toracica)")

    ax3 = fig.add_subplot(2, 2, 3)
    if volumes_reais:
        ax3.boxplot(volumes_reais, vert=False, patch_artist=True, 
                    boxprops=dict(facecolor="lightblue", color="blue"),
                    medianprops=dict(color="red", linewidth=2))
        ax3.set_title("Distribuicao Volumetrica dos Nodulos")
        ax3.set_xlabel("Volume em mm3")
        ax3.set_yticklabels(["Nodulos"])
        ax3.grid(True, linestyle=":", alpha=0.6)
        
        v_mean = np.mean(volumes_reais)
        v_std = np.std(volumes_reais)
        print(f"Estatisticas de Volume: Media = {v_mean:.2f} mm3 | Desvio Padrao = {v_std:.2f} mm3")

    ax4 = fig.add_subplot(2, 2, 4)
    if valores_hu_sementes:
        ax4.hist(valores_hu_sementes, bins=20, density=True, alpha=0.6, color="gray", edgecolor="black")
        
        hu_media = np.mean(valores_hu_sementes)
        hu_mediana = np.median(valores_hu_sementes)
        hu_moda = float(stats.mode(valores_hu_sementes, keepdims=True)[0][0])
        
        ax4.axvline(hu_media, color="red", linestyle="--", linewidth=2, label=f"Media: {hu_media:.1f} HU")
        ax4.axvline(hu_mediana, color="green", linestyle="-.", linewidth=2, label=f"Mediana: {hu_mediana:.1f} HU")
        ax4.axvline(hu_moda, color="blue", linestyle=":", linewidth=2, label=f"Moda: {hu_moda:.1f} HU")
        
        ax4.set_title("Perfil de Atenuacao Radiologica das Sementes")
        ax4.set_xlabel("Unidades Hounsfield (HU)")
        ax4.set_ylabel("Densidade de Probabilidade")
        ax4.legend(loc="upper right")
        ax4.grid(True, linestyle=":", alpha=0.6)
        
        print(f"Estatisticas de Atenuacao: Media = {hu_media:.1f} HU | Mediana = {hu_mediana:.1f} HU | Moda = {hu_moda:.1f} HU")

    plt.tight_layout()
    plt.show()