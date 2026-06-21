import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np


def obter_texto(elem, tag, ns, prefixo):
    no = elem.find(f"{prefixo}{tag}", ns)
    return no.text if no is not None else None


def extrair_dados_xml(caminho_xml):

    dados_nodulos = []

    try:
        tree = ET.parse(caminho_xml)
        root = tree.getroot()

    except Exception as e:
        print(f"Erro ao abrir {caminho_xml}")
        print(e)
        return []

    ns = {'nih': 'http://www.nih.gov'} if 'http://www.nih.gov' in root.tag else {}
    prefixo = 'nih:' if ns else ''

    series_uid_elem = root.find(
        f'.//{prefixo}SeriesInstanceUid',
        ns
    )

    series_uid = (
        series_uid_elem.text
        if series_uid_elem is not None
        else "DESCONHECIDO"
    )

    sessoes = root.findall(
        f'.//{prefixo}readingSession',
        ns
    )

    for id_medico, sessao in enumerate(sessoes):

        nodulos = sessao.findall(
            f'.//{prefixo}unblindedReadNodule',
            ns
        )

        for nodulo in nodulos:

            nodule_id = obter_texto(
                nodulo,
                "noduleID",
                ns,
                prefixo
            )

            # =====================
            # Características
            # =====================

            characteristics = nodulo.find(
                f'{prefixo}characteristics',
                ns
            )

            malignancy = None
            subtlety = None
            internal_structure = None
            calcification = None
            sphericity = None
            margin = None
            lobulation = None
            spiculation = None
            texture = None

            if characteristics is not None:

                malignancy = obter_texto(
                    characteristics,
                    "malignancy",
                    ns,
                    prefixo
                )

                subtlety = obter_texto(
                    characteristics,
                    "subtlety",
                    ns,
                    prefixo
                )

                internal_structure = obter_texto(
                    characteristics,
                    "internalStructure",
                    ns,
                    prefixo
                )

                calcification = obter_texto(
                    characteristics,
                    "calcification",
                    ns,
                    prefixo
                )

                sphericity = obter_texto(
                    characteristics,
                    "sphericity",
                    ns,
                    prefixo
                )

                margin = obter_texto(
                    characteristics,
                    "margin",
                    ns,
                    prefixo
                )

                lobulation = obter_texto(
                    characteristics,
                    "lobulation",
                    ns,
                    prefixo
                )

                spiculation = obter_texto(
                    characteristics,
                    "spiculation",
                    ns,
                    prefixo
                )

                texture = obter_texto(
                    characteristics,
                    "texture",
                    ns,
                    prefixo
                )

            rois = nodulo.findall(
                f'.//{prefixo}roi',
                ns
            )

            for roi in rois:

                z_elem = roi.find(
                    f'{prefixo}imageZposition',
                    ns
                )

                sop_elem = roi.find(
                    f'{prefixo}imageSOP_UID',
                    ns
                )

                if z_elem is None or sop_elem is None:
                    continue

                z_pos = float(z_elem.text)
                sop_uid = sop_elem.text

                edge_maps = roi.findall(
                    f'{prefixo}edgeMap',
                    ns
                )

                if len(edge_maps) == 0:
                    continue

                x_coords = []
                y_coords = []

                for edge in edge_maps:

                    x_elem = edge.find(
                        f'{prefixo}xCoord',
                        ns
                    )

                    y_elem = edge.find(
                        f'{prefixo}yCoord',
                        ns
                    )

                    if x_elem is None or y_elem is None:
                        continue

                    x_coords.append(int(x_elem.text))
                    y_coords.append(int(y_elem.text))

                if len(x_coords) == 0:
                    continue

                centro_x = np.mean(x_coords)
                centro_y = np.mean(y_coords)

                dados_nodulos.append({

                    "paciente_serie_uid": series_uid,

                    "medico_id": id_medico,

                    "nodulo_original_id": nodule_id,

                    "sop_uid_fat_dicom": sop_uid,

                    "coord_z": z_pos,

                    "centro_x": centro_x,
                    "centro_y": centro_y,

                    "min_x": min(x_coords),
                    "max_x": max(x_coords),

                    "min_y": min(y_coords),
                    "max_y": max(y_coords),

                    "largura_bbox":
                        max(x_coords) - min(x_coords),

                    "altura_bbox":
                        max(y_coords) - min(y_coords),

                    "numero_pontos_contorno":
                        len(x_coords),

                    # CONTORNO COMPLETO
                    "contorno_x":
                        ";".join(map(str, x_coords)),

                    "contorno_y":
                        ";".join(map(str, y_coords)),

                    # CARACTERÍSTICAS LIDC
                    "malignancy": malignancy,
                    "subtlety": subtlety,
                    "internalStructure": internal_structure,
                    "calcification": calcification,
                    "sphericity": sphericity,
                    "margin": margin,
                    "lobulation": lobulation,
                    "spiculation": spiculation,
                    "texture": texture,

                    "caminho_xml_origem":
                        caminho_xml
                })

    return dados_nodulos


def processar_pastas_lidc(
        pastas_alvo,
        diretorio_base):

    todos_dados = []

    print("Procurando XMLs...")

    for pasta in pastas_alvo:

        caminho_pasta = os.path.join(
            diretorio_base,
            pasta
        )

        if not os.path.exists(caminho_pasta):

            print(f"Pasta não encontrada: {caminho_pasta}")
            continue

        for raiz, _, arquivos in os.walk(caminho_pasta):

            for arquivo in arquivos:

                if arquivo.lower().endswith(".xml"):

                    caminho_xml = os.path.join(
                        raiz,
                        arquivo
                    )

                    dados = extrair_dados_xml(
                        caminho_xml
                    )

                    todos_dados.extend(dados)

    if len(todos_dados) == 0:

        print("Nenhum dado encontrado.")
        return None

    df = pd.DataFrame(todos_dados)

    # ==================================
    # CONSENSO ENTRE MÉDICOS
    # ==================================

    df["centro_x_agrup"] = (
        df["centro_x"] / 20
    ).round() * 20

    df["centro_y_agrup"] = (
        df["centro_y"] / 20
    ).round() * 20

    df["numero_de_medicos"] = (
        df.groupby(
            [
                "paciente_serie_uid",
                "sop_uid_fat_dicom",
                "centro_x_agrup",
                "centro_y_agrup"
            ]
        )["medico_id"]
        .transform("nunique")
    )

    # Mantém consenso >= 2

    df = df[
        df["numero_de_medicos"] >= 2
    ].copy()

    df.drop(
        columns=[
            "centro_x_agrup",
            "centro_y_agrup"
        ],
        inplace=True
    )

    df.sort_values(
        [
            "paciente_serie_uid",
            "coord_z"
        ],
        inplace=True
    )

    nome_saida = "dados_segmentacao_nodulos_completo.csv"

    df.to_csv(
        nome_saida,
        index=False,
        encoding="utf-8"
    )

    print()
    print("=" * 60)
    print(f"CSV salvo: {nome_saida}")
    print(f"Registros: {len(df)}")
    print("=" * 60)

    return df


# ======================================
# EXECUÇÃO
# ======================================

if __name__ == "__main__":

    pastas = [
        "157",
        "185",
        "186",
        "187",
        "188",
        "189"
    ]

    diretorio_base = (
        r"C:\Users\jefte\projetos em python"
        r"\ufc 2025 a 2026"
        r"\segmentação de nodulos 2"
        r"\nodulos segmentation"
        r"\LIDC-XML-only"
        r"\tcia-lidc-xml"
    )

    processar_pastas_lidc(
        pastas,
        diretorio_base
    )