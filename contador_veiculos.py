"""
Contador de Veiculos com Jogo de Palpites (YOLOv8 + CNN Validadora)
------------------------------------------------------------------
Como instalar as dependencias (rode uma vez no terminal):
    pip install ultralytics lap opencv-python tensorflow kagglehub pandas scikit-learn

Como executar (modo interativo, igual antes — usado na demo com a turma):
    python contador_veiculos.py

Modo lote (sem janelas e sem palpites; usa o ROI salvo do vídeo) — usado nos experimentos:
    python contador_veiculos.py --video 1125.mp4 --modo cnn --lote
    python contador_veiculos.py --video 1125.mp4 --modo yolo --conf 0.50 --lote
    python contador_veiculos.py --video 1125.mp4 --sem-roi --lote
    python contador_veiculos.py --video 1125.mp4 --condicao escuro --preproc clahe --lote
Cada execução acrescenta uma linha em resultados/experimentos.csv (contagem, resolução,
FPS de origem, tempo médio de inferência e FPS de processamento).

Modo apresentação (dia da demo): pede os palpites, toca o vídeo já processado e mostra o pódio:
    python contador_veiculos.py --apresentar resultados/resultado_YOLO_CNN_1125_....mp4
"""

import argparse
from collections import Counter
import csv
from datetime import datetime
import json
import os
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# === CONFIGURACOES PADRAO ===
PASTA_ATUAL = Path(__file__).resolve().parent
CAMINHO_MODELO_PADRAO = PASTA_ATUAL / "modelo_veiculos.keras"
PASTA_RESULTADOS = PASTA_ATUAL / "resultados"
ARQUIVO_ROIS = PASTA_ATUAL / "rois_salvos.json"

MODELO_YOLO = "yolov8s.pt"    # yolov8n.pt (leve), yolov8s.pt (equilibrado), yolov8m.pt (mais forte)
TAMANHO_IMG_YOLO = 960        # resolucao alta para pescar motos e carros menores
FRAMES_MINIMOS = 5            # minimo de frames que o ID precisa aparecer para contar (evita duplicatas rapidas)
CONF_YOLO = 0.25              # mantenha baixo (0.20-0.30) para o rastreador nao perder o ID do carro
CONF_CNN = 0.55               # confianca minima da CNN para confirmar o veiculo
MAX_TENTATIVAS_CNN = 5        # quantas vezes a CNN reavalia um veiculo antes de rejeitar de vez
INTERVALO_TENTATIVA_CNN = 10  # intervalo em frames entre reavaliacoes (evita rejeitar carro tampado)
JANELA_REASSOCIACAO = 20      # frames recentes para reconhecer carro que perdeu e ganhou novo ID
IOU_DUPLICATA = 0.35          # sobreposicao minima para considerar o mesmo veiculo (evita contar 2x)
INDICE_CLASSE_VEICULO = 1     # no dataset do Kaggle: 0=non-vehicles, 1=vehicles

NOMES_CLASSES = {
    2: "Carro",
    3: "Moto",
    5: "Onibus",
    7: "Caminhao",
}

ARQUIVO_EXPERIMENTOS = PASTA_RESULTADOS / "experimentos.csv"

# === CONDICOES DE TESTE (R6) E PRE-PROCESSAMENTO (R2) ===
# As condicoes simulam a camera em situacao adversa; o tamanho do frame nao muda,
# entao o ROI salvo e as anotacoes continuam valendo.
CONDICOES = ("original", "escuro", "baixa_res", "borrado")
PREPROCS = ("nenhum", "clahe")
FATOR_BAIXA_RES = 3       # baixa_res: reduz para 1/3 da largura/altura e volta ao tamanho original
KERNEL_BORRADO = 15       # borrado: blur de movimento horizontal de 15 px
GAMMA_ESCURO = 1.6        # escuro: curva gamma + ganho 0.6 (simula pouca luz)
GANHO_ESCURO = 0.6

_TABELA_ESCURO = np.clip(
    ((np.arange(256) / 255.0) ** GAMMA_ESCURO) * 255 * GANHO_ESCURO, 0, 255
).astype(np.uint8)
_KERNEL_MOV = np.zeros((KERNEL_BORRADO, KERNEL_BORRADO), np.float32)
_KERNEL_MOV[KERNEL_BORRADO // 2, :] = 1.0 / KERNEL_BORRADO
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def aplicar_condicao(frame: np.ndarray, condicao: str) -> np.ndarray:
    """Degrada o frame para simular iluminacao ruim, baixa resolucao ou movimento."""
    if condicao == "escuro":
        return cv2.LUT(frame, _TABELA_ESCURO)
    if condicao == "baixa_res":
        h, w = frame.shape[:2]
        pequeno = cv2.resize(frame, (w // FATOR_BAIXA_RES, h // FATOR_BAIXA_RES), interpolation=cv2.INTER_AREA)
        return cv2.resize(pequeno, (w, h), interpolation=cv2.INTER_LINEAR)
    if condicao == "borrado":
        return cv2.filter2D(frame, -1, _KERNEL_MOV)
    return frame


def aplicar_preproc(frame: np.ndarray, preproc: str) -> np.ndarray:
    """CLAHE no canal L (LAB): equaliza o contraste local antes do YOLO/CNN."""
    if preproc == "clahe":
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_canal, a_canal, b_canal = cv2.split(lab)
        l_canal = _CLAHE.apply(l_canal)
        return cv2.cvtColor(cv2.merge((l_canal, a_canal, b_canal)), cv2.COLOR_LAB2BGR)
    return frame


def descrever_ambiente() -> dict[str, str]:
    """Versoes e hardware usados (R8 / R7: registrar versoes e condicoes de medicao)."""
    info = {"python": platform.python_version(), "opencv": cv2.__version__,
            "cpu": platform.processor() or platform.machine(), "sistema": platform.platform()}
    try:
        import ultralytics
        info["ultralytics"] = ultralytics.__version__
    except Exception:
        info["ultralytics"] = "?"
    try:
        import torch
        info["torch"] = torch.__version__
        info["dispositivo"] = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        info["torch"] = "?"
        info["dispositivo"] = "?"
    return info


def registrar_experimento(linha: dict[str, object]) -> None:
    """Acrescenta uma linha em resultados/experimentos.csv (cria o cabecalho se precisar)."""
    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    novo = not ARQUIVO_EXPERIMENTOS.is_file()
    with ARQUIVO_EXPERIMENTOS.open("a", newline="", encoding="utf-8-sig") as f:
        escritor = csv.DictWriter(f, fieldnames=list(linha.keys()), delimiter=";")
        if novo:
            escritor.writeheader()
        escritor.writerow(linha)


def escolher_arquivo(titulo: str, tipos: list[tuple[str, str]]) -> str:
    """Abre o seletor de arquivos do Windows; se falhar, pede no terminal."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        caminho = filedialog.askopenfilename(title=titulo, filetypes=tipos)
        root.destroy()
        if caminho:
            return caminho
    except Exception:
        pass

    caminho = input(f"\n{titulo} (cole o caminho do arquivo): ").strip()
    return caminho.strip('"').strip("'")


def localizar_modelo_cnn() -> Path | None:
    """Procura o modelo .keras na pasta atual ou na pasta do PyCharm."""
    candidatos = [
        CAMINHO_MODELO_PADRAO,
        Path.cwd() / "modelo_veiculos.keras",
        Path.home() / "PycharmProjects" / "PythonProject2" / "modelo_veiculos.keras",
    ]
    for path in candidatos:
        if path.is_file():
            return path
    return None


# (slug_do_kaggle, filtrar_por_nome_da_pasta)
# filtrar=True -> pega só pastas de veículos de rua (ignora avião/barco/trem)
# filtrar=False -> dataset já é 100% carros/veículos, coleta tudo
DATASETS_EXTRAS_KAGGLE = [
    ("mmohaiminulislam/vehicles-image-dataset", True),
    ("utkarshsaxenadn/car-vs-bike-classification-dataset", True),
    ("mohamedmaher5/vehicle-classification", True),
    ("kshitij192/cars-image-dataset", False),
]
MAX_POR_DATASET_EXTRA = 5000  # limita por base extra pra não estourar o tempo na CPU

EXTENSOES_IMAGEM = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TERMOS_VEICULOS_RUA = (
    "car", "bike", "motor", "bus", "truck", "van", "suv", "pickup",
    "lorry", "taxi", "scooter", "auto", "sedan", "hatchback", "jeep",
    "vehicle", "ambulan", "limo", "two_wheeler",
)
TERMOS_IGNORAR = (
    "non-vehicle", "non_vehicle", "nonvehicle", "plane", "airplane",
    "aeroplane", "ship", "boat", "trains", "helicopter", "submarine",
    "sailboat", "yacht", "jet", "bicycle",
)


def _coletar_veiculos_extra(raiz: str, filtrar_pasta: bool = True, limite: int = MAX_POR_DATASET_EXTRA) -> list[str]:
    """Percorre um dataset extra e coleta imagens válidas de veículos terrestres."""
    from PIL import Image

    caminhos_validos: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(raiz):
        pasta_atual = Path(dirpath).name.lower()

        if filtrar_pasta:
            if any(ign in pasta_atual for ign in TERMOS_IGNORAR):
                continue
            if not any(termo in pasta_atual for termo in TERMOS_VEICULOS_RUA):
                continue

        for nome_arq in filenames:
            ext = Path(nome_arq).suffix.lower()
            if ext not in EXTENSOES_IMAGEM:
                continue
            caminho = os.path.join(dirpath, nome_arq)
            try:
                if os.path.getsize(caminho) < 500:
                    continue
                with Image.open(caminho) as img:
                    img.verify()
                caminhos_validos.append(caminho)
                if len(caminhos_validos) >= limite:
                    return caminhos_validos
            except Exception:
                continue
    return caminhos_validos


def treinar_cnn(destino: Path = CAMINHO_MODELO_PADRAO) -> Path:
    """Baixa a base principal + 4 bases extras do Kaggle (5 no total) e treina uma CNN reforçada em .keras."""
    import kagglehub
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    from tensorflow import keras

    keras.utils.set_random_seed(123)  # semente fixa para o treino ser reproduzivel (R8)
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adamax
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    filepaths: list[str] = []
    labels: list[str] = []

    print("\n[1/4] Carregando base principal (brsdincer/vehicle-detection-image-set)...")
    cache_principal = Path.home() / ".cache" / "kagglehub" / "datasets" / "brsdincer" / "vehicle-detection-image-set" / "versions" / "1"
    if cache_principal.is_dir():
        dataset_path = str(cache_principal)
    else:
        dataset_path = kagglehub.dataset_download("brsdincer/vehicle-detection-image-set")
    data_dir = os.path.join(dataset_path, "data")
    if not os.path.isdir(data_dir):
        data_dir = dataset_path

    for classe in os.listdir(data_dir):
        pasta_classe = os.path.join(data_dir, classe)
        if not os.path.isdir(pasta_classe):
            continue
        label_padrao = "non-vehicles" if "non" in classe.lower() else "vehicles"
        for arquivo in os.listdir(pasta_classe):
            if Path(arquivo).suffix.lower() in EXTENSOES_IMAGEM:
                filepaths.append(os.path.join(pasta_classe, arquivo))
                labels.append(label_padrao)

    print(f"  Base principal carregada: {len(filepaths)} imagens.")

    pasta_extras_local = PASTA_ATUAL / "datasets_veiculos" / "vehicles_extras"
    if pasta_extras_local.is_dir():
        imgs_locais = [
            str(p) for p in pasta_extras_local.iterdir()
            if p.is_file() and p.suffix.lower() in EXTENSOES_IMAGEM
        ]
        filepaths.extend(imgs_locais)
        labels.extend(["vehicles"] * len(imgs_locais))
        print(f"  Bases extras locais carregadas ({pasta_extras_local}): +{len(imgs_locais)} imagens!")
    else:
        print("\nBaixando bases extras de veículos variados (carros, motos, ônibus, caminhões)...")
        for idx, (slug, filtrar_pasta) in enumerate(DATASETS_EXTRAS_KAGGLE, start=1):
            try:
                print(f"  ({idx}/{len(DATASETS_EXTRAS_KAGGLE)}) Baixando {slug}...")
                extra_path = kagglehub.dataset_download(slug)
                imgs_extra = _coletar_veiculos_extra(extra_path, filtrar_pasta=filtrar_pasta)
                filepaths.extend(imgs_extra)
                labels.extend(["vehicles"] * len(imgs_extra))
                print(f"      -> +{len(imgs_extra)} imagens de veículos adicionadas!")
            except Exception as err:
                print(f"      -> Aviso: não consegui baixar {slug} ({err}). Seguindo com as demais.")

    df = pd.DataFrame({"filepaths": filepaths, "labels": labels})
    contagem_classes = df["labels"].value_counts().to_dict()
    print(f"\nTotal final do dataset combinado: {len(df)} imagens -> {contagem_classes}")

    train_df, dummy_df = train_test_split(
        df, train_size=0.8, shuffle=True, random_state=123, stratify=df["labels"]
    )
    valid_df, test_df = train_test_split(
        dummy_df, train_size=0.5, shuffle=True, random_state=123, stratify=dummy_df["labels"]
    )

    print("\n[2/4] Preparando geradores de imagem (64x64 + Data Augmentation)...")
    img_size = (64, 64)
    batch_size = 64

    tr_gen = ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=15,
        zoom_range=0.15,
        brightness_range=[0.75, 1.25],
    )
    ts_gen = ImageDataGenerator()

    train_gen = tr_gen.flow_from_dataframe(
        train_df, x_col="filepaths", y_col="labels", target_size=img_size,
        class_mode="categorical", color_mode="rgb", shuffle=True, batch_size=batch_size,
    )
    valid_gen = ts_gen.flow_from_dataframe(
        valid_df, x_col="filepaths", y_col="labels", target_size=img_size,
        class_mode="categorical", color_mode="rgb", shuffle=True, batch_size=batch_size,
    )
    test_gen = ts_gen.flow_from_dataframe(
        test_df, x_col="filepaths", y_col="labels", target_size=img_size,
        class_mode="categorical", color_mode="rgb", shuffle=False, batch_size=batch_size,
    )

    # Balanceamento automático para a CNN não ficar enviesada com mais veículos do que não-veículos
    y_train_idx = np.array(train_gen.classes)
    classes_unicas = np.unique(y_train_idx)
    pesos = compute_class_weight(class_weight="balanced", classes=classes_unicas, y=y_train_idx)
    class_weight = {int(c): float(p) for c, p in zip(classes_unicas, pesos)}
    print(f"  Índices das classes: {train_gen.class_indices} | Pesos balanceados: {class_weight}")

    print("\n[3/4] Treinando CNN reforçada (até 8 épocas com EarlyStopping)...")
    model = Sequential([
        keras.layers.Input(shape=(64, 64, 3)),
        keras.layers.Rescaling(1.0 / 255),
        Conv2D(32, (3, 3), padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.35),
        Dense(len(train_gen.class_indices), activation="softmax"),
    ])

    model.compile(
        Adamax(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
    ]
    model.fit(
        train_gen,
        epochs=8,
        verbose=1,
        validation_data=valid_gen,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    print("\n[4/4] Avaliando e salvando novo modelo .keras...")
    _, acc = model.evaluate(test_gen, verbose=1)
    model.save(destino)
    print(f"\nModelo reforçado treinado (acurácia de teste: {acc * 100:.2f}%) e salvo em:\n  {destino}")
    return destino


def cadastrar_palpites() -> dict[str, int]:
    """Permite cadastrar os palpites da sala antes do vídeo começar."""
    print("\n=== CADASTRO DE PALPITES DA SALA ===")
    print("Digite o nome de cada participante e o palpite de quantos veículos vão passar.")
    print("Deixe o nome vazio e aperte ENTER quando terminar.\n")

    palpites: dict[str, int] = {}
    while True:
        nome = input("Nome do participante (ENTER para iniciar): ").strip()
        if not nome:
            break
        while True:
            valor = input(f"  Palpite de {nome} (número de veículos): ").strip()
            if valor.isdigit():
                palpites[nome] = int(valor)
                break
            print("  Digite apenas um número inteiro (ex: 25).")
    return palpites


def carregar_roi_salvo(video_path: str, w: int, h: int) -> np.ndarray | None:
    """Carrega um ROI salvo anteriormente para o mesmo vídeo, se existir."""
    if not ARQUIVO_ROIS.is_file():
        return None
    try:
        dados = json.loads(ARQUIVO_ROIS.read_text(encoding="utf-8"))
        pontos = dados.get(Path(video_path).name)
        if not pontos or len(pontos) < 3:
            return None
        poly = np.array(pontos, dtype=np.int32)
        if np.all((poly[:, 0] >= 0) & (poly[:, 0] <= w) & (poly[:, 1] >= 0) & (poly[:, 1] <= h)):
            return poly
    except Exception:
        pass
    return None


def salvar_roi(video_path: str, pontos: list[list[int]]) -> None:
    """Salva as coordenadas do ROI por nome de vídeo para reutilizar depois."""
    dados: dict[str, list[list[int]]] = {}
    if ARQUIVO_ROIS.is_file():
        try:
            dados = json.loads(ARQUIVO_ROIS.read_text(encoding="utf-8"))
        except Exception:
            dados = {}
    dados[Path(video_path).name] = pontos
    ARQUIVO_ROIS.write_text(json.dumps(dados, indent=2, ensure_ascii=False), encoding="utf-8")


def selecionar_roi_interativo(primeiro_frame: np.ndarray, video_path: str) -> np.ndarray | None:
    """Deixa o usuário clicar os cantos da área de detecção (ROI) direto no 1º frame."""
    h, w = primeiro_frame.shape[:2]
    roi_salvo = carregar_roi_salvo(video_path, w, h)
    if roi_salvo is not None:
        resp = input(
            f"\nJá existe um ROI salvo para '{Path(video_path).name}' ({len(roi_salvo)} pontos). "
            "Usar o mesmo? [S/n]: "
        ).strip().lower()
        if resp in ("", "s", "sim", "y"):
            return roi_salvo

    print("\n=== DEFINIR ÁREA DE DETECÇÃO (ROI) ===")
    print("  Clique ESQUERDO : adicionar canto da pista (mínimo 3 pontos)")
    print("  Clique DIREITO  : desfazer último ponto")
    print("  [R]             : limpar tudo")
    print("  [ENTER/ESPAÇO]  : confirmar ROI")
    print("  [ESC]           : cancelar (usar tela inteira)\n")

    pontos: list[list[int]] = []
    janela_roi = "Definir ROI - Clique nos cantos da pista e aperte ENTER"
    cv2.namedWindow(janela_roi, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(janela_roi, min(w, 1280), min(h, 720))
    cv2.moveWindow(janela_roi, 60, 40)
    cv2.setWindowProperty(janela_roi, cv2.WND_PROP_TOPMOST, 1)
    cv2.imshow(janela_roi, primeiro_frame)
    cv2.waitKey(1)
    cv2.setWindowProperty(janela_roi, cv2.WND_PROP_TOPMOST, 0)

    def ao_clicar(evento: int, x: int, y: int, _flags: int, _param: object) -> None:
        if evento == cv2.EVENT_LBUTTONDOWN:
            pontos.append([int(x), int(y)])
        elif evento == cv2.EVENT_RBUTTONDOWN and pontos:
            pontos.pop()

    cv2.setMouseCallback(janela_roi, ao_clicar)

    try:
        while True:
            vis = primeiro_frame.copy()

            if len(pontos) >= 3:
                poly = np.array(pontos, dtype=np.int32)
                overlay = vis.copy()
                cv2.fillPoly(overlay, [poly], (0, 255, 255))
                cv2.addWeighted(overlay, 0.22, vis, 0.78, 0, vis)
                cv2.polylines(vis, [poly], isClosed=True, color=(0, 255, 255), thickness=3)
            elif len(pontos) == 2:
                cv2.line(vis, tuple(pontos[0]), tuple(pontos[1]), (0, 255, 255), 2)

            for idx, (px, py) in enumerate(pontos, start=1):
                cv2.circle(vis, (px, py), 6, (0, 0, 255), -1)
                cv2.putText(vis, str(idx), (px + 8, py - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            barra = (
                f"Pontos: {len(pontos)} | [Esq] Marcar  [Dir/Z] Desfazer  "
                "[R] Limpar  [ENTER] Confirmar  [ESC] Tela inteira"
            )
            cv2.rectangle(vis, (0, 0), (w, 38), (15, 15, 15), -1)
            cv2.putText(vis, barra, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow(janela_roi, vis)
            tecla = cv2.waitKey(30) & 0xFF

            if tecla in (13, 32):  # ENTER ou ESPACO
                if len(pontos) >= 3:
                    salvar_roi(video_path, pontos)
                    print(f"ROI salvo ({len(pontos)} pontos): {pontos}")
                    return np.array(pontos, dtype=np.int32)
                print("Marque pelo menos 3 pontos antes de confirmar!")
            elif tecla in (ord("z"), ord("Z")) and pontos:
                pontos.pop()
            elif tecla in (ord("r"), ord("R")):
                pontos.clear()
            elif tecla == 27 or cv2.getWindowProperty(janela_roi, cv2.WND_PROP_VISIBLE) < 1:
                print("ROI ignorado: detectando na tela inteira.")
                return None
    finally:
        try:
            cv2.destroyWindow(janela_roi)
            cv2.waitKey(1)
        except cv2.error:
            pass


def calcular_sobreposicao(
    caixa_a: tuple[int, int, int, int],
    caixa_b: tuple[int, int, int, int],
) -> tuple[float, float]:
    """Retorna (IoU, IoMin) entre duas caixas (x1, y1, x2, y2).
    IoMin detecta quando uma caixa menor (ex: cabine vista como Carro)
    está dentro de uma caixa maior (ex: Caminhão completo).
    """
    ix1 = max(caixa_a[0], caixa_b[0])
    iy1 = max(caixa_a[1], caixa_b[1])
    ix2 = min(caixa_a[2], caixa_b[2])
    iy2 = min(caixa_a[3], caixa_b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0, 0.0
    area_a = max(0, caixa_a[2] - caixa_a[0]) * max(0, caixa_a[3] - caixa_a[1])
    area_b = max(0, caixa_b[2] - caixa_b[0]) * max(0, caixa_b[3] - caixa_b[1])
    uniao = area_a + area_b - inter
    menor_area = min(area_a, area_b)
    iou = inter / uniao if uniao > 0 else 0.0
    io_min = inter / menor_area if menor_area > 0 else 0.0
    return iou, io_min


def eh_mesmo_veiculo(
    caixa_a: tuple[int, int, int, int],
    caixa_b: tuple[int, int, int, int],
    outro_sumiu: bool,
) -> bool:
    """Verifica se duas caixas pertencem ao mesmo veículo (inclui troca Carro -> Caminhão)."""
    iou, io_min = calcular_sobreposicao(caixa_a, caixa_b)
    if iou >= IOU_DUPLICATA:
        return True
    if outro_sumiu and io_min >= 0.55:
        return True
    return False


def desenhar_roi(frame: np.ndarray, roi_poly: np.ndarray, mascara_fora: np.ndarray) -> np.ndarray:
    """Escurece levemente o que está fora do ROI e destaca o contorno ativo."""
    escurecido = (frame * 0.45).astype(np.uint8)
    frame = np.where(mascara_fora[:, :, None] > 0, escurecido, frame)
    cv2.polylines(frame, [roi_poly], isClosed=True, color=(0, 255, 255), thickness=3)
    return frame


def desenhar_hud(
    frame: np.ndarray,
    contagem: dict[str, int],
    usa_cnn: bool,
    aprovados_cnn: int,
    rejeitados: int,
) -> None:
    """Desenha o placar ao vivo com fundo semitransparente."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    altura_hud = 228 if usa_cnn else 200
    cv2.rectangle(overlay, (10, 10), (315, altura_hud), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    total = sum(contagem.values())
    modo = "YOLO + CNN" if usa_cnn else "SOMENTE YOLO"
    linhas = [
        (f"TOTAL: {total}", (0, 255, 255), 0.8),
        (f"Carros:    {contagem['Carro']}", (255, 255, 255), 0.65),
        (f"Motos:     {contagem['Moto']}", (255, 255, 255), 0.65),
        (f"Onibus:    {contagem['Onibus']}", (255, 255, 255), 0.65),
        (f"Caminhoes: {contagem['Caminhao']}", (255, 255, 255), 0.65),
        (f"Modo: {modo}", (180, 220, 255), 0.55),
    ]
    if usa_cnn:
        linhas.append((f"CNN OK: {aprovados_cnn} | Rej: {rejeitados}", (120, 255, 160), 0.52))
    for i, (texto, cor, escala) in enumerate(linhas):
        cv2.putText(frame, texto, (22, 42 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, escala, cor, 2)

    # Rodape com comandos
    rodape = "[ESPACO] Pausar  |  [Q] Encerrar"
    cv2.rectangle(frame, (0, h - 32), (w, h), (15, 15, 15), -1)
    cv2.putText(frame, rodape, (15, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)


def gerar_tela_podio(
    w: int,
    h: int,
    contagem: dict[str, int],
    palpites: dict[str, int],
    modo_label: str = "YOLO",
) -> np.ndarray:
    """Cria a tela final com o total real e o ranking dos palpites."""
    tela = np.full((h, w, 3), 24, dtype=np.uint8)
    total_real = sum(contagem.values())

    cv2.putText(tela, f"RESULTADO FINAL - {modo_label}", (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    resumo = (
        f"TOTAL REAL: {total_real}  |  "
        f"Carros: {contagem['Carro']}  Motos: {contagem['Moto']}  "
        f"Onibus: {contagem['Onibus']}  Caminhoes: {contagem['Caminhao']}"
    )
    cv2.putText(tela, resumo, (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.line(tela, (40, 130), (w - 40, 130), (80, 80, 80), 2)

    if palpites:
        ranking = sorted(palpites.items(), key=lambda item: abs(item[1] - total_real))
        menor_erro = abs(ranking[0][1] - total_real)
        cv2.putText(tela, "RANKING DOS PALPITES:", (40, 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 120), 2)

        for pos, (pessoa, palpite) in enumerate(ranking[:8], start=1):
            erro = abs(palpite - total_real)
            vencedor = "  <-- VENCEDOR!" if erro == menor_erro else ""
            cor = (0, 255, 120) if erro == menor_erro else (230, 230, 230)
            linha = f"{pos}o - {pessoa}: palpite {palpite} (erro: {erro}){vencedor}"
            cv2.putText(tela, linha, (55, 220 + (pos - 1) * 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)
    else:
        cv2.putText(tela, "Nenhum palpite cadastrado nesta rodada.", (40, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

    cv2.putText(tela, "Pressione qualquer tecla para fechar.", (40, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)
    return tela


def executar_contagem(
    video_path: str,
    cnn_path: Path | None,
    palpites: dict[str, int],
    conf_yolo_cfg: float = CONF_YOLO,
    usar_roi: bool = True,
    condicao: str = "original",
    preproc: str = "nenhum",
    lote: bool = False,
    max_frames: int | None = None,
) -> dict[str, object]:
    """Executa o pipeline YOLOv8 (rastreamento) + CNN (validação) no vídeo.

    lote=True: sem janelas, sem palpites e usando o ROI salvo (para rodar experimentos em sequência).
    Retorna a linha registrada em resultados/experimentos.csv.
    """
    from ultralytics import YOLO

    print(f"\nCarregando YOLOv8 ({MODELO_YOLO}, imgsz={TAMANHO_IMG_YOLO}, conf={conf_yolo_cfg})...")
    yolo_model = YOLO(MODELO_YOLO)

    cnn_model = None
    alt_cnn = larg_cnn = 64
    if cnn_path is not None and cnn_path.is_file():
        from tensorflow import keras

        print(f"Carregando CNN validadora: {cnn_path}")
        cnn_model = keras.models.load_model(cnn_path)
        alt_cnn, larg_cnn = int(cnn_model.input_shape[1]), int(cnn_model.input_shape[2])
        print(f"CNN carregada! Entrada esperada: {larg_cnn}x{alt_cnn}")
    else:
        print("Rodando sem CNN (somente YOLOv8).")

    # R1 - validacao da entrada
    if not Path(video_path).is_file():
        raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Não consegui abrir o vídeo (formato/codec?): {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    # Lê o 1º frame para o usuário delimitar o ROI real (onde o YOLO vai detectar)
    ret_init, primeiro_frame = cap.read()
    if not ret_init:
        cap.release()
        raise RuntimeError("Não consegui ler nenhum frame do vídeo.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    duracao_s = total_frames / fps if fps else 0
    print(f"Vídeo: {Path(video_path).name} | {w}x{h} | {fps:.2f} FPS | {total_frames} frames (~{duracao_s:.1f}s)")
    print(f"Configuração: condição={condicao} | pré-processamento={preproc} | ROI={'sim' if usar_roi else 'não'}")

    roi_poly = None
    if usar_roi:
        if lote:
            roi_poly = carregar_roi_salvo(video_path, w, h)
            if roi_poly is None:
                cap.release()
                raise RuntimeError(
                    f"Nenhum ROI salvo para '{Path(video_path).name}'. Rode uma vez no modo normal "
                    "(sem --lote) para marcar o ROI, ou use --sem-roi."
                )
            print(f"Usando ROI salvo ({len(roi_poly)} pontos).")
        else:
            roi_poly = selecionar_roi_interativo(primeiro_frame, video_path)
    mascara_roi = None
    mascara_fora = None
    if roi_poly is not None:
        mascara_roi = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mascara_roi, [roi_poly], 255)
        mascara_fora = cv2.bitwise_not(mascara_roi)

    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    agora = datetime.now()
    carimbo = agora.strftime("%Y-%m-%d_%H-%M-%S")
    nome_video_limpo = "".join(c if c.isalnum() or c in "-_" else "_" for c in Path(video_path).stem)
    tag_metodo = "YOLO_CNN" if cnn_model is not None else "SO_YOLO"
    tag_config = f"conf{conf_yolo_cfg:.2f}_{'roi' if roi_poly is not None else 'semroi'}_{condicao}_{preproc}"
    caminho_saida = PASTA_RESULTADOS / f"resultado_{tag_metodo}_{nome_video_limpo}_{tag_config}_{carimbo}.mp4"

    out = cv2.VideoWriter(str(caminho_saida), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not out.isOpened():
        caminho_saida = caminho_saida.with_suffix(".avi")
        out = cv2.VideoWriter(str(caminho_saida), cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))
        if not out.isOpened():
            raise RuntimeError("Não consegui criar o arquivo de vídeo de saída.")

    contagem = {"Carro": 0, "Moto": 0, "Onibus": 0, "Caminhao": 0}
    validados_cnn: dict[int, tuple[bool, float, int, str]] = {}       # tid -> (ok, prob, tentativas, origem)
    ultimo_frame_cnn: dict[int, int] = {}                             # tid -> ultimo frame em que a CNN avaliou
    aparicoes_id: dict[int, list[str]] = {}                           # tid -> classes detectadas por frame
    classe_contada: dict[int, str] = {}                               # tid -> classe final contabilizada
    ultimas_caixas_contadas: dict[int, tuple[int, int, int, int, int]] = {}  # tid -> (x1, y1, x2, y2, frame_idx)
    contados: set[int] = set()
    pausado = False
    frames_gravados = 0
    frames_video = 0
    aprovados_cnn = 0
    rejeitados = 0

    # Medicoes de desempenho (R7). O 1o frame e descartado (aquecimento do modelo).
    tempos_inferencia_ms: list[float] = []   # so a rede YOLO (res.speed["inference"])
    tempos_track_ms: list[float] = []        # chamada completa do track (pre + rede + pos + ByteTrack)
    tempos_frame_ms: list[float] = []        # frame inteiro (condição, máscara, YOLO, CNN, contagem, desenho, gravação)
    tempos_cnn_ms: list[float] = []          # cada chamada da CNN
    inicio_total = time.perf_counter()

    janela = "Contador de Veiculos (YOLO + CNN)"
    if not lote:
        cv2.namedWindow(janela, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(janela, min(w, 1280), min(h, 720))
        cv2.moveWindow(janela, 60, 40)
        cv2.setWindowProperty(janela, cv2.WND_PROP_TOPMOST, 1)
        cv2.imshow(janela, primeiro_frame)
        cv2.waitKey(1)
        cv2.setWindowProperty(janela, cv2.WND_PROP_TOPMOST, 0)

    print("\nIniciando vídeo... Contando automaticamente por ID único do YOLO dentro do ROI.", flush=True)
    if not lote:
        print("Controles: [ESPAÇO] para pausar | [Q] para sair.", flush=True)

    try:
        while True:
            if not pausado:
                if max_frames is not None and frames_video >= max_frames:
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                t_frame = time.perf_counter()
                frames_video += 1

                # 0. Condição de teste (R6) e pré-processamento (R2) — o pipeline inteiro enxerga este frame
                frame = aplicar_condicao(frame, condicao)
                frame = aplicar_preproc(frame, preproc)

                # 1. Aplica a máscara real de ROI: o YOLO só enxerga o que está dentro da área marcada
                frame_para_yolo = (
                    cv2.bitwise_and(frame, frame, mask=mascara_roi)
                    if mascara_roi is not None
                    else frame
                )
                t_track = time.perf_counter()
                res = yolo_model.track(
                    frame_para_yolo, persist=True, classes=[2, 3, 5, 7],
                    conf=conf_yolo_cfg, imgsz=TAMANHO_IMG_YOLO,
                    agnostic_nms=True, tracker="bytetrack.yaml", verbose=False,
                )[0]
                if frames_video > 1:
                    tempos_track_ms.append((time.perf_counter() - t_track) * 1000)
                    tempos_inferencia_ms.append(float(res.speed.get("inference", 0.0)))

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if (cnn_model is not None and res.boxes.id is not None) else None
                if roi_poly is not None and mascara_fora is not None:
                    frame = desenhar_roi(frame, roi_poly, mascara_fora)

                if res.boxes.id is not None:
                    caixas = res.boxes.xyxy.cpu().numpy().astype(int)
                    ids = res.boxes.id.cpu().numpy().astype(int)
                    clss = res.boxes.cls.cpu().numpy().astype(int)
                    confs = res.boxes.conf.cpu().numpy().astype(float)
                    ids_ativos_frame = set(int(i) for i in ids)

                    for (x1, y1, x2, y2), tid, cid, conf_yolo in zip(caixas, ids, clss, confs):
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                        # Garante que o centro do veículo está dentro do ROI
                        if roi_poly is not None and cv2.pointPolygonTest(roi_poly, (float(cx), float(cy)), False) < 0:
                            continue

                        nome = NOMES_CLASSES.get(cid, "Veiculo")
                        caixa_atual = (x1, y1, x2, y2)

                        # Se este ID acabou de nascer no lugar de um veículo já contado
                        # (ex: Carro que virou Caminhão ou trocou de ID), herda a contagem sem somar 2x
                        if tid not in aparicoes_id and tid not in contados:
                            for outro_id, (ox1, oy1, ox2, oy2, f_idx) in list(ultimas_caixas_contadas.items()):
                                if frames_gravados - f_idx <= JANELA_REASSOCIACAO:
                                    outro_sumiu = outro_id not in ids_ativos_frame
                                    if eh_mesmo_veiculo(caixa_atual, (ox1, oy1, ox2, oy2), outro_sumiu):
                                        contados.add(tid)
                                        classe_antiga = classe_contada.get(outro_id, nome)
                                        # Se era "Carro" de longe e virou "Caminhao"/"Onibus" de perto, corrige a categoria
                                        if classe_antiga == "Carro" and nome in ("Caminhao", "Onibus") and contagem["Carro"] > 0:
                                            contagem["Carro"] -= 1
                                            contagem[nome] += 1
                                            classe_contada[tid] = nome
                                        else:
                                            classe_contada[tid] = classe_antiga
                                        if outro_id in validados_cnn:
                                            validados_cnn[tid] = validados_cnn[outro_id]
                                        ultimas_caixas_contadas.pop(outro_id, None)
                                        break

                        aparicoes_id.setdefault(tid, []).append(nome)

                        # Se um veículo já contado como "Carro" passar a ser visto como "Caminhao"/"Onibus"
                        # de forma consistente (>= 5 frames), corrige a categoria sem alterar o TOTAL
                        if tid in contados and classe_contada.get(tid) == "Carro":
                            votos_grandes = Counter(
                                c for c in aparicoes_id[tid] if c in ("Caminhao", "Onibus")
                            )
                            if votos_grandes:
                                nova_classe, qtd_votos = votos_grandes.most_common(1)[0]
                                if qtd_votos >= FRAMES_MINIMOS and contagem["Carro"] > 0:
                                    contagem["Carro"] -= 1
                                    contagem[nova_classe] += 1
                                    classe_contada[tid] = nova_classe

                        # 2. Na Opção 1 (YOLO + CNN), 100% dos veículos passam pela CNN
                        #    Reavalia de forma espaçada (a cada INTERVALO_TENTATIVA_CNN frames)
                        #    para dar chance de confirmar caso o carro tenha surgido tampado/longe
                        if cnn_model is not None:
                            confirmado, prob_val, tent, origem = validados_cnn.get(tid, (False, 0.0, 0, "CNN"))
                            pode_avaliar = (
                                tid not in ultimo_frame_cnn
                                or (frames_gravados - ultimo_frame_cnn[tid]) >= INTERVALO_TENTATIVA_CNN
                            )
                            if not confirmado and tent < MAX_TENTATIVAS_CNN and pode_avaliar:
                                recorte = frame_rgb[y1:y2, x1:x2]
                                if recorte.size > 0:
                                    t_cnn = time.perf_counter()
                                    recorte = cv2.resize(recorte, (larg_cnn, alt_cnn))
                                    batch = np.expand_dims(recorte, axis=0).astype(np.float32)
                                    pred = cnn_model(batch, training=False).numpy()
                                    tempos_cnn_ms.append((time.perf_counter() - t_cnn) * 1000)
                                    prob_val = float(pred[0][INDICE_CLASSE_VEICULO])
                                    confirmado = prob_val >= CONF_CNN
                                    origem = "CNN"
                                    tent += 1
                                    validados_cnn[tid] = (confirmado, prob_val, tent, origem)
                                    ultimo_frame_cnn[tid] = frames_gravados
                        else:
                            confirmado, prob_val, tent, origem = True, conf_yolo, 0, "YOLO"

                        # 3. Contagem direta por ID único com proteção anti-duplicata
                        #    (evita contar de novo se o mesmo carro perder/trocar de ID ou virar caminhão)
                        if tid not in contados and confirmado and len(aparicoes_id[tid]) >= FRAMES_MINIMOS:
                            classe_final = Counter(aparicoes_id[tid]).most_common(1)[0][0]
                            eh_duplicado = False
                            for outro_id, (ox1, oy1, ox2, oy2, f_idx) in list(ultimas_caixas_contadas.items()):
                                if frames_gravados - f_idx <= JANELA_REASSOCIACAO:
                                    outro_sumiu = outro_id not in ids_ativos_frame
                                    if eh_mesmo_veiculo(caixa_atual, (ox1, oy1, ox2, oy2), outro_sumiu):
                                        eh_duplicado = True
                                        classe_antiga = classe_contada.get(outro_id, classe_final)
                                        if classe_antiga == "Carro" and classe_final in ("Caminhao", "Onibus") and contagem["Carro"] > 0:
                                            contagem["Carro"] -= 1
                                            contagem[classe_final] += 1
                                            classe_contada[tid] = classe_final
                                        else:
                                            classe_contada[tid] = classe_antiga
                                        ultimas_caixas_contadas.pop(outro_id, None)
                                        break
                            if not eh_duplicado:
                                contagem[classe_final] += 1
                                classe_contada[tid] = classe_final
                            contados.add(tid)

                        if tid in contados:
                            ultimas_caixas_contadas[tid] = (x1, y1, x2, y2, frames_gravados)

                        nome_exibir = classe_contada.get(tid, nome)
                        if cnn_model is not None:
                            pcts = f"Y:{conf_yolo * 100:.0f}% | CNN:{prob_val * 100:.0f}%"
                        else:
                            pcts = f"YOLO:{conf_yolo * 100:.0f}%"

                        if tid in contados:
                            cor = (0, 255, 0)  # verde: já contado
                            tag = f"#{tid} {nome_exibir} ({pcts})"
                        elif cnn_model is not None and not confirmado and tent >= MAX_TENTATIVAS_CNN:
                            cor = (0, 0, 255)  # vermelho: rejeitado pela CNN
                            tag = f"#{tid} Rej ({pcts})"
                        else:
                            cor = (0, 255, 255)  # amarelo: confirmando
                            tag = f"#{tid} {nome_exibir}... ({pcts})"

                        cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
                        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
                        ty = max(y1 - 6, th + 6)
                        cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 6, ty + 3), (15, 15, 15), -1)
                        cv2.putText(frame, tag, (x1 + 3, ty),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, cor, 2)

                aprovados_cnn = sum(1 for ok, _, _, orig in validados_cnn.values() if ok and orig == "CNN")
                rejeitados = sum(1 for ok, _, tent, _ in validados_cnn.values() if not ok and tent >= MAX_TENTATIVAS_CNN)
                desenhar_hud(frame, contagem, cnn_model is not None, aprovados_cnn, rejeitados)
                out.write(frame)
                frames_gravados += 1
                if frames_video > 1:
                    tempos_frame_ms.append((time.perf_counter() - t_frame) * 1000)
                if not lote:
                    cv2.imshow(janela, frame)

                if frames_gravados == 1 or frames_gravados % 30 == 0:
                    progresso = f"{frames_gravados}/{total_frames}" if total_frames else str(frames_gravados)
                    fps_proc = 1000 / np.mean(tempos_frame_ms) if tempos_frame_ms else 0.0
                    print(f"  Frame {progresso} | TOTAL: {sum(contagem.values())} | CNN OK: {aprovados_cnn} | "
                          f"Rej: {rejeitados} | {fps_proc:.1f} FPS", flush=True)

            if lote:
                continue
            tecla = cv2.waitKey(1 if not pausado else 30) & 0xFF
            if tecla in (ord("q"), ord("Q"), 27):  # Q ou ESC
                break
            elif tecla == ord(" "):
                pausado = not pausado

            # Se o usuário fechou a janela no X, encerra limpo
            if cv2.getWindowProperty(janela, cv2.WND_PROP_VISIBLE) < 1:
                break

        tempo_total_s = time.perf_counter() - inicio_total

        # Grava 4 segundos do pódio e FINALIZA o arquivo ANTES de esperar tecla
        modo_podio = "YOLO + CNN" if cnn_model is not None else "SOMENTE YOLO"
        tela_podio = gerar_tela_podio(w, h, contagem, palpites, modo_podio)
        for _ in range(int(fps * 4)):
            out.write(tela_podio)
        out.release()

        # Mostra o pódio até apertar uma tecla ou fechar no X
        if not lote and cv2.getWindowProperty(janela, cv2.WND_PROP_VISIBLE) >= 1:
            cv2.imshow(janela, tela_podio)
            while cv2.getWindowProperty(janela, cv2.WND_PROP_VISIBLE) >= 1:
                if cv2.waitKey(50) != -1:
                    break
    finally:
        cap.release()
        out.release()
        if not lote:
            cv2.destroyAllWindows()

    # Imprime resumo no terminal
    total_real = sum(contagem.values())
    media = lambda valores: float(np.mean(valores)) if valores else 0.0  # noqa: E731
    inferencia_ms = media(tempos_inferencia_ms)
    track_ms = media(tempos_track_ms)
    frame_ms = media(tempos_frame_ms)
    cnn_ms = media(tempos_cnn_ms)
    fps_proc = 1000 / frame_ms if frame_ms else 0.0
    ambiente = descrever_ambiente()

    print("\n" + "=" * 45)
    print("         RESULTADO FINAL DA CONTAGEM")
    print("=" * 45)
    for tipo, qtd in contagem.items():
        print(f"  {tipo:<12}: {qtd}")
    print("-" * 45)
    print(f"  TOTAL REAL  : {total_real}")
    print("=" * 45)
    print(f"  Inferência YOLO (rede)      : {inferencia_ms:.1f} ms/frame")
    print(f"  YOLO + ByteTrack (chamada)  : {track_ms:.1f} ms/frame")
    print(f"  Frame completo (pipeline)   : {frame_ms:.1f} ms/frame -> {fps_proc:.2f} FPS "
          f"(vídeo original: {fps:.2f} FPS)")
    if tempos_cnn_ms:
        print(f"  CNN por recorte             : {cnn_ms:.1f} ms ({len(tempos_cnn_ms)} chamadas)")
    print(f"  Hardware: {ambiente['cpu']} | dispositivo: {ambiente['dispositivo']} | "
          f"ultralytics {ambiente['ultralytics']} | torch {ambiente['torch']}")

    if palpites:
        print("\n=== RANKING DOS PALPITES ===")
        ranking = sorted(palpites.items(), key=lambda item: abs(item[1] - total_real))
        menor_erro = abs(ranking[0][1] - total_real)
        for pos, (pessoa, palpite) in enumerate(ranking, start=1):
            erro = abs(palpite - total_real)
            trofeu = "  <-- VENCEDOR!" if erro == menor_erro else ""
            print(f"  {pos}º lugar: {pessoa:<15} | Palpite: {palpite:<4} | Diferença: {erro}{trofeu}")

    tamanho_mb = caminho_saida.stat().st_size / (1024 * 1024) if caminho_saida.is_file() else 0
    print(f"\nVídeo salvo com {frames_gravados} frames do vídeo + 4s de pódio ({tamanho_mb:.2f} MB) em:\n  {caminho_saida}")

    # Registra também um resumo acumulado em texto para facilitar análises depois
    arquivo_historico = PASTA_RESULTADOS / "historico_contagens.txt"
    modo_usado = "YOLO + CNN" if cnn_model is not None else "YOLO"
    with arquivo_historico.open("a", encoding="utf-8") as f:
        f.write(f"[{agora.strftime('%d/%m/%Y %H:%M:%S')}] Vídeo: {Path(video_path).name} | Modo: {modo_usado}\n")
        f.write(f"  Entrada: {w}x{h} @ {fps:.2f} FPS | frames processados: {frames_video}/{total_frames}\n")
        f.write(f"  Config: {MODELO_YOLO} imgsz={TAMANHO_IMG_YOLO} conf={conf_yolo_cfg} | "
                f"ROI={'sim' if roi_poly is not None else 'não'} | condição={condicao} | pré-proc={preproc}\n")
        f.write(f"  Arquivo salvo: {caminho_saida.name}\n")
        f.write(
            f"  TOTAL: {total_real} | Carros: {contagem['Carro']} | Motos: {contagem['Moto']} | "
            f"Ônibus: {contagem['Onibus']} | Caminhões: {contagem['Caminhao']}\n"
        )
        f.write(f"  Desempenho: inferência {inferencia_ms:.1f} ms | frame {frame_ms:.1f} ms | {fps_proc:.2f} FPS | "
                f"{ambiente['cpu']} ({ambiente['dispositivo']})\n")
        if palpites:
            ranking = sorted(palpites.items(), key=lambda item: abs(item[1] - total_real))
            resumo_palpites = ", ".join(
                f"{p} (palpite {v}, erro {abs(v - total_real)})" for p, v in ranking
            )
            f.write(f"  Palpites: {resumo_palpites}\n")
        f.write("-" * 60 + "\n")
    print(f"Histórico atualizado em:\n  {arquivo_historico}")

    linha = {
        "data_hora": agora.strftime("%d/%m/%Y %H:%M:%S"),
        "video": Path(video_path).name,
        "resolucao": f"{w}x{h}",
        "fps_origem": round(fps, 2),
        "frames_processados": frames_video,
        "modo": modo_usado,
        "modelo_yolo": MODELO_YOLO,
        "imgsz": TAMANHO_IMG_YOLO,
        "conf_yolo": conf_yolo_cfg,
        "roi": "sim" if roi_poly is not None else "nao",
        "condicao": condicao,
        "preproc": preproc,
        "total": total_real,
        "carros": contagem["Carro"],
        "motos": contagem["Moto"],
        "onibus": contagem["Onibus"],
        "caminhoes": contagem["Caminhao"],
        "cnn_aprovados": aprovados_cnn,
        "cnn_rejeitados": rejeitados,
        "inferencia_ms": round(inferencia_ms, 1),
        "yolo_track_ms": round(track_ms, 1),
        "cnn_ms": round(cnn_ms, 1),
        "frame_ms": round(frame_ms, 1),
        "fps_processamento": round(fps_proc, 2),
        "tempo_total_s": round(tempo_total_s, 1),
        "cpu": ambiente["cpu"],
        "dispositivo": ambiente["dispositivo"],
        "ultralytics": ambiente["ultralytics"],
        "arquivo_saida": caminho_saida.name,
    }
    registrar_experimento(linha)
    print(f"Linha registrada em:\n  {ARQUIVO_EXPERIMENTOS}")
    return linha


def buscar_resultado_registrado(arquivo_video: Path) -> dict[str, str] | None:
    """Procura no experimentos.csv a contagem registrada para um vídeo de resultado."""
    if not ARQUIVO_EXPERIMENTOS.is_file():
        return None
    with ARQUIVO_EXPERIMENTOS.open(encoding="utf-8-sig") as f:
        for linha in csv.DictReader(f, delimiter=";"):
            if linha.get("arquivo_saida") == arquivo_video.name:
                return linha
    return None


def apresentar_resultado(arquivo_resultado: str, total_manual: int | None = None) -> None:
    """Modo apresentação: recebe os palpites, toca o vídeo JÁ PROCESSADO (saída real do sistema,
    sem edição) e mostra o pódio. Evita esperar o processamento ao vivo na CPU."""
    caminho = Path(arquivo_resultado)
    if not caminho.is_file():
        caminho = PASTA_RESULTADOS / arquivo_resultado
    if not caminho.is_file():
        raise FileNotFoundError(f"Vídeo de resultado não encontrado: {arquivo_resultado}")

    registro = buscar_resultado_registrado(caminho)
    if registro is not None:
        contagem = {"Carro": int(registro["carros"]), "Moto": int(registro["motos"]),
                    "Onibus": int(registro["onibus"]), "Caminhao": int(registro["caminhoes"])}
        modo_label = "YOLO + CNN" if registro["modo"] == "YOLO + CNN" else "SOMENTE YOLO"
    elif total_manual is not None:
        contagem = {"Carro": total_manual, "Moto": 0, "Onibus": 0, "Caminhao": 0}
        modo_label = "YOLO + CNN" if "YOLO_CNN" in caminho.name else "SOMENTE YOLO"
        print("Aviso: resultado não está no experimentos.csv; usando --total (sem separar por classe).")
    else:
        raise RuntimeError("Esse vídeo não está no experimentos.csv. Informe o total com --total N.")

    palpites = cadastrar_palpites()

    cap = cv2.VideoCapture(str(caminho))
    if not cap.isOpened():
        raise FileNotFoundError(f"Não consegui abrir: {caminho}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frames_video = max(0, total_frames - int(fps * 4))  # os últimos 4s são o pódio gravado sem palpites
    atraso_ms = max(1, int(1000 / fps))

    janela = "Contador de Veiculos (YOLO + CNN)"
    cv2.namedWindow(janela, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(janela, min(w, 1280), min(h, 720))
    cv2.moveWindow(janela, 60, 40)
    print("\nTocando o resultado processado... [ESPAÇO] pausa | [Q] pula para o pódio", flush=True)

    pausado = False
    lidos = 0
    try:
        while lidos < frames_video or total_frames == 0:
            if not pausado:
                inicio = time.perf_counter()
                ret, frame = cap.read()
                if not ret:
                    break
                lidos += 1
                cv2.imshow(janela, frame)
                espera = max(1, atraso_ms - int((time.perf_counter() - inicio) * 1000))
            else:
                espera = 30
            tecla = cv2.waitKey(espera) & 0xFF
            if tecla in (ord("q"), ord("Q"), 27):
                break
            if tecla == ord(" "):
                pausado = not pausado
            if cv2.getWindowProperty(janela, cv2.WND_PROP_VISIBLE) < 1:
                break

        tela_podio = gerar_tela_podio(w, h, contagem, palpites, modo_label)
        if cv2.getWindowProperty(janela, cv2.WND_PROP_VISIBLE) < 1:
            cv2.namedWindow(janela, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(janela, min(w, 1280), min(h, 720))
        cv2.imshow(janela, tela_podio)
        while cv2.getWindowProperty(janela, cv2.WND_PROP_VISIBLE) >= 1:
            if cv2.waitKey(50) != -1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    total_real = sum(contagem.values())
    print(f"\nTOTAL DO SISTEMA: {total_real}")
    if palpites:
        ranking = sorted(palpites.items(), key=lambda item: abs(item[1] - total_real))
        menor_erro = abs(ranking[0][1] - total_real)
        for pos, (pessoa, palpite) in enumerate(ranking, start=1):
            erro = abs(palpite - total_real)
            trofeu = "  <-- VENCEDOR!" if erro == menor_erro else ""
            print(f"  {pos}º lugar: {pessoa:<15} | Palpite: {palpite:<4} | Diferença: {erro}{trofeu}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Contador de veículos (YOLOv8 + CNN) com jogo de palpites")
    parser.add_argument("--apresentar", metavar="VIDEO_RESULTADO",
                        help="modo apresentação: palpites + toca o vídeo já processado + pódio")
    parser.add_argument("--total", type=int, help="total contado (só se o vídeo não estiver no experimentos.csv)")
    parser.add_argument("--video", help="caminho do vídeo (pula o seletor de arquivos)")
    parser.add_argument("--modo", choices=["cnn", "yolo"], help="cnn = YOLO + CNN | yolo = somente YOLO")
    parser.add_argument("--conf", type=float, default=CONF_YOLO, help=f"confiança mínima do YOLO (padrão {CONF_YOLO})")
    parser.add_argument("--sem-roi", action="store_true", help="detecta na tela inteira (sem ROI)")
    parser.add_argument("--condicao", choices=CONDICOES, default="original", help="condição simulada (R6)")
    parser.add_argument("--preproc", choices=PREPROCS, default="nenhum", help="pré-processamento antes do YOLO")
    parser.add_argument("--lote", action="store_true", help="sem janelas e sem palpites; usa o ROI salvo")
    parser.add_argument("--max-frames", type=int, help="processa só os N primeiros frames (teste rápido)")
    args = parser.parse_args()

    if args.apresentar:
        apresentar_resultado(args.apresentar, args.total)
        return

    opcoes = dict(conf_yolo_cfg=args.conf, usar_roi=not args.sem_roi, condicao=args.condicao,
                  preproc=args.preproc, lote=args.lote, max_frames=args.max_frames)

    # === Modo por linha de comando (experimentos) ===
    if args.video or args.lote:
        if not args.video:
            parser.error("--lote precisa de --video")
        modo = args.modo or "cnn"
        cnn_path = None
        if modo == "cnn":
            cnn_path = localizar_modelo_cnn()
            if cnn_path is None:
                print("ERRO: modo cnn pedido mas 'modelo_veiculos.keras' não foi encontrado. Use --modo yolo.")
                sys.exit(1)
        palpites = {} if args.lote else cadastrar_palpites()
        executar_contagem(args.video, cnn_path, palpites, **opcoes)
        return

    # === Modo interativo (demo com a turma) ===
    print("=" * 50)
    print("  CONTADOR DE VEÍCULOS + JOGO (YOLOv8 + CNN)")
    print("=" * 50)

    cnn_encontrada = localizar_modelo_cnn()
    if cnn_encontrada:
        print(f"\nModelo CNN encontrado automaticamente:\n  {cnn_encontrada}")
    else:
        print("\nNenhum arquivo 'modelo_veiculos.keras' encontrado ainda.")

    print("\nEscolha o que deseja fazer:")
    print("  1 - Rodar contagem + jogo em um vídeo (YOLO + CNN)")
    print("  2 - Treinar CNN reforçada (base principal + 4 bases extras do Kaggle)")
    print("  3 - Rodar contagem + jogo usando somente YOLO")
    opcao = input("\nOpção [1/2/3]: ").strip() or "1"

    cnn_path: Path | None = cnn_encontrada

    if opcao == "2":
        cnn_path = treinar_cnn()
        continuar = input("\nQuer rodar um vídeo agora? [s/n]: ").strip().lower()
        if continuar not in ("s", "sim", "y", ""):
            return
    elif opcao == "1":
        if cnn_path is None:
            print("\nNão achei o 'modelo_veiculos.keras'.")
            print("  1 - Escolher o arquivo .keras no computador")
            print("  2 - Treinar agora (~1-2 min)")
            print("  3 - Seguir só com YOLO")
            sub = input("Escolha [1/2/3]: ").strip()
            if sub == "1":
                escolhido = escolher_arquivo("Selecione o modelo_veiculos.keras", [("Keras Model", "*.keras *.h5")])
                cnn_path = Path(escolhido) if escolhido else None
            elif sub == "2":
                cnn_path = treinar_cnn()
            else:
                cnn_path = None
    elif opcao == "3":
        cnn_path = None
    else:
        print("Opção inválida.")
        sys.exit(1)

    video = escolher_arquivo("Selecione o vídeo da rua", [("Vídeos", "*.mp4 *.avi *.mov *.mkv"), ("Todos", "*.*")])
    if not video:
        print("Nenhum vídeo selecionado. Encerrando.")
        return

    palpites = cadastrar_palpites()
    executar_contagem(video, cnn_path, palpites, **opcoes)


if __name__ == "__main__":
    main()
