"""
Avaliacao quantitativa da deteccao (requisito R7 do Capstone)
--------------------------------------------------------------
Fica na mesma pasta do contador_veiculos.py e reaproveita as configuracoes dele.

1) Separar 30 frames espaçados do video (evita frames quase iguais):
       python avaliar_deteccao.py extrair --video 1125.mp4

2) Anotar a mao os veiculos de cada frame (ground truth). Da pra dividir entre o grupo:
       python avaliar_deteccao.py anotar --inicio 0  --fim 10
       python avaliar_deteccao.py anotar --inicio 10 --fim 20
       python avaliar_deteccao.py anotar --inicio 20 --fim 30
   (se cada um anotar num PC, depois junte os .txt de avaliacao/rotulos/ numa pasta so)

3) Calcular precisao, recall, F1, IoU, tempo de inferencia e FPS:
       python avaliar_deteccao.py avaliar                      # configuracao final
       python avaliar_deteccao.py avaliar --conf 0.5 --cnn     # outra configuracao
       python avaliar_deteccao.py lote                         # todas as configuracoes de uma vez

Saidas: avaliacao/metricas.csv, avaliacao/metricas.md e avaliacao/visual/<config>/
(imagens com acertos em azul, falsos positivos em vermelho e falsos negativos em amarelo).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from contador_veiculos import (
    ARQUIVO_ROIS, CONDICOES, CONF_CNN, CONF_YOLO, INDICE_CLASSE_VEICULO, MODELO_YOLO,
    NOMES_CLASSES, PASTA_ATUAL, PREPROCS, TAMANHO_IMG_YOLO,
    aplicar_condicao, aplicar_preproc, calcular_sobreposicao, descrever_ambiente, localizar_modelo_cnn,
)

PASTA_AVAL = PASTA_ATUAL / "avaliacao"
PASTA_FRAMES = PASTA_AVAL / "frames"
PASTA_ROTULOS = PASTA_AVAL / "rotulos"
PASTA_VISUAL = PASTA_AVAL / "visual"
ARQUIVO_INFO = PASTA_AVAL / "info.json"
ARQUIVO_METRICAS = PASTA_AVAL / "metricas.csv"
ARQUIVO_METRICAS_MD = PASTA_AVAL / "metricas.md"
IOU_LIMIAR = 0.50

CLASSES_ANOTACAO = [2, 3, 5, 7]                 # ids COCO: Carro, Moto, Onibus, Caminhao
CORES_CLASSE = {2: (0, 200, 0), 3: (255, 160, 0), 5: (200, 0, 200), 7: (0, 140, 255)}


# ----------------------------------------------------------------------------- utilidades
def carregar_info() -> dict:
    if not ARQUIVO_INFO.is_file():
        raise SystemExit("Rode primeiro: python avaliar_deteccao.py extrair --video SEU_VIDEO.mp4")
    return json.loads(ARQUIVO_INFO.read_text(encoding="utf-8"))


def roi_do_info(info: dict) -> np.ndarray | None:
    return np.array(info["roi"], dtype=np.int32) if info.get("roi") else None


def centro_dentro(roi: np.ndarray | None, caixa) -> bool:
    if roi is None:
        return True
    cx, cy = (caixa[0] + caixa[2]) / 2, (caixa[1] + caixa[3]) / 2
    return cv2.pointPolygonTest(roi, (float(cx), float(cy)), False) >= 0


def ler_rotulos(caminho: Path, w: int, h: int) -> list[tuple[int, tuple[int, int, int, int]]]:
    """Le um .txt no formato YOLO (classe xc yc largura altura, normalizados) -> caixas em pixels."""
    caixas = []
    for linha in caminho.read_text(encoding="utf-8").splitlines():
        partes = linha.split()
        if len(partes) != 5:
            continue
        cls = int(partes[0])
        xc, yc, bw, bh = (float(v) for v in partes[1:])
        x1, y1 = int(round((xc - bw / 2) * w)), int(round((yc - bh / 2) * h))
        x2, y2 = int(round((xc + bw / 2) * w)), int(round((yc + bh / 2) * h))
        caixas.append((cls, (x1, y1, x2, y2)))
    return caixas


def salvar_rotulos(caminho: Path, caixas, w: int, h: int) -> None:
    linhas = []
    for cls, (x1, y1, x2, y2) in caixas:
        xc, yc = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        linhas.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    caminho.write_text("\n".join(linhas) + ("\n" if linhas else ""), encoding="utf-8")


def escurecer_fora(img: np.ndarray, roi: np.ndarray | None) -> np.ndarray:
    if roi is None:
        return img.copy()
    mascara = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(mascara, [roi], 255)
    escuro = (img * 0.45).astype(np.uint8)
    saida = np.where(mascara[:, :, None] > 0, img, escuro)
    cv2.polylines(saida, [roi], True, (255, 255, 255), 2)
    return saida


# ----------------------------------------------------------------------------- 1) extrair
def extrair(video: str, n: int) -> None:
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Não consegui abrir o vídeo: {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total < n:
        raise SystemExit(f"O vídeo tem só {total} frames.")

    # Amostragem uniforme: 1 frame a cada (total/n), começando no meio do 1º intervalo
    passo = total / n
    indices = [int(passo / 2 + i * passo) for i in range(n)]

    PASTA_FRAMES.mkdir(parents=True, exist_ok=True)
    PASTA_ROTULOS.mkdir(parents=True, exist_ok=True)
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            print(f"  aviso: não li o frame {idx}")
            continue
        cv2.imwrite(str(PASTA_FRAMES / f"frame_{idx:06d}.png"), frame)
    cap.release()

    roi = None
    if ARQUIVO_ROIS.is_file():
        roi = json.loads(ARQUIVO_ROIS.read_text(encoding="utf-8")).get(Path(video).name)
    info = {
        "video": Path(video).name, "resolucao": f"{w}x{h}", "fps": round(fps, 2), "total_frames": total,
        "amostragem": f"uniforme: {n} frames, 1 a cada {passo:.1f} frames (~{passo / fps:.1f}s)",
        "indices": indices, "roi": roi, "criado_em": datetime.now().strftime("%d/%m/%Y %H:%M"),
    }
    ARQUIVO_INFO.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(indices)} frames salvos em {PASTA_FRAMES}")
    print(f"Amostragem: {info['amostragem']}")
    print("ROI: " + ("usando o ROI salvo do vídeo" if roi else "NENHUM ROI salvo (vai anotar a tela toda)"))


# ----------------------------------------------------------------------------- 2) anotar
def anotar(inicio: int, fim: int | None) -> None:
    info = carregar_info()
    roi = roi_do_info(info)
    frames = sorted(PASTA_FRAMES.glob("frame_*.png"))[inicio:fim]
    if not frames:
        raise SystemExit("Nenhum frame nesse intervalo.")

    estado = {"classe": 2, "caixas": [], "arrastando": None, "atual": None}

    def ao_mouse(evento, x, y, _flags, _param):
        if evento == cv2.EVENT_LBUTTONDOWN:
            estado["arrastando"] = (x, y)
            estado["atual"] = (x, y)
        elif evento == cv2.EVENT_MOUSEMOVE and estado["arrastando"]:
            estado["atual"] = (x, y)
        elif evento == cv2.EVENT_LBUTTONUP and estado["arrastando"]:
            (xa, ya), (xb, yb) = estado["arrastando"], (x, y)
            x1, x2 = sorted((xa, xb))
            y1, y2 = sorted((ya, yb))
            if x2 - x1 >= 4 and y2 - y1 >= 4:
                estado["caixas"].append((estado["classe"], (x1, y1, x2, y2)))
            estado["arrastando"] = None
            estado["atual"] = None
        elif evento == cv2.EVENT_RBUTTONDOWN:
            dentro = [
                (i, (c[1][2] - c[1][0]) * (c[1][3] - c[1][1]))
                for i, c in enumerate(estado["caixas"])
                if c[1][0] <= x <= c[1][2] and c[1][1] <= y <= c[1][3]
            ]
            if dentro:
                estado["caixas"].pop(min(dentro, key=lambda t: t[1])[0])

    janela = "Anotacao - ground truth"
    cv2.namedWindow(janela, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(janela, ao_mouse)

    print("\n=== ANOTAÇÃO ===")
    print("Marque TODO veículo cujo centro esteja dentro da ROI (área clara). Caixa justa na parte visível.")
    print("  [1] Carro  [2] Moto  [3] Ônibus  [4] Caminhão  -> escolhe a classe")
    print("  Botão ESQUERDO arrastando: nova caixa | Botão DIREITO: apaga a caixa clicada | [Z] desfaz")
    print("  [D] ou [ENTER]: salva e vai pro próximo | [A]: salva e volta | [Q]/[ESC]: salva e sai\n")

    teclas_classe = {ord("1"): 2, ord("2"): 3, ord("3"): 5, ord("4"): 7}
    i = 0
    while 0 <= i < len(frames):
        caminho = frames[i]
        img = cv2.imread(str(caminho))
        h, w = img.shape[:2]
        if i == 0:
            cv2.resizeWindow(janela, min(w, 1400), min(h, 800))
        rotulo = PASTA_ROTULOS / (caminho.stem + ".txt")
        estado["caixas"] = ler_rotulos(rotulo, w, h) if rotulo.is_file() else []
        base = escurecer_fora(img, roi)
        acao = None
        while acao is None:
            vis = base.copy()
            for cls, (x1, y1, x2, y2) in estado["caixas"]:
                cor = CORES_CLASSE[cls]
                cv2.rectangle(vis, (x1, y1), (x2, y2), cor, 2)
                cv2.putText(vis, NOMES_CLASSES[cls], (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)
            if estado["arrastando"] and estado["atual"]:
                cv2.rectangle(vis, estado["arrastando"], estado["atual"], CORES_CLASSE[estado["classe"]], 1)
            barra = (f"Frame {inicio + i + 1}/{inicio + len(frames)} ({caminho.stem}) | Classe: "
                     f"{NOMES_CLASSES[estado['classe']]} | Caixas: {len(estado['caixas'])} | "
                     "[1-4] classe [Dir] apagar [Z] desfaz [D] prox [A] volta [Q] sai")
            cv2.rectangle(vis, (0, 0), (w, 34), (15, 15, 15), -1)
            cv2.putText(vis, barra, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow(janela, vis)
            tecla = cv2.waitKey(20) & 0xFF
            if tecla in teclas_classe:
                estado["classe"] = teclas_classe[tecla]
            elif tecla in (ord("z"), ord("Z")) and estado["caixas"]:
                estado["caixas"].pop()
            elif tecla in (ord("d"), ord("D"), 13):
                acao = 1
            elif tecla in (ord("a"), ord("A")):
                acao = -1
            elif tecla in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(janela, cv2.WND_PROP_VISIBLE) < 1:
                acao = 0
        salvar_rotulos(rotulo, estado["caixas"], w, h)
        print(f"  {caminho.stem}: {len(estado['caixas'])} caixas salvas")
        if acao == 0:
            break
        i += acao
    cv2.destroyAllWindows()
    resumo_anotacoes()


def resumo_anotacoes() -> None:
    rotulos = sorted(PASTA_ROTULOS.glob("frame_*.txt"))
    frames = sorted(PASTA_FRAMES.glob("frame_*.png"))
    por_classe = {c: 0 for c in CLASSES_ANOTACAO}
    for r in rotulos:
        for linha in r.read_text(encoding="utf-8").splitlines():
            if linha.split():
                por_classe[int(linha.split()[0])] = por_classe.get(int(linha.split()[0]), 0) + 1
    total = sum(por_classe.values())
    print(f"\nAnotados: {len(rotulos)}/{len(frames)} frames | {total} veículos -> "
          + ", ".join(f"{NOMES_CLASSES[c]}: {q}" for c, q in por_classe.items()))
    if total < 50:
        print("  Atenção: o R7 pede pelo menos 50 instâncias no total.")


# ----------------------------------------------------------------------------- 3) avaliar
def casar(preds, gts, limiar: float, mesma_classe: bool):
    """Correspondência um-para-um gulosa: preds em ordem de confiança, cada uma pega o GT livre de maior IoU."""
    usados = set()
    pares = []
    fps_idx = []
    for pi in sorted(range(len(preds)), key=lambda k: -preds[k][2]):
        p_cls, p_box, _ = preds[pi]
        melhor, melhor_iou = None, limiar
        for gi, (g_cls, g_box) in enumerate(gts):
            if gi in usados or (mesma_classe and g_cls != p_cls):
                continue
            iou, _ = calcular_sobreposicao(p_box, g_box)
            if iou >= melhor_iou:
                melhor, melhor_iou = gi, iou
        if melhor is None:
            fps_idx.append(pi)
        else:
            usados.add(melhor)
            pares.append((pi, melhor, melhor_iou))
    fns_idx = [gi for gi in range(len(gts)) if gi not in usados]
    return pares, fps_idx, fns_idx


def razao(a: float, b: float) -> float | None:
    return a / b if b > 0 else None


def fmt(v: float | None) -> str:
    return "n/d" if v is None else f"{v:.3f}"


def nome_config(cfg: dict) -> str:
    partes = [Path(cfg["modelo"]).stem, f"img{cfg['imgsz']}", f"conf{cfg['conf']:.2f}",
              "roi" if cfg["roi"] else "semroi", cfg["condicao"], cfg["preproc"]]
    if cfg["cnn"]:
        partes.append("cnn")
    return "_".join(partes)


class Avaliador:
    def __init__(self):
        self.info = carregar_info()
        self.roi = roi_do_info(self.info)
        self.frames = []
        for img_path in sorted(PASTA_FRAMES.glob("frame_*.png")):
            rot = PASTA_ROTULOS / (img_path.stem + ".txt")
            if rot.is_file():
                self.frames.append((img_path, rot))
        faltando = len(list(PASTA_FRAMES.glob("frame_*.png"))) - len(self.frames)
        if not self.frames:
            raise SystemExit("Nenhum frame anotado ainda. Rode: python avaliar_deteccao.py anotar")
        if faltando:
            print(f"Aviso: {faltando} frame(s) ainda sem anotação ficaram de fora.")
        self.modelos = {}
        self.cnn = None

    def yolo(self, nome: str):
        if nome not in self.modelos:
            from ultralytics import YOLO
            self.modelos[nome] = YOLO(nome)
        return self.modelos[nome]

    def modelo_cnn(self):
        if self.cnn is None:
            caminho = localizar_modelo_cnn()
            if caminho is None:
                raise SystemExit("modelo_veiculos.keras não encontrado (necessário para --cnn).")
            from tensorflow import keras
            self.cnn = keras.models.load_model(caminho)
        return self.cnn

    def rodar(self, cfg: dict, salvar_visual: bool = True) -> list[dict]:
        nome = nome_config(cfg)
        modelo = self.yolo(cfg["modelo"])
        cnn = self.modelo_cnn() if cfg["cnn"] else None
        roi = self.roi if cfg["roi"] else None
        pasta_vis = PASTA_VISUAL / nome
        if salvar_visual:
            pasta_vis.mkdir(parents=True, exist_ok=True)

        # aquecimento (o 1o predict é sempre mais lento)
        img0 = cv2.imread(str(self.frames[0][0]))
        modelo.predict(img0, imgsz=cfg["imgsz"], conf=cfg["conf"], classes=CLASSES_ANOTACAO, verbose=False)

        agn = {"tp": 0, "fp": 0, "fn": 0, "ious": []}
        cls_tot = {c: {"tp": 0, "fp": 0, "fn": 0} for c in CLASSES_ANOTACAO}
        trocas = 0
        rejeitados_cnn = 0
        n_gt = 0
        t_inf, t_total, t_cnn = [], [], []

        for img_path, rot_path in self.frames:
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            gts = [(c, b) for c, b in ler_rotulos(rot_path, w, h) if centro_dentro(self.roi, b)]
            n_gt += len(gts)

            t0 = time.perf_counter()
            frame = aplicar_preproc(aplicar_condicao(img, cfg["condicao"]), cfg["preproc"])
            entrada = frame
            if roi is not None:
                mascara = np.zeros((h, w), np.uint8)
                cv2.fillPoly(mascara, [roi], 255)
                entrada = cv2.bitwise_and(frame, frame, mask=mascara)
            res = modelo.predict(entrada, imgsz=cfg["imgsz"], conf=cfg["conf"], classes=CLASSES_ANOTACAO,
                                 agnostic_nms=True, verbose=False)[0]
            t_inf.append(float(res.speed.get("inference", 0.0)))
            preds = []
            for (x1, y1, x2, y2), c, s in zip(res.boxes.xyxy.cpu().numpy().astype(int),
                                              res.boxes.cls.cpu().numpy().astype(int),
                                              res.boxes.conf.cpu().numpy().astype(float)):
                caixa = (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
                if roi is not None and not centro_dentro(roi, caixa):
                    continue
                if cnn is not None:
                    recorte = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[caixa[1]:caixa[3], caixa[0]:caixa[2]]
                    if recorte.size == 0:
                        continue
                    tc = time.perf_counter()
                    entrada_cnn = np.expand_dims(cv2.resize(recorte, (64, 64)), 0).astype(np.float32)
                    prob = float(cnn(entrada_cnn, training=False).numpy()[0][INDICE_CLASSE_VEICULO])
                    t_cnn.append((time.perf_counter() - tc) * 1000)
                    if prob < CONF_CNN:
                        rejeitados_cnn += 1
                        continue
                preds.append((int(c), caixa, float(s)))
            t_total.append((time.perf_counter() - t0) * 1000)

            # agnóstico de classe: "achou o veículo?" (o que importa para a contagem)
            pares, fp_idx, fn_idx = casar(preds, gts, IOU_LIMIAR, mesma_classe=False)
            agn["tp"] += len(pares)
            agn["fp"] += len(fp_idx)
            agn["fn"] += len(fn_idx)
            agn["ious"] += [iou for _, _, iou in pares]
            trocas += sum(1 for pi, gi, _ in pares if preds[pi][0] != gts[gi][0])

            # por classe: só conta acerto se a classe também bater
            for c in CLASSES_ANOTACAO:
                pc = [p for p in preds if p[0] == c]
                gc = [g for g in gts if g[0] == c]
                p2, f2, n2 = casar(pc, gc, IOU_LIMIAR, mesma_classe=True)
                cls_tot[c]["tp"] += len(p2)
                cls_tot[c]["fp"] += len(f2)
                cls_tot[c]["fn"] += len(n2)

            if salvar_visual:
                vis = escurecer_fora(frame, self.roi)
                casados_gt = {gi for _, gi, _ in pares}
                for gi, (_, b) in enumerate(gts):
                    cor = (0, 255, 0) if gi in casados_gt else (0, 255, 255)
                    cv2.rectangle(vis, b[:2], b[2:], cor, 4 if gi not in casados_gt else 1)
                for pi, (c, b, s) in enumerate(preds):
                    cor = (0, 0, 255) if pi in fp_idx else (255, 120, 0)
                    cv2.rectangle(vis, b[:2], b[2:], cor, 2)
                    cv2.putText(vis, f"{NOMES_CLASSES[c]} {s:.2f}", (b[0], max(15, b[1] - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)
                legenda = (f"{nome} | azul=acerto  vermelho=falso positivo  amarelo grosso=falso negativo  "
                           f"verde fino=anotacao  branco=ROI")
                cv2.rectangle(vis, (0, 0), (w, 30), (15, 15, 15), -1)
                cv2.putText(vis, legenda, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.imwrite(str(pasta_vis / (img_path.stem + ".jpg")), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])

        amb = descrever_ambiente()
        inf_ms = float(np.mean(t_inf))
        tot_ms = float(np.mean(t_total))
        base = {
            "data_hora": datetime.now().strftime("%d/%m/%Y %H:%M"), "config": nome,
            "modelo": Path(cfg["modelo"]).name, "imgsz": cfg["imgsz"], "conf": cfg["conf"],
            "roi": "sim" if cfg["roi"] else "nao", "condicao": cfg["condicao"], "preproc": cfg["preproc"],
            "cnn": "sim" if cfg["cnn"] else "nao", "iou_limiar": IOU_LIMIAR, "frames": len(self.frames),
        }
        linhas = []
        p = razao(agn["tp"], agn["tp"] + agn["fp"])
        r = razao(agn["tp"], agn["tp"] + agn["fn"])
        f1 = razao(2 * p * r, p + r) if p is not None and r is not None else None
        linhas.append({**base, "classe": "TODAS (agnostico)", "gt": n_gt, "tp": agn["tp"], "fp": agn["fp"],
                       "fn": agn["fn"], "precisao": fmt(p), "recall": fmt(r), "f1": fmt(f1),
                       "iou_medio_tp": fmt(float(np.mean(agn["ious"])) if agn["ious"] else None),
                       "classe_trocada": trocas, "rejeitados_cnn": rejeitados_cnn,
                       "inferencia_ms": round(inf_ms, 1), "total_ms_por_frame": round(tot_ms, 1),
                       "fps": round(1000 / tot_ms, 2) if tot_ms else 0,
                       "cnn_ms": round(float(np.mean(t_cnn)), 2) if t_cnn else "",
                       "cpu": amb["cpu"], "dispositivo": amb["dispositivo"], "ultralytics": amb["ultralytics"]})
        for c, t in cls_tot.items():
            gt_c = t["tp"] + t["fn"]
            if gt_c == 0 and t["fp"] == 0:
                continue
            pc = razao(t["tp"], t["tp"] + t["fp"])
            rc = razao(t["tp"], gt_c)
            f1c = razao(2 * pc * rc, pc + rc) if pc is not None and rc is not None else None
            linhas.append({**base, "classe": NOMES_CLASSES[c], "gt": gt_c, "tp": t["tp"], "fp": t["fp"],
                           "fn": t["fn"], "precisao": fmt(pc), "recall": fmt(rc), "f1": fmt(f1c),
                           "iou_medio_tp": "", "classe_trocada": "", "rejeitados_cnn": "",
                           "inferencia_ms": "", "total_ms_por_frame": "", "fps": "", "cnn_ms": "",
                           "cpu": "", "dispositivo": "", "ultralytics": ""})

        print(f"\n[{nome}] {len(self.frames)} frames, {n_gt} veículos anotados, IoU >= {IOU_LIMIAR}")
        print(f"  {'classe':<18}{'GT':>5}{'TP':>5}{'FP':>5}{'FN':>5}{'Prec':>8}{'Rec':>8}{'F1':>8}")
        for l in linhas:
            print(f"  {l['classe']:<18}{l['gt']:>5}{l['tp']:>5}{l['fp']:>5}{l['fn']:>5}"
                  f"{l['precisao']:>8}{l['recall']:>8}{l['f1']:>8}")
        print(f"  IoU médio dos acertos: {linhas[0]['iou_medio_tp']} | classe trocada: {trocas}"
              + (f" | rejeitados pela CNN: {rejeitados_cnn}" if cfg["cnn"] else ""))
        print(f"  Inferência: {inf_ms:.1f} ms | por frame (com pré-proc/filtros): {tot_ms:.1f} ms "
              f"-> {1000 / tot_ms:.2f} FPS | {amb['cpu']} ({amb['dispositivo']})")
        if not cfg["roi"]:
            print("  (sem ROI: detecções fora da área de interesse contam como falso positivo)")
        self.registrar(linhas)
        return linhas

    @staticmethod
    def registrar(linhas: list[dict]) -> None:
        PASTA_AVAL.mkdir(parents=True, exist_ok=True)
        novo = not ARQUIVO_METRICAS.is_file()
        with ARQUIVO_METRICAS.open("a", newline="", encoding="utf-8-sig") as f:
            escritor = csv.DictWriter(f, fieldnames=list(linhas[0].keys()), delimiter=";")
            if novo:
                escritor.writeheader()
            escritor.writerows(linhas)
        with ARQUIVO_METRICAS_MD.open("a", encoding="utf-8") as f:
            if novo:
                f.write("| config | classe | GT | TP | FP | FN | Precisão | Recall | F1 | IoU médio | FPS |\n")
                f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for l in linhas:
                f.write(f"| {l['config']} | {l['classe']} | {l['gt']} | {l['tp']} | {l['fp']} | {l['fn']} | "
                        f"{l['precisao']} | {l['recall']} | {l['f1']} | {l['iou_medio_tp']} | {l['fps']} |\n")


def cfg_padrao(**mudancas) -> dict:
    cfg = {"modelo": MODELO_YOLO, "imgsz": TAMANHO_IMG_YOLO, "conf": CONF_YOLO, "roi": True,
           "condicao": "original", "preproc": "nenhum", "cnn": False}
    cfg.update(mudancas)
    return cfg


CONFIGS_LOTE = [
    ("Final, só YOLO",               cfg_padrao()),
    ("Final, YOLO + CNN",            cfg_padrao(cnn=True)),
    ("Confiança 0,50",               cfg_padrao(conf=0.50)),
    ("Sem ROI",                      cfg_padrao(roi=False)),
    ("Original + CLAHE",             cfg_padrao(preproc="clahe")),
    ("Escuro",                       cfg_padrao(condicao="escuro")),
    ("Escuro + CLAHE",               cfg_padrao(condicao="escuro", preproc="clahe")),
    ("Baixa resolução",              cfg_padrao(condicao="baixa_res")),
    ("Baixa resolução + CLAHE",      cfg_padrao(condicao="baixa_res", preproc="clahe")),
    ("Borrado (movimento)",          cfg_padrao(condicao="borrado")),
    ("Borrado + CLAHE",              cfg_padrao(condicao="borrado", preproc="clahe")),
    ("Início: YOLOv8n 640px",        cfg_padrao(modelo="yolov8n.pt", imgsz=640)),
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Avaliação da detecção (precisão, recall, F1, IoU, FPS)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extrair", help="separa N frames espaçados do vídeo")
    e.add_argument("--video", required=True)
    e.add_argument("--n", type=int, default=30)

    a = sub.add_parser("anotar", help="ferramenta de anotação (ground truth)")
    a.add_argument("--inicio", type=int, default=0, help="índice do 1º frame (0 a 29)")
    a.add_argument("--fim", type=int, default=None, help="índice final (exclusivo)")

    sub.add_parser("resumo", help="mostra quantos frames/veículos já foram anotados")

    v = sub.add_parser("avaliar", help="avalia uma configuração")
    v.add_argument("--modelo", default=MODELO_YOLO)
    v.add_argument("--imgsz", type=int, default=TAMANHO_IMG_YOLO)
    v.add_argument("--conf", type=float, default=CONF_YOLO)
    v.add_argument("--sem-roi", action="store_true")
    v.add_argument("--condicao", choices=CONDICOES, default="original")
    v.add_argument("--preproc", choices=PREPROCS, default="nenhum")
    v.add_argument("--cnn", action="store_true", help="filtra as detecções com a CNN (>= CONF_CNN)")

    l = sub.add_parser("lote", help="avalia todas as configurações dos experimentos")
    l.add_argument("--sem-cnn", action="store_true", help="pula a configuração com CNN")

    args = ap.parse_args()
    if args.cmd == "extrair":
        extrair(args.video, args.n)
    elif args.cmd == "anotar":
        anotar(args.inicio, args.fim)
    elif args.cmd == "resumo":
        resumo_anotacoes()
    elif args.cmd == "avaliar":
        Avaliador().rodar(cfg_padrao(modelo=args.modelo, imgsz=args.imgsz, conf=args.conf, roi=not args.sem_roi,
                                     condicao=args.condicao, preproc=args.preproc, cnn=args.cnn))
    elif args.cmd == "lote":
        av = Avaliador()
        resumo_anotacoes()
        for rotulo, cfg in CONFIGS_LOTE:
            if args.sem_cnn and cfg["cnn"]:
                continue
            print(f"\n>>> {rotulo}")
            try:
                av.rodar(cfg)
            except Exception as err:  # uma config com problema não derruba as outras
                print(f"  ERRO nessa configuração: {err}")
        print(f"\nTabelas salvas em:\n  {ARQUIVO_METRICAS}\n  {ARQUIVO_METRICAS_MD}\n  Imagens: {PASTA_VISUAL}")


if __name__ == "__main__":
    main()
