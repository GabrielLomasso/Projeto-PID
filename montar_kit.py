"""Monta a pasta/zip para levar ao PC da apresentação.

Rodar na pasta do projeto (onde está o contador_veiculos.py), com o venv ativado:
    python montar_kit.py                      # kit do jogo (leve: só precisa de opencv + numpy)
    python montar_kit.py --completo           # + vídeo original, modelos, ROI e anotações (rodar ao vivo)
    python montar_kit.py --video-resultado resultados/resultado_YOLO_CNN_....mp4
"""

import argparse
import csv
import shutil
import sys
import zipfile
from pathlib import Path

BASE = Path(__file__).resolve().parent
RESULTADOS = BASE / "resultados"
CSV_EXP = RESULTADOS / "experimentos.csv"
DESTINO = BASE / "kit_apresentacao"

REQ_APRESENTACAO = "opencv-python\nnumpy\n"

BAT = r"""@echo off
cd /d "%~dp0"
set PY=py -3
%PY% --version >nul 2>&1 || set PY=python
%PY% --version >nul 2>&1 || (echo Python nao encontrado. Instale pelo python.org marcando "Add python.exe to PATH". & pause & exit /b 1)
%PY% -c "import cv2, numpy" >nul 2>&1 || %PY% -m pip install -r requirements-apresentacao.txt
%PY% contador_veiculos.py --apresentar "resultados\{video}"
pause
"""

LEIA_ME = """KIT DA APRESENTACAO - Contagem de Veiculos (YOLOv8 + CNN)

JOGO (o que vamos usar):
  1. Dois cliques em JOGAR.bat
  2. Digitar nome e palpite de cada um (ENTER vazio termina)
  3. O video ja processado toca e no fim aparece o podio

  Precisa so de Python + opencv-python + numpy (o JOGAR.bat instala se faltar, com internet).
  Video usado: resultados/{video}  (contagem registrada em resultados/experimentos.csv)

  Plano B sem Python: abrir o video da pasta resultados no VLC e fazer o podio na mao
  (total do sistema: {total}).
{extra}"""

EXTRA_COMPLETO = """
RODAR AO VIVO (so se pedirem; demora ~3 min por 30 s de video na CPU):
  pip install -r requirements.txt   (instala tensorflow/ultralytics/torch, pesado: fazer ANTES)
  python contador_veiculos.py --video 1125.mp4 --modo cnn --lote
"""


def ler_experimentos() -> list[dict[str, str]]:
    if not CSV_EXP.is_file():
        sys.exit(f"Não achei {CSV_EXP}. Rode na pasta do projeto.")
    with CSV_EXP.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f, delimiter=";"))


def escolher_video(linhas: list[dict[str, str]], pedido: str | None) -> tuple[Path, dict[str, str]]:
    registrados = {l["arquivo_saida"]: l for l in linhas if l.get("arquivo_saida")}
    if pedido:
        caminho = Path(pedido)
        if not caminho.is_file():
            caminho = RESULTADOS / Path(pedido).name
        if not caminho.is_file():
            sys.exit(f"Vídeo não encontrado: {pedido}")
        if caminho.name not in registrados:
            sys.exit(f"{caminho.name} não está no experimentos.csv; o --apresentar não acharia a contagem.")
        return caminho, registrados[caminho.name]

    # padrão: a rodada mais recente da configuração final (YOLO + CNN, conf 0.25, ROI, original, sem pré-proc)
    candidatos = []
    for nome, linha in registrados.items():
        caminho = RESULTADOS / nome
        final = (linha["modo"] == "YOLO + CNN" and linha["roi"] == "sim" and linha["condicao"] == "original"
                 and linha["preproc"] == "nenhum" and float(linha["conf_yolo"]) == 0.25
                 and int(linha["frames_processados"]) > 100)
        if final and caminho.is_file():
            candidatos.append(caminho)
    if not candidatos:
        sys.exit("Nenhum vídeo da configuração final achado em resultados/. Use --video-resultado.")
    caminho = max(candidatos, key=lambda p: p.stat().st_mtime)
    return caminho, registrados[caminho.name]


def copiar(origem: Path, destino_rel: str, faltando: list[str]) -> None:
    if not origem.exists():
        faltando.append(origem.name)
        return
    destino = DESTINO / destino_rel
    destino.parent.mkdir(parents=True, exist_ok=True)
    if origem.is_dir():
        shutil.copytree(origem, destino, dirs_exist_ok=True)
    else:
        shutil.copy2(origem, destino)


def main() -> None:
    parser = argparse.ArgumentParser(description="Monta o kit da apresentação.")
    parser.add_argument("--video-resultado", help="vídeo processado a usar no jogo (padrão: config final mais recente)")
    parser.add_argument("--completo", action="store_true",
                        help="inclui vídeo original, modelos, ROI, anotações e scripts de avaliação")
    args = parser.parse_args()

    linhas = ler_experimentos()
    video, registro = escolher_video(linhas, args.video_resultado)

    if DESTINO.exists():
        shutil.rmtree(DESTINO)
    DESTINO.mkdir()
    faltando: list[str] = []

    copiar(BASE / "contador_veiculos.py", "contador_veiculos.py", faltando)
    copiar(video, f"resultados/{video.name}", faltando)
    copiar(CSV_EXP, "resultados/experimentos.csv", faltando)
    (DESTINO / "requirements-apresentacao.txt").write_text(REQ_APRESENTACAO, encoding="utf-8")
    (DESTINO / "JOGAR.bat").write_text(BAT.format(video=video.name), encoding="ascii")

    if args.completo:
        nome_original = registro.get("video") or "1125.mp4"
        copiar(BASE / nome_original, nome_original, faltando)
        for nome in ("modelo_veiculos.keras", "yolov8s.pt", "rois_salvos.json", "avaliar_deteccao.py",
                     "rodar_experimentos.py", "requirements.txt", "requirements-versoes.txt", "README.md"):
            copiar(BASE / nome, nome, faltando)
        copiar(BASE / "avaliacao" / "rotulos", "avaliacao/rotulos", faltando)
        copiar(BASE / "avaliacao" / "info.json", "avaliacao/info.json", faltando)
        copiar(BASE / "avaliacao" / "metricas.csv", "avaliacao/metricas.csv", faltando)

    (DESTINO / "LEIA-ME.txt").write_text(
        LEIA_ME.format(video=video.name, total=registro["total"],
                       extra=EXTRA_COMPLETO if args.completo else ""),
        encoding="utf-8")

    arquivo_zip = BASE / "kit_apresentacao.zip"
    with zipfile.ZipFile(arquivo_zip, "w") as zf:
        for arq in sorted(DESTINO.rglob("*")):
            if arq.is_file():
                metodo = zipfile.ZIP_STORED if arq.suffix in {".mp4", ".pt", ".keras"} else zipfile.ZIP_DEFLATED
                zf.write(arq, Path("kit_apresentacao") / arq.relative_to(DESTINO), compress_type=metodo)

    print(f"Vídeo do jogo: {video.name}  (total registrado: {registro['total']})")
    if faltando:
        print("Não encontrados (copie na mão se precisar): " + ", ".join(faltando))
    print(f"Pronto: {arquivo_zip}  ({arquivo_zip.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
