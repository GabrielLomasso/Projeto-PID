"""
Roda todos os experimentos de contagem (R6) em sequência, sem precisar clicar em nada.
Fica na mesma pasta do contador_veiculos.py. Antes, marque o ROI do vídeo uma vez no modo normal.

    python rodar_experimentos.py --video 1125.mp4
    python rodar_experimentos.py --video 1125.mp4 --so 1,2,3      # só alguns
    python rodar_experimentos.py --video 1125.mp4 --a-partir 5    # retoma do 5 em diante

Cada rodada gera um vídeo em resultados/, uma linha em resultados/experimentos.csv
e um log em resultados/logs/. Se uma rodada falhar, as outras continuam.
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

PASTA = Path(__file__).resolve().parent
SCRIPT = PASTA / "contador_veiculos.py"
PASTA_LOGS = PASTA / "resultados" / "logs"
ARQUIVO_CSV = PASTA / "resultados" / "experimentos.csv"

# Em ordem de prioridade: se a noite acabar no meio, os mais importantes já rodaram.
EXPERIMENTOS = [
    ("final_yolo_cnn",     "Versão final (YOLO + CNN)",           ["--modo", "cnn"]),
    ("final_so_yolo",      "Versão final, só YOLO",               ["--modo", "yolo"]),
    ("conf_050",           "Confiança 0,50",                      ["--modo", "cnn", "--conf", "0.5"]),
    ("sem_roi",            "Sem ROI",                             ["--modo", "cnn", "--sem-roi"]),
    ("escuro",             "Condição: escuro",                    ["--modo", "cnn", "--condicao", "escuro"]),
    ("escuro_clahe",       "Condição: escuro + CLAHE",            ["--modo", "cnn", "--condicao", "escuro", "--preproc", "clahe"]),
    ("baixa_res",          "Condição: baixa resolução",           ["--modo", "cnn", "--condicao", "baixa_res"]),
    ("borrado",            "Condição: borrado (movimento)",       ["--modo", "cnn", "--condicao", "borrado"]),
    ("baixa_res_clahe",    "Condição: baixa resolução + CLAHE",   ["--modo", "cnn", "--condicao", "baixa_res", "--preproc", "clahe"]),
    ("borrado_clahe",      "Condição: borrado + CLAHE",           ["--modo", "cnn", "--condicao", "borrado", "--preproc", "clahe"]),
    ("original_clahe",     "Original + CLAHE",                    ["--modo", "cnn", "--preproc", "clahe"]),
]


def linhas_csv() -> list[dict]:
    if not ARQUIVO_CSV.is_file():
        return []
    with ARQUIVO_CSV.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f, delimiter=";"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--so", help="números separados por vírgula (ex.: 1,2,3)")
    ap.add_argument("--a-partir", type=int, default=1, help="começa deste número")
    ap.add_argument("--sem-cnn", action="store_true", help="troca --modo cnn por --modo yolo em todos")
    args = ap.parse_args()

    if not Path(args.video).is_file():
        sys.exit(f"Vídeo não encontrado: {args.video}")

    escolhidos = set(int(x) for x in args.so.split(",")) if args.so else None
    fila = [(i, *e) for i, e in enumerate(EXPERIMENTOS, 1)
            if i >= args.a_partir and (escolhidos is None or i in escolhidos)]

    PASTA_LOGS.mkdir(parents=True, exist_ok=True)
    ambiente = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    linhas_antes = len(linhas_csv())
    inicio_geral = time.time()
    duracoes = []
    falhas = []

    print(f"{len(fila)} experimento(s) na fila. Log de cada um em {PASTA_LOGS}\n")
    for pos, (num, chave, descricao, extra) in enumerate(fila, 1):
        if args.sem_cnn:
            extra = ["yolo" if x == "cnn" else x for x in extra]
        cmd = [sys.executable, str(SCRIPT), "--video", args.video, "--lote", *extra]
        print(f"[{pos}/{len(fila)}] #{num} {descricao}")
        print("   " + " ".join(cmd[1:]))
        t0 = time.time()
        log = PASTA_LOGS / f"{num:02d}_{chave}.log"
        with log.open("w", encoding="utf-8") as f:
            proc = subprocess.Popen(cmd, cwd=PASTA, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    env=ambiente, text=True, encoding="utf-8", errors="replace")
            for linha in proc.stdout:
                f.write(linha)
                if "Frame " in linha or "TOTAL REAL" in linha or "FPS (" in linha or "Error" in linha:
                    print("   " + linha.rstrip(), flush=True)
            proc.wait()
        dur = time.time() - t0
        duracoes.append(dur)
        if proc.returncode != 0:
            falhas.append(f"#{num} {descricao}")
            print(f"   FALHOU (código {proc.returncode}). Veja {log}")
        restantes = len(fila) - pos
        if restantes:
            estimativa = sum(duracoes) / len(duracoes) * restantes / 60
            print(f"   levou {dur / 60:.1f} min | faltam ~{estimativa:.0f} min\n")

    print("\n" + "=" * 70)
    print(f"Fim. Tempo total: {(time.time() - inicio_geral) / 60:.1f} min")
    novas = linhas_csv()[linhas_antes:]
    if novas:
        print(f"\n{'modo':<12}{'conf':>6}{'roi':>5}  {'condicao':<11}{'preproc':<8}{'TOTAL':>6}"
              f"{'carro':>7}{'cam.':>6}{'rej':>5}{'FPS':>7}")
        for l in novas:
            print(f"{l['modo']:<12}{l['conf_yolo']:>6}{l['roi']:>5}  {l['condicao']:<11}{l['preproc']:<8}"
                  f"{l['total']:>6}{l['carros']:>7}{l['caminhoes']:>6}{l['cnn_rejeitados']:>5}"
                  f"{l['fps_processamento']:>7}")
    if falhas:
        print("\nFalharam: " + "; ".join(falhas))
    print(f"\nTabela completa: {ARQUIVO_CSV}")


if __name__ == "__main__":
    main()
