# Contagem Automática de Veículos — YOLOv8 + CNN + Jogo de Palpites

Capstone de Processamento de Imagens Digitais e Visão Computacional — Centro Universitário Dom Helder (7º período).
Grupo: Bruno Sampaio, Gabriel Fajardo e Sofia Ferreira.

O sistema detecta e rastreia veículos (carro, moto, ônibus e caminhão) em um vídeo de rua, conta cada veículo uma única vez
dentro de uma região de interesse (ROI) e transforma a contagem em um jogo: a turma dá palpites e, no fim, aparece o pódio.

## Pipeline

Entrada (vídeo, validação, resolução e FPS) → condição/pré-processamento (opcional: CLAHE) → máscara da ROI →
YOLOv8s + ByteTrack (imgsz 960, conf 0,25, classes 2/3/5/7, agnostic NMS) → filtro (centro na ROI + CNN ≥ 0,55) →
contagem por ID único com anti-duplicata (IoU/IoMin) → placar, vídeo anotado, histórico e CSV de experimentos.

## Instalação

```bash
pip install -r requirements.txt
```

Para as mesmas versões usadas nos experimentos (Ultralytics 8.4.218, torch 2.15.0 CPU, TensorFlow 2.21.0): `pip install -r requirements-versoes.txt`.

Arquivos necessários na mesma pasta: `contador_veiculos.py`, `modelo_veiculos.keras` (CNN treinada; dá para treinar pela opção 2
do menu), `rois_salvos.json` (criado ao marcar a ROI). Os pesos `yolov8s.pt` são baixados automaticamente pela Ultralytics.

## Como rodar

| O que | Comando |
|---|---|
| Modo interativo (menu, ROI, palpites, pódio) | `python contador_veiculos.py` |
| Uma rodada sem janelas (usa a ROI salva) | `python contador_veiculos.py --video 1125.mp4 --modo cnn --lote` |
| Variações | `--modo yolo`, `--conf 0.5`, `--sem-roi`, `--condicao escuro\|baixa_res\|borrado`, `--preproc clahe` |
| Todos os experimentos em sequência | `python rodar_experimentos.py --video 1125.mp4` |
| Apresentação (palpites + vídeo já processado + pódio) | `python contador_veiculos.py --apresentar resultados/<video_resultado>.mp4` |
| Kit para o PC da apresentação (zip com tudo que o jogo precisa) | `python montar_kit.py` (ou `--completo`) |

Cada rodada salva o vídeo anotado em `resultados/`, acrescenta uma linha em `resultados/experimentos.csv`
(contagem por classe, resolução, FPS de origem, tempo de inferência, FPS de processamento, hardware e versões)
e registra um resumo em `resultados/historico_contagens.txt`.

## Avaliação da detecção (precisão, recall, F1, IoU)

```bash
python avaliar_deteccao.py extrair --video 1125.mp4     # 30 frames com espaçamento uniforme
python avaliar_deteccao.py anotar                         # ground truth manual (pode dividir com --inicio/--fim)
python avaliar_deteccao.py lote                           # avalia todas as configurações
```

- Correspondência um-para-um entre caixa prevista e caixa anotada, com **IoU ≥ 0,50**.
- Métricas agnósticas de classe ("achou o veículo?") e por classe; casos de denominador zero aparecem como `n/d`.
- Só entram veículos com o centro dentro da ROI (a área de interesse da contagem).
- Saídas: `avaliacao/metricas.csv`, `avaliacao/metricas.md` e `avaliacao/visual/` (acertos, falsos positivos e falsos negativos).

## Dados

- Vídeo: `1125.mp4` — vídeo público encontrado na internet (trecho inicial de uma gravação original de ~5 min), usado só
  para fins acadêmicos; link da origem: **[preencher]**. 1920×1080, 30 FPS, 910 frames (~30 s). Não versionado no repositório.
- Ground truth da contagem: **47 veículos** dentro da ROI, contados manualmente (inclui 1 moto, 3 caminhões e 3 vans; vans
  contam como caminhão porque o COCO não tem classe van; fica fora o carro estacionado fora da ROI e a ponta de carro no início).
- Treino da CNN (Kaggle): `brsdincer/vehicle-detection-image-set`, `mmohaiminulislam/vehicles-image-dataset`,
  `utkarshsaxenadn/car-vs-bike-classification-dataset`, `mohamedmaher5/vehicle-classification`, `kshitij192/cars-image-dataset`.
- Detector: YOLOv8s pré-treinado no COCO (Ultralytics). Não treinamos o YOLO.
- As condições `escuro`, `baixa_res` e `borrado` são degradações sintéticas do vídeo original, para simular iluminação ruim,
  menor resolução e movimento.

## Resultados (1125.mp4, CPU AMD Ryzen, sem GPU)

Contagem (verdade: 47) — `resultados/experimentos.csv` (a 1ª linha é um teste de 60 frames, ignorar):

| Configuração | Total | Rejeitados pela CNN | FPS |
|---|---|---|---|
| **Final: YOLO + CNN, conf 0,25, ROI** | **43** | 1 | 5,9 |
| Só YOLO | 45 | – | 6,1 |
| conf 0,50 | 48 | 1 | 5,9 |
| Sem ROI | 46 | 1 | 7,5 |
| Original + CLAHE | 45 | 0 | 4,9 |
| Escuro / + CLAHE | 43 / 45 | 2 / 0 | 5,9 / 4,9 |
| Baixa resolução / + CLAHE | 33 / 45 | 10 / 1 | 5,8 / 4,9 |
| Borrado / + CLAHE | 44 / 49 | 2 / 0 | 4,8 / 4,1 |

Detecção por frame (30 frames anotados, 230 veículos, IoU ≥ 0,50) — `avaliacao/metricas.csv`:

| Configuração | Precisão | Recall | F1 | FPS |
|---|---|---|---|---|
| **YOLOv8s 960, conf 0,25, ROI** | **0,68** | **0,67** | **0,68** | 7,2 |
| + CNN | 0,69 | 0,65 | 0,67 | 4,1 |
| conf 0,50 | 0,71 | 0,60 | 0,65 | 7,5 |
| Sem ROI | 0,51 | 0,68 | 0,59 | 7,6 |
| + CLAHE | 0,69 | 0,68 | 0,69 | 5,6 |
| Escuro / + CLAHE | 0,66 / 0,68 | 0,67 / 0,69 | 0,66 / 0,69 | 7,4 / 5,8 |
| Baixa resolução / + CLAHE | 0,69 / 0,71 | 0,67 / 0,70 | 0,68 / 0,71 | 7,1 / 5,7 |
| Borrado / + CLAHE | 0,76 / 0,77 | 0,49 / 0,60 | 0,59 / 0,67 | 5,0 / 4,2 |
| YOLOv8n 640 | 0,69 | 0,50 | 0,58 | 27,6 |

IoU médio dos acertos ≈ 0,67: as caixas anotadas à mão são mais folgadas que as do YOLO, então parte dos FP/FN é o mesmo
carro com caixa diferente. Exemplos visuais em `avaliacao/exemplos/` (azul = acerto, vermelho = FP, amarelo = FN,
verde = anotação, branco = ROI).

## Vídeos de resultado

Ficam na pasta `videos/` do zip de entrega (não vão para o GitHub por tamanho; recodificados em H.264 para ficarem leves).

| Arquivo | Configuração (1125.mp4, verdade = 47) | Total |
|---|---|---|
| `01_final_yolo_cnn.mp4` | **Final: YOLO + CNN, conf 0,25, ROI** | **43** |
| `02_so_yolo.mp4` | Só YOLO | 45 |
| `03_conf_0.50.mp4` | conf 0,50 | 48 |
| `04_sem_roi.mp4` | Sem ROI | 46 |
| `05_escuro.mp4` / `06_escuro_clahe.mp4` | Escuro / + CLAHE | 43 / 45 |
| `07_baixa_res.mp4` / `08_baixa_res_clahe.mp4` | Baixa resolução / + CLAHE | 33 / 45 |
| `09_borrado.mp4` / `10_borrado_clahe.mp4` | Borrado / + CLAHE | 44 / 49 |
| `11_original_clahe.mp4` | Original + CLAHE | 45 |
| `12_jogo_2min_apresentacao.mp4` | Vídeo do jogo (2 min), rodado na apresentação | 146 |

### Vídeo do jogo (2 min)

O sistema contou **146** (112 carros, 3 motos, 8 ônibus, 23 caminhões). A contagem de referência é **135 ± 2**,
feita fora do sistema: uma linha virtual fixa na pista, empilhada frame a frame numa imagem (cada veículo que cruza
vira uma mancha), com os casos duvidosos conferidos nos frames, mais os veículos que ainda estavam na ROI no último
frame. Nos primeiros 30 s (o trecho do `1125.mp4`) esse método dá exatamente os 47 da contagem manual.

No vídeo longo o sistema passou a **contar a mais (+11, cerca de 8%)**, ao contrário do trecho de 30 s (−4).
Hipótese: o vídeo longo tem congestionamento e muitos veículos compridos (carretas, caçamba, ônibus, caminhões-baú),
o que favorece troca de ID e contagem dupla; as classes também se confundem (o sistema marcou 8 ônibus onde há 2).

## Estrutura

```
contador_veiculos.py     # sistema completo (menu, treino da CNN, contagem, jogo, modo lote, --apresentar)
avaliar_deteccao.py      # extração de frames, ferramenta de anotação e métricas
rodar_experimentos.py    # roda todas as configurações de contagem em sequência
montar_kit.py            # gera kit_apresentacao.zip para levar ao PC da apresentação
rois_salvos.json         # ROI usada no vídeo
avaliacao/rotulos/       # anotações manuais (formato YOLO)
avaliacao/info.json      # frames amostrados e ROI usada na avaliação
avaliacao/metricas.csv   # métricas de detecção (também em metricas.md)
avaliacao/exemplos/      # imagens de acertos e falhas
resultados/experimentos.csv
requirements-versoes.txt # versões exatas do ambiente dos experimentos
```

Não versionados (pesados): vídeos, `yolov8*.pt`, `modelo_veiculos.keras`, frames extraídos e vídeos de saída.

## Limitações conhecidas

- Parâmetros ajustados e avaliados no mesmo vídeo (uma câmera, um horário).
- A CNN tem viés de dataset (poucas vans/caminhões) e de domínio (fotos ao nível do chão × câmera alta).
- A regra anti-duplicata pode juntar dois veículos muito próximos (subcontagem).

## Uso de ferramentas de IA

Parte do código foi desenvolvida com auxílio de assistentes de IA (Google Antigravity e Claude). Todo o código foi revisado,
testado e é explicado pelo grupo.
