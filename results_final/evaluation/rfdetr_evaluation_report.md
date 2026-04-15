# RF-DETR Evaluation Report

## Métricas Principales

| Modelo | mAP@0.5:0.95 | mAP@0.50 | mAP@0.75 | mAP_S | mAP_M | mAP_L |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| Best_EMA | 0.4729 | 0.8237 | 0.4923 | 0.1110 | 0.4613 | 0.6288 |
| Best_Regular | 0.4562 | 0.8028 | 0.4696 | 0.1030 | 0.4439 | 0.6112 |
| Best_Total | 0.4729 | 0.8237 | 0.4923 | 0.1110 | 0.4613 | 0.6288 |

## Recall

| Modelo | AR@1 | AR@10 | AR@100 | AR_S | AR_M | AR_L |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| Best_EMA | 0.3232 | 0.5356 | 0.5963 | 0.2098 | 0.5874 | 0.7061 |
| Best_Regular | 0.3170 | 0.5339 | 0.6039 | 0.2120 | 0.5946 | 0.7122 |
| Best_Total | 0.3232 | 0.5356 | 0.5963 | 0.2098 | 0.5874 | 0.7061 |

## Per-class AP

| Modelo | Clase | AP@0.5:0.95 | AP@0.50 |
|--------|-------|:---:|:---:|
| Best_EMA | player | 0.5840 | 0.9656 |
| Best_EMA | goalkeeper | 0.3535 | 0.6548 |
| Best_EMA | referee | 0.4812 | 0.8506 |
| Best_Regular | player | 0.5695 | 0.9633 |
| Best_Regular | goalkeeper | 0.3401 | 0.6250 |
| Best_Regular | referee | 0.4589 | 0.8201 |
| Best_Total | player | 0.5840 | 0.9656 |
| Best_Total | goalkeeper | 0.3535 | 0.6548 |
| Best_Total | referee | 0.4812 | 0.8506 |

## Rendimiento

| Modelo | FPS | Tamaño (MB) |
|--------|:---:|:---:|
| Best_EMA | 17.4 | 368.8 |
| Best_Regular | 17.5 | 370.1 |
| Best_Total | 17.8 | 127.6 |

## Veredicto

**Mejor modelo (mAP@0.5:0.95):** `Best_EMA` con **0.4729**
