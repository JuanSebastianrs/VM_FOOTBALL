# Plan B: Saneamiento del Sistema de Identificación de Dorsales (Jersey Number)

> **Fecha:** 2026-05-12
> **Contexto:** Tras la implementación inicial (v1.1) y la primera ronda de correcciones de bugs críticos (v1.2), se realizó una auditoría que reveló deficiencias arquitectónicas en la evaluación, el postproceso (Hungarian global) y la validación del equipo (team clustering). Este documento establece el plan detallado para sanear el módulo de identificación de dorsales antes de su integración final en la tesis.

---

## 1. Estado Actual (Snapshot Post-v1.2)

### 1.1. Lo que funciona
- **Dataset construido:** `datasets/jersey_tracking_v1/` con 1,742 tracklets, 41,795 crops, extraídos de SoccerNet Tracking.
- **Modelo entrenado:** EfficientNet-B0 con MIL (Multi-Instance Learning) y cabezales composicionales (length, tens, ones).
- **Rutas de crops corregidas:** Ya no hay duplicación de prefijos ni zero-fill silencioso en el loader.
- **Lógica de lock corregida:** Usa la confianza del número asignado por Hungarian, no el máximo global.
- **Pipeline integrado:** `src/tactical_vision_pipeline.py` acepta `--jersey_model` y ejecuta Phase 10 opcionalmente.

### 1.2. Métricas actuales (checkpoint `runs/jersey_digit_mil_v2/best.pt`)

#### Dataset-native (val / test)
| Split | Raw Top-1 | Raw Top-3 | Assigned | Locked Acc | False Lock |
|-------|-----------|-----------|----------|------------|------------|
| Val   | 57.1%     | 72.9%     | 55.7%    | 88.4%      | 11.6%      |
| Test  | 27.3%     | 42.3%     | 36.2%    | 67.5%      | 32.5%      |

#### End-to-end SNMOT-148 (pipeline ↔ GT, IoU matched)
| Version | Roster | Threshold | GT Raw Top-1 | GT Assigned | GT Locked Acc | GT False Lock | GT Locked Cov | Frag Raw Top-1 |
|---------|--------|-----------|--------------|-------------|---------------|---------------|---------------|----------------|
| v4      | per-team | p1=0.60 | —            | —           | —             | —             | —             | 32.6%          |
| v5      | per-team | p1=0.85 | 44.4%        | 38.9%       | 75.0%         | 25.0%         | 44.4%         | 32.6%          |
| **v6 (canon)** | per-team | p1=0.85 | **44.4%**    | **38.9%**   | **75.0%**     | **25.0%**     | **44.4%**     | 32.6%          |

**Análisis:** El modelo lee números razonablemente bien (raw top-1 de 57% en val, 27% en test). El postproceso original con **Hungarian global por equipo degradaba masivamente** la salida (4.4% en test). Tras el rediseño (roster mask + top-1 + resolución temporal de duplicados), el assigned accuracy subió a **36.2% en test** y **55.7% en val** (post-migración con `all_frame_ids`). Sobre SNMOT-148 end-to-end, la conservación de threshold (`p1=0.85`) mejora la **locked accuracy GT-level del 66.7% (fragment) al 75.0% (por jugador)** y reduce el **false lock rate del 33.3% al 25.0%**. El artefacto canónico `SNMOT-148_jersey_identity.json` ahora se genera directamente con `p1=0.85`, `--seed 42` y determinismo garantizado.

### 1.3. Bugs críticos identificados en auditoría
1. **Colisión de `track_id` entre secuencias:** `infer_jersey_on_dataset.py` guarda asignaciones en un dict por `track_id` únicamente. Como los `track_id` se repiten entre secuencias (hasta 80 veces), las asignaciones de una secuencia pueden pisar las de otra.
2. **Evaluación que oculta misses:** `evaluate_jersey_tracklets.py` evalúa solo la intersección Pred/GT. Los tracklets GT sin predicción no penalizan el denominador, inflando métricas.
3. **Modelo duplicado y carga permisiva:** `DigitCompositionalMIL` está definido en 3 archivos distintos y se carga con `strict=False`. Un cambio en la arquitectura de entrenamiento puede pasar desapercibido en inferencia.
4. **Resume training roto:** Al hacer resume, `best_acc` se reinicia a 0.0, pudiendo sobrescribir el mejor checkpoint con uno peor.
5. **Zero-fill en inferencia:** `jersey_identity_phase.py` aún rellena con tensores de ceros cuando un crop no se encuentra, en lugar de marcar el tracklet como `unknown`.
6. **Team ID desconocido mapeado a team 0:** `jersey_assignment.py:295` envía `team_id=-1` al grupo del equipo 0, contaminando sus asignaciones.

---

## 2. Objetivo del Plan B

Transformar el módulo de dorsales de un **prototipo funcional pero frágil** a un **sistema auditable y thesis-defensible** con las siguientes características:

1. **Métricas desglosadas y justas:** Separar claramente la capacidad del modelo (`raw`) del beneficio del roster (`oracle/external`) y del postproceso (`assigned`).
2. **Postproceso robusto:** Reemplazar el Hungarian global rígido por una máscara de roster + resolución de duplicados temporal.
3. **Validación de equipo (Team Audit):** Antes de aplicar cualquier restricción por equipo, verificar que el clustering de equipos es correcto mediante matching IoU/Hungarian contra GT.
4. **Infraestructura saneada:** Un único modelo definido, checkpoints con estado completo, y manejo de errores sin zero-fill.

---

## 3. Fases de Implementación

### Fase 1: Arreglar Infraestructura de Evaluación y Claves
**Archivos afectados:** `scripts/infer_jersey_on_dataset.py`, `scripts/evaluate_jersey_tracklets.py`

- **Corregir claves:** Todas las asignaciones y resultados deben usar la tupla `(sequence, track_id)` como clave primaria.
- **Modos de evaluación explícitos:**
  - `mode=dataset_native`: Predicciones y GT comparten `track_id` (evaluación actual del dataset construido).
  - `mode=sequence_coverage`: Predice sobre una secuencia del pipeline; reporta coverage (% de tracklets GT que tienen predicción) y accuracy sobre los emparejados.
  - `mode=end_to_end_iou`: Predice sobre el pipeline; empareja Pred↔GT por IoU+Hungarian por frame; luego compara dorsales. **Este es el modo más realista y debe ser el objetivo final.**
- **Reportar métricas separadas:** `raw_top1`, `raw_top3`, `assigned_top1`, `roster_top1`.

### Fase 2: Extraer Rosters desde la Fuente de Verdad
**Archivos afectados:** `training/identification/build_tracking_jersey_dataset.py` (o nuevo script)

- Parsear `gameinfo.ini` por secuencia.
- Extraer dorsales únicos numéricos por lado (`left` / `right`).
- Excluir no-numéricos (`Y`, `W`, `A`, etc.).
- Guardar `rosters.json` con estructura:
  ```json
  {
    "SNMOT-148": {
      "left": [1, 4, 7, 10, ...],
      "right": [2, 5, 9, 21, ...]
    }
  }
  ```
- Validar que los tamaños sean razonables (11-18 por equipo). Si hay más, investigar si hay datos de entrenamiento mezclados o cambios no documentados.

### Fase 3: Auditar Team Clustering (Pre-requisito para usar Roster)
**Archivos afectados:** Nuevo script `scripts/audit_team_clustering.py`

- Leer `*_team_assignments.json` (predicción del pipeline) y `gt/gt.txt` + `gameinfo.ini` (GT).
- Por frame, emparejar bboxes Pred↔GT usando IoU + Hungarian.
- Construir matriz de confusión `pred_team_id` (0, 1, -1) vs `gt_team_side` (`left`, `right`, `goalkeeper`, `referee`).
- Resolver el mapping arbitrario `pred_team_id -> gt_team_side` usando Hungarian 2x2 sobre la matriz de votos.
- Reportar:
  - `team_purity`: % de tracklets emparejados donde el equipo predicho (mapeado) coincide con GT.
  - `gk_detection_rate`: % de porteros GT detectados y clasificados correctamente.
  - `referee_detection_rate`: % de árbitros detectados.
  - `coverage`: % de jugadores GT que tienen un track Pred emparejado.
- **Regla de Oro:** Si `team_purity < 90%`, NO aplicar roster por equipo todavía; el problema es upstream.

### Fase 4: Rediseñar Postproceso (Roster Mask + Duplicate Resolution Temporal)
**Archivos afectados:** `core/identity/jersey_assignment.py`, `core/identity/jersey_identity_phase.py`

#### 4.1. Roster Mask (en vez de Hungarian global)
- Opción A (`external_roster`): El usuario pasa `--roster_json`. Se enmascaran probabilidades: si un número no está en el roster del equipo, su probabilidad se fuerza a 0. Se re-normaliza.
- Opción B (`oracle_roster`): Para evaluación interna, se lee el roster de `gameinfo.ini` del mismo partido. Esto es un **upper bound**; no se reporta como métrica principal, solo como análisis.
- Opción C (`no_roster`): Evaluación base sobre 1-99.

#### 4.2. Duplicate Resolution (temporal, no global)
- **Problema actual:** Hungarian obliga a que todos los tracklets de un equipo tengan dorsales distintos. ByteTrack fragmenta jugadores, así que un mismo jugador puede tener 2-3 `track_id`. Forzar unicidad destruye predicciones correctas.
- **Solución:**
  - Dos tracklets con el mismo dorsal y **frames solapados** → probable duplicado. Mantener el de mayor confianza (`locked` o `tentative`); el otro pasa a `unknown`.
  - Dos tracklets con el mismo dorsal y **frames disjuntos** → probable mismo jugador fragmentado. Permitir duplicado.
  - Más de 16 dorsales distintos en un equipo a lo largo del partido → permitir (cambios de jugadores).

#### 4.3. Lock/Confidence
- Calibrar `p1_threshold` y `margin_threshold` barriendo sobre el split de validación.
- Objetivo: minimizar `false_lock_rate` (dorsal equivocado marcado como `locked`), no maximizar accuracy bruto.
- Reportar curva `locked_coverage` vs `locked_accuracy`.

### Fase 5: Centralizar Modelo y Arreglar Checkpoints
**Archivos afectados:** Nuevo `core/identity/jersey_model.py`, `training/identification/train_jersey_digit_mil.py`, `scripts/infer_jersey_on_dataset.py`, `core/identity/jersey_identity_phase.py`

- Mover la definición de `DigitCompositionalMIL` y las transforms a `core/identity/jersey_model.py`.
- Todos los scripts de entrenamiento e inferencia deben importar desde allí.
- Cambiar `load_state_dict(..., strict=False)` a `strict=True`.
- El resume de entrenamiento debe guardar y restaurar: `model.state_dict()`, `optimizer.state_dict()`, `scheduler.state_dict()`, `epoch`, `best_acc`, `random_state`.
- Guardar checkpoints periódicos cada 5 épocas como ya se hace, pero ahora con estado completo.

### Fase 6: Integrar Roster Opcional al Pipeline End-to-End
**Archivos afectados:** `src/tactical_vision_pipeline.py`, `core/identity/jersey_identity_phase.py`

- Añadir `--roster_json` opcional al pipeline.
- Si se proporciona, `jersey_identity_phase.py` aplica la máscara de roster antes de la selección de top-k.
- Si no se proporciona, el pipeline funciona en modo `no_roster` (base 1-99).
- Permitir mapeo de `pred_team_id -> left/right` ya sea por audit automático o por argumento `--team_mapping`.

### Fase 7: Re-evaluación Completa y Reporte
- Re-entrenar (o reanudar desde el mejor checkpoint actual) solo si se detecta que la arquitectura cambió.
- Evaluar en Val, Test y SNMOT-148 con las nuevas métricas desglosadas.
- Generar tablas comparativas:
  - `raw_top1` vs `roster_top1` vs `assigned_top1`.
  - Impacto del team audit (si el equipo está mal, el roster no ayuda).
- Actualizar `AGENTS.md` con resultados y changelog v1.3.

---

## 4. Métricas Finales a Reportar

| Métrica | Definición | Es métrica principal? |
|---|---|---|
| `raw_top1_accuracy` | `alternatives[0] == GT`, sin roster ni Hungarian | **Sí** |
| `raw_top3_accuracy` | GT dentro de top-3 alternativas, sin roster | **Sí** |
| `roster_top1_accuracy` | Top-1 restringido al roster GT del equipo (oracle) | No (upper bound) |
| `external_roster_top1` | Top-1 restringido a roster externo conocido | Sí, si se dispone del roster |
| `assigned_accuracy` | Predicción final tras lock/duplicados == GT | Sí, pero solo si lock está calibrado |
| `locked_coverage` | % de tracklets con estado `locked` | Diagnóstico |
| `locked_accuracy` | Accuracy solo sobre tracklets `locked` | Diagnóstico |
| `false_lock_rate` | % de `locked` que son incorrectos | **Sí, crítica** |
| `team_purity` | % de tracks con equipo correcto (tras mapping) | **Sí, pre-requisito** |
| `end_to_end_accuracy_iou` | Accuracy tras matching IoU Pred↔GT | **Sí, objetivo final** |

---

## 5. Criterios de Éxito

1. **Métricas separadas:** `raw_top1` y `assigned_accuracy` se reportan en columnas distintas, nunca mezcladas.
2. **Roster como mejora, no como salvavidas:** `raw_top1` debe ser aceptable por sí solo (actualmente 57% val, 27% test). El roster debe mejorarlo, no ocultar un modelo roto.
3. **No más degradación extrema:** `assigned_accuracy` no debe caer de 27% a 4%. El rediseño de postproceso debe mantenerlo por encima del 15-20% como mínimo.
4. **Team audit > 90%:** Si el clustering de equipos no alcanza 90% de pureza, se debe corregir antes de aplicar roster por equipo.
5. **Infraestructura sólida:** Un solo `jersey_model.py`, checkpoints completos, sin zero-fill, sin `strict=False`.

---

## 6. Decisiones Pendientes (para discutir en próxima sesión)

1. **¿Entrenar de nuevo?** El modelo actual (v2) parece suficiente para raw top-1. Si centralizamos el modelo y no cambiamos la arquitectura, podemos reusar `best.pt`. Si añadimos dropout, attention mejorada, etc., se requiere reentrenamiento.
2. **¿Roster externo vs Oracle?** Para la tesis, ¿dispondremos de los dorsales de los equipos antes de procesar un partido? Si la respuesta es "no siempre", el modo `external_roster` es secundario y `raw_top1` sigue siendo la métrica principal.
3. **¿Incluir portero en roster?** En `gameinfo.ini`, los porteros tienen su propio descriptor. El roster puede separar `field_players` y `goalkeeper`, o dejarlos juntos. Esto afecta la máscara.

---

## 7. Archivos a Crear / Modificar

| Acción | Archivo |
|---|---|
| Crear | `docs/jersey_identification_plan_b.md` (este documento) |
| Crear | `scripts/audit_team_clustering.py` |
| Crear | `core/identity/jersey_model.py` (modelo centralizado) |
| Modificar | `scripts/infer_jersey_on_dataset.py` (claves `(seq, tid)`, modos de eval) |
| Modificar | `scripts/evaluate_jersey_tracklets.py` (métricas separadas, modos) |
| Modificar | `core/identity/jersey_assignment.py` (roster mask, duplicate temporal) |
| Modificar | `core/identity/jersey_identity_phase.py` (zero-fill, roster opcional) |
| Modificar | `training/identification/train_jersey_digit_mil.py` (checkpoint completo, import centralizado) |
| Modificar | `src/tactical_vision_pipeline.py` (`--roster_json`, `--team_mapping`) |
| Modificar | `AGENTS.md` (changelog v1.3 tras completar) |

---

## 8. Notas para el Agente Futuro

- **No asumir que `team_id` 0/1 significa left/right.** Siempre usar `team_mapping_pred_to_gt` o el audit de Fase 3.
- **No cambiar nombres de columnas en CSVs de salida** sin actualizar todos los scripts downstream.
- **Validar siempre sobre `SNMOT-148`** antes de declarar una fase como completa.
- **Nunca usar `strict=False`** al cargar checkpoints del modelo de dorsales.
- **Nunca hacer zero-fill de crops faltantes**; mejor marcar `unknown`.

---

## 9. Checklist Operativo para Retomar

### 9.1. Estado del Repositorio y Worktree
- Gran parte del código jersey está **untracked**: `core/identity/*.py`, `training/identification/*.py`, `scripts/*.py`.
- `runs/detect/` and `runs/segment/` are in `.gitignore`, but `runs/jersey_digit_mil_*` is NOT. Add to `.gitignore` to avoid committing checkpoints.
- **Checkpoint actual:** `runs/jersey_digit_mil_v2/best.pt` (30 épocas, ~17 MB). No se versiona automáticamente.
- **Decisión de almacenamiento:** Mantener `best.pt` local en `runs/jersey_digit_mil_v2/` y documentar su hash/tamaño. Si se necesita compartir, copiar a `models/jersey_digit_mil_v2.pt` (también ignorado por `.gitignore`) o subir a artefacto externo.

### 9.2. Comandos Reproducibles

**Entrenamiento (desde cero o resume):**
```bash
python training/identification/train_jersey_digit_mil.py \
    --dataset_dir datasets/jersey_tracking_v1 \
    --output_dir runs/jersey_digit_mil_v2 \
    --epochs 30 --batch_size 16 --K 16 \
    --device cuda:0 --num_workers 0 --lr 1e-4
```

**Inferencia dataset-native (val/test):**
```bash
# Val
python scripts/infer_jersey_on_dataset.py \
    --dataset_dir datasets/jersey_tracking_v1 \
    --model_path runs/jersey_digit_mil_v2/best.pt \
    --split val \
    --output_json outputs/jersey_eval/val_predictions.json \
    --K 16 --device cuda:0

# Test
python scripts/infer_jersey_on_dataset.py \
    --dataset_dir datasets/jersey_tracking_v1 \
    --model_path runs/jersey_digit_mil_v2/best.pt \
    --split test \
    --output_json outputs/jersey_eval/test_predictions.json \
    --K 16 --device cuda:0
```

**Evaluación dataset-native:**
```bash
python scripts/evaluate_jersey_tracklets.py \
    --pred_json outputs/jersey_eval/val_predictions.json \
    --gt_json datasets/jersey_tracking_v1/tracklets.json \
    --output_dir outputs/jersey_eval/val_results
```

**Pipeline end-to-end (SNMOT-148):**
```bash
python src/tactical_vision_pipeline.py \
    --sequence_dir data/tracking/SoccerNet/tracking/test/test/SNMOT-148 \
    --yolo_weights models/yolo26.pt \
    --rfdetr_weights models/models_rfdetr_player_gk_ref_rfdetr_base_448_3class_checkpoint_best_total.pth \
    --sam2_weights models/sam2.1_hiera_small.pt \
    --output_dir outputs/SNMOT-148 \
    --jersey_model runs/jersey_digit_mil_v2/best.pt \
    --roster_json datasets/jersey_tracking_v1/rosters.json \
    --team_mapping outputs/SNMOT-148/team_audit.json
```

**Inferencia sobre SNMOT-148 ya procesado:**
```bash
python core/identity/jersey_identity_phase.py \
    --sequence_dir data/tracking/SoccerNet/tracking/test/test/SNMOT-148 \
    --detections_json outputs/SNMOT-148/SNMOT-148_detections.json \
    --team_assignments_json outputs/SNMOT-148/SNMOT-148_team_assignments.json \
    --model_path runs/jersey_digit_mil_v2/best.pt \
    --output_json outputs/SNMOT-148/SNMOT-148_jersey_identity.json \
    --K 16 --device cuda:0
```

### 9.3. Inventario de Artifacts Generados

| Artifact | Ruta | Descripción |
|---|---|---|
| Dataset crops | `datasets/jersey_tracking_v1/crops/` | 41,795 JPGs extraídos |
| Tracklets JSON | `datasets/jersey_tracking_v1/tracklets.json` | 1,742 tracklets con rutas y GT |
| Splits JSON | `datasets/jersey_tracking_v1/splits.json` | Train/val/test por gameID |
| Metadata CSV | `datasets/jersey_tracking_v1/metadata.csv` | Vista plana sin listas anidadas |
| Checkpoint best | `runs/jersey_digit_mil_v2/best.pt` | Mejor modelo (val jersey_acc 57.1%) |
| Checkpoint final | `runs/jersey_digit_mil_v2/final.pt` | Último epoch |
| Log entrenamiento | `runs/jersey_digit_mil_v2/train.log` | Pérdida y métricas por epoch |
| Pred val | `outputs/jersey_eval/val_predictions.json` | Inferencia sobre 70 val tracklets |
| Pred test | `outputs/jersey_eval/test_predictions.json` | Inferencia sobre 792 test tracklets |
| Eval val | `outputs/jersey_eval/val_final2/jersey_evaluation_summary.json` | Métricas raw + assigned |
| Eval test | `outputs/jersey_eval/test_final2/jersey_evaluation_summary.json` | Métricas raw + assigned |
| Pipeline SNMOT-148 | `outputs/SNMOT-148/SNMOT-148_jersey_identity.json` | Salida end-to-end sobre 41 tracklets |

### 9.4. Detalles del Matching `end_to_end_iou`

Para la evaluación más realista (Fase 1, modo `end_to_end_iou`):

1. **Por frame:** Emparejar bboxes Pred↔GT usando IoU + Hungarian.
   - Umbral IoU sugerido: `0.5` (ajustable a `0.3` si hay muchos misses).
   - Excluir `ball` y `referee` del matching jersey.
2. **Voto por tracklet:** Para cada `track_id` Pred, contar cuántos frames emparejados coinciden con cada `track_id` GT. El GT con mayor cantidad de matches es el `matched_gt_id`.
3. **Manejo de fragmentación:** Si un `track_id` Pred se empareja con más de un GT a lo largo del tiempo (cambio de track), dividirlo en sub-tracklets por segmentos temporales contiguos.
4. **Cobertura:** Reportar `% de tracklets GT con al menos un match Pred`.
5. **Jersey comparison:** Solo comparar dorsales entre pares `(pred_track_id, matched_gt_id)`.

### 9.5. Schema Esperado para Duplicados Temporales

Para que `jersey_assignment.py` pueda resolver duplicados por solapamiento temporal, la salida de inferencia debe incluir:

```json
{
  "tracklets": [
    {
      "track_id": 5,
      "sequence": "SNMOT-148",
      "team_id": 0,
      "predicted_number": 21,
      "state": "tentative",
      "frame_ids": [120, 125, 130, ...],
      "start_frame": 120,
      "end_frame": 340,
      "alternatives": [[21, 0.54], [20, 0.33], ...]
    }
  ]
}
```

Sin `frame_ids` o rango temporal, no se puede determinar si dos tracklets se solapan.

### 9.6. Filtros por Rol

En todas las etapas de evaluación y roster:
- **Incluir:** `player`, `goalkeeper` (si su dorsal es numérico).
- **Excluir:** `ball`, `referee`, `team_id=-1`, `team_id=-2`, labels no numéricos (`Y`, `W`, `A`, `B`, etc.).
- **Opcional:** Reportar porteros por separado si se quiere analizar su precisión aparte.

### 9.7. Reproducibilidad

- **Split val:** Se calcula con hash MD5 del nombre de secuencia (`hashlib.md5(seq.encode()).hexdigest()`), no aleatorio. Es determinista.
- **Semillas:** Actualmente no se fijan semillas en entrenamiento. Si se requiere reproducibilidad estricta, agregar al inicio de `train_jersey_digit_mil.py`:
  ```python
  random.seed(42)
  np.random.seed(42)
  torch.manual_seed(42)
  torch.cuda.manual_seed_all(42)
  torch.backends.cudnn.deterministic = True
  ```
- **Entorno:** PyTorch 2.5.1+cu121, CUDA 12.1, RTX 2080 (11GB).
- **Metadata de entrenamiento:** Se recomienda guardar `training_metadata.json` con: `args`, `epoch`, `best_acc`, `timestamp`, `pytorch_version`, `cuda_version`.

### 9.8. Riesgo de Roster Oracle (Recordatorio)

- `oracle_roster` lee `gameinfo.ini` del **mismo partido** que se está evaluando. Esto es un **upper bound** teórico.
- **Nunca reportar `oracle_roster` como métrica principal** en la tesis. Usarlo solo para análisis interno ("¿cuánto ganaríamos si tuviéramos el roster perfecto?").
- La métrica principal siempre debe ser `raw_top1_accuracy` (modelo solo) o `external_roster_top1` (si se dispone de roster externo real).

### 9.9. Consistencia Crop Entrenamiento vs Inferencia

Actualmente tanto `build_tracking_jersey_dataset.py` como `jersey_identity_phase.py` calculan el torso crop con la misma lógica (10%-90% horizontal, 10%-70% vertical). Sin embargo, están duplicados. Si se cambia uno, el otro puede divergir. **Centralizar la función `extract_torso_crop` en `core/identity/crop_utils.py` (o similar) y usarla en ambos.**

### 9.10. Estado de Fases del Plan B

| Fase | Estado | Notas |
|------|--------|-------|
| Fase 1: Infraestructura de evaluación | **Completada** | Claves `(seq, tid)`, evaluación dataset-native y end-to-end IoU implementadas. |
| Fase 2: Extraer rosters | **Completada** | `scripts/extract_rosters.py` genera `rosters.json` con 106 secuencias. |
| Fase 3: Auditar team clustering | **Completada** | SNMOT-148: `assigned_purity=100%`. Roster per-team es seguro. |
| Fase 4: Rediseñar postproceso | **Completada** | Roster mask + top-1 + resolución temporal de duplicados activa. |
| Fase 5: Centralizar modelo | **Completada** | `core/identity/jersey_model.py` es fuente única de verdad. |
| Fase 6: Integrar roster al pipeline | **Completada** | `--roster_json` y `--team_mapping` en pipeline y `jersey_identity_phase.py`. |
| Fase 7: Re-evaluación y reporte | **Completada** | Métricas finales en val/test y SNMOT-148 documentadas arriba. |
| Post-audit (v1.3.3/v1.3.4) | **Completada** | GT-level aggregation, full temporal range, per-frame crop labels, schema alignment. |

### 9.11. Próximos Pasos Recomendados (post-cierre)
1. **Auditoría visual de crops locked:** Revisar `jersey_crops_v6/incorrect` para identificar modos de fallo (calidad de crop, orientación, confusión de dígitos, swap de equipo).
2. **Calibración de threshold por split:** Barrer `p1_threshold` sobre val para encontrar el punto óptimo locked_accuracy/coverage, y aplicar ese threshold a test.
3. **Reentrenamiento opcional:** Si la auditoría visual muestra que el cuello de botella es el modelo (no el postproceso), considerar reentrenar con dropout, data augmentation o arquitectura alternativa.
4. **Demo final:** Generar video overlay con `jersey_visual_qa.py` ocultando estados `tentative`/`unknown` para un demo más limpio.
5. **Ejecutar pipeline completo:** Validar que `src/tactical_vision_pipeline.py` con `--run_team_audit` reproduce estos mismos números en un directorio limpio.
