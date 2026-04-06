# Team Clustering Pipeline (Detalle Completo)

Este documento describe el pipeline actual de `eval_team_clustering.py` con el comportamiento por defecto

## 1. Objetivo

Separar jugadores de campo en dos equipos (A/B), asignar porteros al equipo correcto y mantener arbitros fuera del clustering de equipos, usando deteccion RF-DETR role-aware + tracking + logica temporal.

## 2. Entradas y Salidas

### Entradas principales
- Frames del clip (ejemplo: `data/test_sequences/SNMOT-116/`).
- Checkpoint RF-DETR 3 clases (`player`, `goalkeeper`, `referee`).
- Parametros CLI de evaluacion.

### Salidas principales
- CSV de diagnostico por secuencia en `outputs/team_clustering_debug/`.
- Reporte post-run por secuencia en `outputs/team_clustering_debug/`.
- Video anotado (si aplica) en `outputs/visualizations/`.

## 3. Pipeline End-to-End

1. Deteccion por frame:
- RF-DETR produce cajas, `class_id` y `confidence`.
- Las clases se normalizan al vocabulario role-aware.

2. Tracking temporal:
- ByteTrack enlaza detecciones entre frames.
- Cada track acumula historial de posicion, clase y atributos visuales.

3. Construccion de candidatos para clustering:
- Por defecto se excluyen `goalkeeper` y `referee` del set de clustering.
- Se usan outfield players para formar prototipos de equipo.

4. Clustering de equipos (A/B):
- Modo por defecto: `hsv`.
- Alternativa: `dbscan` (post-procesado + K=2 para identidad final).
- Identidad A/B se normaliza con reglas espaciales consistentes (evitar flips arbitrarios).

5. Asignacion de porteros:
- Modo por defecto: `fused`.
- Para cada candidato GK se calcula score por equipo:
  - afinidad espacial,
  - afinidad por vecinos,
  - afinidad visual debil.
- Se aplica consistencia temporal (histeresis + fallback) para reducir cambios espurios.

6. Rol de arbitros:
- Se etiquetan como `REF`.
- Quedan fuera del clustering de equipos por defecto.
- Solo entran al clustering si se activa flag explicito.

7. Export de diagnosticos:
- Decision final por track/ventana.
- `decision_confidence`, `reason_code`.
- Componentes fused por equipo.

## 4. Defaults Actuales

Comando base recomendado:

```bash
python eval_team_clustering.py --mode hsv --k 2
```

Defaults efectivos:
- Uso de separacion por clase de portero: activo.
- Modo asignacion GK: `fused`.
- Referee en clustering: desactivado.

Flags de override/debug:
- `--no-use-gk-class`: desactiva ruta role-aware para GK.
- `--gk-assignment-mode legacy`: usa heuristica legacy.
- `--gk-assignment-mode fused`: fuerza modo fused.
- `--cluster-referee`: incluye referee en clustering (solo debug/ablacion).

## 5. Modos de Asignacion GK

### Legacy
- Basado en heuristica espacial + movimiento + voto de vecinos.
- Util para comparabilidad historica y regression testing.

### Fused (default)
- Combina evidencia multi-cue (espacial, vecinos, visual debil).
- Mas robusto en escenas congestionadas y cambios de contexto.
- Mantiene fallback temporal cuando la evidencia instantanea es baja.

## 6. Mecanismos de Robustez

- Fallback seguro cuando no hay suficientes tracks outfield para clusterizar.
- Continuidad temporal de etiquetas para evitar oscilacion frame a frame.
- Soporte a taxonomias heterogeneas de salida RF-DETR (normalizacion de clases).

## 7. Artefactos de Diagnostico

Por secuencia, se esperan al menos:
- `*_gk_diagnostics.csv`
- `*_post_report.txt`

Campos clave a revisar:
- `decision_confidence`
- `reason_code`
- `fused_spatial_*`, `fused_neighbor_*`, `fused_appearance_*`, `fused_score_*`

## 8. Validacion Realizada

Flujo validado con checkpoint 3 clases en:
- SNMOT-116
- SNMOT-117
- SNMOT-143

Conclusiones operativas actuales:
- Ruta role-aware fused estable en los clips validados.
- Legacy y fused pueden coincidir en algunos clips, pero fused se mantiene como default por robustez general.

## 9. Fallas Tipicas y Como Depurarlas

1. Portero mal asignado en corner o area congestionada:
- Comparar `legacy` vs `fused`.
- Revisar componentes fused y `reason_code` en CSV.

2. Clustering inestable entre A/B:
- Verificar cantidad/calidad de tracks outfield.
- Probar `--mode dbscan` para inspeccion.

3. Referee contaminando equipos:
- Confirmar que no se uso `--cluster-referee` accidentalmente.

## 10. Comandos de Trabajo Rapido

```bash
# Default validado
python eval_team_clustering.py --mode hsv --k 2

# Ablacion legacy
python eval_team_clustering.py --mode hsv --k 2 --gk-assignment-mode legacy

# Desactivar uso de clase GK
python eval_team_clustering.py --mode hsv --k 2 --no-use-gk-class

# Incluir referee (solo diagnostico)
python eval_team_clustering.py --mode hsv --k 2 --cluster-referee
```

## 11. Relacion con Fases del Plan

- Phase 7: validacion y matriz de ablation (base lista y ejecutada en clips clave).
- Phase 8: rollout de defaults (completado).
- Phase 9: riesgos y mitigacion (siguiente foco: generalizacion en clips no vistos adicionales).
