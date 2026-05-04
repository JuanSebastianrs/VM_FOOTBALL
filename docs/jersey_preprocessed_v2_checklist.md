# Checklist de ejecucion - Jersey preprocessed v2

## 0) Objetivo

- Mantener 4 lineas comparables: `full`, `processed`, `preprocessed_full`, `preprocessed_processed`.
- Los 2 nuevos (`preprocessed_*`) deben ser persistentes y reanudables sobre su propio `checkpoint_last.pt`.

## 1) Validacion local de integridad

- [ ] Ejecutar validacion y reporte JSON:

```bash
python training/identification/check_and_package_preprocessed.py \
  --report-path tmp/preprocessed_dataset_validation_report.json
```

- [ ] Revisar reporte en `tmp/preprocessed_dataset_validation_report.json`:
  - `required_file_errors` debe quedar vacio.
  - `count_mismatches` debe quedar vacio.
  - `zero_bytes` debe ser 0 en train/test.

## 2) Empaquetado de datasets preprocesados

- [ ] Crear tarballs locales:

```bash
python training/identification/check_and_package_preprocessed.py \
  --create-archives \
  --archive-dir tmp \
  --archive-suffix v2
```

- [ ] Se deben generar:
  - `tmp/sn_jersey_2023_preprocessed_full_keepall_v2.tar.gz`
  - `tmp/sn_jersey_2023_preprocessed_processed_keepall_v2.tar.gz`

## 3) Subida a GCS

- [ ] Subir archivos al bucket:

```bash
gcloud storage cp tmp/sn_jersey_2023_preprocessed_full_keepall_v2.tar.gz gs://vm-football-data/
gcloud storage cp tmp/sn_jersey_2023_preprocessed_processed_keepall_v2.tar.gz gs://vm-football-data/
```

## 4) Configuracion persistente de experimentos

- [ ] Usar `cloud/jersey_number/config_preprocessed_v2.yaml`.
- [ ] Verificar nombres fijos de experimento:
  - `jersey_full_preprocessed_v2`
  - `jersey_processed_preprocessed_v2`
- [ ] Verificar output GCS fijo:
  - `gs://vm-football-data/models/jersey_number/jersey_preprocessed_v2`

## 5) Ejecutar job en Vertex (manual)

- [ ] Lanzar job (corrida completa):

```powershell
pwsh .\cloud\submit_jersey_job.ps1 -GPU T4 -ConfigFile config_preprocessed_v2.yaml
```

- [ ] Dry run opcional:

```powershell
pwsh .\cloud\submit_jersey_job.ps1 -GPU T4 -ConfigFile config_preprocessed_v2.yaml -DryRun
```

## 6) Re-ejecucion de los mismos 2 modelos

- [ ] Volver a correr con el mismo comando del paso 5.
- [ ] Esperado:
  - Se intenta descargar `checkpoint_last.pt` de cada experimento fijo.
  - Continua entrenamiento del mismo experimento.
  - `best.pt` solo cambia si mejora metrica.

## 7) Reporte comparativo final (4 modelos)

- [ ] Consolidar por modelo:
  - `best val top1`, `best val top3`
  - `best_epoch`
  - curva `train_loss`/`val_loss`
  - gap de generalizacion
- [ ] Emitir recomendacion final de integracion al pipeline.
