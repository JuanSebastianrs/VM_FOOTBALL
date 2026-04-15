# Estabilización Temporal de la Calibración de Cámara para Reconstrucción 2D en Video de Fútbol

## Tabla de Contenidos

1. [Definición del Problema](#1-definición-del-problema)
2. [Modelo de Cámara: Fundamentos Matemáticos](#2-modelo-de-cámara-fundamentos-matemáticos)
3. [Análisis de Causa Raíz](#3-análisis-de-causa-raíz)
4. [Enfoques Considerados](#4-enfoques-considerados)
5. [Solución Implementada: Decompose → Smooth → Reconstruct](#5-solución-implementada-decompose--smooth--reconstruct)
6. [Referencia de Funciones](#6-referencia-de-funciones)
7. [Justificación de Decisiones de Diseño](#7-justificación-de-decisiones-de-diseño)

---

## 1. Definición del Problema

### 1.1 Contexto

El pipeline TacticalVision reconstruye una vista cenital (minimap 2D) del campo de fútbol a partir de video de transmisión broadcast. Para lograr esto, cada cuadro del video debe ser **calibrado espacialmente**: se necesita una función matemática que mapee coordenadas de píxel $(u, v)$ en la imagen a coordenadas métricas $(x_w, y_w)$ sobre el plano del campo (105 × 68 metros, norma FIFA).

El modelo **PnLCalib** (Points and Lines Calibration) realiza esta calibración detectando keypoints y líneas del campo en cada cuadro, y estimando los parámetros intrínsecos y extrínsecos de la cámara.

### 1.2 Síntoma Observado

Cuando la cámara de transmisión **rota horizontalmente** (pan) para seguir la acción del juego, el minimapa presenta **saltos bruscos** ("jumping"): los jugadores proyectados cambian de posición abruptamente entre cuadros consecutivos, a pesar de que en la realidad se mueven de forma suave y continua.

### 1.3 Impacto

Este artefacto visual invalida toda la reconstrucción 2D: heatmaps, diagramas de Voronoi, análisis táctico y cualquier métrica espacial derivada del minimapa se vuelven inutilizables.

---

## 2. Modelo de Cámara: Fundamentos Matemáticos

### 2.1 Modelo Pinhole y Matriz de Proyección

Una cámara broadcast se modela matemáticamente como una proyección perspectiva. Un punto 3D en coordenadas del mundo $\mathbf{X}_w = (X, Y, Z, 1)^T$ (coordenadas homogéneas) se proyecta a un punto 2D en la imagen $\mathbf{x} = (u, v, 1)^T$ mediante:

$$\mathbf{x} = \mathbf{P} \cdot \mathbf{X}_w$$

donde $\mathbf{P}$ es la **matriz de proyección** de dimensión $3 \times 4$, que se descompone como:

$$\mathbf{P} = \mathbf{K} \cdot [\mathbf{R} \mid -\mathbf{R}\mathbf{t}]$$

#### 2.1.1 Matriz de Calibración Intrínseca $\mathbf{K}$

$$\mathbf{K} = \begin{pmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{pmatrix}$$

donde:
- $f_x, f_y$: distancias focales en píxeles (horizontal, vertical)
- $(c_x, c_y)$: punto principal (generalmente el centro de la imagen)

En una cámara broadcast, $f_x \approx f_y$ (sensor cuadrado) y ambos valores son **casi constantes** durante una secuencia (el camarógrafo rara vez hace zoom).

#### 2.1.2 Parámetros Extrínsecos: Rotación $\mathbf{R}$ y Posición $\mathbf{t}$

- $\mathbf{R} \in SO(3)$: matriz de rotación $3 \times 3$ (ortonormal, $\det(\mathbf{R}) = 1$)
- $\mathbf{t} \in \mathbb{R}^3$: posición de la cámara en coordenadas del mundo

La matriz de transformación extrínseca completa es:

$$[\mathbf{R} \mid \mathbf{d}] = \begin{pmatrix} R_{00} & R_{01} & R_{02} & d_0 \\ R_{10} & R_{11} & R_{12} & d_1 \\ R_{20} & R_{21} & R_{22} & d_2 \end{pmatrix}, \quad \mathbf{d} = -\mathbf{R}\mathbf{t}$$

### 2.2 Descomposición de la Rotación en Ángulos de Euler

La matriz de rotación $\mathbf{R}$ se descompone en tres ángulos de Euler que tienen interpretación física directa para una cámara broadcast:

$$\mathbf{R}^T = \mathbf{R}_{\text{pan}} \cdot \mathbf{R}_{\text{tilt}} \cdot \mathbf{R}_{\text{roll}}$$

donde $\mathbf{R}^T$ es la **orientación** (transpuesta de la rotación):

$$\mathbf{R}_{\text{pan}}(\phi) = \begin{pmatrix} \cos\phi & -\sin\phi & 0 \\ \sin\phi & \cos\phi & 0 \\ 0 & 0 & 1 \end{pmatrix}$$

$$\mathbf{R}_{\text{tilt}}(\theta) = \begin{pmatrix} 1 & 0 & 0 \\ 0 & \cos\theta & -\sin\theta \\ 0 & \sin\theta & \cos\theta \end{pmatrix}$$

$$\mathbf{R}_{\text{roll}}(\psi) = \begin{pmatrix} \cos\psi & -\sin\psi & 0 \\ \sin\psi & \cos\psi & 0 \\ 0 & 0 & 1 \end{pmatrix}$$

- **Pan** ($\phi$): rotación horizontal — cambia cuando la cámara sigue la acción izquierda-derecha
- **Tilt** ($\theta$): inclinación vertical — ángulo entre la cámara y el plano horizontal
- **Roll** ($\psi$): rotación sobre el eje óptico — casi siempre $\approx 0$ (el camarógrafo mantiene la cámara nivelada)

La extracción de ángulos desde $\mathbf{R}$ se realiza mediante (implementada en `rotation_matrix_to_pan_tilt_roll`):

$$\theta = \arccos(R^T_{2,2})$$

$$\phi = \arctan2\left(\text{sgn}(\sin\theta) \cdot R^T_{0,2},\ -\text{sgn}(\sin\theta) \cdot R^T_{1,2}\right)$$

$$\psi = \arctan2\left(\text{sgn}(\sin\theta) \cdot R^T_{2,0},\ \text{sgn}(\sin\theta) \cdot R^T_{2,1}\right)$$

> [!NOTE]
> Existen dos soluciones ($\theta$ y $-\theta$). PnLCalib selecciona la solución con **mínimo roll**, ya que los camarógrafos profesionales mantienen la cámara nivelada.

### 2.3 Homografía del Plano del Campo (Z=0)

Para proyectar jugadores al minimap, solo necesitamos el mapeo sobre el **plano del campo** ($Z = 0$). Sustituyendo $Z = 0$ en $\mathbf{P} \cdot \mathbf{X}_w$:

$$\mathbf{x} = \mathbf{P} \begin{pmatrix} X \\ Y \\ 0 \\ 1 \end{pmatrix} = \begin{pmatrix} P_{:,0} & P_{:,1} & P_{:,3} \end{pmatrix} \begin{pmatrix} X \\ Y \\ 1 \end{pmatrix} = \mathbf{H} \begin{pmatrix} X \\ Y \\ 1 \end{pmatrix}$$

donde la **homografía** $\mathbf{H}$ es la submatriz $3 \times 3$ formada por las columnas 0, 1 y 3 de $\mathbf{P}$ (eliminando la columna de $Z$).

Para ir de la imagen al mundo (dirección que necesitamos), usamos la **homografía inversa**:

$$\begin{pmatrix} X_w \\ Y_w \\ 1 \end{pmatrix} \sim \mathbf{H}^{-1} \begin{pmatrix} u \\ v \\ 1 \end{pmatrix}$$

donde $\sim$ denota igualdad hasta un factor de escala (se normaliza dividiendo por la tercera componente).

### 2.4 Sistema de Coordenadas de PnLCalib

PnLCalib usa un sistema con **origen en el centro del campo**:

$$X_w \in [-52.5, +52.5], \quad Y_w \in [-34.0, +34.0]$$

Para convertir a coordenadas absolutas del campo (origen en esquina superior izquierda):

$$x_{\text{pitch}} = X_w + 52.5, \quad y_{\text{pitch}} = Y_w + 34.0$$

---

## 3. Análisis de Causa Raíz

### 3.1 Causa Raíz #1: PnLCalib es un calibrador de imagen única — sin memoria temporal

> [!CAUTION]
> Este es el problema fundamental. PnLCalib fue diseñado para el benchmark SoccerNet Camera Calibration (imágenes independientes), **NO para video**.

#### El método `heuristic_voting()` — anatomía del problema

Para cada cuadro, `heuristic_voting()` ejecuta internamente **18 combinaciones** de parámetros:

| `mode` | Keypoints usados | `ransac` threshold |
|--------|------------------|--------------------|
| `full` | 57 keypoints + 16 auxiliares + 23 líneas | 0, 5, 10, 15, 25, 50 |
| `ground_plane` | Solo keypoints del plano Z=0 | 0, 5, 10, 15, 25, 50 |
| `main` | 30 keypoints principales | 0, 5, 10, 15, 25, 50 |

Cada combinación ejecuta `cv2.calibrateCamera()` independientemente y produce un **conjunto completo de parámetros** $(\mathbf{K}, \mathbf{R}, \mathbf{t})$. El método selecciona la combinación con menor error de reproyección:

$$e_{\text{reproj}} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \|\mathbf{x}_i - \hat{\mathbf{x}}_i\|^2}$$

donde $\hat{\mathbf{x}}_i = \pi(\mathbf{P}, \mathbf{X}_{w,i})$ es la proyección del punto 3D $\mathbf{X}_{w,i}$ usando la calibración estimada.

**El problema**: dos cuadros consecutivos pueden seleccionar combinaciones completamente diferentes. Por ejemplo:

| Cuadro | Combinación ganadora | $f_x$ | pan | $e_{\text{reproj}}$ |
|--------|---------------------|--------|-----|---------------------|
| $N$ | `full, ransac=0` | 1200 | +15° | 3.2 px |
| $N+1$ | `ground_plane, ransac=25` | 850 | +22° | 2.8 px |

Ambas tienen bajo error de reproyección, pero parámetros **radicalmente diferentes**. Esto produce un salto de $+7°$ en pan y $-350$ en focal entre dos cuadros consecutivos a 25 fps.

#### El loop de `heuristic_voting()` en pseudocódigo

```
para cada mode en {full, ground_plane, main}:
    para cada ransac en {0, 5, 10, 15, 25, 50}:
        cam_params, err = calibrate(mode, ransac)   ← cv2.calibrateCamera()
        resultados.append((cam_params, err))

ordenar resultados por (err, mode)

si existe (full, ransac=0) con err ≤ 5px:
    retornar ese resultado          ← preferencia por el caso "limpio"
sino:
    retornar el de menor error      ← PUEDE SER CUALQUIER combinación
```

### 3.2 Causa Raíz #2: Suavizado EMA sobre $\mathbf{H}^{-1}$ — matemáticamente inválido

El código original implementaba:

$$\mathbf{H}^{-1}_{\text{smooth}} = \alpha \cdot \mathbf{H}^{-1}_{\text{new}} + (1 - \alpha) \cdot \mathbf{H}^{-1}_{\text{old}}$$

#### ¿Por qué esto es incorrecto?

Una homografía $\mathbf{H}$ pertenece al **grupo proyectivo** $PGL(3)$, no a un espacio vectorial euclidiano. La interpolación lineal de dos matrices en $PGL(3)$ **no produce una transformación proyectiva válida** en general.

Para ilustrar, consideremos dos homografías que representan rotaciones puras de $+10°$ y $-10°$ respectivamente:

$$\mathbf{H}_1 = \mathbf{K} \mathbf{R}(+10°) \mathbf{K}^{-1}, \quad \mathbf{H}_2 = \mathbf{K} \mathbf{R}(-10°) \mathbf{K}^{-1}$$

La interpolación lineal $\frac{1}{2}(\mathbf{H}_1 + \mathbf{H}_2)$ **no es igual** a $\mathbf{K} \mathbf{R}(0°) \mathbf{K}^{-1}$ (la identidad). De hecho, la interpolación lineal puede:

1. **Distorsionar las líneas rectas** del campo (las convierte en curvas)
2. **Mover el punto de fuga** a una posición inconsistente
3. **Colapsar la transformación** si $\mathbf{H}_1$ y $\mathbf{H}_2$ son suficientemente diferentes (determinante cercano a cero)

Para cambios muy pequeños entre cuadros ($\Delta\phi < 0.5°$), la interpolación lineal es una **aproximación de primer orden** razonable por la linealidad local de $PGL(3)$. Pero es exactamente durante **paneos rápidos** ($\Delta\phi > 2°$/cuadro) cuando la aproximación falla — y es exactamente cuando se necesita.

### 3.3 Causa Raíz #3: Sin rechazo de calibraciones anómalas

Cuando PnLCalib no detecta suficientes keypoints (imagen borrosa durante paneo rápido, o la cámara apunta a una zona del campo con pocas líneas), puede retornar:

1. `None` → el código usa `last_valid_H_inv` → **correcto pero insuficiente**
2. Un resultado con params incorrectos pero $e_{\text{reproj}}$ bajo (pocos puntos → pocos residuos → bajo error) → **se acepta sin filtrar** → salto

No existía ningún mecanismo para validar que la calibración del cuadro $N+1$ sea **geométricamente consistente** con la del cuadro $N$.

---

## 4. Enfoques Considerados

### 4.1 Enfoque A: Suavizado EMA mejorado sobre $\mathbf{H}^{-1}$ (descartado)

**Idea**: Mantener el suavizado directo pero con $\alpha$ adaptativo y normalización de $\mathbf{H}^{-1}$ después de cada interpolación.

**Razón de descarte**: Incluso con normalización ($\mathbf{H} / \mathbf{H}_{2,2}$), la interpolación lineal sigue sin ser geométricamente significativa. No hay forma de garantizar que $\alpha \mathbf{H}_1 + (1-\alpha) \mathbf{H}_2$ preserve el significado geométrico de la transformación para diferencias grandes.

### 4.2 Enfoque B: Filtro de Kalman sobre $\mathbf{H}^{-1}$ (descartado)

**Idea**: Tratar los 9 elementos de $\mathbf{H}^{-1}$ como un vector de estado en $\mathbb{R}^9$ y aplicar un filtro de Kalman con modelo de velocidad constante.

**Razón de descarte**:
- Los 9 elementos de $\mathbf{H}$ no son independientes (la homografía tiene 8 grados de libertad, no 9)
- El modelo dinámico lineal ($\mathbf{H}_{t+1} = \mathbf{H}_t + \dot{\mathbf{H}} \cdot \Delta t$) hereda el mismo problema de la interpolación lineal en $PGL(3)$
- La covarianza de proceso sería no trivial de definir en este espacio

### 4.3 Enfoque C: Integración de CMC (Camera Motion Compensation) (evaluado, no prioritario)

**Idea**: Usar las transformaciones afines inter-cuadro estimadas por ORB features (ya computadas en Phase 2: CMC) para propagar la calibración del cuadro anterior.

**Razón de no-priorización**: CMC estima la transformación **relativa** 2D entre cuadros consecutivos, pero no proporciona calibración **absoluta**. Solo podría servir como consistencia check secundario, pero PnLCalib ya provee calibración absoluta que es lo que necesitamos. Añadir CMC incrementaría la complejidad sin resolver el problema fundamental (el ruido de PnLCalib). Se puede integrar en el futuro como verificación cruzada.

### 4.4 Enfoque D ✅: Decompose → Smooth → Reconstruct (seleccionado)

**Idea**: Descomponer cada resultado de PnLCalib en sus parámetros naturales, suavizar cada uno independientemente en su espacio correcto, y reconstruir la proyección.

**Razón de selección**:
- Los parámetros de cámara ($\phi, \theta, \psi, f_x, f_y, \mathbf{t}$) **sí** habitan espacios donde la interpolación es válida
- Los ángulos se suavizan con EMA circular (maneja wraparound)
- Las distancias focales y posición son escalares/vectores euclidianos → EMA estándar es correcto
- Cada parámetro puede tener su propio $\alpha$ reflejando su dinámica real
- Es el enfoque estándar en sistemas de tracking de cámara broadcast (Vizrt, ChyronHego, Hawk-Eye)

### 4.5 Sub-decisión: Fallback a `heuristic_voting_ground()` (incluido)

**Idea**: Cuando `heuristic_voting()` falla completamente (retorna `None`), usar `heuristic_voting_ground()` que estima solo la homografía del plano del campo.

**Razón de inclusión**: `heuristic_voting_ground()` es más robusto que `heuristic_voting()` en ciertos escenarios porque:
- Solo necesita correspondencias en el plano $Z=0$ (más numerosas)
- Usa `cv2.findHomography()` en vez de `cv2.calibrateCamera()` (menos parámetros → mejor condicionado)
- Itera 6 valores de RANSAC (vs 18 combinaciones del full)

El costo es una calibración intrínseca menos precisa (no usa el modelo PnP completo), pero como los parámetros se suavizan temporalmente, las imprecisiones instantáneas se filtran.

---

## 5. Solución Implementada: Decompose → Smooth → Reconstruct

### 5.1 Arquitectura General

```
Cuadro N
   │
   ├──► HRNet (keypoints) ──┐
   │                         ├──► FramebyFrameCalib ──► heuristic_voting()
   ├──► HRNet (lines) ──────┘          │
   │                                   ▼
   │                          ¿Resultado válido?
   │                           │            │
   │                          SÍ           NO
   │                           │            │
   │                           │    heuristic_voting_ground()
   │                           │            │
   │                           ▼            ▼
   │                    cam_params     cam_params (from cam.R, cam.t, cam.K)
   │                           │            │
   │                           └──────┬─────┘
   │                                  ▼
   │                    ┌─── CameraParamsSmoother ───┐
   │                    │                             │
   │                    │  1. Descomponer:            │
   │                    │     φ,θ,ψ,fx,fy,cx,cy,t    │
   │                    │                             │
   │                    │  2. Validar consistencia:   │
   │                    │     rate-of-change check    │
   │                    │     + reproj error check    │
   │                    │                             │
   │                    │  3. Si OK → EMA smooth      │
   │                    │     Si NO → hold state      │
   │                    │                             │
   │                    │  4. Reconstruir H_inv       │
   │                    │     desde params suavizados │
   │                    └─────────────────────────────┘
   │                                  │
   │                                  ▼
   │                          H_inv (suavizado)
   │                                  │
   ▼                                  ▼
 Detections ───────────► project_point(u,v, H_inv) ──► Minimap 2D
```

### 5.2 Paso 1: Descomposición de Parámetros

Dado el diccionario `cam_params` de PnLCalib:

```python
pan  = deg2rad(cam_params['pan_degrees'])     # φ ∈ (-π, π]
tilt = deg2rad(cam_params['tilt_degrees'])     # θ ∈ [0, π]
roll = deg2rad(cam_params['roll_degrees'])     # ψ ∈ (-π, π]
fx   = cam_params['x_focal_length']            # px
fy   = cam_params['y_focal_length']            # px
cx   = cam_params['principal_point'][0]        # px
cy   = cam_params['principal_point'][1]        # px
pos  = cam_params['position_meters']           # (x, y, z) metros
```

Para el path de fallback (`heuristic_voting_ground()`), los parámetros se extraen de los **side-effects** del objeto `FramebyFrameCalib`:

```python
# Después de cam.heuristic_voting_ground(), internamente from_homography()
# establece cam.rotation, cam.position, cam.calibration

pan, tilt, roll = rotation_matrix_to_pan_tilt_roll(cam.rotation)
fx = cam.calibration[0, 0]
fy = cam.calibration[1, 1]
pos = cam.position
```

### 5.3 Paso 2: Validación de Consistencia (Outlier Gate)

Antes de incorporar nuevos parámetros al estado suavizado, se aplica un doble filtro:

#### 5.3.1 Filtro de Error de Reproyección

$$\text{RECHAZAR si } e_{\text{reproj}} > \tau_{\text{reproj}}$$

Con $\tau_{\text{reproj}} = 20$ píxeles. Si PnLCalib reporta un error de reproyección alto, la calibración es inherentemente poco confiable independientemente de la tasa de cambio.

#### 5.3.2 Filtro de Tasa de Cambio (Rate-of-Change)

Para cada parámetro, se verifica que el cambio entre el valor nuevo y el valor suavizado actual no exceda un umbral físicamente plausible:

| Parámetro | Umbral máximo/cuadro | Justificación física |
|-----------|----------------------|-----------------------|
| Pan ($\Delta\phi$) | 5° | A 25 fps, 5°/cuadro = 125°/s = paneo muy rápido |
| Tilt ($\Delta\theta$) | 3° | Tilt cambia más lentamente que pan |
| Roll ($\Delta\psi$) | 2° | Roll casi nunca cambia en broadcast |
| Focal ($\Delta f / f$) | 20% | Zoom súbito del 20% es extremadamente raro |
| Posición ($\|\Delta\mathbf{t}\|$) | 20 m | Cámara en trípode fijo, no se mueve |

Para ángulos, el cambio se calcula con **diferencia circular** para manejar el wraparound en $\pm\pi$:

$$\Delta\phi = \text{wrap}(\phi_{\text{new}} - \phi_{\text{smooth}})$$

$$\text{wrap}(\delta) = ((\delta + \pi) \mod 2\pi) - \pi$$

Si **cualquier** parámetro excede su umbral, el cuadro completo se rechaza.

#### 5.3.3 Mecanismo de Recuperación

Si se acumulan $N_{\text{max}} = 15$ rechazos consecutivos, se **fuerza la aceptación** con $\alpha$ alto (0.8) para permitir la reconvergencia. Esto maneja el escenario donde la cámara genuinamente se ha movido a una nueva posición (e.g., un corte de cámara o un paneo muy largo).

```
Estado del gate:
  reject_count = 0  →  aceptar normalmente (α = α_nominal)
  reject_count < 15 →  rechazar, hold estado suavizado
  reject_count ≥ 15 →  force-accept (α = 0.8), reset reject_count
```

#### 5.3.4 Período de Warm-up

Los primeros $W = 3$ cuadros se aceptan incondicionalmente con $\alpha = 1.0$ (sin suavizado) para establecer el estado inicial rápidamente. Sin warm-up, los primeros cuadros serían rechazados por el gate (no hay estado previo contra el cual comparar).

### 5.4 Paso 3: Suavizado EMA por Parámetro

#### 5.4.1 EMA Estándar (para escalares euclidianos)

Para $f_x, f_y, c_x, c_y$ y cada componente de $\mathbf{t}$:

$$\hat{p}_{t} = \alpha \cdot p_t + (1 - \alpha) \cdot \hat{p}_{t-1}$$

donde:
- $p_t$: valor observado (ruidoso) en el cuadro $t$
- $\hat{p}_{t-1}$: valor suavizado hasta el cuadro $t-1$
- $\hat{p}_t$: nuevo valor suavizado
- $\alpha$: peso de la nueva observación ($0 < \alpha \leq 1$)

#### 5.4.2 EMA Circular (para ángulos)

Para pan ($\phi$), tilt ($\theta$), roll ($\psi$), la EMA estándar puede producir resultados incorrectos cerca de los puntos de wraparound (e.g., $+179°$ y $-179°$ están a $2°$ de distancia, no a $358°$).

La **EMA circular** resuelve esto:

$$\hat{\phi}_t = \hat{\phi}_{t-1} + \alpha \cdot \text{wrap}(\phi_t - \hat{\phi}_{t-1})$$

Expandiendo:

$$\hat{\phi}_t = \hat{\phi}_{t-1} + \alpha \cdot \left[ \left((\phi_t - \hat{\phi}_{t-1} + \pi) \bmod 2\pi\right) - \pi \right]$$

Esto garantiza que la actualización siempre toma el **camino más corto** alrededor del círculo.

#### 5.4.3 Valores de $\alpha$ por Familia de Parámetros

| Familia | $\alpha$ | Razón |
|---------|----------|-------|
| Ángulos (pan, tilt, roll) | 0.25 | Deben ser responsivos al paneo real de la cámara pero rechazar saltos |
| Focales ($f_x, f_y, c_x, c_y$) | 0.05 | Casi constantes → $\alpha$ muy bajo para máxima estabilidad |
| Posición ($t_x, t_y, t_z$) | 0.10 | Constante (trípode fijo) pero PnLCalib es ruidoso en posición |

La elección de $\alpha = 0.25$ para ángulos representa un compromiso: la ventana efectiva de promediado es $\sim 1/\alpha = 4$ cuadros. A 25 fps, esto son 160 ms — suficiente para filtrar ruido cuadro-a-cuadro pero no tanto como para crear lag visible en el minimap.

### 5.5 Paso 4: Reconstrucción de $\mathbf{H}^{-1}$ desde Parámetros Suavizados

Una vez suavizados todos los parámetros, se reconstruye la cadena completa:

#### 5.5.1 Orientación → Rotación

```python
orientation = pan_tilt_roll_to_orientation(φ_smooth, θ_smooth, ψ_smooth)
R = orientation^T
```

Matemáticamente:

$$\mathbf{O} = \mathbf{R}_{\text{pan}}(\hat\phi) \cdot \mathbf{R}_{\text{tilt}}(\hat\theta) \cdot \mathbf{R}_{\text{roll}}(\hat\psi)$$

$$\mathbf{R} = \mathbf{O}^T$$

#### 5.5.2 Construcción de $\mathbf{P}$

$$\mathbf{K} = \begin{pmatrix} \hat{f}_x & 0 & \hat{c}_x \\ 0 & \hat{f}_y & \hat{c}_y \\ 0 & 0 & 1 \end{pmatrix}$$

$$\mathbf{I}_t = \begin{pmatrix} 1 & 0 & 0 & -\hat{t}_x \\ 0 & 1 & 0 & -\hat{t}_y \\ 0 & 0 & 1 & -\hat{t}_z \end{pmatrix}$$

$$\mathbf{P} = \mathbf{K} \cdot \mathbf{R} \cdot \mathbf{I}_t$$

#### 5.5.3 Extracción de $\mathbf{H}^{-1}$

$$\mathbf{H} = (\mathbf{P}_{:,0} \mid \mathbf{P}_{:,1} \mid \mathbf{P}_{:,3})$$

$$\mathbf{H}^{-1} = \mathbf{H}^{-1}$$

> [!IMPORTANT]
> Aquí $\mathbf{H}^{-1}$ está **garantizado** ser una homografía válida, porque fue construido a partir de una cámara con parámetros físicamente plausibles (ángulos suaves, focal estable, posición coherente). Esto es cualitativamente diferente de interpolar dos $\mathbf{H}^{-1}$ arbitrarios.

### 5.6 Proyección Final: Imagen → Minimap

Dado un punto de pie de jugador $(u, v)$ en la imagen:

$$\begin{pmatrix} X_w \\ Y_w \\ w \end{pmatrix} = \mathbf{H}^{-1}_{\text{smooth}} \begin{pmatrix} u \\ v \\ 1 \end{pmatrix}$$

$$X_w \leftarrow X_w / w, \quad Y_w \leftarrow Y_w / w$$

$$x_{\text{pitch}} = X_w + 52.5, \quad y_{\text{pitch}} = Y_w + 34.0$$

$$u_{\text{minimap}} = \lfloor x_{\text{pitch}} \cdot s \rfloor + m, \quad v_{\text{minimap}} = \lfloor y_{\text{pitch}} \cdot s \rfloor + m$$

donde $s = 8$ px/m es la escala del minimap y $m = 40$ px es el margen.

**Validación de bounds**: si $x_{\text{pitch}} \notin [-5, 110]$ o $y_{\text{pitch}} \notin [-5, 73]$, el punto se descarta (está fuera del campo con un margen de tolerancia de 5 m).

---

## 6. Referencia de Funciones

### 6.1 Clase `CameraParamsSmoother`

| Método | Entrada | Salida | Descripción |
|--------|---------|--------|-------------|
| `__init__()` | — | — | Inicializa estado interno a `None`, contadores a 0 |
| `_wrap(diff)` | $\delta \in \mathbb{R}$ | $\delta' \in [-\pi, \pi)$ | Wrap de diferencia angular |
| `_circular_ema(old, new, α)` | ángulos, peso | ángulo suavizado | EMA circular: `old + α·wrap(new-old)` |
| `_is_consistent(φ,θ,ψ,fx,fy,t,err)` | params nuevos | `(bool, str)` | Gate de consistencia: verifica reproj error + rate-of-change |
| `update(cam_params, rep_err)` | dict PnLCalib, error | $\mathbf{H}^{-1}_{3\times3}$ | Core: descompone → valida → suaviza → reconstruye |
| `_build_H_inv()` | (usa estado interno) | $\mathbf{H}^{-1}_{3\times3}$ | Reconstruye H_inv desde params suavizados |
| `get_current_H_inv()` | — | $\mathbf{H}^{-1}_{3\times3}$ o `None` | Retorna H_inv actual sin actualizar |
| `summary()` | — | `str` | Estadísticas: aceptados/rechazados/force-accepted |

### 6.2 Funciones Auxiliares

| Función | Descripción |
|---------|-------------|
| `build_P_from_cam_params(cam_params)` | Construye $\mathbf{P}_{3\times4}$ desde un dict de PnLCalib |
| `ground_homography_from_P(P)` | Extrae $\mathbf{H}^{-1}$ (ground plane) desde $\mathbf{P}$ |
| `draw_pitch_cv2(scale, margin)` | Renderiza minimap del campo con OpenCV |
| `project_point(u, v, H_inv, ...)` | Proyecta pixel imagen → pixel minimap |

### 6.3 Funciones de PnLCalib (vendor, no modificadas)

| Función | Archivo | Descripción |
|---------|---------|-------------|
| `rotation_matrix_to_pan_tilt_roll(R)` | `utils_calib.py` | $\mathbf{R} \to (\phi, \theta, \psi)$ con min-roll |
| `pan_tilt_roll_to_orientation(φ,θ,ψ)` | `utils_calib.py` | $(\phi, \theta, \psi) \to \mathbf{R}^T$ |
| `FramebyFrameCalib.heuristic_voting()` | `utils_calib.py` | Calibración 3D completa (18 combos) |
| `FramebyFrameCalib.heuristic_voting_ground()` | `utils_calib.py` | Calibración ground-plane (6 combos) |
| `FramebyFrameCalib.get_cam_params()` | `utils_calib.py` | Core: PnP via `cv2.calibrateCamera` |
| `FramebyFrameCalib.get_homography_from_ground_plane()` | `utils_calib.py` | Core: homografía via `cv2.findHomography` |
| `FramebyFrameCalib.from_homography()` | `utils_calib.py` | Extrae $\mathbf{K}, \mathbf{R}, \mathbf{t}$ desde $\mathbf{H}$ (Alg. 8.2, Hartley & Zisserman) |

### 6.4 Pipeline del Main Loop

```
POR CADA cuadro:
  1. Leer imagen
  2. Inferencia HRNet → keypoints + lines
  3. cam.update(kp_dict, lines_dict)
  4. result = cam.heuristic_voting(refine_lines=True)
  5. Si result ≠ None:
       cam_params = result['cam_params']
       rep_err = result['rep_err']
     Sino:
       result_g = cam.heuristic_voting_ground(refine_lines=True)
       Si result_g ≠ None:
         cam_params = extraer de cam.R, cam.t, cam.K
         rep_err = result_g['rep_err']
       Sino:
         cam_params = None   ← fallback
  6. Si cam_params ≠ None:
       H_inv = smoother.update(cam_params, rep_err)
     Sino:
       H_inv = smoother.get_current_H_inv()
  7. Proyectar jugadores y balón con H_inv → minimap
  8. Renderizar frame combinado (video + minimap) al video de salida
```

---

## 7. Justificación de Decisiones de Diseño

### 7.1 ¿Por qué descomponer en params en vez de suavizar $\mathbf{H}^{-1}$?

| Criterio | EMA sobre $\mathbf{H}^{-1}$ | EMA sobre parámetros descompuestos |
|----------|----------------------------|------------------------------------|
| Validez matemática | ❌ No (interpolación en $PGL(3)$) | ✅ Sí (cada param en su espacio natural) |
| Granularidad de $\alpha$ | Un solo $\alpha$ global | Un $\alpha$ por familia de params |
| Outlier detection | Solo verificar `None` vs no-`None` | Rate-of-change por parámetro |
| Resultado garantizado válido | ❌ (puede producir $\mathbf{H}$ singular) | ✅ (siempre producido desde camera model) |
| Complejidad | Trivial | Moderada |

### 7.2 ¿Por qué EMA y no Kalman?

Un filtro de Kalman (KF) sería teóricamente superior porque:
- Modela explícitamente la incertidumbre de la observación
- Puede usar un prior de velocidad ($\dot\phi, \dot\theta$)
- Produce intervalos de confianza

Sin embargo, para esta aplicación:
- La covarianza de observación de PnLCalib es **desconocida** y varía drásticamente entre cuadros (depende de cuántos keypoints detectó, cuál `mode` ganó, etc.)
- El modelo dinámico de la cámara broadcast no es estrictamente de velocidad constante (el camarógrafo acelera y desacelera)
- La EMA con outlier gating logra el mismo efecto práctico con mucha menos complejidad
- Si en el futuro se quiere pasar a KF, la descomposición en params ya está hecha — solo se reemplaza el step de EMA

### 7.3 ¿Por qué el fallback a `heuristic_voting_ground()`?

`heuristic_voting()` puede fallar (retornar `None`) cuando:
1. Se detectan < 4 keypoints con confianza suficiente
2. Los keypoints caen en una configuración degenerada (colineales)
3. `cv2.calibrateCamera()` no converge

`heuristic_voting_ground()` es más robusto porque:
- Solo necesita correspondencias 2D→2D (no 3D→2D)
- `cv2.findHomography()` solo requiere 4 puntos y es numéricamente más estable que `cv2.calibrateCamera()` (4 DoF vs 6 DoF para extrínsecos + 2 DoF para intrínsecos)
- Las líneas del campo son features extensas que se detectan incluso con motion blur moderado

El tradeoff es que la calibración intrínseca extraída vía `from_homography()` (Algoritmo 8.2 de Hartley & Zisserman) es más ruidosa que la de `cv2.calibrateCamera()`. Pero como el smoother aplica $\alpha_f = 0.05$ a los focales, este ruido se filtra efectivamente.

### 7.4 ¿Por qué mantener `refine_lines=True`?

PnLCalib ofrece un refinamiento post-calibración usando **Levenberg-Marquardt** que minimiza simultáneamente:

$$\mathcal{L} = (1-\alpha_{\text{opt}}) \cdot \|\mathbf{e}_{\text{kp}}\|^2 + \alpha_{\text{opt}} \cdot \|\mathbf{e}_{\text{lines}}\|^2$$

donde:
- $\mathbf{e}_{\text{kp}}$: error de reproyección de keypoints
- $\mathbf{e}_{\text{lines}}$: distancia punto-a-línea para las líneas detectadas
- $\alpha_{\text{opt}} = 0.7$: peso relativo (favorece las líneas)

Este refinamiento mejora la precisión absoluta a costa de ~50-100ms/cuadro. Como el usuario priorizó **precisión sobre latencia**, se mantiene activado. Se puede desactivar con `--disable_pnl_refine` para procesamiento más rápido.

### 7.5 ¿Por qué los umbrales específicos elegidos?

Los umbrales del gate fueron derivados de la cinemática real de una cámara broadcast profesional:

| Parámetro | Velocidad máxima real | A 25fps | Umbral elegido | Factor de seguridad |
|-----------|----------------------|---------|----------------|---------------------|
| Pan | ~100°/s (paneo rápido) | 4°/frame | 5°/frame | 1.25× |
| Tilt | ~50°/s | 2°/frame | 3°/frame | 1.5× |
| Roll | ~5°/s (raro) | 0.2°/frame | 2°/frame | 10× (muy conservador) |
| Focal | Casi nunca cambia | ~0% | 20% | Muy conservador |
| Posición | 0 (trípode fijo) | 0 m | 20 m | Maneja ruido de PnLCalib |

Los factores de seguridad altos garantizan que **nunca se rechace una calibración correcta**. El gate solo filtra saltos que son físicamente imposibles.

---

> [!TIP]
> **Para debugging futuro**: Ejecutar con `--debug` imprime cada 25 cuadros los ángulos suavizados, la fuente de calibración (voting/ground/hold), y el contador de rechazos. Si el minimap sigue inestable, el primer paso es revisar estos logs para determinar si el gate está aceptando demasiados outliers (bajar umbrales) o rechazando demasiados frames buenos (subir umbrales).
