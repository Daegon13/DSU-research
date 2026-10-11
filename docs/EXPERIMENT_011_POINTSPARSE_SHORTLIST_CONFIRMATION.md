# S3-T11 — Confirmación de la shortlist PointSparse iso-presupuesto

## Preregistro anterior a las corridas nuevas

TASK ID: S3-T11. OBJECTIVE: confirmar la shortlist fija de T10 y contrastar el piloto de 3 épocas con ER zero-cost. WHY: ER raw falló en T10 y el piloto seleccionó tres arquitecturas. CONTEXT: ocho PointSparse de 9610 parámetros; sólo `80×109`, `160×49` y anchor `128×64` pasan a Stage 3. ALLOWED FILES: runner/análisis/tests T11, JSON de preregistro/resultados, este informe, framework, resultados, hipótesis, protocolo y estado. FORBIDDEN FILES: baselines, arquitectura T10, splits y cohortes. REQUIREMENTS: auditar reutilización, sellar regla antes de entrenar, seis corridas nuevas reanudables, test sólo después de época 25, análisis histórico por familia. OUT OF SCOPE: otros candidatos, tuning, fitting de proxies, Line B, revisión formal Sprint 3. ACCEPTANCE CRITERIA: tres arquitecturas × seeds 1–3 con resultados finales válidos, dos nuevas si anchor reutilizable, comparaciones piloto/final y costes. TESTS: auditoría, test firewall, resume, suite y diff check. EXPECTED OUTPUT: JSON por corrida, resultado agregado y clasificación metodológica. EXPECTED COMMIT: `experiment: preregister PointSparse shortlist confirmation`, luego `experiment: confirm PointSparse iso-budget shortlist`. STOP CONDITIONS: discrepancia material de anchor, artefacto T10/S3-T07 modificado, preregistro sin commit, acceso anticipado a test o corrida incompleta.

### Shortlist y protocolo

Congelada: **A `80×109`**, **B `160×49`**, **anchor `128×64`**. Cada una tiene 9610 parámetros. No entran `240×29`, `64×139` ni otro brazo. Seeds 1, 2, 3; `training_seed` y `topology_seed` son campos distintos aunque toman el mismo entero. Fashion-MNIST 54000/6000/10000, split NumPy por seed, píxeles `/255`, CPU float32, 25 épocas, batch 128, Adam `lr=.001`, betas `.9/.999`, epsilon `1e-7`, weight decay cero, CrossEntropy sobre logits y LeakyReLU 0.1. Se usa **`final_epoch_25`** sin escoger best epoch. Se guardan train loss y validation loss/accuracy por época; después de la época 25 se evalúa test una vez por brazo/seed fijado: loss/NLL, accuracy, norma de logits, entropía, margen correcto y ECE-15 con el evaluador S3-T07. Test no altera shortlist, hiperparámetros ni seeds. Cada JSON válido se conserva y revalida en resume.

Costes analíticos ya congelados: `80×109`: 9520 MAC principales, 69760 bytes de topología int64, payload tensorial estimado 108200 bytes; `160×49`: 9440, 62720, 101160; `128×64`: 9472, 65536, 103976. No son FLOPs, RAM pico, latencia ni energía medidas.

### Auditoría objetiva del anchor

El [artefacto de preregistro](../data/s3_t11_preregistered_confirmation.json) registra para cada seed el SHA-256 del JSON histórico S3-T07, del estado inicial reconstruido, de `G64`, de los índices train/validation y del orden de 25 épocas. `validate_record` de S3-T07 vuelve a verificar el registro original. `build_model(128,64,seed,seed)` de T10 devuelve la **misma clase PointSparse**, usa `base_topology(seed)`, `paired_values(seed)` y el mismo forward que `build_arm` de S3-T07. H=128, K=64, LeakyReLU 0.1, pesos/biases iniciales y G64 coinciden exactamente en seeds 1–3. Ambos protocolos fijan split, optimizer, LR, batch, 25 épocas y checkpoint final. En seed 1, T10 y S3-T07 tienen idénticos hashes de split y orden de minibatches de las tres primeras épocas, e idéntica validation accuracy por época. Las pérdidas difieren ligeramente a escala numérica CPU; S3-T07 no fijó número de threads, mientras T10 usó uno. No se observa diferencia material de arquitectura o protocolo. **Decisión preregistrada: reutilizar las tres corridas S3-T07, sin reentrenar el anchor.** No se ha usado la calidad final para decidirlo.

### Análisis temprano congelado

Proxy principal **EARLY-PILOT-3E = validation accuracy de época 3**. Épocas 1, 2 y 5 son diagnósticos secundarios, nunca candidatas retrospectivas a proxy principal. Targets: validation accuracy de época 25 y test accuracy final sólo cuando todos los brazos del grupo la tienen. Por familia, promediar cada arquitectura sobre seeds 1–3 en cada epoch y target; calcular Spearman, Kendall tau-b y pares, tanto con todos los pares como excluyendo `|Δaccuracy final|<0.20 pp` (medio crédito neutral al incluir near-ties). Inspeccionar estabilidad por seed por separado. No unir arquitecturas de familias distintas en una correlación global.

Grupos históricos fijados con curvas de 25 épocas completas: S3-T07 parameter-neutral (`PointSparse`, `1×64`, `2×32`, `4×16`, `8×8`), S3-T05 original (`1×64`, `2×32`, `4×16`, `8×8`) y S2-T06 MLP capacity (`h16`, `h19`, `h20`, `h21`, `h22`, `h25`). S2-T06 carece de test para toda la cohorte, así que sólo admite target final validation. Comparar ER raw con piloto de época 3 dentro de S3-T07/S3-T05 usando proxies T09, y dentro de la shortlist T11 usando los proxies T10. No inventar ER para los MLP faltantes ni fitear combinaciones.

La validación prospectiva crucial conserva el orden T10 de la shortlist en seed 1: **`80×109 > 128×64 > 160×49`**. Contrastarlo con validation de época 25 en seed 1 y medias N=3, y con test N=3. Las diferencias pequeñas y near-ties se muestran; no se elige el epoch que mejor correlacionó. ER no queda rehabilitado automáticamente si `160×49` termina ganando.

`EARLY-PILOT-3E = PROMISING` si al menos dos grupos históricos tienen Spearman ≥.50 y >.70 de pares correctos para validation final, sin grupo con Spearman negativo o <.50 en pares; `WEAK` si al menos dos grupos dan señal positiva pero hay inconsistencia material; `FAILED` en otro caso o si una familia es sistemáticamente inversa (Spearman ≤−.50 y pares ≤.30). Son reglas descriptivas internas con grupos pequeños, sin prueba formal de significancia. La shortlist T11 es prueba prospectiva aparte.

`H-WIDTH-FANIN = SUPPORTED_WITHIN_TESTED_RANGE` si algún candidato frente al anchor presenta `|Δ|≥.50 pp` de accuracy media tanto en validation como test, mismo signo, al menos 2/3 seeds en esa dirección y NLL sin oposición ≥.005. `WEAK_SUPPORT` requiere `|Δ|≥.20 pp` en ambas accuracies, mismo signo y 2/3 seeds, sin alcanzar el criterio fuerte. `NOT_SUPPORTED` si ambos contrastes tienen `|Δ|<.20 pp` en validation/test y `|ΔNLL|<.005`. Los demás patrones son `INCONCLUSIVE`. Esta clasificación cubre **sólo el rango ensayado de PointSparse bajo presupuesto fijo**.

### Contabilidad y siguiente decisión

Búsqueda ingenua: `8×25×3=600` epoch-equivalents. T10 realizó 24; T11 planifica **150** nuevos (`2×25×3`), total **174**, ahorro `426/600=71%` frente a la búsqueda ingenua, antes de costes futuros. Las 75 épocas/seed del anchor fueron ejecutadas históricamente en S3-T07: se reutilizan, no se vuelven a ejecutar ni se cuentan como nuevas. Informar por separado esta dependencia histórica; los epoch-equivalents no son energía.

Después de ver los resultados se diseñarán próximos proxies (gradientes, sensibilidad SNIP-like, SynFlow-like, Jacobiano/NTK aproximados o pendiente de curva), sin implementarlos ni ajustarlos en T11. Si los ocho asuntos de cierre de Sprint 3 están cubiertos, recomendar **S3-REVIEW** como tarea separada; no ejecutar la revisión aquí.

## Resultados y análisis

Pendientes del commit de preregistro y las seis corridas nuevas.
