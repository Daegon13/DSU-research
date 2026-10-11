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

El preregistro quedó sellado en el commit `33a626ee06dae4318628369c3bf9f2854deee45e` antes del primer entrenamiento nuevo. El runner completó y persistió `80×109` y `160×49` con seeds 1–3, **seis corridas nuevas** de 25 épocas. Reutilizó los tres JSON `128×64` de S3-T07 tras revalidar hashes, configuración y curvas; no reentrenó el anchor. La prueba `resume` relee los seis JSON sin repetir épocas. Los resultados y curvas completos, con fingerprints y diagnósticos por seed, están en [el artefacto versionado](../data/s3_t11_confirmation_results.json); los JSON originales nuevos están en `runs/s3_t11_point_sparse_shortlist/`.

### Resultados finales por seed

Valores del checkpoint **época 25**, sin selección retrospectiva. `Val acc.` y `Test acc.` están en porcentaje; loss es CrossEntropy media. Todas las corridas conservan 9610 parámetros. `80×109`: 9520 MAC, 69760 bytes de topología y payload estimado 108200 bytes. `160×49`: 9440/62720/101160. `128×64`: 9472/65536/103976.

| Arquitectura | Seed | Train loss | Val loss | Val acc. | Test NLL | Test acc. |
|---|---:|---:|---:|---:|---:|---:|
| 80×109 | 1 | .308498 | .329622 | 88.317 | .371177 | 86.69 |
| 80×109 | 2 | .304921 | .347708 | 87.800 | .379859 | 86.57 |
| 80×109 | 3 | .305469 | .334889 | 88.417 | .370523 | 86.78 |
| 160×49 | 1 | .295356 | .319269 | 88.383 | .363762 | 86.92 |
| 160×49 | 2 | .299084 | .339565 | 88.167 | .369113 | 86.80 |
| 160×49 | 3 | .295891 | .320844 | 88.250 | .366997 | 86.69 |
| 128×64, S3-T07 | 1 | .300638 | .322322 | 88.533 | .371451 | 86.91 |
| 128×64, S3-T07 | 2 | .301687 | .338940 | 87.717 | .371541 | 87.04 |
| 128×64, S3-T07 | 3 | .303372 | .329999 | 88.267 | .373933 | 86.79 |

| Arquitectura | Val acc. media ± DE, pp | Test acc. media ± DE, pp | Val loss media ± DE | Test NLL media ± DE |
|---|---:|---:|---:|---:|
| 80×109 | 88.178 ± .331 | 86.680 ± .105 | .337406 ± .009302 | .373853 ± .005212 |
| 160×49 | **88.267 ± .109** | 86.803 ± .115 | **.326559 ± .011291** | **.366624 ± .002695** |
| 128×64 | 88.172 ± .416 | **86.913 ± .125** | .330420 ± .008317 | .372309 ± .001408 |

Las medias ± DE muestral de train loss, norma de logits, entropía predictiva, margen correcto, fracción de márgenes positivos, ECE-15 y duración están también en el JSON. El tiempo de entrenamiento+validation de las seis corridas nuevas sumó **1009.6 s** en este CPU, ~168.3 s por corrida; no representa latencia de inferencia. La norma de logits media por arquitectura fue 27.669/27.339/27.539, entropía .3342/.3281/.3325, margen correcto 4.138/4.166/4.158 y ECE-15 .01068/.01139/.01181 para 80/160/128 respectivamente. Estas medidas no muestran una mejora global inequívoca.

**Ranking validation final por medias:** `160×49 > 80×109 > 128×64`; **ranking test final por medias:** `128×64 > 160×49 > 80×109`. Los tres pares de validation media son near-ties según `<.20 pp`: el máximo supera al mínimo sólo .094 pp. En test, anchor supera a `160×49` por .110 pp y a `80×109` por .233 pp. El contraste `80×109 − anchor` es +.006 pp validation pero −.233 pp test; `160×49 − anchor` es +.094 pp validation y −.110 pp test, aunque mejora NLL por .005685. Es evidencia mixta entre métricas y seeds.

**H-WIDTH-FANIN = INCONCLUSIVE**, conforme a la regla congelada. Ningún contraste candidato–anchor muestra una diferencia de accuracy ≥.20 pp en validation **y** test con el mismo signo y victorias pareadas suficientes. Tampoco son todos los contrastes <.20 pp por test. El ancho/fan-in puede afectar el aprendizaje, pero estos tres brazos no establecen una ventaja final material bajo el protocolo fijado. El resultado de 3 épocas de T10 permanece como señal preliminar, no capacidad final confirmada.

### Predictividad histórica de épocas tempranas

Cada celda muestra `Spearman / pares correctos sin near-ties` para predecir **validation accuracy de época 25** desde validation accuracy de la época indicada. Se calculó sobre medias de tres seeds por arquitectura dentro de cada familia; las correlaciones por seed, Kendall tau-b, pares con near-ties y la misma extracción de validation loss temprana están en el JSON. Near-tie final: `<.20 pp`.

| Familia | Época 1 | Época 2 | **Época 3 primaria** | Época 5 diagnóstica |
|---|---|---|---|---|
| S3-T07, 5 brazos | 1.00 / 8/8 | .90 / 7/8 | **.90 / 7/8** | 1.00 / 8/8 |
| S3-T05, 4 brazos | .00 / 1/2 | −.40 / 0/2 | **−.20 / 1/2** | −.40 / 0/2 |
| S2-T06 MLP, 6 brazos | .60 / 8/10 | .029 / 6/10 | **.486 / 8/10** | .771 / 9/10 |

Frente a **test accuracy final**, época 3 dio S3-T07 `ρ=.70`, `7/9` pares no near-tie; S3-T05 `ρ=−.40`, sin pares puntuables porque sus seis pares de test fueron near-ties. S2-T06 no tiene test para todos sus brazos, por lo que se excluye de esta comparación. No se mezclaron estas familias ni se eligió época 5 por su resultado.

**EARLY-PILOT-3E = WEAK** por las reglas preregistradas: ayuda en S3-T07 y, más modestamente, MLP; S3-T05 es plana/inconsistente. La evidencia no respalda que tres épocas sean universalmente superiores a zero-cost ni que Stage 2 ordene correctamente arquitecturas cercanas.

### ER raw frente al piloto de 3 épocas

Con los mismos targets y grupos: en S3-T07, ER raw y piloto dieron `ρ=.90` contra validation final, pero ER acertó `8/8` pares frente a `7/8`; contra test ER dio `ρ=1.00`, `9/9`, frente al piloto `.70`, `7/9`. En S3-T05, ER dio `ρ=.00`, `1/2` contra validation y el piloto `−.20`, `1/2`; para test no hubo pares no near-tie. No hay ER completo para la cohorte MLP. Son contrastes históricos y no rehabilitan ER fuera de esos grupos.

La prueba **prospectiva T10→T11** usa los tres candidatos fijados. T10 predijo por piloto `80×109 > 128×64 > 160×49`. En **seed 1**, validation final ordenó `128×64 > 160×49 > 80×109`; test final `160×49 > 128×64 > 80×109`. En medias N=3, validation final ordenó `160×49 > 80×109 > 128×64`; test final `128×64 > 160×49 > 80×109`. El piloto tiene Spearman `−.50` frente a ambas medias finales; en validation los tres pares son near-ties y no hay pares puntuables, mientras en test acierta `0/1` par no near-tie. ER raw tiene Spearman `+.50` frente a ambas medias y `1/1` par test no near-tie, pero `0` pares puntuables en validation. Con sólo tres arquitecturas cercanas no se puede afirmar que ER generaliza: **H-ZEROCOST-ER sigue FAILED para la cohorte T10 completa**, y la victoria final de `160×49` en validation es un contraste exploratorio entre objetivos distintos. El caso observado combina elementos de B/C: el piloto no separó bien al anchor, y `160×49` mejora validation loss y NLL sin ser el mejor test accuracy.

### Coste, límites y decisión

T10 ejecutó 24 epoch-equivalents; T11 ejecutó **150 nuevos**. Total incremental del funnel: **174** frente a 600 de entrenar ingenuamente los ocho brazos con N=3/25 épocas, **71% menos**. Los **75 epoch-equivalents** del anchor provienen de S3-T07 y se reutilizan: el ahorro incremental depende de esa inversión histórica. Estas cuentas no equivalen a joules, tiempo total histórico ni ahorro de latencia.

Diseño futuro, sin implementar proxies: priorizar una medida de aprendizaje inicial que observe gradientes sobre train (norma por bloque y sensibilidad tipo SNIP con batch/seed fijos) y una pendiente de curva tras pocos minibatches; evaluar estabilidad entre seeds y coste antes de usarla para descartar modelos. SynFlow-like y estadísticas Jacobianas/NTK pueden quedar como controles si se fija de antemano su presupuesto computacional; no se ajustará una regresión con resultados de T11. La evidencia mixta de los extremos H/K aconseja controles que distingan escala de activación, entrenabilidad y redundancia, con una cohorte nueva si se valida un proxy futuro.

**Gate técnico S3-T11: PASS.** Shortlist, reutilización del anchor, seeds, análisis y criterios se sellaron antes de las corridas nuevas; se ejecutaron sólo seis corridas necesarias, se esperó hasta época 25 para test, se evaluó piloto vs final, se analizaron curvas históricas sin mezclar protocolos, no se fitteó un proxy y se registró el coste incremental. Limitaciones: N=3, diferencias pequeñas, splits distintos entre seeds aunque pareados dentro de seed, topologías y escalas dependientes de H/K, resultados S3-T07 reutilizados desde archivos locales cuyo contenido queda copiado en el artefacto final, distinta configuración de threads entre T07 y T11, test usado sólo tras selección, sin RAM pico/latencia/energía. No se declara arquitectura globalmente superior.

Las condiciones sugeridas de Sprint 3 están cubiertas a nivel de experimentos y resultados negativos: fan-in, storage, compartimentos estáticos, PointSparse, línea sparse H/K, framework prospectivo y límites de zero-cost, con programa A/B ya formalizado. **Recomendación: S3-REVIEW** como tarea separada para evaluar formalmente el cierre, resolver límites abiertos y decidir los inicios separados de Línea A/B. No se ejecutó esa revisión ni Line B.
