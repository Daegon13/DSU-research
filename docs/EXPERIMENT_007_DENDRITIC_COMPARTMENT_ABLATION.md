# EXPERIMENT_007 — Ablación de compartimentalización dendrítica (S3-T05)

## Ficha de tarea

| Campo | Valor |
|---|---|
| TASK ID | S3-T05 |
| OBJECTIVE | Comparar `1×64`, `2×32`, `4×16`, `8×8` con las mismas 64 entradas por soma. |
| WHY | Probar H-011 separando la partición dendrítica de la cobertura de entradas. |
| CONTEXT | S3-T04 cerrado; DSU-S v0 permanece intacta. |
| ALLOWED FILES | Clase y runner experimentales, config, tests, este informe, `RESULTS.md`, `HYPOTHESES.md`, diseño Sprint 3 y `PROJECT_STATE.md`. |
| FORBIDDEN FILES | DSU-S original, baselines, otros datasets, estado temporal y kernels. |
| REQUIREMENTS | G64 compartida, particiones nested, inicialización y minibatches pareados, 12 corridas, artefactos reanudables. |
| OUT OF SCOPE | Tuning, seeds 4–5, latencia, DSU-T y S3-T06. |
| ACCEPTANCE CRITERIA | Las 12 corridas válidas completan época 25; hashes pareados y tests pasan; H-011 se clasifica por umbrales prefijados. |
| TESTS | Suite completa, `git diff --check`, inspección de JSON y `git status`. |
| EXPECTED OUTPUT | Informe R-012, curvas por época y JSON por arquitectura/seed más resumen. |
| EXPECTED COMMIT | `experiment: ablate dendritic compartmentalization` (si se crea commit). |
| STOP CONDITIONS | Topología, inicialización o minibatches no pareados; métricas inválidas; bug metodológico. |

## Hipótesis y diseño congelado

H-011 pregunta si distribuir exactamente 64 conexiones input por soma entre más compartimentos no lineales mejora la capacidad de clasificación de esta arquitectura. `G64[s]` contiene 64 índices distintos tomados uniformemente sin reemplazo de `0..783` por `np.random.default_rng(topology_seed)`. Cada modelo reinterpreta la misma secuencia `[128,64]` en bloques contiguos `[128,B,64/B]`. Una rama de B=2 es la unión ordenada de dos de B=4, y una de B=4 de dos de B=8. No se vuelven a muestrear aristas por brazo. La variante `4×16` usa cobertura sin solapamiento dentro del soma y **no reutiliza** ni representa exactamente la topología histórica de DSU-S v0.

Se mantienen 128 somas, 784 entradas, 10 logits, LeakyReLU 0.1 en dendritas y somas, y almacenamiento compacto. Los sesgos empiezan en cero. Cada brazo tiene 8192 pesos input→dendrita. El número de pesos y sesgos dendríticos y somáticos cambia con B:

| Brazo | Dendritas | Parámetros entrenables | MAC principales/muestra | Ratio MAC vs 1×64 |
|---|---:|---:|---:|---:|
| 1×64 | 128 | 9866 | 9600 | 1.0000 |
| 2×32 | 256 | 10122 | 9728 | 1.0133 |
| 4×16 | 512 | 10634 | 9984 | 1.0400 |
| 8×8 | 1024 | 11658 | 10496 | 1.0933 |

Parámetros = `8192 + 128B + 128B + 128 + 1280 + 10 = 9610 + 256B`. MAC principales = `8192 + 128B + 1280 = 9472 + 128B`. Los MAC excluyen activaciones, indexación y backward. La conectividad input queda igualada, los parámetros totales y las no linealidades no. Por tanto una diferencia de calidad no aísla exclusivamente la no linealidad local.

## Inicialización y control pareado

Para cada training seed, un `torch.Generator` local genera 8192 valores base `N(0,1)` de pesos input, 1024 valores base `N(0,1)` para ocho posiciones dendrita→soma por soma, y una única matriz output uniforme Xavier `[10,128]`. Las cuatro variantes reutilizan exactamente estos tres tensores base. Para fan-in K, `W_d = base_input × sqrt(2/(1+0.1²))/sqrt(K)`. Para B, se agrupan las ocho posiciones somáticas en bloques contiguos de `8/B`, se suman y dividen por `sqrt(8/B)`, y luego se aplica `sqrt(2/(1+0.1²))/sqrt(B)`. Los grupos somáticos también son nested y conservan varianza inicial. Los tres biases se fijan a cero. Esta política Kaiming para LeakyReLU se fijó antes de ver accuracy; las activaciones iniciales son diagnósticas, no objetivo de ajuste.

Para cada seed, el split Fashion-MNIST 54000/6000 se fija por el mismo `training_seed`, como en el loader histórico. El `RandomSampler` usa un generador PyTorch con esa seed; se guardan hashes SHA-256 del orden real de índices fuente para las 25 épocas. Se registran además hashes del split, `G64`, `G` particionada, los tres tensores base de inicialización y las versiones del entorno. Los hashes se cotejan entre los cuatro brazos de la seed antes de resumir.

## Protocolo y artefactos

Fashion-MNIST oficial con `ToTensor()` float32 `/255`, train/validation/test 54000/6000/10000, CPU, Adam (`lr=0.001`, betas 0.9/0.999, epsilon `1e-7`, weight decay 0), batch 128, 25 épocas, CrossEntropy sobre logits, sin scheduler, clipping, mixed precision ni early stopping. Se evalúa `final_epoch_25`. `training_seed=topology_seed` toma sólo 1, 2 y 3. Se captura un mismo primer batch de validation para las medias y desviaciones iniciales de preactivaciones/activaciones dendríticas y somáticas, y la desviación de logits. La duración incluye train y validation y no es un benchmark de rendimiento.

Comando: `.venv/Scripts/python.exe -m dsu_research.compartment_ablation --config experiments/configs/compartment_ablation_fmnist_s3_t05.json --output runs/compartment_s3_t05`. Cada corrida se escribe atómicamente al terminar en un JSON propio. El runner valida un JSON existente y lo reutiliza; `--force` es necesario para repetirlo. Las 25 filas de historia por corrida y el resumen quedan en `runs/compartment_s3_t05/`, ignorado por Git.

## Criterios de interpretación prefijados

La comparación principal es `4×16 − 1×64` en test de época 25. `STRONG_PRELIMINARY_SUPPORT` requiere ≥+1.0 pp de accuracy media, tres victorias pareadas de accuracy y tres mejoras de loss. `WEAK_SUPPORT` requiere +0.30 a <+1.0 pp con ≥2 victorias y sin deterioro relevante de loss, o tendencia media estrictamente creciente 1→2→4 sin deterioro relevante de loss. Operacionalmente se fijó `Δloss medio ≤+0.01` para ese deterioro. `NEGLIGIBLE` requiere |Δaccuracy|<0.30 pp y |Δloss|<0.01. `CONTRARY` requiere ≤−0.50 pp y dirección de accuracy adversa en tres seeds. El resto es `INCONCLUSIVE`. N=3 sólo permite lectura exploratoria. `8×8` puede revelar un óptimo intermedio; no se supone monotonicidad hasta B=8.

## Auditoría y diagnóstico inicial

**Gate técnico S3-T05: PASS. H-011: NEGLIGIBLE.** Los 12 JSON contienen 25 épocas finitas, test final y los conteos estructurales esperados. Una segunda invocación del runner reanudó las 12 corridas sin entrenar de nuevo y reprodujo el resumen. Dentro de cada seed coincidieron exactamente los hashes SHA-256 de `G64`, los tres tensores base de inicialización, los índices de train/validation y las 25 secuencias de minibatches. Los tests verifican las uniones nested 1→2→4→8 y ausencia de matrices ocultas densas.

Los diagnósticos siguientes se midieron antes de entrenar con el mismo primer batch de validation de cada seed. Cada celda `media/DE` usa la desviación estándar de activaciones del batch y todas sus unidades; `logits` registra sólo DE. No se ajustaron escalas tras verlos.

| Seed | Brazo | PreD media/DE | ActD media/DE | PreS media/DE | ActS media/DE | Logits DE |
|---:|---|---:|---:|---:|---:|---:|
| 1 | 1×64 | 0.026/0.653 | 0.238/0.423 | 0.056/0.660 | 0.171/0.526 | 0.567 |
| 1 | 2×32 | 0.018/0.660 | 0.232/0.432 | 0.037/0.655 | 0.188/0.490 | 0.521 |
| 1 | 4×16 | 0.013/0.649 | 0.225/0.416 | 0.083/0.615 | 0.227/0.455 | 0.557 |
| 1 | 8×8 | 0.009/0.650 | 0.218/0.413 | 0.037/0.615 | 0.221/0.409 | 0.540 |
| 2 | 1×64 | −0.011/0.697 | 0.238/0.446 | 0.130/0.651 | 0.217/0.545 | 0.619 |
| 2 | 2×32 | −0.008/0.686 | 0.236/0.423 | 0.075/0.532 | 0.188/0.404 | 0.452 |
| 2 | 4×16 | −0.006/0.655 | 0.220/0.407 | 0.027/0.606 | 0.195/0.397 | 0.477 |
| 2 | 8×8 | −0.004/0.638 | 0.210/0.404 | 0.005/0.620 | 0.200/0.417 | 0.527 |
| 3 | 1×64 | 0.017/0.665 | 0.241/0.413 | 0.061/0.654 | 0.166/0.524 | 0.441 |
| 3 | 2×32 | 0.012/0.675 | 0.239/0.414 | 0.048/0.693 | 0.203/0.508 | 0.478 |
| 3 | 4×16 | 0.009/0.671 | 0.232/0.433 | 0.019/0.708 | 0.218/0.474 | 0.647 |
| 3 | 8×8 | 0.006/0.675 | 0.228/0.435 | 0.025/0.684 | 0.223/0.486 | 0.873 |

Las DE de preactivación dendrítica estuvieron entre 0.638 y 0.697; las somáticas, entre 0.532 y 0.708. Logits varió más (0.441–0.873), en especial 8×8 seed 3, por las proyecciones compartidas sobre representaciones iniciales distintas. No hubo colapso, explosión ni métricas no finitas; la diferencia residual de escala sigue siendo una limitación del pareo y no se retocó para favorecer un brazo.

## Resultados por seed

Accuracy está en porcentaje. La duración cubre train y validation, no mide rendimiento de inferencia.

| Seed | Brazo | Train loss | Validation loss / acc. | Test loss / acc. | Duración s |
|---:|---|---:|---:|---:|---:|
| 1 | 1×64 | 0.289950 | 0.322785 / 88.900% | 0.373047 / 86.69% | 166.8 |
| 1 | 2×32 | 0.292864 | 0.319501 / 88.417% | 0.362687 / 87.03% | 182.9 |
| 1 | 4×16 | 0.293035 | 0.324760 / 88.650% | 0.368889 / 86.87% | 178.3 |
| 1 | 8×8 | 0.298259 | 0.329192 / 88.450% | 0.361178 / 87.33% | 187.3 |
| 2 | 1×64 | 0.286666 | 0.339435 / 87.700% | 0.365676 / 87.03% | 167.2 |
| 2 | 2×32 | 0.285022 | 0.336864 / 87.667% | 0.369293 / 86.63% | 178.4 |
| 2 | 4×16 | 0.288663 | 0.331678 / 88.133% | 0.359479 / 86.95% | 176.2 |
| 2 | 8×8 | 0.290330 | 0.326952 / 88.383% | 0.364024 / 86.65% | 187.0 |
| 3 | 1×64 | 0.288857 | 0.321728 / 88.400% | 0.369768 / 86.88% | 167.0 |
| 3 | 2×32 | 0.287229 | 0.320891 / 88.283% | 0.366707 / 86.83% | 184.3 |
| 3 | 4×16 | 0.293058 | 0.320864 / 88.067% | 0.364970 / 86.85% | 185.8 |
| 3 | 8×8 | 0.297197 | 0.326759 / 88.317% | 0.362746 / 87.00% | 195.4 |

## Agregados y presupuesto de calidad

Los valores son media ± DE muestral entre tres seeds. `Quality/param` es test accuracy en puntos porcentuales por 1000 parámetros; `quality/MAC` usa puntos porcentuales por 1000 MAC principales. Son ratios descriptivos, no métricas físicas ni prueba única de superioridad.

| Brazo | Test loss | Test acc. | Validation loss | Validation acc. | Quality/param | Quality/MAC |
|---|---:|---:|---:|---:|---:|---:|
| 1×64 | 0.369497 ± 0.003693 | 86.867 ± 0.170% | 0.327983 ± 0.009932 | 88.333 ± 0.603% | 8.805 ± 0.017 | 9.049 ± 0.018 |
| 2×32 | 0.366229 ± 0.003328 | 86.830 ± 0.200% | 0.325752 ± 0.009648 | 88.122 ± 0.400% | 8.578 ± 0.020 | 8.926 ± 0.021 |
| 4×16 | 0.364446 ± 0.004727 | 86.890 ± 0.053% | 0.325767 ± 0.005477 | 88.283 ± 0.319% | 8.171 ± 0.005 | 8.703 ± 0.005 |
| 8×8 | 0.362649 ± 0.001426 | 86.993 ± 0.340% | 0.327634 ± 0.001352 | 88.383 ± 0.067% | 7.462 ± 0.029 | 8.288 ± 0.032 |

## Comparaciones y curvas

En la comparación principal `4×16 − 1×64`, Δ test accuracy fue **+0.023 ± 0.138 pp**: +0.18, −0.08 y −0.03 pp por seed, una victoria de tres. Δ test loss fue **−0.005051 ± 0.001043**, mejor en las tres seeds. Como `|Δaccuracy|<0.30 pp` y `|Δloss|<0.01`, H-011 cae exactamente en **NEGLIGIBLE** según el criterio fijado. El menor loss sugiere una mejora pequeña de probabilidades predichas, pero no alcanza el criterio prefijado de ventaja material de clasificación.

| Contraste (segundo − primero) | Δ test acc. pp, media ± DE | Δ test loss, media ± DE | Victorias accuracy |
|---|---:|---:|---:|
| 2×32 − 1×64 | −0.037 ± 0.370 | −0.003268 ± 0.006990 | 1/3 |
| 4×16 − 2×32 | +0.060 ± 0.242 | −0.001783 ± 0.008008 | 2/3 |
| 8×8 − 4×16 | +0.103 ± 0.382 | −0.001797 ± 0.006140 | 2/3 |
| 4×16 − 1×64 | +0.023 ± 0.138 | −0.005051 ± 0.001043 | 1/3 |

El ranking **descriptivo** por accuracy media es `8×8 > 4×16 > 1×64 > 2×32`; sus separaciones son pequeñas respecto a la variación entre seeds. No hay tendencia monótona de accuracy desde B=1 a B=4 ni señal de un óptimo B=4. El loss medio de test desciende con B, pero los deltas adyacentes de loss no mantienen signo en las tres seeds. Los ratios quality/parameter y quality/MAC decrecen al aumentar B: el pequeño cambio de accuracy no compensa descriptivamente los costes adicionales.

Las curvas completas están en cada JSON. Esta selección muestra train loss medio / validation accuracy media en épocas prefijadas; todos los brazos aprenden, con trayectorias cercanas y sin separación sostenida de validation a favor de más ramas.

| Brazo | Época 1 | Época 5 | Época 10 | Época 15 | Época 20 | Época 25 |
|---|---:|---:|---:|---:|---:|---:|
| 1×64 | 0.916 / 80.71% | 0.411 / 85.91% | 0.354 / 86.96% | 0.323 / 87.46% | 0.303 / 87.98% | 0.288 / 88.33% |
| 2×32 | 0.938 / 80.55% | 0.408 / 85.93% | 0.353 / 87.09% | 0.323 / 87.40% | 0.303 / 87.87% | 0.288 / 88.12% |
| 4×16 | 0.951 / 80.05% | 0.409 / 85.73% | 0.355 / 86.92% | 0.326 / 87.36% | 0.306 / 87.86% | 0.292 / 88.28% |
| 8×8 | 0.950 / 80.09% | 0.408 / 85.84% | 0.356 / 86.92% | 0.328 / 87.40% | 0.309 / 87.93% | 0.295 / 88.38% |

## Interpretación, límites y siguiente decisión

En Fashion-MNIST y este protocolo N=3, particionar las mismas 64 conexiones por soma en cuatro ramas no produjo una mejora material de test accuracy frente a una rama. La conclusión es exploratoria y acotada. B también aumenta sesgos, pesos soma, número de activaciones y MAC; la escala inicial de logits no es idéntica pese al pareo de random base values y la escala fan-in-aware. La topología de `4×16` no es la DSU-S v0 histórica. No se infiere causalidad exclusiva de la no linealidad, ventaja general de arquitectura, FLOPs reales, latencia, RAM ni energía. **Bugs:** el primer test estructural detectó pesos sin inicializar en el constructor experimental; se añadió inicialización Glorot por defecto antes de comenzar las corridas, que el runner sobrescribe con la política pareada. No se detectaron bugs metodológicos en los 12 artefactos.

La siguiente tarea científica recomendada para S3-T06 es **diagnosticar la ausencia de ventaja material**: fijar por adelantado un control que separe la contribución de los pesos/biases dendríticos extra de la partición no lineal y revise el efecto de la escala inicial de logits. Diseñar ese control antes de abrir nuevas seeds o resultados. S3-T06 no se ejecutó aquí.
