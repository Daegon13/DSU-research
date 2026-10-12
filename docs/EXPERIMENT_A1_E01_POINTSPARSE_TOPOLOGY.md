# A1-E01 — contraste causal de topología PointSparse

**Estado (2026-10-11):** gate técnico **PASS**; H-A1-TOPOLOGY = **NEGLIGIBLE** dentro de los grafos con grados controlados probados. Track A = **BACKLOG / DEPRIORITIZED**. Sprint 3 permanece CLOSED; Line B/S4 no se inició.

TASK ID: A1-E01. OBJECTIVE: medir el efecto de la topología G en PointSparse `128×64`. WHY: determinar cuánto de su calidad depende de grados y organización de overlap antes de estudiar mecanismos neuronales dinámicos. CONTEXT: [A1-DESIGN](A1_POINTSPARSE_TOPOLOGY_DESIGN.md), commit `4ce1fc7`. ALLOWED FILES: runner/análisis/tests A1-E01, JSON de corridas y resultado, este informe, estado/resultados/hipótesis/programa/roadmap/decisiones. FORBIDDEN FILES: preregistro, generador A1, PointSparse, baselines, datasets/splits, Sprint 3. REQUIREMENTS: preflight, pareo, 25 épocas, test final, resume y reglas congeladas. OUT OF SCOPE: brazos nuevos, pilotos, proxies, búsqueda H/K, Line B/S4. ACCEPTANCE CRITERIA: nueve corridas nuevas válidas si A0 reutilizable, cuatro contrastes pareados y decisión Track A. TESTS: suite, auditoría de hashes, firewall de test y `git diff --check`. EXPECTED OUTPUT: JSON por corrida y resultado agregado. EXPECTED COMMIT: `experiment: test PointSparse topology causally`. STOP CONDITIONS: deriva de G, pérdida del pareo, test anticipado o corrida incompleta.

## Preregistro heredado y preflight

Los brazos y umbrales son los de A1-DESIGN, sin cambios: A0 IID histórico, A1 balanceado, A2 menor overlap y A3 mayor overlap. H=128, K=64, E=8192, 9610 parámetros, 9472 MAC principales, LeakyReLU 0.1. A1/A2/A3 mantienen la misma secuencia **y vector** de grados de input (352×11, 432×10), cobertura 784/784 y la misma base balanceada por seed.

Antes del entrenamiento se regeneraron las 12 G y se compararon generación, hash y todas las métricas con [el artefacto congelado](../data/a1_topology_structural.json). Seeds 1–3: **PASS** de grados, cobertura, unicidad, presupuestos y los cuatro márgenes estructurales de A1-DESIGN. SHA-256 del artefacto: `8de4f3a87049d500a1ee1bcf90b47be9a4f3f465602ded9058170ad4dd20a609`. El runner repite este preflight al iniciar o reanudar y verifica también hashes del diseño y código congelados; ante deriva se detiene antes de entrenar.

La auditoría S3-T11 `audit_anchor()` volvió a validar las tres corridas PointSparse S3-T07: misma clase/función, G, estado inicial, seeds, split, transform, Adam, minibatches registrados, 25 épocas y `final_epoch_25`. Decisión: **REUSE_S3_T07_N3**. El número de threads CPU de T07 no quedó congelado; A1-E01 usa uno y no promete identidad bit a bit de pérdidas con A0. Los hashes de orden de las 25 épocas se comparan por seed con A0 antes de abrir test en cada corrida nueva.

## Protocolo y registro

Fashion-MNIST oficial 54000/6000/10000, `ToTensor()` `/255`, CPU float32, seeds 1–3. `training_seed` y `topology_seed` son campos distintos y valen el mismo entero por pareja. PointSparse y `paired_values(seed)` son los históricos; sólo cambia G. Adam lr `.001`, betas `.9/.999`, epsilon `1e-7`, weight decay 0; batch 128; CrossEntropy sobre logits; 25 épocas; sin scheduler, clipping, mixed precision, early stopping ni best epoch. Validation loss/accuracy y train loss se guardan por época. Test se instancia y evalúa una vez sólo después de época 25 y de comprobar el orden de minibatches. Se guardan accuracy, NLL, norma de logits, entropía predictiva, margen correcto y ECE-15 con el evaluador S3-T07.

Cada corrida nueva persiste JSON completo en `runs/a1_e01/` con G explícita, hash de G y estado inicial, fingerprints de pesos base, split, orden por época, métricas estructurales, historia, test y duración. En resume se valida el JSON existente y no se repite una corrida válida. El resumen versionado en `data/a1_e01_results.json` se produce únicamente al completar las nueve corridas y validar el pareo entre los cuatro brazos.

## Resultados por seed

El [resultado versionado](../data/a1_e01_results.json) contiene historia de 25 épocas, métricas finales, hashes de cada corrida y resumen verificable. A0 son los tres JSON históricos; A1/A2/A3 son las nueve corridas nuevas. Accuracy se expresa en %, NLL es CrossEntropy media en el split completo.

| Seed | Brazo | Validation accuracy | Test accuracy | Test NLL | Train + validation, s |
|---:|---|---:|---:|---:|---:|
| 1 | A0 | 88.533 | 86.910 | .371451 | 165.7 histórico |
| 1 | A1 | 88.467 | 86.820 | .366580 | 251.5 |
| 1 | A2 | 88.683 | 86.940 | .366108 | 267.3 |
| 1 | A3 | 88.567 | 86.790 | .366931 | 244.7 |
| 2 | A0 | 87.717 | 87.040 | .371541 | 157.0 histórico |
| 2 | A1 | 88.067 | 86.900 | .365913 | 245.4 |
| 2 | A2 | 88.067 | 87.030 | .369515 | 241.8 |
| 2 | A3 | 87.850 | 86.730 | .368641 | 241.4 |
| 3 | A0 | 88.267 | 86.790 | .373933 | 161.7 histórico |
| 3 | A1 | 88.817 | 87.100 | .361424 | 242.2 |
| 3 | A2 | 88.267 | 86.960 | .369118 | 244.9 |
| 3 | A3 | 88.367 | 87.100 | .359895 | 246.3 |

| Brazo | Validation accuracy media ± DE | Test accuracy media ± DE | Test NLL media ± DE |
|---|---:|---:|---:|
| A0 | 88.172 ± .416 | 86.913 ± .125 | .372309 ± .001408 |
| A1 | 88.450 ± .375 | 86.940 ± .144 | .364639 ± .002804 |
| A2 | 88.339 ± .315 | 86.977 ± .047 | .368247 ± .001863 |
| A3 | 88.261 ± .370 | 86.873 ± .199 | .365156 ± .004636 |

## Contrastes pareados congelados

Delta = segundo brazo menos primero. Accuracy en puntos porcentuales; NLL en unidades de loss. DE es muestral de los tres deltas. No se seleccionó checkpoint por validation ni se cambiaron seeds o umbrales al ver test.

| Contraste | Seed 1: Δacc / ΔNLL | Seed 2 | Seed 3 | Media ± DE: Δacc | Media ± DE: ΔNLL | Estado |
|---|---:|---:|---:|---:|---:|---|
| C1 A1−A0 | −.090 / −.004871 | −.140 / −.005629 | +.310 / −.012509 | +.027 ± .247 | −.007670 ± .004208 | **INCONCLUSIVE** |
| C2 A2−A1 | +.120 / −.000472 | +.130 / +.003602 | −.140 / +.007694 | +.037 ± .153 | +.003608 ± .004083 | **NEGLIGIBLE** |
| C3 A3−A1 | −.030 / +.000351 | −.170 / +.002728 | .000 / −.001529 | −.067 ± .091 | +.000517 ± .002134 | **NEGLIGIBLE** |
| C4 A3−A2 | −.150 / +.000823 | −.300 / −.000874 | +.140 / −.009223 | −.103 ± .224 | −.003091 ± .005377 | **NEGLIGIBLE** |

En C1, A1 mejora NLL en 3/3 seeds y por `.007670` de media, pero accuracy cambia sólo `+.027 pp`, con signos mezclados. Como `|ΔNLL|` supera `.005`, **no** corresponde `NEGLIGIBLE` según A1-DESIGN; tampoco se alcanza el umbral de accuracy para `MATERIAL` o `SMALL`. Se conserva `INCONCLUSIVE`. La distinta configuración histórica de threads puede cambiar pequeñas pérdidas numéricas; no permite explicar ni descartar por sí sola esta diferencia. C1 interviene en el régimen de grados **y** en las aristas concretas, por lo que tampoco identifica un efecto puro de grado.

En C2/C3/C4, todos los promedios cumplen simultáneamente `|Δaccuracy|<.20 pp` y `|ΔNLL|<.005`. El contraste C4, con los grados exactamente pareados y las colas estructurales más separadas, termina **NEGLIGIBLE** bajo el criterio fijado. Las victorias individuales cambian de seed y no justifican elegir una topología por ranking de medias. Se registraron norma de logits, entropía, margen correcto y ECE-15 en cada JSON, sin usarlas como criterio de selección.

## Diferencia estructural efectivamente probada

El preflight y los JSON nuevos coinciden exactamente con A1-DESIGN. En las tres seeds todos los grafos tienen 128 filas, 64 entradas únicas/fila, 8192 aristas, 9610 parámetros y 9472 MAC principales. A1/A2/A3 mantienen cobertura 784/784, el mismo **vector** de grados 352×11 + 432×10 (media 10.449, DE .497, min 10, max 11), y conectividad bipartita completa. A0 tuvo cobertura 784/784 en estas seeds, grado variable (mínimo 2–3, máximo 20–22, DE 3.000–3.129). Los histogramas completos de intersecciones, mediana, DE y p90 de Jaccard están en el artefacto estructural.

| Brazo | Mean Jaccard, seeds 1–3 | p90 Jaccard | Max Jaccard | Swaps de objetivo aceptados |
|---|---:|---:|---:|---:|
| A0 | .042610–.042945 | .066667 | .113043–.132743 | 0 |
| A1 | .039015–.039023 | .057851 | .122807–.132743 | 0; mezcla aleatoria 85695–86050 |
| A2 | .038805–.038806 | .049180 | .066667–.075630 | 2768–2829 |
| A3 | .040294–.040310 | .084746 | .454545–.523810 | 5586–5639 |

La suma de intersecciones entre pares está fijada en 38800 al fijar grados. A2 sólo podía reducir modestamente Jaccard medio y logró ~`.00021` de descenso frente a A1, cerca del 80 % del margen inferior matemático disponible; el máximo cayó aproximadamente a la mitad. A3 elevó fuertemente la cola y el máximo manteniendo esos grados. Por ello, el resultado negativo de C2/C3/C4 sí contrasta organizaciones de G distintas; no demuestra que cualquier topología posible sea equivalente.

## Hipótesis, límites y decisión

**H-A1-TOPOLOGY = NEGLIGIBLE** para la intervención con distribución/vector de grados controlados de C2/C3/C4 y este protocolo Fashion-MNIST N=3. No se observó efecto material de reducir la cola de overlap ni de concentrarla/clustering. **Balance de grados A0→A1 = INCONCLUSIVE**: accuracy casi igual, NLL mejor en A1 y una comparación que también cambia aristas. No se concluye que width sea la causa de la señal PointSparse, ni que el balance o el overlap nunca importen fuera de los grafos y seeds probados.

**Track A: BACKLOG / DEPRIORITIZED como prioridad activa.** Ningún contraste alcanzó `MATERIAL`; PointSparse IID `128×64` sigue siendo el baseline simple y comparable con la historia. La discordancia C1 de NLL permite, como máximo, **una** tarea diagnóstica específica futura si se considera necesaria; no se abre aquí otra búsqueda de G ni se ejecuta esa tarea. La siguiente prioridad de investigación es una decisión separada de diseño para Line B / S4, empezando por PointSparse estática frente a PointSparse con estado temporal en una tarea temporal. Este experimento no inició Line B ni S4.

Se completaron **9 corridas nuevas × 25 = 225 epoch-equivalents**, reutilizando 75 históricos de A0. Train + validation de las nuevas corridas sumó **2225.4 s**; excluye evaluación final de test, preflight, análisis y overhead, y no constituye un benchmark de latencia ni energía. Limitaciones: N=3, sólo Fashion-MNIST/CPU, A0 con threads históricos no fijados frente a uno en A1-E01, split variable entre seeds aunque pareado dentro de cada una, y ninguna medida nueva de RAM, latencia o energía.

**Gate A1-E01: PASS técnico.** Topologías y separación idénticas al preregistro; A0 reutilizado tras auditoría objetiva; nueve corridas de 25 épocas completas y reanudables; pares verificados por hashes de inicialización, split y minibatches; test abierto sólo después de época 25; cuatro criterios aplicados sin tuning; hipótesis y Track A con estado explícito. La suite pasó (`109 passed, 2 skipped`) y `git diff --check` quedó limpio.
