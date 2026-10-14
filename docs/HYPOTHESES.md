# HYPOTHESES

Estados permitidos:

- UNTESTED
- SUPPORTED
- WEAK SUPPORT
- NOT SUPPORTED / NEGLIGIBLE
- NEGLIGIBLE
- MIXED / NOT SUPPORTED
- CONTRARY
- REFUTED
- INCONCLUSIVE
- FAILED (validación prospectiva de proxy)
- WEAK (predictividad de piloto)

## H-B1-INTERFERENCE — estado EMA frente a distractores

**Status:** SUPPORTED sólo en `delayed_recall_noise_v1` con amplitud `.1`, B1-E02 (2026-10-13). **Preregistro:** [B1-NOISE-DESIGN](B1_NOISE_DESIGN.md). **Resultado:** [B1-E02](EXPERIMENT_B1_E02_STATEFUL_INTERFERENCE.md).

Con la unidad, λ y presupuesto de B1-E01, PointSparse-T v0 conservó ventaja sobre PointSparse estática cuando cada paso intermedio contenía ruido IID independiente del target de amplitud fija `.1`, incluso en canales de señal. En el endpoint preregistrado Δ=16, S0 obtuvo 50.00 ± 0.00%, S1 100.00 ± 0.00%, ventaja pareada +50.00 ± 0.00 pp en 3/3 seeds y ΔNLL −.691904 ± .000386. Cumple literalmente el criterio SUPPORTED. La accuracy de S1 quedó en techo en los cuatro delays; NLL aumentó frente a EASY. Este resultado no establece límite de memoria, tolerancia a ruido más fuerte o utilidad general del mecanismo.

---

## H-B1-STATE — estado temporal mínimo en PointSparse

**Status:** SUPPORTED sólo en delayed recall EASY, B1-E01 (2026-10-13). **Preregistro:** [B1-DESIGN / S4-T01](B1_POINTSPARSE_TEMPORAL_DESIGN.md). **Resultado:** [B1-E01](EXPERIMENT_B1_E01_STATEFUL_POINTSPARSE.md).

Con igual arquitectura sparse y número de parámetros entrenables, un estado temporal leaky permite retener información útil a través del tiempo que PointSparse estática no puede recuperar del timestep final. B1-E01 comparó S0/S1 en delayed recall binario con `D=32, H=64, K=8`, tres seeds pareadas y delays de 4/8/16/32 transiciones; λ fijo por vida media 16. En el endpoint Δ=16, S0 50.00 ± 0.00%, S1 100.00 ± 0.00%, delta pareado +50.00 ± 0.00 pp y ΔNLL −.692323 ± .000190. Cumple el criterio preregistrado. Todos los delays de EASY dieron accuracy en techo para S1; el horizonte límite, interferencia por ruido y eficiencia física siguen sin probarse.

---

## H-A1-TOPOLOGY — organización del grafo sparse

**Status:** NEGLIGIBLE dentro de A1-E01 para C2/C3/C4 con grados controlados; C1 = INCONCLUSIVE.

Con H=128, K=64, 9610 parámetros, 9472 MAC principales, cobertura completa y vector de grados de input 352×11 + 432×10 controlados, la organización de overlap/topología de G cambia materialmente la calidad de PointSparse. Los contrastes A1→A2/A3 y A2→A3 son los que controlan todos esos factores. A0→A1 prueba el régimen IID frente a balanceado, con grados distintos. Criterios congelados en [A1-DESIGN](A1_POINTSPARSE_TOPOLOGY_DESIGN.md).

**Resultado A1-E01 (2026-10-11):** C2 A2−A1 = +.037 ± .153 pp / +.003608 ± .004083 NLL; C3 A3−A1 = −.067 ± .091 pp / +.000517 ± .002134; C4 A3−A2 = −.103 ± .224 pp / −.003091 ± .005377. Todos son NEGLIGIBLE según los umbrales prefijados, aunque A2 redujo la cola y A3 la elevó claramente. C1 A1−A0 = +.027 ± .247 pp / −.007670 ± .004208 NLL: INCONCLUSIVE por discordancia accuracy/NLL; cambia régimen de grados y aristas. No se concluye que toda topología sea irrelevante ni que width sea la causa. [Informe](EXPERIMENT_A1_E01_POINTSPARSE_TOPOLOGY.md).

---

**Cierre S3-REVIEW (2026-10-11):** los estados finales de H-009–H-016, H-ZEROCOST-ER, H-WIDTH-FANIN y EARLY-PILOT-3E están reunidos con evidencia, alcance y límites en [SPRINT_3_REVIEW.md](SPRINT_3_REVIEW.md). Sprint 3 = PASS / CLOSED; ninguna hipótesis abierta se promueve por ello a SUPPORTED. H-002/H-006 continúan abiertas para RAM/runtime y eficiencia física general.

---

## H-001 — Eficiencia espacial dendrítica

**Status:** SUPPORTED

### Statement
Una arquitectura dendrítica compacta puede mantener rendimiento comparable al baseline dANN-R de igual anchura usando muchos menos parámetros físicamente almacenados; comparaciones posteriores con MLP de presupuestos igualados evaluarán la ventaja más general.

### Reason
Cada dendrita procesa sólo un subconjunto de las entradas y realiza agregación local antes del soma.

### Experiment
S2-T04: comparar vANN, dANN-R dense-mask y DSU-S v0 compact en Fashion-MNIST con el mismo split y protocolo cuando sea posible, inicialmente N=3 seeds. S2-T05/T06: comparaciones posteriores igualadas por parámetros y por calidad.

### Prediction
DSU-S almacenaría 10634 parámetros entrenables frente a 468874 de dANN-R y vANN, manteniendo calidad cercana. El coste de índices se reportaría aparte.

### Resultado S2-T04 (2026-10-06)
En Fashion-MNIST N=3, DSU-S obtuvo test loss **0.357825 ± 0.002485** y accuracy **87.253 ± 0.277 %**; dANN-R obtuvo **0.358398 ± 0.004885** y **87.237 ± 0.235 %**. Deltas pareados DSU-S − dANN-R: **−0.000574 ± 0.002440** de loss y **+0.017 ± 0.117 pp** de accuracy. Tres entrenamientos estables y 10634 parámetros almacenados frente a 468874 de dANN-R satisfacen la señal fuerte predefinida para esta comparación. `SUPPORTED` se limita a este benchmark de igual anchura; N=3 no aísla fan-in/topología/inicialización ni demuestra ventaja general frente a MLP igualado. Detalles en [EXPERIMENT_001_DSU_S_LEARNING.md](EXPERIMENT_001_DSU_S_LEARNING.md).

### Auditoría S2-REVIEW
**SUPPORTED** se conserva para la comparación de Fashion-MNIST N=3 de igual anchura y protocolo. Es una reducción de parámetros entrenables físicamente almacenados frente a **dANN-R dense-mask**, no prueba de menor payload tensorial que un MLP quality-matched ni atribución causal a las dendritas. Véase [SPRINT_2_REVIEW.md](SPRINT_2_REVIEW.md).

---

## H-002 — Utilidad física de la conectividad regular

**Status:** UNTESTED

### Statement
Una estructura compacta de fan-in fijo puede convertir la sparsity dendrítica en una reducción física de almacenamiento y un cómputo estructural utilizable.

### Reason
DSU-S v0 almacena sólo pesos conectados y 16 índices por dendrita; dANN-R almacena matrices densas enmascaradas. La indexación puede añadir coste y debe medirse.

### Experiment
S2-T04: medir bytes de parámetros, bytes de topología y `state_dict` de DSU-S junto con calidad. RAM pico CPU y latencia controlada frente a dANN-R quedan pendientes; la comparación de distintas topologías y kernels queda para Sprint 3/5, con variables controladas.

### Prediction
DSU-S reducirá almacenamiento de pesos entrenables; el resultado de latencia queda abierto. Menos parámetros o MAC teóricos no garantizan aceleración.

### Evidencia parcial S2-T04
DSU-S almacena 42536 bytes de tensores entrenables más 65536 bytes de índices; un `state_dict` representativo ocupa 111508 bytes. Esto confirma tamaño tensorial compacto, pero H-002 sigue **UNTESTED** en cuanto a eficiencia física/latencia de ejecución: no se midió RAM pico CPU ni una comparación controlada de latencia y los índices/gather tienen coste propio.

### Auditoría S2-REVIEW
El payload tensorial conocido de DSU-S (**108072 bytes**) supera los **66064 bytes** de pesos del MLP `20→26`. H-002 permanece **UNTESTED** para utilidad de memoria runtime y cómputo; el resultado observado impide afirmar una ventaja de almacenamiento total frente a ese MLP con la representación int64 actual. Optimizar topología requiere estudio separado de checkpoint y runtime.

---

## H-003 — Permutación entre capas evita islas

**Status:** UNTESTED

### Statement
Una permutación barata de características entre capas aumentará capacidad sin necesidad de conectividad global densa.

### Experiment
DSU-S con y sin permutación.

### Prediction
La permutación mejorará calidad a coste computacional muy pequeño.

---

## H-004 — Memoria dendrítica mejora tareas temporales

**Status:** UNTESTED

### Statement
Permitir que cada dendrita mantenga estado temporal independiente mejora tareas secuenciales frente a DSU-S.

### Experiment
Comparar:
- DSU-S;
- DSU-T con lambda fija;
- DSU-T con lambda aprendible;
- RNN/GRU;
- modelos temporales relevantes.

### Prediction
DSU-T obtendrá mejor relación calidad/parámetros en tareas temporales.

---

## H-005 — Lambda aprendible > lambda fija

**Status:** UNTESTED

### Statement
Constantes temporales aprendibles permitirán que diferentes ramas representen distintas escalas de memoria.

### Prediction
Mejor calidad o igual calidad con menor tamaño.

---

## H-006 — Menos parámetros no implica necesariamente menor latencia

**Status:** UNTESTED

### Statement
Una DSU puede reducir parámetros sin ser más rápida en PyTorch si su patrón de acceso a memoria es ineficiente.

### Purpose
Esta hipótesis existe para prevenir conclusiones erróneas.

---

## H-007 — DSU puede sustituir un FFN pequeño

**Status:** UNTESTED

### Statement
Una DSU suficientemente madura puede sustituir parcial o totalmente el FFN de un Transformer pequeño manteniendo calidad comparable con menos memoria.

### Gate
No probar hasta completar los milestones previos.

---

## H-008 — Capacidad de clasificación por parámetro frente a MLP denso

**Status:** SUPPORTED

**Alcance:** familia MLP densa y protocolo S2-T05/T06.

### Statement
Con un presupuesto de aproximadamente 10.6k parámetros, DSU-S v0 logra mejor calidad de clasificación Fashion-MNIST que un MLP denso de profundidad no lineal comparable.

### Experiment
S2-T05: comparar DSU-S v0 ya entrenada en seeds 1, 2 y 3 con el MLP congelado 784→13→17→10, LeakyReLU 0.1 en ambas capas ocultas, 10623 parámetros, 25 épocas y el mismo split por seed. Se evaluará el estado final. Criterios fijados antes del entrenamiento en `docs/EXPERIMENT_002_PARAMETER_MATCHED_MLP.md`.

### Prediction
DSU-S distribuiría sus pocos pesos entre 512 activaciones dendríticas y 128 somáticas, frente a 13 y 17 activaciones ocultas del MLP. Estas activaciones no son parámetros; mayor cantidad no garantiza mayor calidad.

### Resultado S2-T05 (2026-10-06)
DSU-S obtuvo 87.253 ± 0.277 % y loss 0.357825 ± 0.002485; el MLP 85.883 ± 0.287 % y loss 0.398187 ± 0.007987. Los deltas pareados MLP − DSU-S son −1.370 ± 0.565 pp y +0.040362 ± 0.010470 de loss, con ventaja DSU-S en las tres seeds. Señal **MODERADA/PROMETEDORA** según criterios previos; H-008 queda en **WEAK SUPPORT** por N=3 y una única arquitectura MLP fijada. No aísla la estructura dendrítica de otras diferencias ni demuestra superioridad general.

### Resultado S2-T06 (2026-10-06)
El search prefijado usó sólo validation para seleccionar dentro de la familia `784→h1→round(17h1/13)→10`. Los MLP `13→17`, `16→21` y `19→25` fallaron el criterio dual de calidad; `22→29` fue el primer full match de escalera y el refinamiento seleccionó `20→26` con **16516 parámetros**, **1.553×** DSU-S. Después de congelar la selección, sólo ese MLP se evaluó en test: **86.700 ± 0.191 %**, loss **0.375776 ± 0.003194**, frente a DSU-S **87.253 ± 0.277 %** y **0.357825 ± 0.002485**. La ventaja de test DSU-S conserva dirección en las tres seeds. H-008 pasa a **SUPPORTED en este alcance estrecho**: eficiencia de parámetros entrenables dentro de la familia y protocolo evaluados. N=3, tolerancias heurísticas y una familia limitada impiden afirmar superioridad general o causalidad dendrítica. Los **65536 bytes de topología** de DSU-S hacen que su payload tensorial total sea mayor que el de este MLP; no se afirma ventaja de almacenamiento total, latencia o energía. Detalles en [EXPERIMENT_003_MLP_CAPACITY_SEARCH.md](EXPERIMENT_003_MLP_CAPACITY_SEARCH.md).

### Auditoría S2-REVIEW
**SUPPORTED** se conserva para el MLP 13→17 probado y la familia/criterios S2-T06. El factor 1.553× es el mínimo **entre los candidatos evaluados**, no un mínimo demostrado entre todas las anchuras y MLP posibles. La decisión de Sprint 3 se fundamenta en esta señal acotada y en la necesidad de ablaciones causales; véase [SPRINT_2_REVIEW.md](SPRINT_2_REVIEW.md).

---

## H-009 — Fan-in fijo 16

**Status:** NOT SUPPORTED / NEGLIGIBLE

### Statement
El fan-in fijo de 16 aporta parte de la diferencia de rendimiento observada entre dANN-R y DSU-S.

### Experiment
S3-E01: dANN-R dense-mask frente a Fixed-dANN dense-mask, ambos con 8192 aristas de entrada y protocolo pareado. El contraste cambia también la distribución de topología; no identifica el efecto exclusivo del número 16. Véase [SPRINT_3_CAUSAL_DESIGN.md](SPRINT_3_CAUSAL_DESIGN.md).

### Resultado S3-T03 (2026-10-08)
En Fashion-MNIST N=5, Fixed-dANN menos dANN-R obtuvo **−0.042 ± 0.329 pp** de test accuracy y **+0.001066 ± 0.007721** de test loss. B ganó accuracy en 2/5 seeds y loss en 3/5; ambos deltas medios quedaron dentro del criterio prefijado de efecto despreciable. **NOT SUPPORTED / NEGLIGIBLE** dentro de este protocolo: regularizar a K=16 exacto no mostró señal material. No se concluye que fan-in nunca importe; A/B también cambia aristas concretas. Véase [EXPERIMENT_005_FIXED_FANIN_ABLATION.md](EXPERIMENT_005_FIXED_FANIN_ABLATION.md).

---

## H-010 — Equivalencia de representación

**Status:** SUPPORTED

### Statement
Una representación compacta puede ser funcionalmente equivalente a una dense-mask con la misma topología y pesos efectivos, usando muchos menos parámetros entrenables físicamente almacenados.

### Experiment
S3-E02: par Fixed-dANN/DSU-S con `G` idéntica y transferencia exacta de pesos; comparar intermedios, logits, gradientes, trayectoria y bytes. Los conteos estructurales existentes son evidencia preliminar de almacenamiento de pesos, no prueba de equivalencia pareada ni de menor RAM total.

### Resultado S3-T02 (2026-10-08)
**SUPPORTED en el alcance estructural y funcional CPU float32 probado.** Fixed-dANN y DSU-S compartieron `G`, pesos activos y sesgos. El máximo error observado fue 1.19e-7 en forward, 3.73e-9 en gradientes y 3.35e-8 en parámetros activos tras cinco pasos Adam; los gradientes y momentos inactivos fueron exactamente cero. Fixed-dANN almacenó 468874 parámetros frente a 10634 de DSU-S. Esto no prueba menor checkpoint/RAM total, performance, energía, equivalencia GPU ni trayectoria larga. Véase [EXPERIMENT_004_FIXED_DANN_EQUIVALENCE.md](EXPERIMENT_004_FIXED_DANN_EQUIVALENCE.md).

---

## H-011 — Partición dendrítica

**Status:** NEGLIGIBLE en S3-T05; evidencia exploratoria Fashion-MNIST N=3.

### Statement
Con igual número total de conexiones input por soma, dividirlas entre múltiples subunidades dendríticas no lineales proporciona más capacidad que una sola integración.

### Experiment
S3-T05 comparó `1×64`, `2×32`, `4×16`, `8×8` con 64 índices únicos y la misma cobertura por soma, partición nested, inicialización base y minibatches pareados. `4×16 − 1×64` produjo +0.023 ± 0.138 pp de test accuracy (una victoria de tres) y −0.005051 ± 0.001043 de test loss (tres mejoras). Ambos deltas medios quedan bajo los umbrales de efecto material prefijados. Los parámetros suben de 9866 a 10634 en ese contraste; no se aísla sólo la no linealidad. [Protocolo y resultados](EXPERIMENT_007_DENDRITIC_COMPARTMENT_ABLATION.md).

---

## H-012 — Almacenamiento físico B/C

**Status:** SUPPORTED para Fixed-dANN frente a DSU-S en CPU float32.

### Statement
La representación compacta reduce materialmente el almacenamiento físico del modelo y del estado de entrenamiento frente a la dense-mask funcionalmente equivalente.

### Resultado S3-T04 (2026-10-09)
Con `G` y pesos activos pareados, DSU-S reduce 22.281× el payload tensorial del modelo, 21.737× el checkpoint de modelo, 44.080× el estado tensorial Adam y 30.752× el checkpoint de entrenamiento. Son bytes medidos para las implementaciones actuales, incluyendo `G` y máscaras persistentes; no equivalen a RAM pico. Véase [EXPERIMENT_006_PHYSICAL_REPRESENTATION_BENCHMARK.md](EXPERIMENT_006_PHYSICAL_REPRESENTATION_BENCHMARK.md).

---

## H-013 — Latencia física B/C

**Status:** MIXED / NOT SUPPORTED como ventaja general.

### Statement
La representación compacta reduce latencia de ejecución frente a la dense-mask funcionalmente equivalente en hardware convencional.

### Resultado S3-T04 (2026-10-09)
En este CPU y PyTorch, DSU-S favorece claramente forward batch 1 (Fixed/DSU 4.245× con 1 thread; 2.381× con 6), pero Fixed es más rápida en forward batches 128 y 512 (ratios 0.252–0.512). El paso Adam batch 128 es similar. La dirección depende de batch y modo; no hay ventaja general de latencia ni conclusión para otros equipos. Véase [EXPERIMENT_006_PHYSICAL_REPRESENTATION_BENCHMARK.md](EXPERIMENT_006_PHYSICAL_REPRESENTATION_BENCHMARK.md).

---

## H-014 — H-SPARSE: anchura sparse sin compartimentos

**Status:** SUPPORTED dentro del contraste histórico acotado S3-T07.

La mayor parte de la eficiencia paramétrica observada frente al MLP denso proviene de mantener muchas unidades sparse con fan-in restringido, incluso sin compartimentalización dendrítica. S3-T06 implementó PointSparse (128 somas, 64 entradas únicas/soma, 9610 parámetros) como control. [Diseño](S3_T06_SPARSE_VS_COMPARTMENT_CONTROLS.md).

### Resultado S3-T07 (2026-10-10)
**SUPPORTED en el alcance histórico acotado.** PointSparse obtuvo 86.913 ± .125% test accuracy y `.372309 ± .001408` NLL, +1.030 pp frente al MLP 13→17 histórico y −.340 pp frente a DSU-S histórica. Retuvo 0.752 del gap histórico de accuracy. La comparación histórica no iguala todos los confusores; [detalle](EXPERIMENT_008_SPARSE_VS_COMPARTMENT_CAUSAL.md).

---

## H-015 — H-COMPARTMENT-PURE: partición sin parámetros extra

**Status:** CONTRARY dentro de S3-T07.

Con igual número de parámetros, las mismas 64 entradas por soma y escala inicial controlada, dividir entradas entre múltiples no linealidades dendríticas mejora la capacidad. S3-T06 implementó B=1,2,4,8 con 9610 parámetros cada uno, G64 nested y calibración sin labels. [Diseño](S3_T06_SPARSE_VS_COMPARTMENT_CONTROLS.md).

### Resultado S3-T07 (2026-10-10)
**CONTRARY** según el criterio prefijado de NLL adversa consistente: `4×16 − 1×64` = −.397 pp accuracy, +.011383 NLL, NLL peor en 3/3 seeds. La menor ECE no revierte la pérdida de top-1/NLL. [Detalle](EXPERIMENT_008_SPARSE_VS_COMPARTMENT_CAUSAL.md).

---

## H-016 — H-DENDRITIC-STAGE: etapa dendrítica con una rama

**Status:** INCONCLUSIVE en S3-T07 (registrada antes del entrenamiento).

Con la misma conectividad sparse y el mismo presupuesto entrenable, insertar una etapa dendrítica no lineal antes del soma mejora la capacidad frente a PointSparse. El contraste primario es `1×64 − PointSparse`; ambos tienen 9610 parámetros, 9472 MAC principales y G64 idéntica. La composición de LeakyReLU para entradas negativas y la calibración afín fija también difieren. Criterios prefijados: [EXPERIMENT_008_SPARSE_VS_COMPARTMENT_CAUSAL.md](EXPERIMENT_008_SPARSE_VS_COMPARTMENT_CAUSAL.md).

### Resultado S3-T07 (2026-10-10)
**INCONCLUSIVE.** `1×64 − PointSparse` = −.273 pp accuracy y +.003573 NLL; ambas direcciones desfavorecen la etapa extra, pero no alcanzan el umbral contrario prefijado. [Detalle](EXPERIMENT_008_SPARSE_VS_COMPARTMENT_CAUSAL.md).

---

## H-WIDTH-FANIN — distribución del presupuesto PointSparse

**Status final tras S3-T11:** INCONCLUSIVE para calidad final dentro de la shortlist; T10 había dado PRELIMINARY_SIGNAL temprano.

Con un presupuesto PointSparse fijo de 9610 parámetros, distribuirlo entre ancho H y fan-in K afecta materialmente la capacidad. El piloto de 3 épocas y 1 seed sólo permitirá PRELIMINARY_SIGNAL, NO_PRELIMINARY_SIGNAL o INCONCLUSIVE. [Protocolo](EXPERIMENT_010_PROSPECTIVE_ZERO_COST_SCREEN.md).

El rango de validation accuracy de época 3 fue 0.583 pp; 80×109 tuvo el máximo de accuracy y 240×29 el mínimo loss. La curva no fue monótona y 13/28 pares fueron near-ties. La señal temprana no establece un óptimo ni capacidad final.

### Resultado S3-T11 (2026-10-11)

`160×49` obtuvo la mayor validation accuracy media (88.267%) y menor test NLL (.366624), pero `128×64` obtuvo la mayor test accuracy media (86.913%); las tres medias de validation son near-ties bajo 0.20 pp. Ningún contraste candidato–anchor cumplió el criterio preregistrado de efecto material coherente entre validation y test. [Detalle](EXPERIMENT_011_POINTSPARSE_SHORTLIST_CONFIRMATION.md).

---

## H-ZEROCOST-ER — ranking prospectivo por effective rank

**Status S3-T10:** FAILED en el piloto prospectivo; era UNTESTED en el preregistro.

ER raw puede priorizar prospectivamente arquitecturas PointSparse de presupuesto fijo mejor que ordenar sólo por ancho H. Se evaluará frente al target de validation en época 3 y al baseline width con los criterios preregistrados. [Protocolo](EXPERIMENT_010_PROSPECTIVE_ZERO_COST_SCREEN.md).

ER raw obtuvo Spearman −.190 y 5/15 pares no near-tie correctos contra accuracy; width H obtuvo −.048 y 6/15. No agregó información al baseline trivial en esta cohorte. El resultado retrospectivo acotado de S3-T09 queda registrado separadamente.

S3-T11 no cambia esta clasificación de la cohorte T10 completa. Dentro de la shortlist de sólo tres brazos, ER raw correlacionó positivamente con las medias finales, pero validation tuvo cero pares no near-tie y test sólo uno; el subconjunto fue seleccionado por la regla T10. No interpretar esto como una validación general de ER.

---

## EARLY-PILOT-3E — ranking temprano de calidad final

**Status S3-T11:** WEAK como predictor entre las familias estudiadas.

La validation accuracy de época 3 se fijó antes del análisis como proxy primario de calidad final. Ordenó bien S3-T07, fue parcial en la familia MLP S2-T06 y adversa en la familia casi plana S3-T05. En la shortlist prospectiva T10→T11, su ranking tuvo Spearman −.50 frente a las medias finales de validation y test; los tres pares de validation fueron near-ties. Es un filtro exploratorio de aprendizaje inicial, no una regla general para elegir el mejor modelo o descartar arquitecturas cercanas. [Protocolo, curvas y límites](EXPERIMENT_011_POINTSPARSE_SHORTLIST_CONFIRMATION.md).
