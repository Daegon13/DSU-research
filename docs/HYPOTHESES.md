# HYPOTHESES

Estados permitidos:

- UNTESTED
- SUPPORTED
- WEAK SUPPORT
- NOT SUPPORTED / NEGLIGIBLE
- CONTRARY
- REFUTED
- INCONCLUSIVE

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

**Status:** UNTESTED

### Statement
Con igual número total de conexiones input por soma, dividirlas entre múltiples subunidades dendríticas no lineales proporciona más capacidad que una sola integración.

### Experiment
Comparación prioritaria futura `1×64`, `2×32`, `4×16`, `8×8`, con 64 índices por soma y cobertura controlada. Cambian también sesgos y pesos soma; registrar el presupuesto de parámetros y añadir control si esa diferencia resulta decisiva.
