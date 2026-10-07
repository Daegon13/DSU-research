# S2-REVIEW — Auditoría y decisión de Sprint 2

**Fecha:** 2026-10-06. **Gate S2-REVIEW: PASS. Gate Sprint 2: PASS. Decisión: GO TO SPRINT 3**, limitado a diseño y ablaciones causales de topología. Esta revisión no implementa ni entrena modelos nuevos.

## Preguntas del sprint y evidencia auditada

¿Aprende DSU-S v0 de forma estable, conserva aproximadamente la calidad de dANN-R de igual anchura y muestra alguna ventaja de calidad por parámetro frente a un MLP denso razonable? La respuesta acumulada es **sí dentro de Fashion-MNIST, seeds 1–3 y el protocolo fijado**, con una limitación importante de almacenamiento total.

| Etapa | Evidencia y alcance |
|---|---|
| S2-T02/T03 | `DSUSv0` tiene 128 somas, cuatro dendritas por soma y 16 entradas distintas por dendrita; 10634 parámetros entrenables, almacenados y efectivos, sin matriz oculta densa 784→512. Tests de shapes, topología, forward, backward, serialización y batch sintético. |
| S2-T04 | DSU-S: test loss **0.357825 ± 0.002485**, accuracy **87.253 ± 0.277 %**. dANN-R: **0.358398 ± 0.004885** y **87.237 ± 0.235 %**. DSU-S almacena 10634 parámetros entrenables frente a 468874 de dANN-R (44.09× menos) y sus tres historias de train redujeron loss sin inestabilidad visible. La diferencia pareada de calidad es pequeña y cambia de signo por seed. |
| S2-T05 | MLP prefijado `784→13→17→10`, 10623 parámetros: test **0.398187 ± 0.007987**, **85.883 ± 0.287 %**. DSU-S obtuvo menor loss y mayor accuracy en las tres parejas. |
| S2-T06 | Search con train+validation únicamente y test cerrado hasta congelar arquitectura. El menor full match **entre candidatos probados** fue `784→20→26→10`: **16516 parámetros = 1.553× DSU-S**, **16460 MAC principales = 1.649× DSU-S**. Tras congelar, test del seleccionado: **0.375776 ± 0.003194**, **86.700 ± 0.191 %**; DSU-S conservó ventaja en las tres parejas. |

Los artefactos locales muestran los mismos hashes de train y validation por seed entre DSU-S, dANN-R, MLP 13→17 y MLP 20→26. Los tres JSON DSU-S usan `final_epoch_25`; sus pérdidas de train bajaron aproximadamente de 1.07–1.09 a 0.287–0.294. Los conteos estructurales se contrastaron con clases y tests actuales. Referencias detalladas: [DSU_S_SPEC.md](DSU_S_SPEC.md), [EXPERIMENT_001_DSU_S_LEARNING.md](EXPERIMENT_001_DSU_S_LEARNING.md), [EXPERIMENT_002_PARAMETER_MATCHED_MLP.md](EXPERIMENT_002_PARAMETER_MATCHED_MLP.md) y [EXPERIMENT_003_MLP_CAPACITY_SEARCH.md](EXPERIMENT_003_MLP_CAPACITY_SEARCH.md).

## Matriz de afirmaciones

| Claim | Estado | Justificación y límite |
|---|---|---|
| A. DSU-S reproduce aproximadamente la capacidad de dANN-R con muchos menos parámetros físicamente almacenados | **SUPPORTED** | Calidad media muy próxima con igual anchura, split y protocolo; 10634 frente a 468874 parámetros almacenados. El resultado no aísla fan-in ni inicialización. |
| B. A ~10.6k parámetros DSU-S aprovecha mejor el presupuesto que el MLP 13→17 probado | **SUPPORTED** | Ventaja de test en loss y accuracy en las tres seeds frente a ese MLP concreto. |
| C. En la familia S2-T06 el MLP necesitó ~1.55× parámetros para cumplir validation | **SUPPORTED dentro del conjunto evaluado** | `20→26` fue el menor candidato probado que pasó ambos umbrales prefijados. No se probaron todos los anchos enteros por debajo de 20 ni otras proporciones; no se conoce el mínimo matemático de toda la familia. |
| D. DSU-S usa menos almacenamiento total que el MLP quality-matched | **NOT_SUPPORTED** | El payload tensorial conocido de DSU-S es 108072 bytes, frente a 66064 bytes del MLP. RAM de ejecución y tamaños de checkpoint comparables no medidos. |
| E. DSU-S es más rápida | **NOT_SUPPORTED** | No hay benchmark de latencia controlado comparable. MAC principales no son tiempo de ejecución. |
| F. DSU-S consume menos energía | **NOT_SUPPORTED** | No se midió energía. |
| G. La ventaja proviene específicamente de las dendritas | **NOT_SUPPORTED** | Fan-in, distribución de conexiones, inicialización, número de activaciones y representación varían conjuntamente. |
| H. El fan-in fijo 16 explica parte o toda la ventaja | **NOT_SUPPORTED** | No existe aún control Fixed-dANN con fan-in idéntico. |
| I. DSU-S generaliza a otros datasets o dominios | **NOT_SUPPORTED** | Sólo hay Fashion-MNIST para esta comparación. |
| J. DSU-S escalará a modelos mayores o Transformers | **NOT_SUPPORTED** | No se ensayó escalado ni Transformer. |

## Accounting de almacenamiento: limitación central

| Modelo | Pesos entrenables float32 | Topología conocida | Payload tensorial conocido |
|---|---:|---:|---:|
| DSU-S v0 | 42536 bytes | 8192 índices int64 = 65536 bytes | **108072 bytes** |
| MLP `20→26` | 66064 bytes | 0 bytes | **66064 bytes** |

DSU-S usa **5882 parámetros entrenables menos** que el MLP seleccionado, pero su representación actual con `G` int64 ocupa **42008 bytes más** de payload tensorial conocido. Los parámetros entrenables no incluyen índices. El `state_dict` representativo de DSU-S medido en S2-T04 fue de 111508 bytes para una instancia recién inicializada; no se midió un checkpoint final DSU-S comparable ni RAM pico/runtime de ambos modelos. No se infiere una ventaja de memoria física total, latencia o energía. Esta limitación es central para decidir qué investigar a continuación.

## Confusores y problemas abiertos

- dANN-R muestrea 8192 conexiones globalmente, con fan-in variable; DSU-S impone 16 distintas en cada dendrita. Cambian también inicialización Glorot por shape, topología concreta y ruta de operaciones PyTorch. La comparación A no atribuye causalidad a la representación compacta.
- Frente al MLP cambian distribución y cantidad de activaciones ocultas, conectividad, fan-in e inicialización efectiva. N=3 permite evidencia descriptiva pareada, sin prueba de significancia ni garantía de estabilidad en otros datasets.
- S2-T06 probó una escalera y sus anchos de refinamiento, no todas las arquitecturas densas. Los umbrales de match son heurísticos; el seleccionado alcanzó validation match pero siguió por debajo de DSU-S en test. No se usó test para seleccionarlo.
- No hay RAM pico CPU, velocidad controlada, energía, tamaño de checkpoint final DSU-S ni medida de tráfico de memoria. Los 9984 y 16460 MAC principales omiten gather, activaciones y overhead.
- La reproducción PyTorch de dANN-R conserva diferencias conocidas con Keras en inicialización, shuffle, Adam y semántica de pérdida; [REPRODUCTION_001.md](REPRODUCTION_001.md) las registra.

## Decisión Go/No-Go

| Requisito para GO | Evaluación |
|---|---|
| 1. Aprendizaje estable | **Cumplido:** tres entrenamientos completos de DSU-S, loss decreciente, métricas finitas. |
| 2. Calidad próxima a dANN-R reproducida | **Cumplido:** 87.253 % frente a 87.237 % de accuracy media y loss casi igual en N=3. |
| 3. Alguna ventaja de capacidad por parámetro | **Cumplido dentro del alcance:** gana al MLP 13→17 casi igualado; el menor match de validation observado requirió 1.553× parámetros. |
| 4. Preguntas causales claras y baratas | **Cumplido:** controles de fan-in fijo y equivalencia matemática dense-mask/compact pueden separar variables antes de ampliar arquitectura. |

**SPRINT 2: PASS. DECISION: GO TO SPRINT 3.** El GO autoriza la fase de investigación causal, **no** una afirmación de superioridad global de DSU-S ni el inicio automático de entrenamientos. H-001 permanece **SUPPORTED** para dANN-R de igual anchura en este benchmark; H-008 permanece **SUPPORTED** dentro de la familia/protocolo MLP ensayados. H-002 y H-006 siguen abiertos para memoria y rendimiento físico.

## Sprint 3 — Topology and Causal Ablations

**Objetivo formal:** determinar qué parte de la señal observada se relaciona con fan-in regular, organización dendrita→soma, representación compacta, número de dendritas y fan-in por dendrita, variando una dimensión conceptual por experimento siempre que sea posible. No añadir estado temporal, DSU-T, gating apical, spikes ni Transformers.

**Siguiente tarea exacta, sólo especificación:** `S3-T01 — Diseñar el control Fixed-dANN de fan-in 16 y la matriz causal de Sprint 3`. Debe definir tres brazos:

| Brazo | Fan-in de entrada | Almacenamiento |
|---|---|---|
| A. dANN-R | Variable, 16 en promedio | Matrices densas enmascaradas |
| B. Fixed-dANN | Exactamente 16 por dendrita | Matrices densas enmascaradas, sólo como control experimental |
| C. DSU-S | Exactamente 16 por dendrita | Pesos compactos e índices explícitos |

**A vs B** pregunta principalmente por regularizar el fan-in. **B vs C** pregunta por representación física **si** la especificación alinea topología exacta, pesos activos iniciales, sesgos, forward, seeds, datos y optimizador; sin esa equivalencia, la diferencia de calidad seguiría confundida. El diseño debe separar calidad de bytes, latencia y RAM. S3-T01 no implementa Fixed-dANN ni ejecuta experimentos.

**Backlog de almacenamiento topológico para Sprint 3 o posterior:** regenerar `G` desde seed, checkpoint sin buffer `G`, metadata int32/int16 y topología estructurada implícita. Cada opción debe evaluar por separado **bytes de checkpoint** y **representación residente/runtime**; comprimir al guardar y expandir a int64 en ejecución puede reducir el archivo sin reducir RAM runtime. No cambiar dtype ni formato en esta revisión.
