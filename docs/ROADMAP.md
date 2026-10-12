# ROADMAP

## Sprint 0 — Fundación experimental

### Objetivo
Crear un entorno reproducible antes de diseñar la neurona.

### S0-T01 — Estructura del repo
PASS:
- documentación presente;
- entorno instalable;
- tests ejecutables;
- commit inicial limpio.

### S0-T02 — Configuración experimental
Debe permitir:
- modelo;
- dataset;
- seed;
- epochs;
- batch size;
- learning rate;
- device.

### S0-T03 — Harness de métricas
Registrar:
- parámetros;
- loss;
- accuracy;
- duración;
- memoria cuando sea posible.

### S0-T04 — Baseline MLP
PASS:
- entrena;
- converge;
- reproduce resultados;
- exporta métricas.

### Gate S0
No seguir si todavía no podemos medir modelos de forma confiable.

---

## Sprint 1 — Reproducción científica

### Objetivo
Reproducir una arquitectura dendrítica publicada.

### Tareas
- S1-T01 seleccionar paper principal.
- S1-T02 reproducir experimento pequeño.
- S1-T03 comparar con baseline.
- S1-T04 documentar divergencias.

### Gate
Resultado:
- REPRODUCED
- PARTIALLY REPRODUCED
- NOT REPRODUCED

---

## Sprint 2 — DSU-S

### Objetivo
Implementar la primera unidad propia.

### Tareas
- S2-T01 cerrar especificación matemática.
- S2-T02 implementar DSU-S.
- S2-T03 tests de forward/backward/dimensiones.
- S2-T04 benchmark inicial.
- S2-T05 comparación parameter-matched.
- S2-T06 comparación quality-matched.

### Gate
**PASS — S2-REVIEW (2026-10-06).** DSU-S aprendió de forma estable, conservó calidad próxima a dANN-R de igual anchura y mostró señal de capacidad por parámetro frente a la familia MLP probada. **GO TO SPRINT 3** para resolver causalidad. El payload tensorial conocido de DSU-S con topología int64 es mayor que el del MLP quality-matched; no hay evidencia de ventaja de almacenamiento total, latencia o energía. Véase [SPRINT_2_REVIEW.md](SPRINT_2_REVIEW.md).

---

## Sprint 3 — Topology and Causal Ablations

**Objetivo:** separar el efecto de fan-in regular, organización dendrita→soma y representación compacta antes de variar número de dendritas o fan-in. Las comparaciones posteriores podrán incluir solapamiento, grupos y permutación, una variable conceptual por experimento.

**S3-T01: PASS de diseño (2026-10-08).** La matriz A/B/C, el emparejamiento de topología y pesos, y los límites causales están en [SPRINT_3_CAUSAL_DESIGN.md](SPRINT_3_CAUSAL_DESIGN.md). A vs B examina fan-in regular junto a distribución topológica; B vs C requiere equivalencia funcional antes de medir representación. No hubo implementación ni entrenamiento.

**S3-T02: PASS (2026-10-08).** Fixed-dANN dense-mask comparte exactamente `G` y pesos efectivos con DSU-S; forward, loss, gradientes y cinco pasos Adam fueron equivalentes dentro de tolerancia CPU float32. Detalle en [EXPERIMENT_004_FIXED_DANN_EQUIVALENCE.md](EXPERIMENT_004_FIXED_DANN_EQUIVALENCE.md).

**S3-T03 / S3-E01: PASS técnico (2026-10-08).** dANN-R vs Fixed-dANN completó N=5 con inicialización y minibatches pareados. El efecto medio fue despreciable según criterios prefijados; H-009 = NOT SUPPORTED / NEGLIGIBLE. Véase [EXPERIMENT_005_FIXED_FANIN_ABLATION.md](EXPERIMENT_005_FIXED_FANIN_ABLATION.md).

**S3-T04: PASS técnico (2026-10-09).** Benchmark físico B/C: H-012 storage SUPPORTED; H-013 runtime MIXED / NOT SUPPORTED como ventaja general. [Resultados](EXPERIMENT_006_PHYSICAL_REPRESENTATION_BENCHMARK.md).

**S3-T05: PASS técnico (2026-10-09).** Familia original `1×64/2×32/4×16/8×8`, 64 conexiones por soma: H-011 NEGLIGIBLE bajo su protocolo. [Resultados](EXPERIMENT_007_DENDRITIC_COMPARTMENT_ABLATION.md).

**S3-T06: PASS de diseño/diagnóstico (2026-10-10).** Controles PointSparse y compartimentos con presupuesto neutral de 9610 parámetros, sin entrenamiento. [Diseño](S3_T06_SPARSE_VS_COMPARTMENT_CONTROLS.md).

**S3-T07: PASS técnico (2026-10-10).** N=3, H-014 SUPPORTED en contraste histórico acotado, H-015 CONTRARY, H-016 INCONCLUSIVE. [Resultados](EXPERIMENT_008_SPARSE_VS_COMPARTMENT_CAUSAL.md).

**Backlog de topología:** estudiar regeneración desde seed, checkpoint sin `G`, metadata int32/int16 empaquetada, topología implícita y topología estructurada/por bloques. Distinguir ahorro de checkpoint de memoria residente/runtime; no cambiar dtype hasta una tarea propia. No introducir todavía estado temporal, DSU-T, AIS, inhibición, rewiring, gating apical, spikes, otros datasets ni Transformers.

**Actualización S3-T08 (2026-10-10):** S3-T04/T05/T06/T07 ya se ejecutaron o cerraron según [PROJECT_STATE](../PROJECT_STATE.md). La evidencia de S3-T07 favorece investigar sparse width/topology y deja la expansión dendrítica estática en pausa. [Programa A/B](RESEARCH_PROGRAM_A_B.md), [framework analítico](ANALYTICAL_SCREENING_FRAMEWORK.md) y funnel del [protocolo](EXPERIMENT_PROTOCOL.md) fijan la nueva dirección. **S3-T09 siguiente:** implementar y validar primeros proxies zero-cost contra el registro histórico, sin nuevas búsquedas o entrenamiento en T08. Revisar el cierre de Sprint 3 después de T09; no queda cerrado automáticamente. Las líneas dinámicas y Sprint 4 requieren gate separado.

**S3-T09: PASS técnico (2026-10-10).** Primera tanda forward-only, 16 arquitecturas, tres seeds y validación retrospectiva por grupo. Effective rank raw es prometedor sólo en alcance acotado; pattern logdet falló. [Resultados](EXPERIMENT_009_ZERO_COST_FORWARD_PROXIES.md). Su señal motivó la validación prospectiva S3-T10. Sprint 3 continúa abierto.

**S3-T10: PASS técnico (2026-10-11).** Preregistro prospectivo sellado antes de entrenar, ocho PointSparse iso-presupuesto y piloto de tres épocas/una seed sólo con train/validation. ER raw falló frente a accuracy temprana y no superó la heurística de ancho; shortlist fijada: 80×109, 160×49 y 128×64. [Resultados](EXPERIMENT_010_PROSPECTIVE_ZERO_COST_SCREEN.md). Esta tarea no ejecutó S3-T11 ni abrió Línea B.

**S3-T11: PASS técnico (2026-10-11).** Shortlist congelada, anchor S3-T07 reutilizado tras auditoría y seis corridas nuevas N=3/25 épocas. Las tres medias de validation final fueron near-ties; anchor ganó test accuracy media, `160×49` ganó validation y NLL. `EARLY-PILOT-3E = WEAK`, `H-WIDTH-FANIN = INCONCLUSIVE` para calidad final del rango probado. [Resultados](EXPERIMENT_011_POINTSPARSE_SHORTLIST_CONFIRMATION.md). **S3-REVIEW recomendado** como cierre formal separado; no se inició Línea B.

**S3-REVIEW: PASS; SPRINT 3: CLOSED (2026-10-11).** [Review formal](SPRINT_3_REVIEW.md): fan-in fijo negligible en A/B, representación compacta equivalente y menor storage B/C, latencia mixta, compartimentos estáticos sin ventaja causal suficiente, PointSparse `128×64` como baseline experimental por defecto, ER prospectivo fallido y piloto de tres épocas débil. No quedan preguntas necesarias para el objetivo original Topology and Causal Ablations. Lo pendiente se asigna a Research Track A/B y requiere gates nuevos.

---

## Research Track A — Sparse Architecture (después de S3)

Baseline: PointSparse `128×64`. **A1-DESIGN: PASS estructural (2026-10-11).** [Preregistro](A1_POINTSPARSE_TOPOLOGY_DESIGN.md) de cuatro brazos a H/K fijos: A0 IID, A1 balanceado, A2 menor overlap y A3 mayor overlap. A1/A2/A3 conservan vector de grados, coverage, parámetros y MAC. H-WIDTH-FANIN sigue INCONCLUSIVE; no se entrenó. Las preguntas anteriores A2/A3 de topología/solapamiento quedan concentradas en A1 y no son tareas adicionales comprometidas. A4 reorganización de `G`, A5 coste físico y A6 generalización siguen en backlog. Siguiente tarea exacta: **A1-E01**, sólo mediante gate propio y revalidación de A0. Esta línea puede preceder o coexistir con S4 sin renumerarlo. [Programa](RESEARCH_PROGRAM_A_B.md).

**A1-E01: PASS técnico (2026-10-11).** [Informe](EXPERIMENT_A1_E01_POINTSPARSE_TOPOLOGY.md): A0 reutilizado, nueve corridas nuevas N=3/25; C1 IID→balanceado INCONCLUSIVE y C2/C3/C4 con grados idénticos NEGLIGIBLE. No hubo efecto MATERIAL. H-A1-TOPOLOGY = NEGLIGIBLE dentro de las topologías probadas. Track A pasa a BACKLOG / DEPRIORITIZED; máximo una tarea diagnóstica futura si la discordancia NLL/accuracy de C1 requiere resolución. Baseline PointSparse IID `128×64` conservado. No iniciar búsqueda de G ni reabrir H/K.

## Research Track B — Dynamic Neuron (separado)

Primer contraste futuro: PointSparse-static vs PointSparse-stateful en tarea temporal con coste de estado y reset definidos. Sólo tras señal positiva, estudiar PointSparse-stateful vs dendritic-stateful; luego heterogeneidad temporal, AIS/excitabilidad, compartimentos dinámicos y plasticidad según [programa A/B](RESEARCH_PROGRAM_A_B.md). Ninguna tarea B se inicia con S3-REVIEW.

**Siguiente tarea recomendada: B1-DESIGN**, preregistro separado del contraste temporal estático/stateful. A1-E01 no ejecutó Line B ni abrió S4.

---

## Sprint 4 — DSU-T

**Reserva histórica para Línea B; NO INICIADO.** Añadir memoria sólo tras un gate separado que pruebe primero el efecto de estado en PointSparse estática frente a PointSparse-stateful. La numeración S4 se conserva, pero las tareas A1/A2 pueden realizarse antes o de forma independiente. No usar dendritas stateful como primer contraste porque mezclarían efecto de estado y compartimentos.

### Tareas
- estado básico;
- lambda fija;
- lambda entrenable;
- múltiples escalas temporales;
- comparación contra baselines temporales.

### Gate
La memoria sobrevive sólo si compensa su coste.

---

## Sprint 5 — Optimización

Medir:
- kernels;
- allocations;
- memoria;
- bottlenecks;
- tiempo por operación.

Investigar:
- vectorización;
- grouped operations;
- block structure;
- torch.compile;
- Triton/kernel especializado sólo si hace falta.

---

## Sprint 6 — Mini Transformer

Sustituir FFN convencional por DSU.

Mantener atención idéntica inicialmente.

Medir:
- parámetros;
- perplexity;
- memoria;
- tiempo de entrenamiento;
- tokens/s;
- latencia token a token.

---

## Sprint 7 — DSU-A

Sólo si existe una razón experimental.

Explorar:
- gating contextual;
- continual learning;
- contextual routing;
- synthetic ICL.

---

## Sprint 8 — Spiking / neuromorphic

Fase futura.

Convertir únicamente una arquitectura previamente demostrada.

Nunca usar spikes como rescate de una arquitectura base que no funciona.

---

# Milestones

## M0
Infraestructura reproducible.

## M1
Reproducción de literatura.

## M2
DSU aprende correctamente.

## M3
Existe señal de eficiencia.

## M4
La ventaja se transforma en memoria/latencia real.

## M5
Viabilidad en Transformer pequeño.

## M6
Candidato serio a informe técnico o preprint.
