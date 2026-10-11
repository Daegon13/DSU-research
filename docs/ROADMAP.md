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

**S3-T10: PASS técnico (2026-10-11).** Preregistro prospectivo sellado antes de entrenar, ocho PointSparse iso-presupuesto y piloto de tres épocas/una seed sólo con train/validation. ER raw falló frente a accuracy temprana y no superó la heurística de ancho; shortlist futura 80×109, 160×49 y 128×64. [Resultados](EXPERIMENT_010_PROSPECTIVE_ZERO_COST_SCREEN.md). **S3-T11 siguiente:** confirmar esa shortlist y diseñar mejores proxies con protocolo congelado. No se ejecutó S3-T11 ni se abrió Línea B.

---

## Sprint 4 — DSU-T

Añadir memoria sólo si DSU-S lo justifica.

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
