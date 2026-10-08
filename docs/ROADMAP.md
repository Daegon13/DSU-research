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

**S3-E01, siguiente tarea:** ejecutar dANN-R vs Fixed-dANN con el protocolo causal congelado para estudiar regularización de fan-in junto a distribución topológica. No se ejecutó en S3-T02.

**Experimentos siguientes:** S3-E01 A/B con N=5 propuesto y protocolo congelado; S3-E02 B/C primero equivalencia y luego accounting, trayectoria y rendimiento físico. Segunda fase: barridos separados `B∈{1,2,4,8}` y `K∈{8,16,32}`. **Alta prioridad:** `1×64 vs 2×32 vs 4×16 vs 8×8`, 64 conexiones por soma con cobertura controlada, para estudiar compartimentalización dendrítica; sus conteos de sesgos/pesos soma difieren y deben declararse.

**Backlog de topología:** estudiar regeneración desde seed, checkpoint sin `G`, metadata int32/int16 empaquetada, topología implícita y topología estructurada/por bloques. Distinguir ahorro de checkpoint de memoria residente/runtime; no cambiar dtype hasta una tarea propia. No introducir todavía estado temporal, DSU-T, AIS, inhibición, rewiring, gating apical, spikes, otros datasets ni Transformers.

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
