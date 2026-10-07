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

**S3-T01, siguiente tarea:** diseñar únicamente el control Fixed-dANN de fan-in exactamente 16 y la matriz causal dANN-R (variable/dense-mask), Fixed-dANN (fijo/dense-mask), DSU-S (fijo/compact). A vs B pregunta por fan-in; B vs C sólo aísla representación si topología, inicialización y matemática se alinean. No implementar ni entrenar en S3-T01.

**Backlog de topología:** estudiar regeneración desde seed, checkpoint sin `G`, metadata int32/int16 y topología implícita. Distinguir ahorro de checkpoint de memoria residente/runtime; no cambiar dtype hasta una tarea propia. No introducir todavía estado temporal, DSU-T, gating apical, spikes ni Transformers.

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
