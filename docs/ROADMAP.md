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
GO si existe señal de eficiencia paramétrica u otra ventaja reproducible.

---

## Sprint 3 — Topología

Comparar:
- número de dendritas;
- tamaño de grupo;
- solapamiento;
- grupos aleatorios;
- grupos estructurados;
- permutación.

### Pregunta
¿Qué topología maximiza calidad por parámetro sin destruir velocidad?

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
