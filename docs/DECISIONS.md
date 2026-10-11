# DECISIONS

## ADR-001 — Proyecto de software primero

**Status:** ACCEPTED

### Decision
Investigar primero arquitectura y eficiencia en CPU/GPU convencional.

### Reason
No contamos con laboratorio ni hardware neuromórfico especializado.

### Consequence
Las ideas deben poder implementarse y medirse en una PC común.

---

## ADR-002 — No comenzar con spikes

**Status:** ACCEPTED

### Decision
DSU-S y DSU-T usarán inicialmente activaciones continuas.

### Reason
Queremos separar el valor de:
- dendritas;
- estado;
- estructura;

del valor específico de comunicación por spikes.

### Consequence
SNN queda para una fase posterior.

---

## ADR-003 — Evitar sparsity falsa

**Status:** ACCEPTED

### Decision
No usar como implementación principal una gran matriz densa multiplicada por máscaras de ceros.

### Reason
Reduce parámetros efectivos pero puede no reducir coste real.

### Consequence
La implementación debe explotar estructura física compacta.

---

## ADR-004 — Estado temporal sólo después de DSU-S

**Status:** ACCEPTED

### Decision
No implementar DSU-T hasta medir DSU-S.

### Reason
Queremos conocer qué ventaja proviene de la estructura espacial.

---

## ADR-005 — Transformer sólo después de señal clara

**Status:** ACCEPTED

### Decision
No comenzar por lenguaje ni Transformers.

### Reason
Sería una forma cara de probar una hipótesis todavía básica.

### Consequence
Primero benchmarks pequeños y diagnósticos.

---

## ADR-006 — Tiempo máximo inicial

**Status:** ACCEPTED

### Decision
Mantener el proyecto en aproximadamente 2–4 horas semanales.

### Reason
Facultad, empleo y proyectos de portfolio tienen prioridad práctica inmediata.

### Consequence
El diseño documental debe permitir pausar y retomar sin pérdida de contexto.

---

## ADR-007 — Baseline de Sprint 0

**Status:** ACCEPTED

### Decision
Usar un MLP denso 784→128→10 con ReLU, Adam y CrossEntropy sobre MNIST, sin normalización adicional. Un JSON estricto configura la ejecución. El smoke test utiliza prefijos fijos de los splits oficiales, con orden de entrenamiento determinado por la seed.

### Reason
Es un pipeline pequeño y público que se descarga automáticamente, permite tests rápidos y evita introducir dependencias de orquestación o decisiones de arquitectura DSU.

### Consequence
El smoke test comprueba funcionamiento, no compara arquitecturas ni estima calidad final. Versiones exactas se fijan en `uv.lock`; las métricas incluyen configuración y entorno. RAM pico en CPU queda sin medir por ahora.

---

## ADR-008 — DSU-S v0 con fan-in fijo y pesos compactos

**Status:** ACCEPTED

### Decision

Fijar la primera DSU-S en 128 somas, cuatro dendritas por soma y exactamente 16 índices distintos por dendrita, muestreados uniformemente sin reemplazo dentro de cada dendrita. Separar `topology_seed` de `training_seed`; mantener la topología fija y no entrenable. Almacenar sólo los 10634 pesos y sesgos conectados, más 8192 índices como metadata o buffers. La definición completa y el experimento inicial están en [DSU_S_SPEC.md](DSU_S_SPEC.md).

### Reason

La reproducción dANN-R mostró 10634 conexiones y sesgos efectivos, pero 468874 parámetros físicamente almacenados por sus matrices densas enmascaradas. El fan-in fijo permite una representación compacta regular sin añadir estado ni optimizaciones complejas.

### Consequence

DSU-S ya no replica la máscara aleatoria global de dANN-R: distribuye sus 8192 conexiones como 16 por dendrita. El primer benchmark medirá calidad, tamaño y latencia y declarará ambos cambios. El coste de índices se contabilizará aparte; los MAC teóricos no se interpretarán como velocidad medida.

---

## ADR-009 — Inicialización y persistencia de DSU-S v0

**Status:** ACCEPTED

### Decision

Inicializar explícitamente los pesos compactos con Glorot uniform (`W_d` visto como `[512,16]`, `W_s` como `[128,4]`, `W_o` como `[10,128]`) y los sesgos en cero. Generar `G` con un RNG NumPy local de `topology_seed`, guardarlo como buffer `int64` persistente en `state_dict` y registrar seed y hash.

### Reason

Glorot mantiene coherencia de familia con los baselines sin trasladar su matriz densa; el buffer conserva la topología exacta al serializar y se mueve con el modelo entre dispositivos. La escala concreta difiere de dANN-R y deberá declararse al comparar calidad.

### Consequence

`G` añade 65536 bytes nominales a los 42536 bytes de parámetros float32. La seed guardada como atributo no reemplaza al buffer en un checkpoint; al cargar, `G` del `state_dict` es la fuente de verdad.

---

## ADR-010 — Continuar con ablaciones causales tras Sprint 2

**Status:** ACCEPTED

### Decision

Cerrar Sprint 2 como PASS y avanzar a Sprint 3 — Topology and Causal Ablations. La primera tarea, S3-T01, será sólo el diseño de un control Fixed-dANN con fan-in exactamente 16 y una matriz causal dANN-R / Fixed-dANN / DSU-S. No se autoriza por esta decisión implementar el control ni entrenar modelos nuevos.

### Reason

DSU-S v0 aprendió en tres seeds, mantuvo calidad próxima a dANN-R de igual anchura almacenando 44.09× menos parámetros entrenables y mostró señal de capacidad por parámetro frente a MLP densos probados. Persisten confusores de fan-in, topología, inicialización y representación, y el payload tensorial conocido con índices int64 supera al del MLP quality-matched. El siguiente paso útil es aislar causas y costes físicos, no ampliar todavía la arquitectura.

### Consequence

El GO es un gate de investigación causal, no una afirmación de eficiencia total. Sprint 3 mantendrá separados calidad, parámetros entrenables, bytes de topología, checkpoint, RAM runtime y velocidad. Las opciones para comprimir o regenerar topología quedan en backlog; cada una requiere experimento propio y no cambia `G` en esta revisión. Véase [SPRINT_2_REVIEW.md](SPRINT_2_REVIEW.md).

---

## ADR-011 — Controles pareados de Sprint 3

**Status:** ACCEPTED

### Decision

Usar Fixed-dANN dense-mask sólo como control experimental: A vs B conserva almacenamiento denso y protocolo de dANN-R; B vs C debe compartir exactamente el buffer `G`, los pesos activos, sesgos y función antes de comparar representación. El PASS funcional y de gradientes es requisito para interpretar B/C. El diseño completo está en [SPRINT_3_CAUSAL_DESIGN.md](SPRINT_3_CAUSAL_DESIGN.md).

### Reason

La seed común no garantiza topología ni inicialización común. A/C mezcla fan-in, topología, escala de inicialización y almacenamiento.

### Consequence

A/B sólo estima el efecto conjunto de regularizar fan-in y cambiar la distribución topológica. B/C separa representación si se verifica equivalencia; pequeñas diferencias numéricas no prueban distinto mecanismo neuronal. La implementación y los experimentos requieren tareas posteriores.

---

## ADR-012 — Controles de parámetros y escala de S3-T06

**Status:** ACCEPTED para diseño preentrenamiento.

### Decision

Comparar PointSparse con una familia de compartimentos B=1,2,4,8 usando la misma G64 nested, 8192 pesos input, 128 sesgos soma y salida compartida en estructura: 9610 parámetros por brazo. No hay bias dendrítico ni peso entrenable rama→soma. La agregación fija es `sum(d)/sqrt(B)`, seguida por una transformación afín escalar sin labels que iguala media y DE a la preactivación PointSparse sobre 256 ejemplos train predeterminados por seed. Congelar ganancia y offset antes del entrenamiento.

### Reason

S3-T05 mezcló partición, 256B parámetros adicionales y escala de logits. La regla `1/sqrt(B)` preserva varianza sólo aproximadamente si las ramas son independientes; la media positiva tras LeakyReLU requiere medición. La calibración reduce diferencias triviales de escala sin optimizar accuracy.

### Consequence

El control sigue teniendo distinta cantidad de activaciones y operaciones fijas; media/DE igualadas del agregado no implican logits iguales. H-014/H-015 quedan UNTESTED hasta S3-T07. [Diseño y diagnóstico](S3_T06_SPARSE_VS_COMPARTMENT_CONTROLS.md).

---

## ADR-013 — Priorizar sparse architecture y posponer dendritas estáticas adicionales

**Status:** ACCEPTED para dirección de investigación en S3-T08; no autoriza nuevos entrenamientos.

### Decision

Organizar el trabajo en Línea A (sparse architecture, baseline PointSparse) y Línea B (dinámica neuronal, empezando en el futuro por PointSparse vs PointSparse + estado). Conservar DSU-T/DSU-A/DSU-AIS/DSU-R como candidatos posteriores de Línea B. Reabrir compartimentos estáticos sólo con hipótesis causal y control nuevos. [Programa A/B](RESEARCH_PROGRAM_A_B.md).

### Reason

H-009 negligible, H-010 equivalencia funcional, H-011 negligible, H-014 supported sólo en contraste histórico acotado, H-015 contrary y H-016 inconclusive. H-012 apoya almacenamiento compacto B/C, H-013 no apoya ventaja general de latencia. La evidencia no atribuye capacidad extra a más ramas estáticas bajo el control neutral estudiado; sparse width/topology es una explicación prioritaria todavía por verificar.

### Consequence

S3-T08 introduce contabilidad y registro, no cambia arquitecturas existentes. S3-T09 validará proxies. Sprint 3 requiere revisión posterior antes de cerrarse.

---

## ADR-014 — Cierre S3 y baseline PointSparse

**Status:** ACCEPTED en S3-REVIEW (2026-10-11).

### Decision

Cerrar Sprint 3 como PASS / CLOSED. Fijar PointSparse `128×64` como **default experimental baseline within the tested budget** para futuros controles de Línea A y B: 9610 parámetros, 9472 MAC principales por muestra y N=3/25 épocas disponibles. Conservar DSU-S v0, Fixed-dANN y MLP históricos para sus comparaciones propias. Pausar y despriorizar más compartimentos dendríticos **estáticos** bajo Fashion-MNIST hasta tener una predicción causal nueva. No declarar óptimo global ni abrir S4/DSU-T automáticamente. [Review](SPRINT_3_REVIEW.md).

### Reason

PointSparse retuvo descriptivamente 0.752 del gap histórico DSU-S−MLP pequeño sin compartimentos. En S3-T11 las tres variantes H/K fueron near-ties en validation y ninguna superó al anchor con el efecto material coherente preregistrado; `160×49` sí obtuvo mejor validation loss y test NLL, por lo que la selección del anchor no es una afirmación de dominancia. H-011 fue negligible, H-015 contrary y H-016 inconclusive bajo los controles probados. Compactar B/C explica bytes, no calidad; H-013 no sustenta speedup general.

### Consequence

La próxima decisión recomendada es A1 de sólo diseño, con presupuesto y contraste causal prefijados. Reabrir dendritas estáticas requiere tarea/mecanismo específico (temporal, contextual/apical, plasticidad local u otra predicción) y control nuevo. Línea B empieza, si se aprueba por gate propio, con PointSparse-static vs PointSparse-stateful en tarea temporal antes de añadir compartimentos dinámicos. S4 conserva su etiqueta histórica DSU-T como reserva, no como tarea iniciada.

---

## ADR-015 — Proxies y pilotos como filtros auxiliares

**Status:** ACCEPTED en S3-REVIEW (2026-10-11).

### Decision

Mantener el Structural Cost Model para contabilidad; tratar signal propagation como diagnóstico y zero-cost/pilotos baratos como filtros auxiliares. Ningún proxy pasa a criterio primario de selección hasta superar validación prospectiva preregistrada en cohorte nueva y target pertinente. Tres épocas no se usan como oracle para ranking final ni para descartar near-ties. Mantener separación entre parámetros, bytes, MAC, latencia, RAM y energía, y entre familias de protocolo.

### Reason

ER raw ordenó bien retrospectivamente S3-T07, pero falló frente a validation temprana en la cohorte completa T10; la correlación favorable de tres candidatos T11 no lo rehabilita. `activation_pattern_logdet` fue misleading. El piloto de época 3 fue WEAK entre familias y Spearman −.50 frente a las medias finales de la shortlist. El modelo estructural sí reprodujo conteos conocidos y los límites de payload/temporales están declarados. [Review](SPRINT_3_REVIEW.md).

### Consequence

El funnel de [EXPERIMENT_PROTOCOL](EXPERIMENT_PROTOCOL.md) sigue siendo orientativo. Cada promoción de candidato o proxy exige preregistro, presupuesto y criterio adverso; el test no calibra reglas de selección. No se autoriza aquí ningún proxy nuevo ni entrenamiento.
