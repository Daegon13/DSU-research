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
