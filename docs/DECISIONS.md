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
