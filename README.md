# DSU Research

**Dendritic Stateful Unit — research project**

Proyecto experimental para investigar si una unidad neuronal artificial con:

- agregación dendrítica local,
- conectividad estructurada,
- estado temporal aprendible,
- integración somática,

puede ofrecer una mejor relación entre **calidad, parámetros, memoria y latencia** que capas densas convencionales ejecutadas en hardware común.

## Objetivo

La pregunta principal del proyecto es:

> ¿Puede una Dendritic Stateful Unit (DSU) proporcionar mejor calidad por parámetro y por byte de memoria que capas densas convencionales, manteniendo una inferencia eficiente en CPU/GPU?

Una pregunta posterior, sólo si la hipótesis inicial funciona, será:

> ¿Puede una DSU sustituir parcial o totalmente el FFN de un Transformer pequeño?

## Estado actual

El proyecto está en fase de fundación experimental.

**Prioridad:** proyecto lateral de investigación.  
**Dedicación recomendada:** 2–4 horas por semana.  
**Primer Go/No-Go:** después de 6–8 semanas parciales de trabajo.

## Documentación principal

- `PROJECT_STATE.md` — fuente de verdad operacional.
- `docs/PROJECT.md` — visión, alcance y objetivos.
- `docs/HYPOTHESES.md` — hipótesis registradas y su estado.
- `docs/ARCHITECTURE.md` — especificación técnica de DSU-S, DSU-T y futuras variantes.
- `docs/ROADMAP.md` — sprints, gates y milestones.
- `docs/EXPERIMENT_PROTOCOL.md` — reglas para que los experimentos sean comparables.
- `docs/BENCHMARKS.md` — baselines y métricas.
- `docs/LITERATURE.md` — mapa inicial del estado del arte.
- `docs/DECISIONS.md` — decisiones arquitectónicas.
- `docs/RESULTS.md` — resultados positivos, negativos e inconclusos.
- `templates/SESSION_TEMPLATE.md` — plantilla de cierre de sesión.
- `templates/EXPERIMENT_TEMPLATE.md` — plantilla para cada experimento.

## Regla principal

No escalar por entusiasmo.

Escalar por evidencia.

```text
Hipótesis
   ↓
Experimento pequeño
   ↓
Medición
   ↓
Ablación
   ↓
Decisión
```

## Próximo paso

Ejecutar `S0-T01`: inicializar el repositorio usando esta documentación como base.

Después, continuar con el Sprint 0 sin implementar todavía la neurona DSU.
