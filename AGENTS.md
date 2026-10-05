# AGENTS.md

## Propósito

Este repositorio contiene una investigación experimental sobre una arquitectura llamada provisionalmente **DSU — Dendritic Stateful Unit**.

Toda herramienta o agente que trabaje en el repositorio debe priorizar:

1. reproducibilidad;
2. cambios pequeños;
3. comparaciones justas;
4. trazabilidad;
5. documentación del estado real.

## Fuente de verdad

Antes de modificar código o documentación:

1. Leer `PROJECT_STATE.md`.
2. Leer `docs/PROJECT.md`.
3. Leer `docs/ARCHITECTURE.md`.
4. Revisar `docs/DECISIONS.md`.
5. Consultar `docs/EXPERIMENT_PROTOCOL.md` antes de tocar benchmarks.

No reconstruir decisiones desde memoria si ya están documentadas.

## Reglas de implementación

- No introducir varias ideas arquitectónicas simultáneamente.
- No añadir optimizaciones prematuras.
- No modificar baselines para favorecer DSU.
- No cambiar splits, seeds o métricas silenciosamente.
- No usar matrices densas llenas de ceros para simular sparsity salvo experimento explícito.
- Preferir operaciones estructuradas, agrupadas y reproducibles.
- Añadir tests a toda nueva unidad neuronal.
- Registrar resultados negativos.
- Mantener commits pequeños y de una sola intención.

## Formato de tareas

Toda tarea técnica debería incluir:

```text
TASK ID
OBJECTIVE
WHY
CONTEXT
ALLOWED FILES
FORBIDDEN FILES
REQUIREMENTS
OUT OF SCOPE
ACCEPTANCE CRITERIA
TESTS
EXPECTED OUTPUT
EXPECTED COMMIT
STOP CONDITIONS
```

## Definition of Done

Una tarea termina sólo cuando:

- el código ejecuta;
- los tests relevantes pasan;
- los resultados fueron inspeccionados;
- las métricas relevantes fueron registradas;
- no se introdujeron errores conocidos;
- el estado del proyecto fue actualizado si corresponde.

## Principio de investigación

Una modificación por experimento siempre que sea posible.

Nunca confundir:

- menos parámetros con menos FLOPs;
- menos FLOPs con menor latencia;
- menor latencia con menor energía;
- mejor accuracy con una arquitectura globalmente mejor.
