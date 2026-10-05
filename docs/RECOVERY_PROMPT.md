# RECOVERY PROMPT

Usar este texto al iniciar una conversación nueva sobre el proyecto:

> Estamos trabajando en DSU Research, una investigación sobre una Dendritic Stateful Unit orientada a mejorar calidad por parámetro, memoria y latencia en hardware convencional. Antes de proponer cambios, lee PROJECT_STATE.md, docs/PROJECT.md, docs/ARCHITECTURE.md, docs/DECISIONS.md y docs/EXPERIMENT_PROTOCOL.md. No reconstruyas decisiones desde memoria si ya están documentadas. Dime el estado actual, el último resultado confirmado, la hipótesis activa, los blockers y la siguiente tarea pequeña. Después continuamos desde allí.

## Regla

Si `PROJECT_STATE.md` contradice una conversación anterior, usar `PROJECT_STATE.md` como fuente de verdad operacional salvo evidencia clara de que quedó desactualizado.
