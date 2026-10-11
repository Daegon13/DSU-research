# RECOVERY PROMPT

Usar este texto al iniciar una conversación nueva sobre el proyecto:

> Estamos trabajando en DSU Research, una investigación sobre una Dendritic Stateful Unit orientada a mejorar calidad por parámetro, memoria y latencia en hardware convencional. Antes de proponer cambios, lee PROJECT_STATE.md, docs/PROJECT.md, docs/ARCHITECTURE.md, docs/DECISIONS.md y docs/EXPERIMENT_PROTOCOL.md. No reconstruyas decisiones desde memoria si ya están documentadas. Dime el estado actual, el último resultado confirmado, la hipótesis activa, los blockers y la siguiente tarea pequeña. Después continuamos desde allí.

## Regla

Si `PROJECT_STATE.md` contradice una conversación anterior, usar `PROJECT_STATE.md` como fuente de verdad operacional salvo evidencia clara de que quedó desactualizado.

## Handoff tras S3-REVIEW (2026-10-11)

Sprint 3 = **PASS / CLOSED** por [SPRINT_3_REVIEW](SPRINT_3_REVIEW.md). No se iniciaron tareas de Línea A/B, S4 ni DSU-T. La expansión de compartimentos **estáticos** quedó PAUSED / DEPRIORITIZED: H-011 negligible, H-015 contrary bajo control neutral y H-016 inconclusive; esto no invalida dendritas en tareas temporales/contextuales. PointSparse `128×64` es el baseline experimental por defecto dentro del presupuesto probado (9610 parámetros, 9472 MAC principales, N=3/25 épocas), sin óptimo global demostrado; T11 dio near-ties en validation y métricas finales discordantes.

ER raw mostró señal retrospectiva acotada en T09 y **falló prospectivamente** en la cohorte T10; no usarlo como selector único. El piloto de tres épocas fue **WEAK** entre familias y falló al ordenar la shortlist cercana T10→T11; sirve sólo como filtro exploratorio. Structural Cost Model sirve para contabilidad, no para deducir RAM pico, latencia o energía.

**Línea A** estudia sparse architecture, primero ancho/conectividad/topología y luego solapamiento, `G`, hardware y generalización. **Línea B** estudia dinámica por separado: primero PointSparse-static vs PointSparse-stateful en una tarea temporal, antes de dendritic-stateful. S4 conserva la reserva histórica DSU-T pero no está iniciado. **Próxima decisión recomendada:** `A1-DESIGN`, sólo diseño/preregistro de un contraste causal acotado con PointSparse `128×64` como control; no entrenar ni iniciar B automáticamente. Leer [ROADMAP](ROADMAP.md), [DECISIONS](DECISIONS.md) ADR-014/015 y [RESEARCH_PROGRAM_A_B](RESEARCH_PROGRAM_A_B.md) antes de abrir otra tarea.
