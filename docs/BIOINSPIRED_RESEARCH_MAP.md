# Mapa bioinspirado — cierre S3-REVIEW

**Estado:** Sprint 3 PASS / CLOSED. Este mapa conserva candidatos y condiciones; ninguna línea futura se inició. [Decisión y límites](SPRINT_3_REVIEW.md).

| Idea | Estado y línea | Próxima condición |
|---|---|---|
| DSU-S, ramas estáticas | PAUSED / DEPRIORITIZED; H-011 negligible, H-015 contrary y H-016 inconclusive en controles estudiados | Reabrir sólo con predicción específica, control causal y coste nuevos; no inferir inutilidad universal |
| Conectividad sparse con muchas unidades | Línea A; PointSparse `128×64` es baseline dentro del presupuesto probado, H-014 acotada; H-WIDTH-FANIN inconclusive | A1 diseño causal acotado; después topología/solapamiento; sin grid H×K grande ni ER como selector único |
| Estado temporal por unidad | Línea B; no iniciada | B1: PointSparse-static vs PointSparse-stateful en tarea temporal, reset/coste/control fijados |
| Escalas temporales heterogéneas | Línea B, B2 | Justificación tras B1 y comparación separada |
| Excitabilidad adaptativa / AIS | Línea B, B3 | Predicción específica y control tras estado simple |
| DSU-T con compartimentos dinámicos | Línea B, B4; S4 reservado, no iniciado | Señal de PointSparse-stateful y comparación contra dendritic-stateful con coste medido |
| DSU-A, modulación contextual/apical | Línea B; backlog condicionado | Tarea que necesite contexto y control pareado |
| DSU-R, plasticidad local o rewiring | Línea B, B5; backlog condicionado | Mecanismo falsable y protocolo de estabilidad |
| Spikes y hardware neuromórfico | Backlog lejano | Arquitectura continua previamente justificada |

Esta organización no cambia implementaciones ni interpreta los resultados estáticos como prueba universal. El [programa A/B](RESEARCH_PROGRAM_A_B.md) define los gates.
