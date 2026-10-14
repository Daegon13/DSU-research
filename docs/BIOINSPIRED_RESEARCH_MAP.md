# Mapa bioinspirado — cierre S3-REVIEW

**Estado:** Sprint 3 PASS / CLOSED; Línea B completó B1-E01 y B1-E02 (2026-10-13). Los mecanismos posteriores siguen condicionados a tareas propias. [Decisión y límites S3](SPRINT_3_REVIEW.md); [B1-E02](EXPERIMENT_B1_E02_STATEFUL_INTERFERENCE.md).

| Idea | Estado y línea | Próxima condición |
|---|---|---|
| DSU-S, ramas estáticas | PAUSED / DEPRIORITIZED; H-011 negligible, H-015 contrary y H-016 inconclusive en controles estudiados | Reabrir sólo con predicción específica, control causal y coste nuevos; no inferir inutilidad universal |
| Conectividad sparse con muchas unidades | Línea A; PointSparse `128×64` es baseline dentro del presupuesto probado, H-014 acotada; H-WIDTH-FANIN inconclusive | A1 diseño causal acotado; después topología/solapamiento; sin grid H×K grande ni ER como selector único |
| Estado temporal por unidad | Línea B, B1-E01/E02 completados; H-B1-STATE SUPPORTED sólo en EASY y H-B1-INTERFERENCE SUPPORTED sólo en NOISE `.1`; accuracy S1 en techo | No inferir límite de memoria, eficiencia física ni beneficio dendrítico; conservar los controles/costes |
| Escalas temporales heterogéneas | Línea B, B2 no iniciado | B2-DESIGN documental: preregistrar escala fija vs heterogéneas fijas en tarea capaz de revelar tradeoff corto/largo; no asumir beneficio |
| Excitabilidad adaptativa / AIS | Línea B, B3 | Predicción específica y control tras estado simple |
| DSU-T con compartimentos dinámicos | Línea B, B4; S4 reservado, no iniciado | Señal de PointSparse-stateful y comparación contra dendritic-stateful con coste medido |
| DSU-A, modulación contextual/apical | Línea B; backlog condicionado | Tarea que necesite contexto y control pareado |
| DSU-R, plasticidad local o rewiring | Línea B, B5; backlog condicionado | Mecanismo falsable y protocolo de estabilidad |
| Spikes y hardware neuromórfico | Backlog lejano | Arquitectura continua previamente justificada |

Esta organización no cambia implementaciones ni interpreta los resultados estáticos como prueba universal. El [programa A/B](RESEARCH_PROGRAM_A_B.md) define los gates.
