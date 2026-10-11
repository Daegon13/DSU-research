# Programa de investigación A/B — S3-T08

**Estado:** programa formalizado tras [S3-REVIEW](SPRINT_3_REVIEW.md), gate PASS y Sprint 3 CLOSED. No es un nuevo resultado experimental ni autoriza iniciar tareas A/B.

## Evidencia que motiva el cambio

H-009: regularizar fan-in a K=16 tuvo efecto medio despreciable (−0.042 pp accuracy, +0.001066 loss). H-010: Fixed-dANN y DSU-S representan la misma función dentro de tolerancias CPU float32; compactar explica bytes, no una ganancia de accuracy. H-012: el payload tensorial y los checkpoints B/C se reducen ~22× y el checkpoint de entrenamiento con Adam ~31×. H-013: la latencia depende de batch y modo; no hay speedup general. H-011: la familia original con 64 conexiones/soma dio accuracy casi plana. H-014: PointSparse retuvo descriptivamente 0.752 del gap histórico DSU-S–MLP pequeño, pero los tres protocolos no están completamente pareados. H-015: la familia neutral perdió NLL, accuracy y margen al aumentar B; H-016 sigue inconclusa. Véanse [S3-T04](EXPERIMENT_006_PHYSICAL_REPRESENTATION_BENCHMARK.md), [S3-T05](EXPERIMENT_007_DENDRITIC_COMPARTMENT_ABLATION.md) y [S3-T07](EXPERIMENT_008_SPARSE_VS_COMPARTMENT_CAUSAL.md).

La hipótesis prioritaria es que muchas unidades con conectividad restringida pueden explicar una parte importante de la eficiencia paramétrica observada. Es una hipótesis, no una atribución causal cerrada. Los controles sólo cubren Fashion-MNIST, CPU, familias concretas y N=3 en la etapa final. No demuestran que toda dendrita estática sea inútil ni que PointSparse sea óptima.

## Línea A — Sparse Architecture

**Objetivo:** entender la relación entre ancho H, fan-in K, cobertura/solapamiento, topología y capacidad bajo presupuestos declarados; usar PointSparse como baseline. Preguntas: ¿H o K domina a presupuesto fijo? ¿Cuánta superposición entre unidades ayuda? ¿Basta topología aleatoria? ¿Cuándo superan topologías estructuradas o aprendidas? ¿Qué medidas previas al entrenamiento anticipan calidad y cuáles anticipan coste real? Separar siempre parámetros, conexiones, bytes, MAC, latencia y energía.

**Orden tras S3-T11:** (A1) diseñar un contraste causal acotado de ancho frente a conectividad/topología con PointSparse `128×64` como control y presupuesto explícito; (A2) topología aleatoria frente a alternativa prefijada; (A3) solapamiento de receptive fields; (A4) `G` aprendible/reorganizable; (A5) implementación sparse físicamente eficiente; (A6) generalización a otras tareas. A1 comienza sólo como decisión/diseño separado, no como entrenamiento autorizado. T10/T11 hicieron un cribado H/K de presupuesto igual y encontraron near-ties finales/objetivos discordantes; no continuar con grid searches extensos. ER raw no es selector primario, y un piloto de tres épocas sólo puede ser filtro exploratorio. Los pasos S3-T08/T09/T10/T11 son historia cerrada, no una receta de selección validada.

## Línea B — Dynamic Neuron

**Objetivo:** medir si una unidad con dinámica sustituye suficientes unidades o capas para compensar su estado y coste. Baseline inicial: PointSparse estática `128×64` contra **PointSparse + estado temporal por unidad** en un problema con estructura temporal y protocolo nuevo. Sólo si hay señal, comparar PointSparse-stateful con dendritic-stateful para separar efecto de estado y de compartimentos. Orden preliminar: B1 estado simple; B2 constantes de tiempo heterogéneas; B3 excitabilidad adaptativa/AIS; B4 compartimentos dendríticos dinámicos; B5 plasticidad local/estructural. DSU-A y variantes quedan en backlog condicionado. DSU-T ocupa una reserva S4 histórica, no una tarea iniciada ni una autorización de implementación.

Cada mecanismo requiere una predicción falsable, control de parámetros/estado, secuencias y reset definidos, coste físico y un baseline temporal razonable. Una ganancia en accuracy sola no basta para justificar complejidad.

## Promoción de backlog a experimento

Una idea pasa a diseño experimental sólo si identifica mecanismo, baseline, variable conceptual, métrica primaria, presupuesto comparable, estimación de coste, criterio adverso y artefactos reproducibles. El protocolo se congela antes de usar validation para selección; test permanece cerrado hasta la decisión final. Las señales pequeñas o contradictorias quedan en backlog o se replican con N mayor mediante tarea separada.

Se aparcan la ampliación de compartimentos dendríticos **estáticos** sin nueva predicción causal, los barridos H×K exhaustivos, DSU-T inmediato, kernels y Transformer. Reabrir dendritas estáticas exige un mecanismo específico que PointSparse no capture, un control pareado de topología/parámetros/escala, una ventaja repetible predefinida y coste físico aceptable. La línea B conserva las dendritas para dinámica y contexto.

El [cierre S3](SPRINT_3_REVIEW.md) fija la política Stage 0–4 y los límites de claims. Tareas A pueden preceder o acompañar una decisión B posterior sin renumerar Sprint 4. La próxima decisión exacta es **A1-DESIGN, sólo diseño/preregistro**; no se ejecuta en este documento.
