# Programa de investigación A/B — S3-T08

**Estado:** programa propuesto y formalizado; no es un nuevo resultado experimental. Sprint 3 sigue abierto.

## Evidencia que motiva el cambio

H-009: regularizar fan-in a K=16 tuvo efecto medio despreciable (−0.042 pp accuracy, +0.001066 loss). H-010: Fixed-dANN y DSU-S representan la misma función dentro de tolerancias CPU float32; compactar explica bytes, no una ganancia de accuracy. H-012: el payload tensorial y los checkpoints B/C se reducen ~22× y el checkpoint de entrenamiento con Adam ~31×. H-013: la latencia depende de batch y modo; no hay speedup general. H-011: la familia original con 64 conexiones/soma dio accuracy casi plana. H-014: PointSparse retuvo descriptivamente 0.752 del gap histórico DSU-S–MLP pequeño, pero los tres protocolos no están completamente pareados. H-015: la familia neutral perdió NLL, accuracy y margen al aumentar B; H-016 sigue inconclusa. Véanse [S3-T04](EXPERIMENT_006_PHYSICAL_REPRESENTATION_BENCHMARK.md), [S3-T05](EXPERIMENT_007_DENDRITIC_COMPARTMENT_ABLATION.md) y [S3-T07](EXPERIMENT_008_SPARSE_VS_COMPARTMENT_CAUSAL.md).

La hipótesis prioritaria es que muchas unidades con conectividad restringida pueden explicar una parte importante de la eficiencia paramétrica observada. Es una hipótesis, no una atribución causal cerrada. Los controles sólo cubren Fashion-MNIST, CPU, familias concretas y N=3 en la etapa final. No demuestran que toda dendrita estática sea inútil ni que PointSparse sea óptima.

## Línea A — Sparse Architecture

**Objetivo:** entender la relación entre ancho H, fan-in K, cobertura/solapamiento, topología y capacidad bajo presupuestos declarados; usar PointSparse como baseline. Preguntas: ¿H o K domina a presupuesto fijo? ¿Cuánta superposición entre unidades ayuda? ¿Basta topología aleatoria? ¿Cuándo superan topologías estructuradas o aprendidas? ¿Qué medidas previas al entrenamiento anticipan calidad y cuáles anticipan coste real? Separar siempre parámetros, conexiones, bytes, MAC, latencia y energía.

**Orden:** (1) S3-T08 contabilidad y registro; (2) S3-T09 proxies de bajo coste y validación retrospectiva, sin usar test para elegir; (3) una sola comparación de H/K/topología previamente congelada si los proxies muestran utilidad; (4) medir hardware de las candidatas que sobrevivan. No iniciar barridos exhaustivos ni optimización PointSparse en S3-T08.

## Línea B — Dynamic Neuron

**Objetivo:** medir si una unidad con dinámica sustituye suficientes unidades o capas para compensar su estado y coste. Baseline inicial: PointSparse estática contra **PointSparse + estado temporal por unidad** en un problema con estructura temporal y protocolo nuevo. Seguir, una variable por experimento, con constantes de tiempo heterogéneas y adaptación de excitabilidad; luego estado dendrítico, modulación contextual/apical (DSU-A/DSU-AIS), plasticidad local/estructural (DSU-R) y sólo después variantes más complejas. DSU-T queda como candidato futuro dentro de esta línea; su implementación no está autorizada por este documento.

Cada mecanismo requiere una predicción falsable, control de parámetros/estado, secuencias y reset definidos, coste físico y un baseline temporal razonable. Una ganancia en accuracy sola no basta para justificar complejidad.

## Promoción de backlog a experimento

Una idea pasa a diseño experimental sólo si identifica mecanismo, baseline, variable conceptual, métrica primaria, presupuesto comparable, estimación de coste, criterio adverso y artefactos reproducibles. El protocolo se congela antes de usar validation para selección; test permanece cerrado hasta la decisión final. Las señales pequeñas o contradictorias quedan en backlog o se replican con N mayor mediante tarea separada.

Se aparcan la ampliación de compartimentos dendríticos **estáticos** sin nueva predicción causal, los barridos H×K exhaustivos, DSU-T inmediato, kernels y Transformer. Reabrir dendritas estáticas exige un mecanismo específico que PointSparse no capture, un control pareado de topología/parámetros/escala, una ventaja repetible predefinida y coste físico aceptable. La línea B conserva las dendritas para dinámica y contexto.
