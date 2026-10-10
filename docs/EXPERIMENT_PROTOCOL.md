# EXPERIMENT_PROTOCOL

## Objetivo

Evitar que los resultados dependan de cambios silenciosos de configuración.

## 1. Seeds

Todo experimento debe registrar seeds de:

- Python;
- NumPy;
- PyTorch;
- CUDA cuando corresponda.

Un resultado importante debe repetirse con varias seeds.

## 2. Datasets

Cada experimento debe registrar:

- nombre;
- versión;
- split;
- transformaciones;
- tamaño de train/validation/test.

Nunca cambiar el split entre modelos comparados.

## 3. Hardware

Registrar:

- CPU;
- RAM;
- GPU;
- VRAM;
- sistema operativo;
- versión de PyTorch;
- backend;
- precisión numérica.

## 4. Entrenamiento

Registrar:

- optimizer;
- learning rate;
- scheduler;
- epochs;
- batch size;
- weight decay;
- gradient clipping;
- mixed precision;
- early stopping.

## 5. Equidad

Cuando comparemos DSU con baselines, declarar qué presupuesto estamos igualando:

- parámetros;
- FLOPs;
- calidad;
- tiempo;
- memoria.

Nunca afirmar superioridad global a partir de una sola clase de igualdad.

## 6. Latencia

Antes de medir:
- ejecutar warm-up;
- sincronizar GPU;
- usar batch size explícito;
- repetir múltiples veces;
- reportar media y dispersión.

Medir batch 1 cuando el objetivo sea inferencia local interactiva.

## 7. Memoria

Registrar:
- tamaño del checkpoint;
- RAM pico;
- VRAM pico;
- tamaño de estados recurrentes.

## 8. Resultados negativos

Los experimentos fallidos deben registrarse si cambian nuestro entendimiento.

## 9. Ablaciones

Una ablación debe modificar una sola variable conceptual siempre que sea posible.

## 10. Repetibilidad

Cada experimento debe poder ejecutarse mediante:
- config;
- comando documentado;
- seed explícita.

## 11. Criterios estadísticos

No interpretar pequeñas diferencias con una sola seed como evidencia fuerte.

Los resultados importantes deben incluir:
- múltiples ejecuciones;
- media;
- desviación o intervalo equivalente.

## 12. Control de hiperparámetros

No optimizar extensivamente DSU sin otorgar un presupuesto comparable al baseline.

Si los presupuestos de tuning son distintos, documentarlo claramente.

## 13. Funnel de selección analítica (desde S3-T08)

Este funnel guía tareas **futuras**; no autoriza búsquedas ni entrenamiento por sí mismo. Registrar en cada etapa candidatos, rechazos, criterio prefijado, coste y protocolo. Mantener `experiment_id`, `protocol_family` y `paired_comparison_group` para impedir contrastes causales entre runs históricos incompatibles.

| Etapa | Presupuesto | Decisión |
|---|---|---|
| 0 — restricciones analíticas | Sin entrenamiento | Revisar parámetros, bytes de pesos/topología/estado, MAC y redundancia; descartar dominancia bajo el presupuesto declarado |
| 1 — proxies zero-cost | Uno o pocos lotes train y, como máximo, backward prefijado | Medir señal/representación/gradientes; descartar degeneración con umbrales congelados |
| 2 — piloto barato | Sólo supervivientes; ejemplo futuro 1 seed, 2–3 épocas | Comprobar aprendizaje inicial sin tocar test |
| 3 — experimento | N=3, 25 épocas según protocolo congelado | Medir calidad y costes, con selección `final_epoch_25` cuando corresponda |
| 4 — confirmación | Señales relevantes; N=5 o más | Replicar efecto y controles antes de afirmación fuerte |

El orden no convierte proxies en predictores de accuracy exacta. No sumar métricas en un score arbitrario; calibrar reglas sólo con validation y dejar test cerrado durante búsqueda. Si el protocolo de una nueva tarea difiere, documentar la razón antes de ejecutar. [Diseño del framework](ANALYTICAL_SCREENING_FRAMEWORK.md).
