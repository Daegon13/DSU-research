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
