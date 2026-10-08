# EXPERIMENT_004 — Equivalencia Fixed-dANN / DSU-S (S3-T02)

**Fecha:** 2026-10-08. **Gate S3-T02: PASS.** Esta tarea implementó el control Fixed-dANN y verificó equivalencia estructural, funcional y de optimización corta con DSU-S. No entrenó Fashion-MNIST, no midió performance y no modificó `DSUSv0`.

## Implementación y alcance

`FixedDANN` recibe una `G [128,4,16]` ya existente; no genera topología para el contraste B/C. Almacena capas dense `784→512→128→10`, aplica máscaras booleanas persistentes en cada forward y usa LeakyReLU 0.1 después de dendritas y somas. `fixed_dann_from_dsu_s` copia `G`, los 8192 pesos dendríticos activos, 512 pesos somáticos, todos los sesgos y la salida. Los pesos inactivos empiezan en cero, pero una prueba los cambia a valores grandes y confirma que la máscara impide su contribución.

| Modelo | Parámetros almacenados | Parámetros efectivos | Input mask | Soma mask |
|---|---:|---:|---:|---:|
| Fixed-dANN | 468874 | 10634 | 8192 | 512 |
| DSU-S | 10634 | 10634 | `G`: 8192 índices | implícita: 512 pesos |

Los conteos se derivaron de shapes, máscaras y tensores reales. Las máscaras y `G` son buffers no entrenables, se mueven con el modelo y sobreviven un round-trip de `state_dict`. Estos conteos no son bytes totales de checkpoint ni RAM runtime.

## Protocolo sintético

CPU, PyTorch 2.14.1+cpu, float32. Seeds explícitas; inputs normales deterministas; batches 1 y 7 para forward; batch 8 para gradientes; CrossEntropyLoss. Tolerancia prefijada `rtol=1e-5`, `atol=1e-6`. Para optimización se usó Adam `lr=0.001`, `weight_decay=0` sobre cinco batches sintéticos prefijados. Las dos rutas compartieron `G`, pesos activos iniciales, sesgos, inputs, targets y loss.

## Forward pareado

Errores absolutos Fixed-dANN frente a DSU-S:

| Etapa | Batch 1: max / media | Batch 7: max / media |
|---|---:|---:|
| Preactivación dendrítica | 1.19e-7 / 1.41e-8 | 1.19e-7 / 1.39e-8 |
| Activación dendrítica | 1.19e-7 / 7.85e-9 | 1.19e-7 / 7.57e-9 |
| Preactivación somática | 2.98e-8 / 2.92e-9 | 2.24e-8 / 2.72e-9 |
| Activación somática | 2.98e-8 / 1.72e-9 | 2.24e-8 / 1.47e-9 |
| Logits | 7.45e-9 / 4.47e-9 | 2.61e-8 / 4.43e-9 |
| Loss absoluta | 0 | 0 |

Las diferencias son consistentes con distinto orden de reducción float32: dense masked sobre 784 posiciones frente a gather y suma sobre 16. No indican una diferencia funcional neuronal.

## Backward y Adam

| Gradiente mapeado | Error máximo | Error medio |
|---|---:|---:|
| `W_d` | 1.86e-9 | 5.60e-11 |
| `bias_d` | 4.66e-10 | 2.05e-11 |
| `W_s` | 1.86e-9 | 3.20e-10 |
| `bias_s` | 3.73e-9 | 3.37e-10 |
| `W_o` | 1.86e-9 | 1.88e-10 |
| `bias_o` | 3.73e-9 | 3.73e-10 |
| Input | 1.16e-10 | 7.45e-12 |

Los gradientes no nulos en posiciones inactivas fueron **0** tanto en input→dendrita como en dendrita→soma. Los momentos Adam de esas posiciones permanecieron exactamente cero después de cinco pasos.

| Paso Adam | Máximo delta absoluto de parámetros activos | Máximo delta de logits |
|---:|---:|---:|
| 1 | 3.35e-8 | 1.86e-8 |
| 2 | 3.35e-8 | 1.49e-8 |
| 3 | 3.35e-8 | 2.05e-8 |
| 4 | 3.35e-8 | 1.49e-8 |
| 5 | 3.35e-8 | 1.86e-8 |

## Interpretación y límites

El par demuestra que Fixed-dANN y DSU-S pueden representar la misma función y seguir una trayectoria Adam extremadamente próxima cuando comparten topología y parámetros efectivos. Esto valida B/C como control de representación para el alcance probado. No demuestra igualdad bit a bit, estabilidad de una trayectoria de 25 épocas, equivalencia GPU, menor checkpoint/RAM, menor latencia, menor energía ni mejora de accuracy por compactación. CUDA quedó sin ejecutar porque el host no dispone del backend.

**H-010 = SUPPORTED en el alcance estructural y funcional CPU float32 probado.** Fixed-dANN almacena 468874 parámetros frente a 10634 de DSU-S para la misma función efectiva; máscaras e índices deben contabilizarse aparte antes de afirmar ahorro total.

**Siguiente tarea exacta:** `S3-E01 — Ejecutar la ablación causal dANN-R vs Fixed-dANN para fan-in/topología bajo el protocolo congelado`. No se ejecutó aquí.
