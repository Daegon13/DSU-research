# S3-T06 — Controles sparse point y compartimentos sin parámetros extra

## Ficha de tarea

| Campo | Valor |
|---|---|
| TASK ID | S3-T06 |
| OBJECTIVE | Implementar PointSparse y familia de compartimentos con parámetros iguales; congelar calibración inicial. |
| WHY | S3-T05 mezcló partición, parámetros y escala de logits. |
| CONTEXT | H-011 = NEGLIGIBLE para accuracy N=3; la cross-entropy media bajó ligeramente con B. |
| ALLOWED FILES | Módulo experimental nuevo, diagnóstico forward, tests, este diseño, hipótesis, diseño causal, decisiones, estado. |
| FORBIDDEN FILES | DSUSv0, baselines, resultados S3-T05, splits y protocolo histórico. |
| REQUIREMENTS | G64 nested, 9610 parámetros en todos, pareo y calibración sin labels. |
| OUT OF SCOPE | Entrenar Fashion-MNIST, S3-T07, tuning, seeds 4–5, rendimiento físico. |
| ACCEPTANCE CRITERIA | Conteos, forward/backward, topología, calibración y diagnósticos iniciales válidos. |
| TESTS | Suite completa, `git diff --check`, `git status`. |
| EXPECTED OUTPUT | Gate S3-T06 y tabla preentrenamiento. |
| EXPECTED COMMIT | `feat: add sparse point and parameter-neutral compartment controls` si se autoriza commit. |
| STOP CONDITIONS | Diferencia grave de escala, topología no pareada, parámetros extra o fuga de labels/test. |

## Lectura de S3-T05

El contraste `4×16 − 1×64` dio +0.023 ± 0.138 puntos de accuracy y −0.005051 ± 0.001043 de cross-entropy. Accuracy mide decisiones argmax; cross-entropy depende también de probabilidades y confianza. La mejora de loss puede deberse a (A) mejores probabilidades o márgenes, (B) escala/confianza de logits, (C) parámetros adicionales, (D) compartimentalización, o una combinación. S3-T05 tenía `9610+256B` parámetros y logits iniciales con DE entre 0.441 y 0.873. No se atribuye el loss a las dendritas.

## Arquitecturas y costes

**PointSparse:** `z_s = Σₖ W[s,k]x[G64[s,k]] + bias_s`, `h_s = LeakyReLU(z_s,0.1)`, `logits = W_o h + bias_o`. Son 128 somas, 64 entradas únicas por soma, una sola no linealidad, sin dendritas ni matriz densa enmascarada.

**NeutralCompartments:** `K=64/B`, `z_sb = Σₖ W[s,b,k]x[G64[s,b,k]]`, `d_sb = LeakyReLU(z_sb,0.1)`, `r_s = Σ_b d_sb/√B`, `u_s = g_B r_s + c_B`, `h_s = LeakyReLU(u_s+bias_s,0.1)`. `g_B,c_B` son buffers escalares fijos después de calibrar; no hay sesgos dendríticos ni pesos rama→soma. Los bloques de G64 son contiguos y nested para B=1,2,4,8. El B=4 de esta familia tampoco es DSUSv0 histórico.

| Brazo | Pesos input | Bias soma | Pesos/bias salida | Total entrenable | MAC principales/muestra | Sumas fijas de ramas/muestra |
|---|---:|---:|---:|---:|---:|---:|
| PointSparse | 8192 | 128 | 1280+10 | 9610 | 9472 | 0 |
| 1×64 | 8192 | 128 | 1280+10 | 9610 | 9472 | 0 |
| 2×32 | 8192 | 128 | 1280+10 | 9610 | 9472 | 128 |
| 4×16 | 8192 | 128 | 1280+10 | 9610 | 9472 | 384 |
| 8×8 | 8192 | 128 | 1280+10 | 9610 | 9472 | 896 |

MAC principales cuentan sólo multiplicaciones por pesos input y output. La suma fija, las activaciones, el gather y los factores afines también cuestan operaciones; no inferir FLOPs totales, latencia, RAM o energía.

## Inicialización, agregación y calibración congeladas

Para cada seed, `base_topology(seed)` de S3-T05 genera **una** G64 `[128,64]`; la misma matriz y orden se usan en los cinco brazos. Un generador PyTorch local crea una base normal estándar `[128,64]` y una única salida Xavier uniforme `[10,128]`; bias soma y salida son cero en todos. El peso input es `base × sqrt(2/((1+0.1²)K))`, con K=64 para PointSparse y K=64/B para la familia. Por eso se emparejan los valores aleatorios base, pero el peso efectivo cambia con K. Se registran SHA-256 de G64 y bancos base.

La regla **fijada antes de entrenar** es suma de ramas dividida por `√B`. Si las ramas fueran independientes y centradas, preservaría aproximadamente la varianza; LeakyReLU produce media positiva y las ramas comparten datos, así que no preserva necesariamente ni media ni varianza real. Para evitar una diferencia inicial trivial se calibra una única transformación afín global por brazo, sin labels: `g_B = std(z_Point)/std(r_B)` y `c_B = mean(z_Point) − g_B mean(r_B)`. Se usan desviaciones poblacionales sobre el mismo lote train. Esto iguala media y DE del agregado a la preactivación PointSparse en el lote; no iguala distribuciones, correlaciones, logits ni gradientes. Los buffers quedan fijos durante el entrenamiento futuro. Está prohibido volver a ajustarlos usando accuracy, validation o test.

Procedimiento reproducible: para cada seed 1–3, aplicar el split de S3-T05 (`default_rng(seed).shuffle(arange(60000))`; últimos 6000 validation), tomar los **primeros 256 índices del tramo train**, convertir píxeles uint8 a float32 `/255`, ignorar labels y hacer sólo forward. Hash de los 256 índices y coeficientes se guardan en `runs/s3_t06_initial_diagnostics.json`; `runs/` está ignorado por Git. Comando: `.venv/Scripts/python.exe -m dsu_research.s3_t06_diagnostics --output runs/s3_t06_initial_diagnostics.json`. El código usa `download=False`; requiere el dataset ya presente. No se entrenó ningún brazo.

## Diagnóstico inicial

Seed 1, 256 imágenes train, media/DE poblacional sobre batch y unidades. `—` indica que PointSparse no tiene etapa dendrítica. Los valores completos de las tres seeds están en el JSON reproducible.

| Brazo | Parámetros | MAC | PreD | ActD | Agregado u | ActS | Logits | Norma L2 | Entropía |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PointSparse | 9610 | 9472 | — | — | — | 0.241/0.430 | 0.126/0.681 | 2.082 | 2.102 |
| 1×64 | 9610 | 9472 | 0.024/0.664 | 0.241/0.430 | 0.024/0.664 | 0.236/0.539 | 0.154/0.804 | 2.391 | 2.053 |
| 2×32 | 9610 | 9472 | 0.017/0.668 | 0.235/0.436 | 0.024/0.664 | 0.239/0.516 | 0.210/0.732 | 2.192 | 2.089 |
| 4×16 | 9610 | 9472 | 0.012/0.657 | 0.228/0.419 | 0.024/0.664 | 0.240/0.504 | 0.190/0.766 | 2.208 | 2.072 |
| 8×8 | 9610 | 9472 | 0.009/0.657 | 0.221/0.416 | 0.024/0.664 | 0.244/0.494 | 0.128/0.743 | 2.065 | 2.075 |

Ganancia/offset por seed para B=1,2,4,8: seed 1 `(1.544,−0.348)`, `(1.526,−0.483)`, `(1.503,−0.661)`, `(1.421,−0.865)`; seed 2 `(1.548,−0.363)`, `(1.539,−0.502)`, `(1.532,−0.659)`, `(1.448,−0.835)`; seed 3 `(1.600,−0.351)`, `(1.587,−0.497)`, `(1.532,−0.666)`, `(1.432,−0.869)`.

En las tres seeds, DE de logits de la familia quedó entre 0.613 y 0.804; PointSparse entre 0.575 y 0.681. No hay colapso ni explosión, aunque la escala no es idéntica: `B=1` conserva una segunda LeakyReLU y puede superar a PointSparse en DE. No se reajustan constantes por ese dato. La comparación futura debe registrar test accuracy, NLL/cross-entropy, norma media de logits, margen medio de clase correcta y entropía predictiva. ECE puede añadirse como exploratoria sin dependencia pesada; nunca criterio primario. No utilizar test durante calibración ni búsqueda.

## Hipótesis y siguiente tarea

H-014 (H-SPARSE) y H-015 (H-COMPARTMENT-PURE) están UNTESTED. PointSparse frente a 1×64 pregunta por añadir una etapa dendrita→soma con una sola rama; PointSparse frente a 4×16 pregunta si la estructura compartimentada supera a 128 unidades sparse. La familia B compara particiones con conectividad, parámetros y escala inicial controlados, aunque cambia el número de no linealidades y las operaciones fijas. Ningún resultado de entrenamiento está disponible aquí.

**Gate S3-T06: PASS de diseño, implementación y diagnóstico forward.** Siguiente tarea exacta: **S3-T07 — Ejecutar el experimento causal PointSparse + parameter-neutral compartment family**, con protocolo y criterios de interpretación congelados antes de abrir resultados.
