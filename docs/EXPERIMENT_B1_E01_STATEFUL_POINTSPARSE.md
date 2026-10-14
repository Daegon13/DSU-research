# B1-E01 — PointSparse estática frente a estado EMA fijo

**Estado (2026-10-13):** **PASS técnico; H-B1-STATE = SUPPORTED** bajo el criterio congelado y sólo para `delayed_recall_easy_v1`. El criterio normativo es el de [B1-DESIGN / S4-T01](B1_POINTSPARSE_TEMPORAL_DESIGN.md), SHA-256 `1081e60304fdb2dace083c8eea6eab2bc072780a71f57b49449c9e043b2cba4e`. Copia machine-readable: [b1_e01_frozen.json](../experiments/configs/b1_e01_frozen.json); [resumen de resultados](../data/b1_e01_results.json). Hubo 24 corridas primarias completas, ninguna selección de checkpoint y ningún oracle activado.

## Preregistro heredado y alcance

S0 = PointSparse-Static-Temporal, S1 = PointSparse-T v0. Ambos usan `D=32, H=64, K=8, C=2`, topología IID pareada, LeakyReLU 0.1, 706 parámetros entrenables y el mismo estado inicial de pesos por par. S1 añade `m_0=0; m_t=λm_(t−1)+(1−λ)a_t`, con `λ=2^(−1/16)` global fijo y salida sólo en `T=Δ+1`. S0 usa sólo `x_T`. Los pesos se actualizan por separado tras la inicialización común.

Tarea primaria `delayed_recall_easy_v1`: señal one-hot inicial; nuisance temprana independiente del label en canales 3–31; pasos intermedios cero; cue fijo al final. Delays `Δ=4,8,16,32`; endpoint científico primario `Δ=16`. Train/validation/test: 50000/10000/10000 por delay; `dataset_seed=41041`, topology/training seeds `1,2,3`. Adam `.001`, betas `.9/.999`, eps `1e−8`, batch 128, CrossEntropy, 20 épocas fijas, sin early stopping ni selección de checkpoint. Test sólo después de época 20. El oracle se activará únicamente bajo la condición preregistrada.

## Gates anteriores al entrenamiento

Tests de generador/unidad: balance exacto, determinismo, cambio al variar `dataset_seed`, longitud de secuencia, señal/cue en sus pasos, `x_T` igual para ambas etiquetas, nuisance sin separación media grande, salida S0 invariante a señal inicial, salida S1 sensible, reset y gradiente temporal. Preflight: `runs/b1_e01/preflight.json`; gate `PASS`, sin acceso al split test. El baseline final-only sobre `x_T` es una predicción constante, por lo que su accuracy esperada en los splits balanceados es exactamente 50%; no se entrenó ningún clasificador de leakage.

| Δ | Retención teórica `λ^Δ` | Retención medida | Error absoluto |
|---:|---:|---:|---:|
| 4 | 0.840896415 | 0.840896398 | 0.000000017 |
| 8 | 0.707106781 | 0.707106752 | 0.000000029 |
| 16 | 0.500000000 | 0.499999959 | 0.000000041 |
| 32 | 0.250000000 | 0.249999959 | 0.000000041 |

Por brazo: 706 parámetros float32 = 2824 bytes; G int64 `[64,8]` = 4096 bytes; payload tensorial persistente = 6920 bytes, sin cabeceras. S0: 640 MAC principales/secuencia, sin estado. S1: `512·(Δ+1)+128` MAC principales/secuencia, 256 bytes de estado runtime/muestra (`32768` para batch 128) y, por paso, 128 multiplicaciones y 64 sumas elementwise de actualización. Los MAC excluyen activación, gathers, bias y backward; los bytes de estado no incluyen activaciones retenidas durante entrenamiento. No inferir latencia, RAM pico o energía de estas cifras.

## Resultados por seed

Accuracy de test en la época 20: **S0 50.00% y S1 100.00% en cada una de las 12 parejas**. La NLL de test por seed muestra variación y aumenta para S1 al alargar el delay, aun con accuracy en techo:

| Δ | Seed | S0 accuracy | S1 accuracy | S0 NLL | S1 NLL | Δ NLL (S1−S0) |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1 | 50.00% | 100.00% | .693225 | .000329 | −.692896 |
| 4 | 2 | 50.00% | 100.00% | .693148 | .000328 | −.692820 |
| 4 | 3 | 50.00% | 100.00% | .693266 | .000277 | −.692989 |
| 8 | 1 | 50.00% | 100.00% | .693225 | .000516 | −.692710 |
| 8 | 2 | 50.00% | 100.00% | .693148 | .000504 | −.692644 |
| 8 | 3 | 50.00% | 100.00% | .693266 | .000387 | −.692879 |
| 16 | 1 | 50.00% | 100.00% | .693225 | .001000 | −.692226 |
| 16 | 2 | 50.00% | 100.00% | .693148 | .000947 | −.692201 |
| 16 | 3 | 50.00% | 100.00% | .693266 | .000723 | −.692543 |
| 32 | 1 | 50.00% | 100.00% | .693225 | .003226 | −.689999 |
| 32 | 2 | 50.00% | 100.00% | .693148 | .002953 | −.690195 |
| 32 | 3 | 50.00% | 100.00% | .693266 | .002175 | −.691091 |

## Curva de memoria y contraste primario

Media ± DE muestral sobre las tres seeds. `memory_advantage` es delta pareado S1−S0 en puntos porcentuales.

| Δ | Retención `λ^Δ` | S0 accuracy | S1 accuracy | Memory advantage | S0 NLL | S1 NLL |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | .8409 | 50.00 ± 0.00% | 100.00 ± 0.00% | +50.00 ± 0.00 pp | .693213 ± .000060 | .000311 ± .000029 |
| 8 | .7071 | 50.00 ± 0.00% | 100.00 ± 0.00% | +50.00 ± 0.00 pp | .693213 ± .000060 | .000469 ± .000071 |
| **16** | **.5000** | **50.00 ± 0.00%** | **100.00 ± 0.00%** | **+50.00 ± 0.00 pp** | **.693213 ± .000060** | **.000890 ± .000147** |
| 32 | .2500 | 50.00 ± 0.00% | 100.00 ± 0.00% | +50.00 ± 0.00 pp | .693213 ± .000060 | .002784 ± .000545 |

En Δ=16, los tres deltas de accuracy son +50 pp, S1 supera 70% de media y el delta medio de NLL es **−.692323 ± .000190**. Cumple literalmente `SUPPORTED`: delta medio ≥10 pp, signo favorable en 3/3 seeds, S1 ≥70% y NLL media menor. La tarea pedía nombres como `STRONG SUPPORT`/`SUPPORT`/`WEAK`; el preregistro no define esos niveles ni sus umbrales. Se conserva **SUPPORTED** como categoría normativa, sin crear una categoría retrospectiva.

La retención del impulso disminuye de .8409 a .25, pero la accuracy de S1 permanece en 100% para los cuatro delays. Este techo impide observar una caída de accuracy o estimar el límite de memoria del mecanismo. La NLL de S1 aumenta descriptivamente con Δ; no se usó para escoger un delay ni λ. `Oracle-current` **no se ejecutó** porque su gatillo exigía que S0 y S1 quedaran entre 45–55% en Δ=16, y S1 obtuvo 100%.

## Trazabilidad, duración y límites

Los 24 JSON por corrida están en `runs/b1_e01/` (ignorados por Git) y conservan G explícita y su hash, inicialización, fingerprints de train/validation/test, hashes de los 20 órdenes de minibatches, curvas completas, test de época 20, duración y entorno. El resumen versionado es `data/b1_e01_results.json`. El runner reanudó los **24/24** artefactos válidos sin repetir entrenamiento. Suma de duraciones registradas de train+validation: **1281.48 s (21.36 min)**; no es un benchmark de latencia de inferencia. Entorno: Windows, AMD Ryzen 5 5600G, CPU float32, PyTorch 2.14.1+cpu, NumPy 2.5.3, un hilo PyTorch por corrida; RAM física ~33.69 GB. RAM pico, energía y latencia de inferencia no se midieron.

Validación local: `113 passed, 2 skipped` en la suite completa; preflight PASS y `git diff --check` sin errores. Los controles de fuga, estado, reset y gradiente pasaron antes de entrenar. Ninguna corrida accedió al test hasta su época final, ni hubo búsqueda de λ, epochs, seeds o topología.

**Conclusión permitida:** en esta tarea sintética EASY, con el presupuesto entrenable igualado y el protocolo fijado, el estado EMA permite recuperar la señal inicial mientras S0 sólo ve un último paso constante. El resultado demuestra capacidad temporal mínima bajo estas condiciones. No establece superioridad frente a modelos recurrentes estándar, una memoria robusta a distractores, ventaja de coste total, menor latencia/RAM pico/energía ni generalización fuera de este generador. B1-NOISE, B2 y DSU-T no se ejecutaron.
