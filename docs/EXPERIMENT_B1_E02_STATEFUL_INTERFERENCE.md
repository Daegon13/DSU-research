# B1-E02 — interferencia temporal sobre PointSparse-T

**Estado (2026-10-13, America/Montevideo): PASS técnico; H-B1-INTERFERENCE = SUPPORTED** sólo para `delayed_recall_noise_v1` y la amplitud `.1` fijada. **Preregistro normativo:** [B1-NOISE-DESIGN](B1_NOISE_DESIGN.md), commit `7bbcc33`, SHA-256 LF `7b30db63056498e1c10d66bd84f6d639152cc21bd56363f8fce8a2e73c3c4191`. B1-E01 EASY y su código están en el commit previo `fc16678`. Ambos commits preceden al entrenamiento NOISE en la rama local `b1-e02-split`; `origin/main` conserva su commit combinado publicado `e810647` sin reescritura ni push.

TASK ID: B1-E02. OBJECTIVE: medir si la EMA fija de PointSparse-T recupera la señal inicial bajo ruido interior independiente del target. WHY: B1-E01 EASY produjo accuracy en techo. CONTEXT: ADR-018–020 y H-B1-INTERFERENCE. ALLOWED FILES: generador/runner/tests B1-E02, resultados B1-E02 y documentación de cierre. FORBIDDEN FILES: preregistro B1-NOISE, modelo/runner B1-E01, baselines históricos y otros mecanismos. REQUIREMENTS: 24 corridas pareadas, época final 20, test cerrado hasta entonces, criterios Δ=16 intactos. OUT OF SCOPE: variar λ, amplitud, seeds, arquitectura, B2, dendritas o DSU-T. ACCEPTANCE CRITERIA: gates de datos/estado/pareo, corridas completas, decisión literal y registro de coste. TESTS: suite completa, resume, `git diff --check`. EXPECTED OUTPUT: informe, JSON por corrida y resumen versionado. EXPECTED COMMIT: `experiment: test PointSparse state under temporal interference`. STOP CONDITIONS: leakage, deriva de protocolo/modelo, pareo roto, artefactos inválidos o presupuesto no asumible.

## Protocolo y condición exacta

S0 PointSparse-Static-Temporal y S1 PointSparse-T v0 conservan `D=32, H=64, K=8, C=2`, 706 parámetros entrenables, LeakyReLU 0.1, G IID y pesos iniciales pareados. S1 usa `m_0=0; m_t=λm_(t−1)+(1−λ)a_t`, con `λ=float32(2^(−1/16))=0.9576032757759094`; S0 decide sólo con `x_T`. Δ=4/8/16/32, T=Δ+1, seeds 1/2/3, train/validation/test 50000/10000/10000, Adam `.001`, batch 128, 20 épocas fijas y test sólo tras época 20. Los modelos no se seleccionan por validation y sus pesos dejan de estar atados después de la inicialización.

En `delayed_recall_noise_v1`, `y=index mod 2`; `x_1` y `x_T` son bit a bit iguales a EASY. El primer paso codifica y en one-hot de canales 0/1 y nuisance `Uniform(−.1,.1)` en 3..31. El último contiene sólo cue fijo `x_T[2]=1`. Cada paso interior recibe ruido IID `Uniform(−.1,.1)` float32 en canales `[0,1,3,…,31]`; canal 2 sigue en cero. El RNG por muestra es `default_rng(SeedSequence([41041,split_id,Δ,index,99]))`, con matriz temporal `(Δ−1,31)` muestreada antes de convertir a float32. No hay padding, posición, ID, máscara ni etiqueta como input. S0 y S1 comparten las secuencias del par.

## Gates previos y fuga

`runs/b1_e02/preflight.json` registró PASS sin leer el split test. Train/validation mantienen balance exacto; la construcción no usa y para muestrear ruido. `x_T` es constante, así que cualquier predictor que sólo vea el último paso tiene 50% de accuracy en los splits balanceados. Las medias de ruido por clase se auditaron por canal: la máxima diferencia absoluta entre medias por canal fue `.001433` entre los ocho split/delay de train/validation, bajo el gate `.005`; la nuisance inicial llegó a `.003964`. Los canales iniciales de señal 0/1 reciben ruido en todos los pasos interiores; cue sólo aparece al final. Las claves de RNG separan splits y delays; el índice controla la generación pero no se entrega al modelo. El mismo `dataset_seed` e índice reproducen bit a bit la secuencia; cambiar la seed modifica ruido/nuisance. Los ocho fingerprints NOISE de train/validation son distintos y difieren de EASY por la intervención. Se verificaron señal, cue, rango, longitud y ausencia de metadata.

S0 permaneció invariante al cambio de señal temprana con último paso igual. S1 respondió al historial; la loss final produjo gradiente hacia `x_1`; forward de secuencias alternadas confirmó reset sin contaminación y el lazo temporal conserva BPTT completo, sin `detach`. `λ`, G y valores iniciales se contrastaron con el protocolo congelado y artefactos EASY. La retención de un impulso coincidió con `λ^Δ` dentro de tolerancia float. Los hashes de orden de minibatches se comparan por par.

## Diagnóstico analítico y coste

Sea `c=1−λ` y `T=Δ+1`. Para una coordenada del estado, `m_T = c λ^Δ a_1 + c Σ_(t=2)^Δ λ^(T−t) a_t + c a_T`. El primer término lleva señal y nuisance inicial; la suma contiene activaciones bajo distractores, ponderadas más cuanto más recientes. La señal diferencial entre clases se escala por `cλ^Δ`; la contribución de ruido no se elimina al fijar λ. A nivel de input por canal, antes de pesos y LeakyReLU, el RMS ilustrativo del ruido interior con desviación `.1/√3` es `c(.1/√3)√Σ_(j=1)^(Δ−1) λ^(2j)`. No predice accuracy ni la distribución real de activaciones aprendidas.

| Δ | Pasos con ruido | T | `λ^Δ` | `cλ^Δ` | RMS ilustrativo | S0 MAC/seq | S1 MAC/seq | S1 actualización elem./seq | S1 MAC + actualización/seq |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 3 | 5 | .840896 | .035651 | .003893 | 640 | 2688 | 960 | 3648 |
| 8 | 7 | 9 | .707107 | .029979 | .005487 | 640 | 4736 | 1728 | 6464 |
| 16 | 15 | 17 | .500000 | .021198 | .006939 | 640 | 8832 | 3264 | 12096 |
| 32 | 31 | 33 | .250000 | .010599 | .007854 | 640 | 17024 | 6336 | 23360 |

Cada brazo almacena 2824 bytes de pesos float32 + 4096 bytes de G int64 = 6920 bytes de payload tensorial. S1 añade 256 bytes de estado de inferencia por muestra (32768 para batch 128); S0 no almacena estado temporal. Las actualizaciones S1 son 128 multiplicaciones y 64 sumas por timestep. Los conteos excluyen gathers, bias, activación, backward, optimizer y generación de datos; no son medidas de latencia, RAM pico ni energía. El coste forward principal de train preregistrado suma `107.52×10^9` MAC para las 24 corridas.

## Resultados

Las 24 corridas primarias (`2×4×3`) llegaron a la época 20; los 24 JSON por corrida están en `runs/b1_e02/` y el resumen versionado en [b1_e02_results.json](../data/b1_e02_results.json). El oracle no cumplió su gatillo y no se ejecutó. No se invalidó ni repitió ninguna seed. Una segunda invocación del runner mostró `REUSE` para los 24 artefactos válidos, sin reentrenar. El tiempo acumulado de train+validation fue **1674.08 s**, no una medida de inferencia. Entorno registrado: Windows 11, CPU AMD Ryzen 5 5600G, Python 3.12.13, NumPy 2.5.3, PyTorch 2.14.1+cpu, float32 y un hilo PyTorch. Había aproximadamente 6.5 GB de RAM física libre antes de entrenar; RAM pico no medida.

### Test por seed, época final 20

| Δ | Seed | S0 accuracy | S1 accuracy | Ventaja S1−S0 | S0 NLL | S1 NLL | ΔNLL S1−S0 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1 | 50% | 100% | +50 pp | .693225 | .000409 | −.692816 |
| 4 | 2 | 50% | 100% | +50 pp | .693148 | .000401 | −.692748 |
| 4 | 3 | 50% | 100% | +50 pp | .693266 | .000352 | −.692914 |
| 8 | 1 | 50% | 100% | +50 pp | .693225 | .000716 | −.692509 |
| 8 | 2 | 50% | 100% | +50 pp | .693148 | .000679 | −.692470 |
| 8 | 3 | 50% | 100% | +50 pp | .693266 | .000528 | −.692738 |
| **16** | **1** | **50%** | **100%** | **+50 pp** | **.693225** | **.001688** | **−.691538** |
| **16** | **2** | **50%** | **100%** | **+50 pp** | **.693148** | **.001281** | **−.691867** |
| **16** | **3** | **50%** | **100%** | **+50 pp** | **.693266** | **.000959** | **−.692307** |
| 32 | 1 | 50% | 100% | +50 pp | .693225 | .005603 | −.687622 |
| 32 | 2 | 50% | 100% | +50 pp | .693148 | .005206 | −.687943 |
| 32 | 3 | 50% | 100% | +50 pp | .693266 | .003541 | −.689725 |

### Agregados por delay

Media ± DE muestral sobre N=3 seeds; ventaja calculada por pareja antes de agregar.

| Δ | S0 accuracy | S1 accuracy | Ventaja | S0 NLL | S1 NLL | ΔNLL |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 50.00 ± 0.00% | 100.00 ± 0.00% | +50.00 ± 0.00 pp | .693213 ± .000060 | .000387 ± .000031 | −.692826 ± .000084 |
| 8 | 50.00 ± 0.00% | 100.00 ± 0.00% | +50.00 ± 0.00 pp | .693213 ± .000060 | .000641 ± .000100 | −.692572 ± .000145 |
| **16** | **50.00 ± 0.00%** | **100.00 ± 0.00%** | **+50.00 ± 0.00 pp** | **.693213 ± .000060** | **.001309 ± .000365** | **−.691904 ± .000386** |
| 32 | 50.00 ± 0.00% | 100.00 ± 0.00% | +50.00 ± 0.00 pp | .693213 ± .000060 | .004783 ± .001094 | −.688430 ± .001133 |

### EASY frente a NOISE y curva observada

La auditoría histórica de los 24 pares confirmó G, inicialización, λ, seeds, coste, optimizador y los 20 órdenes de minibatches idénticos; los 24 fingerprints de dataset cambiaron por el ruido. `x_1` y `x_T` permanecen idénticos por índice. Por eso se puede describir la diferencia pareada NOISE−EASY sin confundir un cambio de protocolo con la intervención.

| Δ | S0 EASY | S1 EASY | S0 NOISE | S1 NOISE | Caída S1 accuracy NOISE−EASY | Ventaja S1−S0 NOISE | S1 NLL EASY → NOISE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 50% | 100% | 50% | 100% | 0 pp | +50 pp | .000311 → .000387 (+.000076) |
| 8 | 50% | 100% | 50% | 100% | 0 pp | +50 pp | .000469 → .000641 (+.000172) |
| 16 | 50% | 100% | 50% | 100% | 0 pp | +50 pp | .000890 → .001309 (+.000419) |
| 32 | 50% | 100% | 50% | 100% | 0 pp | +50 pp | .002784 → .004783 (+.001999) |

La curva de **accuracy vs delay** de S1 NOISE sigue plana en 100%; este nivel de ruido no reveló un límite de memoria en los cuatro delays. La NLL de S1 crece de `.000387` a `.004783` al aumentar Δ, y es peor que EASY en las 12 parejas, aunque las 10000 predicciones de test por corrida permanezcan correctas. Esto describe menor margen/confianza bajo interferencia, sin establecer por sí solo una curva de error de clasificación o robustez universal.

## Decisión científica, límites y siguiente tarea

En el endpoint preregistrado Δ=16, la ventaja media es **+50.00 pp**, positiva en **3/3** seeds; S1 alcanza **100%** medio y el ΔNLL medio es **−.691904**. Cumple literalmente las cuatro condiciones `SUPPORTED` de B1-NOISE-DESIGN. El gate técnico es **PASS**: generador, fuga, estado/gradiente/reset, parámetros, pareo, 24 corridas, test tras época final, criterio fijo y una decisión siguiente explícita. PASS técnico no se interpreta como superioridad frente a RNN/GRU/LSTM ni como ventaja de eficiencia física.

El experimento prueba retención útil frente a este ruido IID de amplitud `.1` en una tarea sintética binaria. No determina el máximo delay, un umbral de amplitud, comportamiento con distractores estructurados, tareas naturales, mecanismo dendrítico ni costo de RAM pico, latencia o energía. La accuracy nuevamente saturada impide afirmar una curva de límite. La subida de NLL es un efecto descriptivo consistente con menor separación bajo ruido, sin descomponer empíricamente cuánto proviene de decay frente a activaciones distractoras; la ecuación analítica muestra que ambos actúan en el mismo estado.

**Decisión única para Línea B:** recomendar **B2-DESIGN**, sólo preregistro de un contraste entre una escala temporal fija y escalas temporales heterogéneas fijas, con hipótesis y tarea que puedan revelar el tradeoff corto/largo. El apoyo a B1 no demuestra que la heterogeneidad vaya a ayudar. No se inició B2 ni DSU-T.

**Reproducibilidad y errores:** `uv run python -m dsu_research.b1_e02_run --preflight-only` y `uv run python -m dsu_research.b1_e02_run`; `uv run python -m dsu_research.b1_e02_audit` audita EASY/NOISE. El primer test local detectó que Windows convirtió a CRLF el preregistro, cuyo SHA-256 se había congelado con LF; la verificación normaliza sólo saltos de línea antes de comparar el hash. Ocurrió antes de entrenar y no invalidó corridas. No se cambiaron generador, modelo, λ, seeds, épocas ni decisión después de ver resultados. Los JSON son UTF-8 sin BOM. Suite completa: **117 passed, 2 skipped**; `git diff --check` sin errores.
