# EXPERIMENT_002 — MLP igualado por parámetros (S2-T05)

**Fecha de especificación previa a entrenamiento:** 2026-10-06. **Pregunta:** ¿DSU-S v0 usa aproximadamente 10.6k parámetros con mayor eficacia de clasificación que un MLP denso convencional de profundidad no lineal comparable?

## Diseño congelado

MLP `784→13→17→10`, LeakyReLU con `negative_slope=0.1` tras las dos capas ocultas y salida de logits. La primera anchura se eligió para colocar la mayor parte del presupuesto en las entradas de 784 píxeles; 17 en la segunda capa ajusta el total a 10623, sólo 11 menos que los 10634 de DSU-S (~0.103 %). No se buscará otra arquitectura tras ver los resultados. Inicialización Glorot uniform por `nn.Linear` y sesgos cero, igual filosofía explícita que vANN/dANN-R; no se puede igualar la distribución exacta de pesos iniciales con DSU-S porque sus matrices y fan-in son distintos.

Conteo analítico: `784×13+13=10205`, `13×17+17=238`, `17×10+10=180`; total **10623**. MAC principales por muestra: `784×13+13×17+17×10=10583`; DSU-S **9984**. Son presupuestos MAC próximos, no equivalencia exacta de FLOPs, latencia, RAM ni energía. El MLP tiene 30 activaciones ocultas (13+17); DSU-S tiene 640 (512 dendríticas+128 somáticas). No son parámetros.

Fashion-MNIST 54000/6000/10000, `ToTensor()` float32/255, split `np.random.default_rng(seed)` idéntico a S2-T04, Adam lr 0.001, betas 0.9/0.999, epsilon 1e-7, batch 128, 25 épocas, cross-entropy sobre logits y test al final de la época 25. Validation es sólo diagnóstico. Seeds de entrenamiento **1, 2, 3**; MLP sin `topology_seed`. Se reutilizan los JSON DSU-S S2-T04 sin reentrenar. No hay tuning.

## Criterios científicos prefijados

- **FUERTE A FAVOR DE DSU:** DSU-S supera al MLP por ≥2.0 pp de accuracy media, o presenta ventaja muy clara y consistente en loss junto con mejor accuracy.
- **MODERADA/PROMETEDORA:** DSU-S supera al MLP por 0.75–2.0 pp de accuracy media con comportamiento razonablemente consistente.
- **SIMILAR/INCONCLUSIVE:** diferencia absoluta de accuracy <0.75 pp, sin ventaja clara y consistente en loss.
- **CONTRARIA:** MLP supera a DSU-S por ≥0.75 pp, o muestra ventaja consistente importante en loss.

Son umbrales heurísticos internos, no tests de significancia. Los deltas se calcularán como **MLP menos DSU-S**: loss positivo favorece DSU-S; accuracy negativa favorece DSU-S. Si el resultado es FUERTE o MODERADA, siguiente tarea S2-T06 (sin ejecutarla aquí). Si es SIMILAR/INCONCLUSIVE o CONTRARIA, definir tarea diagnóstica basada en el resultado antes de avanzar.

## Ejecución y resultados

**Gate técnico:** PASS. **Clasificación científica:** MODERADA/PROMETEDORA. **H-008:** WEAK SUPPORT. Config versionada `experiments/configs/parameter_matched_mlp_fmnist_s2_t05.json`; para cada seed 1, 2 y 3 se cambió únicamente `seed` en una copia local y se ejecutó `.venv/Scripts/python.exe -m dsu_research.run --config runs/parameter_matched_mlp_s2_t05/config_seed<seed>.json --output runs/parameter_matched_mlp_s2_t05/mlp_seed<seed>.json`. Los JSON completos y las copias de config quedan en `runs/` (ignorado por Git). Los JSON DSU-S originales están en `runs/dsu_s_v0_s2_t04/`. No se reentrenó DSU-S ni se cambió el baseline tras observar resultados.

En las tres seeds coinciden exactamente con S2-T04 los SHA-256 de índices train y validation. Cada historia tiene 25 épocas, 1 350 000 ejemplos de train vistos y `checkpoint_selected=final_epoch_25`. Todos los parámetros del MLP son entrenables, almacenados y efectivos: **10623**, verificados por test y JSON. Tensores float32 entrenables: **42492 bytes**; MAC principales teóricos: **10583** por muestra. Los 10634 parámetros, 42536 bytes de pesos y 9984 MAC de DSU-S se reutilizan de S2-T04. El MLP no almacena índices de topología; DSU-S sí (65536 bytes). Comparar sólo bytes de pesos oculta ese coste. Entorno: Windows 11, CPU AMD Ryzen 5 5600G, PyTorch 2.14.1+cpu, float32; versiones exactas en los JSON.

| Seed | MLP test loss | MLP accuracy | DSU-S test loss | DSU-S accuracy | Δ loss MLP−DSU-S | Δ accuracy MLP−DSU-S (pp) | Train+validation MLP (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.398530 | 85.98 % | 0.357590 | 87.18 % | +0.040941 | −1.20 | 147.16 |
| 2 | 0.405998 | 85.56 % | 0.355466 | 87.56 % | +0.050532 | −2.00 | 149.34 |
| 3 | 0.390034 | 86.11 % | 0.360419 | 87.02 % | +0.029615 | −0.91 | 151.12 |
| **Media ± DE muestral** | **0.398187 ± 0.007987** | **85.883 ± 0.287 %** | **0.357825 ± 0.002485** | **87.253 ± 0.277 %** | **+0.040362 ± 0.010470** | **−1.370 ± 0.565** | **149.20 ± 1.99** |

El signo positivo de Δ loss y negativo de Δ accuracy favorece a **DSU-S**. Las DE de los deltas se calcularon sobre las tres diferencias pareadas, no restando DE independientes. Train+validation MLP sumó **447.61 s (7.46 min)**; test e inferencia se excluyen. No se interpreta la diferencia de tiempo frente a DSU-S como benchmark controlado de velocidad.

Train loss inicial→final: seed 1 **0.920933→0.339883**, seed 2 **0.809465→0.321882**, seed 3 **0.844241→0.328650**. Validation loss inicial→final: **0.584478→0.361124**, **0.537369→0.385689**, **0.520083→0.356091**; accuracy final validation **87.38 %, 86.57 %, 87.80 %**. El mínimo de validation loss ocurrió en épocas 25, 22 y 25 respectivamente; en seed 2 el mínimo fue antes del final, pero el test usa la época 25 como se prefijó. No hubo NaN/Inf ni colapso aparente. El resultado inesperado es la ventaja consistente de loss DSU-S pese a que el MLP tiene un presupuesto de parámetros casi idéntico y ~6 % más MAC principales.

La diferencia media de **1.370 pp** cae en el intervalo prefijado de señal **MODERADA/PROMETEDORA** (0.75–2.0 pp), con la misma dirección en las tres seeds y loss también favorable a DSU-S. En esta tarea y presupuesto, los datos apoyan que DSU-S clasifica mejor que **este MLP denso concreto**. N=3, una sola arquitectura MLP y diferencias de inicialización, conectividad y cantidad de activaciones impiden atribuir causalmente toda la diferencia a “dendritas” o extrapolar a otros MLP/datasets. Tampoco demuestran optimización biológica, escalabilidad, velocidad, menor RAM runtime, ahorro energético ni utilidad en Transformers. No se midió RAM de activaciones ni memoria temporal.

**Siguiente tarea exacta:** S2-T06 — Determinar qué tamaño necesita un MLP denso para igualar la calidad de DSU-S v0. No ejecutada aquí.
