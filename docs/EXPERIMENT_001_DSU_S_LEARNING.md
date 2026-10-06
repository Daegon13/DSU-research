# EXPERIMENT_001 — Aprendizaje de DSU-S v0 (S2-T04)

**Fecha:** 2026-10-06. **Gate técnico S2-T04:** PASS. **Clasificación científica predefinida:** FUERTE. La arquitectura de DSU-S v0 permaneció congelada; no hubo tuning ni reentrenamiento de baselines.

## Pregunta y protocolo

¿Puede DSU-S v0 mantener rendimiento cercano a dANN-R con sólo 10634 parámetros entrenables físicamente almacenados? Se comparan tres trials predefinidos de Fashion-MNIST con la pareja fija vANN/dANN-R de `REPRODUCTION_001`. El presupuesto igualado es anchura oculta (512 dendritas/128 somas) y protocolo de datos/entrenamiento; no se igualan parámetros almacenados, cómputo, tiempo ni memoria de ejecución.

Fashion-MNIST oficial: 54000 train, 6000 validation y 10000 test; `ToTensor()` entrega float32 con píxeles divididos por 255. El split de cada seed usa `np.random.default_rng(training_seed)` y sus hashes train/validation coinciden con los dos baselines de esa seed. Adam (lr 0.001, betas 0.9/0.999, epsilon 1e-7), batch 128, 25 épocas, cross-entropy sobre logits, sin scheduler, clipping, dropout, early stopping ni selección por validation. Test se evalúa una vez con el estado final de época 25. El historial de cada JSON registra train loss/accuracy y validation loss/accuracy en cada época.

Trials fijados antes de observar test: `(training_seed, topology_seed) = (1,1), (2,2), (3,3)`. En los JSON, `config.seed` y `seeds.python/numpy/torch` registran `training_seed`, mientras que `config.topology_seed` registra la segunda seed. Son campos conceptualmente independientes: la primera gobierna inicialización, split y shuffle; la segunda genera sólo `G` mediante RNG NumPy local. No se cambió la estrategia entre corridas. La topología tiene exactamente 16 entradas distintas por dendrita, a diferencia de las 16 **en promedio** de la máscara global de dANN-R.

**Reproducción:** plantilla versionada `experiments/configs/dsu_s_v0_fmnist_s2_t04.json`. Para cada seed 1, 2 y 3, generar una copia JSON con ambos campos `seed` y `topology_seed` iguales a esa seed y ejecutar `.venv/Scripts/python.exe -m dsu_research.run --config <copia.json> --output runs/dsu_s_v0_s2_t04/dsu_s_seed<seed>.json`. Las copias concretas (`config_seed1.json` a `config_seed3.json`) y JSON completos están en `runs/dsu_s_v0_s2_t04/`, ignorado por Git. Los baselines originales están en `runs/reproduction_001/`; no se reentrenaron. Entorno de los trials: Windows 11, CPU AMD Ryzen 5 5600G, PyTorch 2.14.1+cpu, torchvision 0.29.1+cpu, NumPy 2.5.3, float32. Los JSON registran versiones, seeds, hashes, configuración, historia, duración e inferencia batch 1.

Los SHA-256 registran los índices de split y el buffer `G` en bytes little-endian int64; train y validation coinciden con los hashes de ambos baselines para la misma seed.

| Seed | Train SHA-256 | Validation SHA-256 | Topología SHA-256 |
|---:|---|---|---|
| 1 | `f3838eee6ae916e4cb9a41b585966f0e1f089433a0abd66f48ef7d03fbed25ea` | `d363e9447f471005e766deb8b12f656c26b6954268c4ad498151b6db2ebf561f` | `86256350473bceeffd4745681f67323bbc6427f54aa02ff48a7de9111cc1a7c3` |
| 2 | `acf787f92794e9268917dc93b6f4d4cfbbeadfe1b615dc9f61dc6a3365b6f4b8` | `56fbaf61830f73d2cedff231c5241b11424ad284c070b56dd8d5872583b2c027` | `f128471bff014b8bcdada26649e6e0014f110ccc286ddf081b69b57bb8464e0d` |
| 3 | `ad16d1e1628fa31055911a4c372dd8aceeb9bfae10f2d0cfa90fd3a12ea4fc0e` | `22a1e84521fc0530406c994d452b7939ca71265a7b08e3b1997496a1acac3581` | `88a0765d2954671798d10ad73ae197a2ac586387171d9a24dd55b87958d3ecd2` |

## Resultados por seed

Accuracy y sus deltas se expresan en porcentaje y puntos porcentuales, respectivamente. Los deltas son **DSU-S menos baseline**. Duración incluye train y validation de las 25 épocas; excluye test e inferencia. El mínimo de validation se registra sólo como diagnóstico.

| Seed train/topology | Test loss | Accuracy | Δ loss vs dANN-R | Δ acc. vs dANN-R (pp) | Δ loss vs vANN | Δ acc. vs vANN (pp) | Train+val (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1/1 | 0.357590 | 87.18 % | −0.001061 | −0.11 | −0.013189 | −2.25 | 198.85 |
| 2/2 | 0.355466 | 87.56 % | +0.002073 | +0.12 | −0.019505 | −1.87 | 184.49 |
| 3/3 | 0.360419 | 87.02 % | −0.002733 | +0.04 | −0.015310 | −2.42 | 172.96 |
| **Media ± DE muestral** | **0.357825 ± 0.002485** | **87.253 ± 0.277 %** | **−0.000574 ± 0.002440** | **+0.017 ± 0.117** | **−0.016001 ± 0.003214** | **−2.180 ± 0.282** | **185.43 ± 12.97** |

La duración total de train+validation fue **556.30 s (9.27 min)**. El throughput del protocolo y la inferencia batch 1 se guardan por trial en los JSON. La media descriptiva de la inferencia automática del harness fue 0.10982 ± 0.01129 ms/batch 1; los tiempos de procesos distintos no permiten inferir una ventaja de latencia frente a las mediciones antiguas de vANN/dANN-R.

Los baselines N=3 originales, con precisión íntegra en sus JSON, son: dANN-R loss 0.358398 ± 0.004885, accuracy 87.237 ± 0.235 %; vANN loss 0.373826 ± 0.002666, accuracy 89.433 ± 0.006 %. La comparación DSU-S/dANN-R cambia de signo entre seeds: en seed 2 DSU-S tiene loss **mayor** (+0.002073), aunque su accuracy es mayor. No se oculta esta inversión. Frente a vANN, DSU-S tiene menor loss en las tres seeds y menor accuracy en las tres. Los deltas agregados de la tabla son medias y DE de las **diferencias pareadas**, no resta de DE independientes.

## Estabilidad y resultado científico

Los tres trials completaron 25 épocas sin NaN ni Inf en historia, test o métricas registradas. Train loss primera→última época: 1.090351→0.293666 (seed 1), 1.073579→0.287105 (seed 2), 1.082579→0.293579 (seed 3). Validation loss primera→última: 0.590124→0.315036, 0.568952→0.324821, 0.582654→0.321524. La accuracy final de validation fue 88.50 %, 88.63 % y 88.33 %. No hay indicio en estas métricas de colapso a una clase, pérdida que no disminuye o comportamiento errático grande entre seeds. El harness no registra normas de gradiente ni predicciones por clase: no se afirma haber medido directamente exploding gradients o distribución de clases.

La accuracy media de DSU-S queda **0.017 pp por encima** de dANN-R; su loss media es **0.000574 menor**. Las tres corridas fueron estables. Satisface ambos umbrales de **señal FUERTE** (degradación de accuracy ≤0.75 pp y aumento de loss ≤0.02) fijados antes de la ejecución. Son umbrales heurísticos internos, no prueba de significancia. Con N=3, las diferencias de calidad son descriptivas y pequeñas; el resultado apoya **H-001 = SUPPORTED** sólo en su comparación inicial de igual anchura contra dANN-R.

## Estructura y tamaño

| Modelo | Parámetros almacenados | Parámetros efectivos | Observación |
|---|---:|---:|---|
| vANN | 468874 | 468874 | Matrices densas |
| dANN-R | 468874 | 10634 | Matrices densas enmascaradas; 16 entradas por dendrita en promedio |
| DSU-S v0 | 10634 | 10634 | Pesos compactos; exactamente 16 entradas por dendrita |

`468874 / 10634 = 44.09197`; DSU-S almacena **97.732 % menos parámetros entrenables** que vANN y dANN-R dense-mask. Sus tensores entrenables float32 ocupan **42536 bytes** y el buffer persistente `G`, 8192 índices int64, **65536 bytes**. Su suma es **108072 bytes de payload tensorial**. Los MAC principales teóricos son **9984 por muestra** (8192 dendrita + 512 soma + 1280 salida); no incluyen gather, activaciones, tráfico de memoria ni backward.

Se serializó con `torch.save(DSUSv0(1).state_dict(), ...)` una instancia recién inicializada de la misma arquitectura y topología seed 1: `runs/dsu_s_v0_s2_t04/dsu_s_state_dict_seed1.pt` mide **111508 bytes**. Incluye los seis tensores entrenables y `G`; la diferencia con 108072 bytes refleja el contenedor de serialización. Es tamaño de un `state_dict` representativo, **no un checkpoint de pesos finales**: el harness guarda métricas, pero no pesos entrenados. Los bytes de tensores y archivo no equivalen a RAM pico, memoria de Adam, activaciones, memoria instalada ni ahorro energético. No se añadió profiler; RAM pico CPU permanece sin medir.

Comando de medición ejecutado desde la raíz (el archivo queda ignorado por Git): `.venv/Scripts/python.exe -c "import torch; from pathlib import Path; from dsu_research.dsu_s import DSUSv0; from dsu_research.harness import seed_everything; seed_everything(1); p=Path('runs/dsu_s_v0_s2_t04/dsu_s_state_dict_seed1.pt'); torch.save(DSUSv0(1).state_dict(), p); print(p.stat().st_size)"`.

## Límites y siguiente tarea

DSU-S y dANN-R difieren en fan-in exacto frente a promedio, topologías concretas, inicialización Glorot sobre shapes compactos frente a matrices densas y operaciones PyTorch. La proximidad de calidad no aísla el efecto del almacenamiento compacto. Tampoco se comparó aún un MLP de presupuesto de parámetros similar ni se evaluaron velocidad o energía con un diseño controlado. Las ablaciones de topología corresponden a Sprint 3.

**Gate técnico S2-T04 = PASS:** tres trials completos, sin bug metodológico conocido, métricas y deltas registrados, baselines originales reutilizados y arquitectura congelada. **Siguiente tarea exacta:** `S2-T05 — Comparar DSU-S v0 contra un MLP aproximadamente igualado por parámetros`. No se ejecutó S2-T05.
