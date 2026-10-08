# EXPERIMENT_005 — Ablación de fan-in fijo (S3-T03 / S3-E01)

**Fecha:** 2026-10-08. **Gate técnico S3-T03: PASS. Clasificación H-009: NOT SUPPORTED / NEGLIGIBLE.** Las diez corridas completaron 25 épocas sin bug metodológico conocido. Esta clasificación sigue exactamente los criterios prefijados; no significa que fan-in nunca importe.

## Pregunta y brazos

¿Qué efecto tiene sustituir la topología global de fan-in variable de dANN-R por exactamente 16 entradas distintas por dendrita, manteniendo representación dense-mask, ancho y presupuesto efectivo?

| Brazo | Topología de entrada | Representación | Stored / effective parameters |
|---|---|---|---:|
| A — dANN-R | 8192 posiciones globales; fan-in variable, media 16 | dense-mask | 468874 / 10634 |
| B — Fixed-dANN | 16 entradas distintas por dendrita; 8192 totales | dense-mask | 468874 / 10634 |

A/B cambia simultáneamente distribución de fan-in, asignación concreta de aristas y regularidad topológica. El contraste estima el efecto de regularizar esta distribución a K=16 exacto dentro de la familia y protocolo probados; no aísla el número 16 manteniendo las mismas aristas.

## Pairing de inicialización y minibatches

Para cada seed se fijaron `training_seed=topology_seed∈{1,2,3,4,5}` como campos conceptualmente separados. Se creó una sola plantilla `VanillaANN` después de `seed_everything(training_seed)`: Glorot uniform para las tres matrices densas y sesgos cero. Sus seis tensores se clonaron y copiaron completos a A y B; después se aplicó la máscara propia de cada brazo. Así la inicialización densa subyacente, biases y salida son idénticos y sólo cambia qué aristas quedan activas. Cada pareja registra el mismo SHA-256 de esos tensores. Igual número de `topology_seed` no implica igual topología: A usa muestreo global y B el algoritmo por dendrita de DSU-S.

Cada brazo recreó independientemente el loader histórico con la misma seed. Un sampler envolvente registró el orden real de índices fuente emitido en cada época sin cambiar `RandomSampler`. Los 25 hashes por época, su hash combinado, los hashes de train y validation y los tamaños de split coincidieron exactamente dentro de cada pareja. Hashes combinados por seed: `85b5a2cd…`, `b71f4745…`, `63d93539…`, `b74befc0…`, `6e426fa7…`.

## Protocolo

Fashion-MNIST oficial, `ToTensor()` float32 `/255`, splits 54000/6000/10000, CPU, 25 épocas, batch 128, Adam (`lr=0.001`, betas 0.9/0.999, epsilon 1e-7, weight decay 0), CrossEntropyLoss sobre logits, LeakyReLU 0.1, sin scheduler, clipping, mixed precision, early stopping ni selección de checkpoint. Test usa `final_epoch_25`. Las duraciones incluyen train y validation; se registran sólo por completitud y no son benchmark de performance.

Comando: `.venv/Scripts/python.exe -m dsu_research.fixed_fanin_ablation --config experiments/configs/fixed_fanin_ablation_fmnist_s3_t03.json --output runs/fixed_fanin_s3_t03`. Los once JSON resultantes están en `runs/`, ignorado por Git.

## Resultados por seed

Accuracy se expresa en porcentaje. Deltas son B−A; loss negativo y accuracy positiva favorecen B.

| Seed | A train loss | B train loss | A val loss / acc. | B val loss / acc. | A test loss / acc. | B test loss / acc. | Δ test loss | Δ test acc. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.293264 | 0.294539 | 0.319539 / 88.583% | 0.317280 / 88.733% | 0.358651 / 87.29% | 0.363519 / 87.18% | +0.004869 | −0.11 pp |
| 2 | 0.293231 | 0.292243 | 0.325129 / 88.483% | 0.330934 / 87.983% | 0.353393 / 87.44% | 0.365908 / 86.94% | +0.012515 | −0.50 pp |
| 3 | 0.294044 | 0.295543 | 0.326166 / 88.217% | 0.319444 / 88.800% | 0.363152 / 86.98% | 0.361536 / 86.83% | −0.001616 | −0.15 pp |
| 4 | 0.289270 | 0.287896 | 0.319407 / 88.283% | 0.325401 / 88.183% | 0.357617 / 87.30% | 0.350562 / 87.52% | −0.007055 | +0.22 pp |
| 5 | 0.293276 | 0.290884 | 0.312151 / 89.417% | 0.312119 / 89.217% | 0.360219 / 87.07% | 0.356838 / 87.40% | −0.003382 | +0.33 pp |

| Métrica final | A mean ± sample SD | B mean ± sample SD | B−A mean ± sample SD |
|---|---:|---:|---:|
| Train loss | 0.292617 ± 0.001902 | 0.292221 ± 0.003037 | −0.000396 ± 0.001708 |
| Validation loss | 0.320478 ± 0.005598 | 0.321036 ± 0.007300 | +0.000558 ± 0.005440 |
| Validation accuracy | 88.597 ± 0.482% | 88.583 ± 0.498% | −0.013 ± 0.407 pp |
| Test loss | 0.358606 ± 0.003586 | 0.359673 ± 0.006087 | +0.001066 ± 0.007721 |
| Test accuracy | 87.216 ± 0.187% | 87.174 ± 0.293% | −0.042 ± 0.329 pp |
| Train+validation duration | 208.22 ± 3.59 s | 209.50 ± 7.91 s | no interpretar como performance |

B mejoró test accuracy en 2/5 seeds y A en 3/5. B mejoró test loss en 3/5 y A en 2/5. La dirección de loss y accuracy no coincide en seed 3. Los deltas medios satisfacen simultáneamente `abs(Δaccuracy)=0.042 pp < 0.15 pp` y `abs(Δloss)=0.001066 < 0.005`; por los criterios congelados, **H-009 = NOT SUPPORTED / NEGLIGIBLE**.

## Distribución de fan-in

| Seed | A min | A max | A mean | A sample SD | B min/max/mean/SD |
|---:|---:|---:|---:|---:|---|
| 1 | 4 | 29 | 16 | 4.047 | 16 / 16 / 16 / 0 |
| 2 | 5 | 29 | 16 | 4.020 | 16 / 16 / 16 / 0 |
| 3 | 7 | 29 | 16 | 3.959 | 16 / 16 / 16 / 0 |
| 4 | 6 | 33 | 16 | 3.835 | 16 / 16 / 16 / 0 |
| 5 | 5 | 29 | 16 | 3.986 | 16 / 16 / 16 / 0 |

La DE media del fan-in de A fue 3.969 ± 0.082 entre seeds. Ambos brazos tuvieron exactamente 8192 conexiones input→dendrite y 512 dendrite→soma por trial.

## Curvas y sanity histórico

Las curvas medias casi se superponen. B tuvo train loss medio ligeramente menor desde la época 1 hasta la 25; la ventaja final fue sólo 0.000396. Los deltas de validation cambiaron de signo: Δaccuracy media B−A fue −0.05 pp en época 1, +0.11 pp en época 10, −0.12 pp en época 15 y −0.013 pp en época 25. No aparece señal consistente de convergencia más rápida, final superior o menor variabilidad de B. Esta lectura es exploratoria.

El nuevo brazo A reprodujo **exactamente por seed** las 25 épocas completas, test loss, test accuracy, hashes de split y hashes de máscara guardados en `REPRODUCTION_001`; por tanto, su media histórica y nueva coinciden: loss 0.358606 y accuracy 87.216%. La nueva política explícita de pairing conserva la inicialización y el orden históricos para A. No hubo regresión ni anomalía de sanity.

## Interpretación y límites

Dentro de Fashion-MNIST, N=5 y este protocolo dense-mask, regularizar la distribución a K=16 exacto mostró efecto medio despreciable sobre calidad final. La variabilidad entre seeds/topologías fue mayor que el delta medio. Esto no demuestra que fan-in nunca importe, que todas las topologías fijas sean equivalentes, ni que el resultado generalice a otros presupuestos, datasets o escalas. Las aristas A/B son distintas; N=5 es descriptivo; no se usaron p-values. La duración no se interpreta como latencia, throughput o energía. S3-E01 no vuelve a probar H-010 ni compara representación compacta.

**Bugs metodológicos encontrados:** ninguno. Tras las corridas, un fixture unitario situado exactamente en un umbral decimal activó comparación float por representación binaria; se movió claramente dentro del intervalo sin cambiar implementación, criterios ni resultados. **Resultados negativos:** la hipótesis material H-009 no recibió apoyo bajo los umbrales prefijados. **Siguiente tarea exacta:** `S3-T04 — Ejecutar S3-E02: benchmark físico controlado Fixed-dANN vs DSU-S`. No se ejecutó aquí.
