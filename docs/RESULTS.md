# RESULTS

Existen resultados diagnósticos de Sprint 0 y una reproducción N=5 de Sprint 1; R-002 conserva el corte intermedio N=3.

Gate S0 auditado el 2026-10-05: **PASS**. Evidencia de cierre en R-001.

## Regla

Registrar:

- resultados positivos;
- resultados negativos;
- resultados inconclusos.

No borrar resultados por no coincidir con la hipótesis.

---

# Tabla maestra

| ID | Modelo | Dataset | Parámetros | Calidad | Latencia | Memoria | Estado |
|---|---|---|---:|---:|---:|---:|---|
| R-000 | MLP/ReLU | MNIST, 1024 train / 256 test | 101770 | accuracy test 0.6914, loss 1.6144 | 0.0468 ms/batch 1 | RAM CPU no medida | COMPLETE (smoke) |
| R-001 | MLP/ReLU | MNIST, 1024 train / 256 test | 101770 | seed 42: accuracy test 0.6914 (1 época), 0.8125 (3 épocas) | 0.0516 ms/batch 1 (primera repetición de 1 época) | RAM CPU no medida | COMPLETE (gate S0) |
| R-002 | vANN / dANN-R | Fashion-MNIST, 54000/6000/10000, N=3 | 468874 almacenados ambos; efectivos 468874 / 10634 | test loss 0.373826 / 0.358398; accuracy 89.433 / 87.237 % | batch 1 por trial abajo | RAM CPU no medida | COMPLETE (S1-T02 PASS) |
| R-003 | vANN / dANN-R | Fashion-MNIST, 54000/6000/10000, N=5 | 468874 almacenados ambos; efectivos 468874 / 10634 | test loss 0.374481 / 0.358606; accuracy 89.322 / 87.216 % | no se infiere aceleración | RAM CPU no medida | COMPLETE (REPRODUCED; Sprint 1 PASS) |

---

# Resultado R-000

**Status:** COMPLETE (smoke; Sprint 0 sigue abierto)

**Fecha:** 2026-10-05. **Comando:** `uv run dsu-run --config experiments/configs/mnist_mlp_smoke.json --output runs/s0_mlp_mnist_smoke.json`.

**Config:** seed 42, CPU, float32, 1 epoch, batch 64, Adam, lr 0.001, sin weight decay, 1024 primeros ejemplos del train oficial y 256 primeros del test oficial de MNIST; `ToTensor()` sin normalización adicional. Python 3.12.13, PyTorch 2.14.1+cpu, torchvision 0.29.1+cpu, NumPy 2.5.3, Windows 11, CPU AMD64 Family 25 Model 80. No hay validation ni tuning.

**Modelo:** `Flatten → Linear(784,128) → ReLU → Linear(128,10)`; 101770 parámetros entrenables y totales.

**Resultado inicial:** train loss 1.940615, train accuracy 0.566406; test loss 1.614402, test accuracy 0.691406. Entrenamiento 0.12345 s, 8294.7 samples/s. Inferencia batch 1 con 5 warm-ups y 20 repeticiones: media 0.046815 ms/batch, desviación estándar 0.000888 ms, 21360.7 samples/s medidos sólo para el forward.

**Repetición:** mismo config y comando con `--output runs/s0_mlp_mnist_smoke_repeat.json`. Loss y accuracy de train/test idénticos al valor completo guardado en JSON. Entrenamiento 0.11697 s; latencia media 0.050255 ms/batch. Los tiempos son sensibles al entorno y no constituyen evidencia de rendimiento comparativo.

**Memoria:** CUDA no aplicable; RAM pico CPU no medida (`null` en JSON). No se midieron FLOPs, checkpoint ni energía en Sprint 0.

**Interpretación:** funciona el pipeline de descarga, entrenamiento, evaluación, medición y exportación. Una época y una seed en subconjuntos pequeños no establecen accuracy final, convergencia sostenida ni ventaja arquitectónica.

---

# Resultado R-001 — auditoría Gate S0

**Status:** COMPLETE (diagnóstico; Gate S0 PASS). **Fecha:** 2026-10-05.

**Instalación:** `uv sync --locked --extra test` en un entorno virtual aislado; 20 paquetes instalados desde `uv.lock`. Suite completa: 6/6 tests. Un test ejecuta el mismo harness con un clasificador distinto y entrada de cuatro características; otro cuenta 3 warm-ups y 7 repeticiones de batch 1; otro verifica loss y accuracy conocidos sobre dos ejemplos.

**Comandos y artefactos locales:** `dsu-run --config experiments/configs/mnist_mlp_smoke.json --output runs/s0_gate_final_seed42_a.json`; repetición idéntica en `runs/s0_gate_final_seed42_b.json`. La configuración derivada con seed 43 quedó en `runs/s0_gate_seed43_config.json` y su resultado en `runs/s0_gate_final_seed43.json`. La configuración derivada con 3 épocas quedó en `runs/s0_gate_convergence_config.json` y su resultado en `runs/s0_gate_convergence.json`. Los JSON de `runs/` son locales e ignorados por Git.

**Config común:** MNIST oficial, primeros 1024 train y 256 test, `ToTensor()` sin normalización extra, batch 64, Adam, lr 0.001, CPU float32. Python 3.12.13, PyTorch 2.14.1+cpu, torchvision 0.29.1, NumPy 2.5.3, Windows 11. Hardware: AMD Ryzen 5 5600G (6 núcleos/12 hilos), 33.69 GB RAM física; GPU integrada AMD Radeon Graphics, no usada. RAM pico CPU, tamaño de checkpoint, FLOPs y energía no medidos.

**Conteo independiente:** `(784 + 1) × 128 + (128 + 1) × 10 = 101770`, igual al total y a los entrenables registrados.

**Reproducibilidad:** las dos ejecuciones finales con seed 42 y 1 época tienen train loss 1.9406150877, train accuracy 0.56640625, test loss 1.6144016087 y test accuracy 0.69140625 exactamente iguales en JSON. Con seed 43: train loss 1.9531081915, train accuracy 0.5517578125, test loss 1.6163665950, test accuracy 0.6796875. La seed modifica el orden de entrenamiento y la inicialización. Los tiempos no son deterministas.

**Última ejecución válida:** seed 42, 3 épocas, `runs/s0_gate_convergence.json`, 2026-10-05T20:27:02Z. Train loss por época: 1.940615 → 1.168307 → 0.728188; train accuracy: 0.566406 → 0.762695 → 0.844727. Test loss 0.755660; test accuracy 0.8125. Entrenamiento 0.317852 s, 9665.0 samples/s aproximadamente. Inferencia batch 1: 5 warm-ups, 20 repeticiones, media 0.046925 ms/batch, desviación estándar 0.001163 ms. Los otros dos procesos seed 42 midieron 0.051600 y 0.033760 ms/batch; la variación impide inferencias de velocidad.

**Interpretación:** instalación, tests, entrenamiento, exportación JSON y reutilización del harness verificados. La bajada de loss durante tres épocas confirma aprendizaje en el subconjunto diagnóstico. No hay todavía comparación entre arquitecturas ni evidencia estadística de calidad final.

---

# Resultado R-002 — S1-T02 vANN vs dANN-R

**Status:** COMPLETE, **gate S1-T02 PASS**. **Fecha:** 2026-10-05. Comando: `.venv/Scripts/python.exe -m dsu_research.run --config experiments/configs/reproduction_001_fmnist.json --output runs/reproduction_001 --seeds 1 2 3`. Artefactos locales ignorados por Git: seis JSON por modelo/seed y su resumen, conservado como `runs/reproduction_001/summary_n3.json` tras el cierre N=5. Suite: 10/10 tests. Los seis JSON contienen train loss, validation loss y validation accuracy de las 25 épocas, además de train accuracy, test loss/accuracy, duración, throughput, latencia, seeds, hashes, entorno, parámetros y estado final evaluado.

**Protocolo:** Fashion-MNIST oficial, píxeles float32 / 255, split por trial de 54000 train / 6000 validation y test oficial de 10000. Cada pareja comparte split (hashes de índices verificados). Adam lr 0.001, betas 0.9/0.999, epsilon 1e-7, batch 128, 25 épocas, cross-entropy sobre logits, sin scheduler, clipping, dropout ni early stopping. Test únicamente después de la época 25. El mínimo de validation es diagnóstico, no checkpoint seleccionado. Seeds de Python, NumPy, PyTorch, máscara y shuffle registradas. Entorno y hardware están en los JSON; CPU Ryzen 5 5600G, PyTorch 2.14.1+cpu, float32.

| Seed | Modelo | Test loss | Test accuracy | Duración train+val (s) | Samples/s de protocolo | Inferencia batch 1 (ms) | Mín/máx entradas por dendrita |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | vANN | 0.370778 | 89.43 % | 179.99 | 7500.3 | 0.09216 | — |
| 1 | dANN-R | 0.358651 | 87.29 % | 189.63 | 7119.2 | 0.28842 | 4/29 |
| 2 | vANN | 0.374970 | 89.43 % | 176.98 | 7627.8 | 0.06795 | — |
| 2 | dANN-R | 0.353393 | 87.44 % | 185.59 | 7274.1 | 0.27382 | 5/29 |
| 3 | vANN | 0.375728 | 89.44 % | 176.02 | 7669.7 | 0.08527 | — |
| 3 | dANN-R | 0.363152 | 86.98 % | 185.88 | 7262.7 | 0.23742 | 7/29 |

**Agregado N=3, media ± desviación estándar muestral:** vANN test loss **0.373826 ± 0.002666**, accuracy **89.433 ± 0.006 %**; dANN-R test loss **0.358398 ± 0.004885**, accuracy **87.237 ± 0.235 %**. dANN-R reduce test loss en 0.015427 (4.13 %) respecto a vANN en estas tres seeds; la accuracy es 2.197 puntos porcentuales menor. La pérdida de dANN-R es menor para cada seed emparejada.

**Referencia archivada N=5 de la misma configuración:** vANN loss 0.401243 ± 0.020207 y accuracy 89.112 ± 0.621 %; dANN-R loss 0.365028 ± 0.003595 y accuracy 86.830 ± 0.184 %. Nuestras medias de loss son 0.027417 (vANN) y 0.006630 (dANN-R) menores que las archivadas. La diferencia entre modelos de nuestro N=3 es menor que la archivada (0.015427 frente a 0.036215). N=3 frente a N=5 y diferencias PyTorch/Keras no permiten reclamar coincidencia estadística; los valores son cercanos en magnitud y la dirección del efecto se conserva. No se modificaron hiperparámetros para acercar las cifras.

**Conectividad y almacenamiento:** ambos modelos almacenan y entrenan 468874 entradas; vANN tiene 468874 efectivas. dANN-R tiene 8192 conexiones input→dendrita y 512 dendrita→soma activas, media 16.0 entradas por dendrita, más 512 sesgos dendríticos, 128 somáticos y 1290 pesos/sesgos de salida: **10634 efectivos**. Las máscaras difieren entre las tres seeds y se registran sus SHA-256. La reducción 44.09× sólo describe conexiones efectivas; la dANN-R reproducida usa operaciones y almacenamiento densos, por lo que su inferencia medida fue más lenta en este entorno. No se infiere reducción de FLOPs, RAM ni energía. Duración total de train+validation para las seis corridas: 1094.10 s (18.24 min). RAM pico CPU, FLOPs, checkpoint físico y energía no medidos.

**Diferencias y límites:** Glorot uniform y biases cero explícitos aproximan los defaults de Keras, pero PyTorch y TensorFlow no generan los mismos pesos con la misma seed. Adam epsilon se fija a 1e-7, aunque su aritmética puede diferir. PyTorch usa cross-entropy sobre logits; Keras softmax y pérdida sobre probabilidades. El shuffle reproducible de DataLoader no tiene el mismo orden que `tf.data.Dataset.shuffle` del original. El reloj de train incluye validation y el throughput divide ejemplos de train por ese tiempo. Estas diferencias son metodológicas conocidas pero no invalidan la arquitectura ni el sentido de la comparación. No se observan errores de arquitectura, máscara, split ni test. La tarea siguiente en el momento de R-002 era **S1-T03**; el cierre posterior consta en R-003.

---

# Resultado R-003 — cierre N=5, S1-T03/S1-T04

**Status:** COMPLETE. **Fecha:** 2026-10-05. **Gate REPRODUCTION_001:** REPRODUCED para la pareja fija D=4/S=128. **Gate Sprint 1:** PASS. Se añadieron únicamente las seeds 4 y 5 mediante `.venv/Scripts/python.exe -m dsu_research.run --config experiments/configs/reproduction_001_fmnist.json --output runs/reproduction_001 --seeds 4 5`. Seeds 1–3 son los JSON originales de R-002. Los diez JSON y `summary.json` N=5 son locales e ignorados por Git; `summary_n3.json` conserva el resumen previo. Suite completa: **10/10** tests.

**Protocolo:** el de R-002, sin cambios: Fashion-MNIST 54000/6000/10000, `ToTensor()` a float32/255, Adam lr 0.001, betas 0.9/0.999, epsilon 1e-7, batch 128, 25 épocas, test sólo tras la época 25. Comparación por igual anchura oculta, no por igual presupuesto de parámetros efectivos. Duración incluye train y validation por época, excluye test e inferencia. Cada pareja comparte split; entre las cinco seeds cambian los hashes de train, validation y máscara. La auditoría completa de seeds y precisión está en `docs/REPRODUCTION_001.md`.

| Seed | Modelo | Test loss | Test accuracy | Train+val (s) | Efectivos | Almacenados |
|---:|---|---:|---:|---:|---:|---:|
| 1 | vANN | 0.370778 | 89.43 % | 179.99 | 468874 | 468874 |
| 1 | dANN-R | 0.358651 | 87.29 % | 189.63 | 10634 | 468874 |
| 2 | vANN | 0.374970 | 89.43 % | 176.98 | 468874 | 468874 |
| 2 | dANN-R | 0.353393 | 87.44 % | 185.59 | 10634 | 468874 |
| 3 | vANN | 0.375728 | 89.44 % | 176.02 | 468874 | 468874 |
| 3 | dANN-R | 0.363152 | 86.98 % | 185.88 | 10634 | 468874 |
| 4 | vANN | 0.354998 | 89.29 % | 168.13 | 468874 | 468874 |
| 4 | dANN-R | 0.357617 | 87.30 % | 188.72 | 10634 | 468874 |
| 5 | vANN | 0.395928 | 89.02 % | 182.77 | 468874 | 468874 |
| 5 | dANN-R | 0.360219 | 87.07 % | 196.04 | 10634 | 468874 |

**Media ± DE muestral (`ddof=1`, N=5):** vANN test loss **0.374481 ± 0.014621**, accuracy **89.322 ± 0.180 %**; dANN-R test loss **0.358606 ± 0.003586**, accuracy **87.216 ± 0.187 %**. La DE de accuracy está en puntos porcentuales.

| Seed | Δ loss = dANN-R − vANN | Δ accuracy (puntos porcentuales) |
|---:|---:|---:|
| 1 | −0.012128 | −2.14 |
| 2 | −0.021578 | −1.99 |
| 3 | −0.012576 | −2.46 |
| 4 | **+0.002618** | −1.99 |
| 5 | −0.035709 | −1.95 |
| **Media ± DE muestral** | **−0.015874 ± 0.014078** | **−2.106 ± 0.211** |

La pérdida media de dANN-R es **4.24 % menor** respecto a la media vANN de esta reproducción. Hay cuatro parejas favorables y una desfavorable para loss. La accuracy de dANN-R es menor en las cinco parejas. Con N=5 no se afirma significancia estadística fuerte.

**Comparación con archivo oficial de la misma fila fija, N=5:** las diferencias son *nuestra media menos media oficial*. El porcentaje relativo divide por la media oficial; para accuracy se informa además la diferencia absoluta en puntos porcentuales.

| Modelo/métrica | Nuestra media ± DE | Oficial media ± DE | Diferencia absoluta | Diferencia relativa | Dirección |
|---|---:|---:|---:|---:|---|
| vANN loss | 0.374481 ± 0.014621 | 0.401243 ± 0.020207 | −0.026762 | −6.67 % | Menor aquí |
| vANN accuracy | 89.322 ± 0.180 % | 89.112 ± 0.621 % | +0.210 pp | +0.236 % | Mayor aquí |
| dANN-R loss | 0.358606 ± 0.003586 | 0.365028 ± 0.003595 | −0.006422 | −1.76 % | Menor aquí |
| dANN-R accuracy | 87.216 ± 0.187 % | 86.830 ± 0.184 % | +0.386 pp | +0.445 % | Mayor aquí |

El archivo oficial presenta Δ medio loss dANN-R − vANN = **−0.036215** y Δ accuracy = **−2.282 pp**; aquí son **−0.015874** y **−2.106 pp**. El efecto medio de loss tiene la misma dirección y es menor en magnitud. Los rendimientos están en un rango comparable, pero las medias no coinciden exactamente. Diferencias conocidas: pesos iniciales no idénticos entre PyTorch/Keras, orden de shuffle de DataLoader frente a `tf.data`, posible aritmética distinta de Adam y pérdida sobre logits frente a softmax/probabilidades. El paper menciona 20 épocas mientras que el archivo/código oficial de esta fila usa 25; su máscara global da 16 entradas por dendrita sólo en promedio. Ninguna diferencia individual fue aislada como causa cuantitativa.

**Eficiencia de parámetros/conectividad:** `468874 / 10634 = 44.09×`; dANN-R tiene **97.73 % menos parámetros/conexiones efectivos** que vANN de igual anchura. Los dos almacenan y entrenan **468874 parámetros densos**. El resultado demuestra eficiencia de conectividad, **no** reducción física de almacenamiento. Tampoco demuestra ahorro de FLOPs, RAM, latencia o energía. La inferencia batch 1 de R-002 fue más lenta para dANN-R en este entorno; no hay ventaja física inferida ni benchmark comparable al Keras original.

**Interpretación:** se reprodujeron la arquitectura de referencia, el protocolo operativo con diferencias de framework declaradas, el rango de rendimiento, la dirección de menor test loss medio de dANN-R, su menor accuracy y la fuerte reducción de conexiones efectivas. La inversión de loss en seed 4 y la menor magnitud de la ventaja media quedan registradas. No se evaluó DSU ni se demostró superioridad general frente a MLP, memoria física menor, aceleración o eficiencia energética. La baja DE de accuracy vANN en N=3 provino de 8943/8943/8944 aciertos guardados sin redondeo prematuro; N=5 elevó su DE a 0.180 pp. El gate Sprint 1 es **PASS**: paper y pareja fijados, reproducción implementada, comparación formal y divergencias documentadas. Siguiente tarea: **S2-T01 — Cerrar la especificación matemática y computacional de DSU-S antes de implementarla**.
