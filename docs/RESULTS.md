# RESULTS

Existen resultados diagnósticos de Sprint 0 y una reproducción N=3 de Sprint 1.

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

**Status:** COMPLETE, **gate S1-T02 PASS**. **Fecha:** 2026-10-05. Comando: `.venv/Scripts/python.exe -m dsu_research.run --config experiments/configs/reproduction_001_fmnist.json --output runs/reproduction_001 --seeds 1 2 3`. Artefactos locales ignorados por Git: seis JSON por modelo/seed y `runs/reproduction_001/summary.json`. Suite: 10/10 tests. Los seis JSON contienen train loss, validation loss y validation accuracy de las 25 épocas, además de train accuracy, test loss/accuracy, duración, throughput, latencia, seeds, hashes, entorno, parámetros y estado final evaluado.

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

**Diferencias y límites:** Glorot uniform y biases cero explícitos aproximan los defaults de Keras, pero PyTorch y TensorFlow no generan los mismos pesos con la misma seed. Adam epsilon se fija a 1e-7, aunque su aritmética puede diferir. PyTorch usa cross-entropy sobre logits; Keras softmax y pérdida sobre probabilidades. El shuffle reproducible de DataLoader no tiene el mismo orden que `tf.data.Dataset.shuffle` del original. El reloj de train incluye validation y el throughput divide ejemplos de train por ese tiempo. Estas diferencias son metodológicas conocidas pero no invalidan la arquitectura ni el sentido de la comparación. No se observan errores de arquitectura, máscara, split ni test. Próxima tarea: **S1-T03, comparar formalmente con el baseline**, sin implementar DSU-S ni avanzar a Sprint 2.
