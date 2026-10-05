# RESULTS

Existe un resultado diagnóstico del baseline de Sprint 0. No es una comparación arquitectónica.

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
