# PROJECT_STATE

## CURRENT PHASE
Sprint 1 — Reproducción científica

## CURRENT SPRINT
S1

## CURRENT TASK
S1-T01 — Seleccionar y fijar el paper/experimento dendrítico principal a reproducir.

## LAST COMPLETED TASK
S0-T01 a S0-T04 y auditoría Gate S0: PASS. Instalación aislada con `uv sync --locked --extra test`, 6/6 tests, baseline reproducible y harness reutilizable con otro modelo y otra forma de entrada.

## LAST RESULT
Gate S0 PASS. MNIST CPU, seed 42, 1 época, 1024 train / 256 test: train loss 1.940615, accuracy 0.566406; test loss 1.614402, accuracy 0.691406; 101770 parámetros entrenables y totales. Segunda ejecución con seed 42: métricas de calidad idénticas; seed 43: test loss 1.616367, accuracy 0.679688. Tres épocas con seed 42: train loss 1.940615 → 1.168307 → 0.728188; test accuracy 0.8125. Detalle en `docs/RESULTS.md`; JSON locales en `runs/`.

## CURRENT BEST MODEL
Ninguno. Todavía no hay implementación.

## CURRENT BASELINE
MLP/ReLU convencional 784→128→10; Adam, lr 0.001; MNIST sin normalización adicional. Hardware de la última ejecución válida: AMD Ryzen 5 5600G (6 núcleos, 12 hilos), 33.69 GB RAM física, Windows 11, CPU float32, PyTorch 2.14.1+cpu. Última ejecución válida: `runs/s0_gate_convergence.json`, 2026-10-05T20:27:02Z, seed 42, 3 épocas, 1024 train / 256 test.

## IMPORTANT NUMBERS
- Dedicación objetivo: 2–4 h/semana.
- Primer Go/No-Go: 6–8 semanas parciales.
- Objetivo aspiracional posterior para DSU-FFN:
  - >= 4x menos parámetros del bloque sustituido;
  - < 2% de degradación de calidad;
  - >= 1.25–1.5x de mejora de throughput o reducción equivalente de latencia.

## ACTIVE HYPOTHESIS
H-001:
Una unidad con agregación dendrítica estructurada puede alcanzar calidad comparable a una capa densa usando menos parámetros.

## OPEN QUESTIONS
- ¿Qué dataset mínimo usaremos para reproducir primero una arquitectura dendrítica publicada?
- ¿Qué topología de grupos dendríticos será más eficiente en PyTorch?
- ¿La ventaja paramétrica se convierte en ventaja real de memoria y latencia?
- ¿El estado temporal por dendrita aporta capacidad suficiente para justificar su coste?

## KNOWN PROBLEMS
- RAM pico de CPU no medida; la interfaz deja el campo en null.
- Los experimentos de cierre usan subconjuntos pequeños y no demuestran calidad final ni comparabilidad estadística.
- La latencia CPU mostró variación entre procesos; no usarla como comparación de rendimiento entre arquitecturas.

## BLOCKERS
Ninguno técnico.
El proyecto debe mantenerse como prioridad lateral frente a Facultad, empleo y proyectos principales.

## NEXT TASK
S1-T01 — Seleccionar y fijar el paper/experimento dendrítico principal a reproducir.

## LAST UPDATE
2026-10-05
