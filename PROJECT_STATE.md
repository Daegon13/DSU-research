# PROJECT_STATE

## CURRENT PHASE
Sprint 2 — DSU-S

## CURRENT SPRINT
S2

## CURRENT TASK
S2-T04 — Ejecutar el primer benchmark de aprendizaje de DSU-S v0 contra vANN y dANN-R. Pendiente; no ejecutado.

## LAST COMPLETED TASK
S2-T02 y S2-T03: PASS. DSU-S v0 implementada con pesos compactos y topología fija; tests estructurales, forward/backward, serialización e integración sintética con el harness verificados. No se entrenó en Fashion-MNIST. Gate REPRODUCTION_001 = REPRODUCED y Gate Sprint 1 = PASS permanecen vigentes.

## LAST RESULT
R-003, N=5 propio: dANN-R test loss 0.358606 ± 0.003586, accuracy 87.216 ± 0.187 %; vANN loss 0.374481 ± 0.014621, accuracy 89.322 ± 0.180 %. Deltas pareados dANN-R − vANN: loss −0.015874 ± 0.014078, accuracy −2.106 ± 0.211 puntos porcentuales. Seed 4 invierte la dirección de loss individual. dANN-R tiene 10634 parámetros efectivos frente a 468874 de vANN, pero ambos almacenan 468874. JSON locales y resumen N=5 en `runs/reproduction_001/`; detalles y referencia oficial en `docs/RESULTS.md`.

## CURRENT BEST MODEL
No se seleccionó un modelo por validation ni test. La pareja vANN/dANN-R fija se evaluó en época 25. DSU-S v0 está implementada y verificada estructuralmente; su calidad en Fashion-MNIST no fue evaluada.

## CURRENT BASELINE
Sprint 0: MLP/ReLU 784→128→10 sobre MNIST diagnóstico, sin cambios. Para reproducción Sprint 1: vANN LeakyReLU 784→512→128→10 sobre Fashion-MNIST, 468874 parámetros. Hardware de R-002: AMD Ryzen 5 5600G, Windows 11, CPU float32, PyTorch 2.14.1+cpu. La pareja tiene la misma anchura oculta, no igual presupuesto de conexiones efectivas.

## IMPORTANT NUMBERS
- Dedicación objetivo: 2–4 h/semana.
- Primer Go/No-Go: 6–8 semanas parciales.
- Objetivo aspiracional posterior para DSU-FFN:
  - >= 4x menos parámetros del bloque sustituido;
  - < 2% de degradación de calidad;
  - >= 1.25–1.5x de mejora de throughput o reducción equivalente de latencia.

## ACTIVE HYPOTHESIS
H-001:
DSU-S v0 compacta podría mantener calidad cercana a dANN-R usando menos parámetros físicamente almacenados. H-002 pregunta si el fan-in fijo puede convertir la sparsity en ahorro físico útil; H-006 mantiene abierta la latencia real.

## OPEN QUESTIONS
- ¿Cuánto difiere la reproducción PyTorch de la semántica Keras de inicialización, máscara y shuffle?
- ¿Qué rendimiento tendrá la topología fija de 16 entradas por dendrita y cuánto costará su indexación en PyTorch?
- ¿La ventaja paramétrica se convierte en ventaja real de memoria y latencia?
- ¿El estado temporal por dendrita aporta capacidad suficiente para justificar su coste?

## KNOWN PROBLEMS
- RAM pico de CPU no medida; la interfaz deja el campo en null.
- Los experimentos de cierre usan subconjuntos pequeños y no demuestran calidad final ni comparabilidad estadística.
- La latencia CPU mostró variación entre procesos; no usarla como comparación de rendimiento entre arquitecturas.
- R-003 N=5 difiere numéricamente del archivo Keras N=5; inicialización, shuffle y detalles de Adam no son bit a bit equivalentes. La pérdida media mantiene la dirección esperada, con inversión individual en seed 4. La pequeña DE inicial de accuracy vANN no provino de redondeo prematuro; N=5 presenta mayor dispersión.
- El tiempo de entrenamiento de R-002/R-003 incluye validación por época; `samples_per_second` divide muestras train por ese tiempo combinado. No se guardan pesos de checkpoints, sólo métricas y selección `final_epoch_25`.
- DSU-S v0 se probó en CPU y con datos sintéticos. CUDA se omite limpiamente si no está disponible; aún no hay evidencia de rendimiento de aprendizaje, latencia comparable ni RAM pico para DSU-S.

## BLOCKERS
Ninguno técnico.
El proyecto debe mantenerse como prioridad lateral frente a Facultad, empleo y proyectos principales.

## NEXT TASK
S2-T04 — Ejecutar el primer benchmark de aprendizaje de DSU-S v0 contra vANN y dANN-R. No iniciado.

## LAST UPDATE
2026-10-06
