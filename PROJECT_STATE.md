# PROJECT_STATE

## CURRENT PHASE
Sprint 1 — Reproducción científica

## CURRENT SPRINT
S1

## CURRENT TASK
S1-T02 — Implementar la reproducción mínima vANN vs dANN-R en Fashion-MNIST.

## LAST COMPLETED TASK
S1-T01: PASS. Paper Chavlis y Poirazi (2025) y pareja fija Fashion-MNIST vANN/dANN-R (D=4, S=128) documentados en `docs/REPRODUCTION_001.md`. Gate S0 previo: PASS.

## LAST RESULT
S1-T01 fijó para la reproducción 25 épocas según código/datos oficiales (el paper dice 20): en `DATA.zip`, N=5, dANN-R D=4/S=128 obtuvo test loss 0.365028 ± 0.003595 y accuracy 86.830 ± 0.184 %; vANN de igual anchura, loss 0.401243 ± 0.020207 y accuracy 89.112 ± 0.621 %. Son datos originales, no resultados propios. Gate S0 y sus mediciones permanecen en `docs/RESULTS.md`.

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
- ¿Cuánto difiere la reproducción PyTorch compacta de la semántica Keras de inicialización, máscara y shuffle?
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
S1-T02 — Implementar la reproducción mínima vANN vs dANN-R en Fashion-MNIST, según `docs/REPRODUCTION_001.md`.

## LAST UPDATE
2026-10-05
