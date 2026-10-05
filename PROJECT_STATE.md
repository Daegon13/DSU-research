# PROJECT_STATE

## CURRENT PHASE
Sprint 1 — Reproducción científica

## CURRENT SPRINT
S1

## CURRENT TASK
S1-T03 — Comparar formalmente la reproducción con el baseline, según ROADMAP.

## LAST COMPLETED TASK
S1-T02: PASS. Reproducción PyTorch Fashion-MNIST vANN/dANN-R (D=4, S=128), 25 épocas × 3 trials; 10 tests pasan. Resultados R-002 en `docs/RESULTS.md`. Sprint 1 continúa con S1-T03 y S1-T04.

## LAST RESULT
R-002, N=3 propio: dANN-R test loss 0.358398 ± 0.004885, accuracy 87.237 ± 0.235 %; vANN loss 0.373826 ± 0.002666, accuracy 89.433 ± 0.006 %. La dirección de menor loss dANN-R se observó en las tres seeds; menor diferencia que en el archivo N=5. Máscara con 8192 conexiones de entrada, 512 de cable y 10634 parámetros efectivos; ambos modelos almacenan 468874. JSON locales en `runs/reproduction_001/`. Referencia archivada y límites en `docs/RESULTS.md`.

## CURRENT BEST MODEL
No se seleccionó un modelo por validation ni test. La pareja vANN/dANN-R fija se evaluó en época 25.

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
- R-002 N=3 difiere numéricamente del archivo Keras N=5; inicialización, shuffle y detalles de Adam no son bit a bit equivalentes. La pérdida mantiene la dirección esperada.
- El tiempo de entrenamiento de R-002 incluye validación por época; `samples_per_second` divide muestras train por ese tiempo combinado. No se guardan pesos de checkpoints, sólo métricas y selección `final_epoch_25`.

## BLOCKERS
Ninguno técnico.
El proyecto debe mantenerse como prioridad lateral frente a Facultad, empleo y proyectos principales.

## NEXT TASK
S1-T03 — Comparar formalmente R-002 con el baseline/referencia fija y separar calidad, conexiones efectivas, almacenamiento y rendimiento físico. Después S1-T04 documentará divergencias; Sprint 1 todavía no está cerrado.

## LAST UPDATE
2026-10-05
