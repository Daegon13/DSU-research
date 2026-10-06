# PROJECT_STATE

## CURRENT PHASE
Sprint 2 — DSU-S

## CURRENT SPRINT
S2

## CURRENT TASK
S2-T05 — Comparar DSU-S v0 contra un MLP aproximadamente igualado por parámetros. Pendiente; no ejecutado.

## LAST COMPLETED TASK
S2-T04: PASS técnico, señal científica FUERTE. DSU-S v0 completó tres trials Fashion-MNIST de 25 épocas con arquitectura congelada; H-001 = SUPPORTED dentro de esta comparación inicial. Gate REPRODUCTION_001 = REPRODUCED y Gate Sprint 1 = PASS permanecen vigentes.

## LAST RESULT
R-005, N=3 DSU-S: test loss 0.357825 ± 0.002485, accuracy 87.253 ± 0.277 %. Deltas pareados DSU-S − dANN-R: loss −0.000574 ± 0.002440, accuracy +0.017 ± 0.117 pp; frente a vANN: loss −0.016001 ± 0.003214, accuracy −2.180 ± 0.282 pp. Seed 2 empeora loss frente a dANN-R (+0.002073). Tres corridas estables; train+validation total 556.30 s. DSU-S almacena 10634 parámetros frente a 468874 de ambos baselines dense-mask. JSON locales y detalle en `docs/EXPERIMENT_001_DSU_S_LEARNING.md` y `docs/RESULTS.md`.

## CURRENT BEST MODEL
No se seleccionó un modelo por validation ni test. La pareja vANN/dANN-R fija y DSU-S v0 se evaluaron en época 25. La calidad DSU-S se midió en tres seeds; no se eligió un checkpoint por validation.

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
H-001 = SUPPORTED para Fashion-MNIST N=3 contra dANN-R de igual anchura: calidad media cercana y 10634 parámetros almacenados. H-002 sigue UNTESTED para eficiencia física/latencia de ejecución; H-006 mantiene abierta la latencia real.

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
- DSU-S v0 se entrenó en CPU con Fashion-MNIST en tres seeds; CUDA se omite limpiamente si no está disponible. No hay RAM pico CPU, comparación controlada de latencia ni evidencia de ventaja energética.
- El `state_dict` medido corresponde a una instancia recién inicializada con la misma estructura, no a los pesos finales: el harness no conserva pesos entrenados. N=3 y fan-in exacto frente a promedio impiden atribuir una diferencia de calidad sólo al almacenamiento compacto.

## BLOCKERS
Ninguno técnico.
El proyecto debe mantenerse como prioridad lateral frente a Facultad, empleo y proyectos principales.

## NEXT TASK
S2-T05 — Comparar DSU-S v0 contra un MLP aproximadamente igualado por parámetros. No iniciado; requiere tarea propia.

## LAST UPDATE
2026-10-06
