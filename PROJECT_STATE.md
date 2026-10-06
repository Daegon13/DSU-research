# PROJECT_STATE

## CURRENT PHASE
Sprint 2 — DSU-S

## CURRENT SPRINT
S2

## CURRENT TASK
S2-T06 — Determinar qué tamaño necesita un MLP denso para igualar la calidad de DSU-S v0. Pendiente; no ejecutado.

## LAST COMPLETED TASK
S2-T05: PASS técnico, señal científica MODERADA/PROMETEDORA. MLP 784→13→17→10 congelado antes de entrenar; tres seeds Fashion-MNIST de 25 épocas y comparación pareada con DSU-S S2-T04 reutilizada. H-008 = WEAK SUPPORT. Gate REPRODUCTION_001 = REPRODUCED y Gate Sprint 1 = PASS permanecen vigentes.

## LAST RESULT
R-006, N=3 MLP igualado: test loss 0.398187 ± 0.007987, accuracy 85.883 ± 0.287 %; DSU-S reutilizada: 0.357825 ± 0.002485 y 87.253 ± 0.277 %. Deltas pareados MLP − DSU-S: loss +0.040362 ± 0.010470, accuracy −1.370 ± 0.565 pp. Ventaja DSU-S en las tres seeds, con mismo split y estado final de época 25. MLP: 10623 parámetros, 42492 bytes de pesos, 10583 MAC principales/muestra. JSON locales y detalle en `docs/EXPERIMENT_002_PARAMETER_MATCHED_MLP.md` y `docs/RESULTS.md`.

## CURRENT BEST MODEL
No se seleccionó un modelo por validation ni test. La pareja vANN/dANN-R fija, DSU-S v0 y el MLP de S2-T05 se evaluaron en época 25. No se eligió un checkpoint por validation.

## CURRENT BASELINE
Sprint 0: MLP/ReLU 784→128→10 sobre MNIST diagnóstico, sin cambios. Sprint 1: vANN LeakyReLU 784→512→128→10 sobre Fashion-MNIST, 468874 parámetros; la pareja vANN/dANN-R tiene igual anchura oculta, no igual presupuesto efectivo. S2-T05: MLP denso LeakyReLU 784→13→17→10, 10623 parámetros y 10583 MAC principales, aproximadamente igualado a DSU-S por parámetros. Hardware CPU AMD Ryzen 5 5600G, Windows 11, float32, PyTorch 2.14.1+cpu.

## IMPORTANT NUMBERS
- Dedicación objetivo: 2–4 h/semana.
- Primer Go/No-Go: 6–8 semanas parciales.
- Objetivo aspiracional posterior para DSU-FFN:
  - >= 4x menos parámetros del bloque sustituido;
  - < 2% de degradación de calidad;
  - >= 1.25–1.5x de mejora de throughput o reducción equivalente de latencia.

## ACTIVE HYPOTHESIS
H-001 = SUPPORTED para Fashion-MNIST N=3 contra dANN-R de igual anchura. H-008 = WEAK SUPPORT frente al MLP denso aproximadamente igualado por parámetros (ventaja DSU-S de 1.370 pp). H-002 sigue UNTESTED para eficiencia física/latencia de ejecución; H-006 mantiene abierta la latencia real.

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
- R-006 compara sólo un MLP denso fijado y N=3; la ventaja de DSU-S no aísla causalmente la estructura dendrítica de inicialización, conectividad y número de activaciones. MAC principales teóricos no son FLOPs exactos ni tiempo medido. RAM de activaciones no medida.

## BLOCKERS
Ninguno técnico.
El proyecto debe mantenerse como prioridad lateral frente a Facultad, empleo y proyectos principales.

## NEXT TASK
S2-T06 — Determinar qué tamaño necesita un MLP denso para igualar la calidad de DSU-S v0. No iniciada; requiere tarea propia.

## LAST UPDATE
2026-10-06
