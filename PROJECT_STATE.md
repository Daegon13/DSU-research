# PROJECT_STATE

## CURRENT PHASE
Sprint 3 — Topology and Causal Ablations

## CURRENT SPRINT
S3

## CURRENT TASK
S3-T01 — Diseñar el control Fixed-dANN de fan-in 16 y la matriz causal de Sprint 3. Pendiente; no iniciada. Es sólo diseño/especificación; no implementar ni entrenar todavía.

## LAST COMPLETED TASK
S2-REVIEW: PASS. Sprint 2: PASS. Decisión formal: GO TO SPRINT 3 para ablaciones causales, sin ejecutar S3-T01. H-001 y H-008 permanecen SUPPORTED sólo dentro de los alcances evaluados. Gate REPRODUCTION_001 = REPRODUCED y Gate Sprint 1 = PASS permanecen vigentes.

## LAST RESULT
R-008: revisión acumulada de Sprint 2. DSU-S aprendió de forma estable, mantuvo calidad próxima a dANN-R de igual anchura con 44.09× menos parámetros entrenables almacenados y mostró señal de capacidad por parámetro frente a MLP densos evaluados. Límite central: 42536 bytes de pesos + 65536 bytes de topología int64 = 108072 bytes de payload tensorial conocido, frente a 66064 bytes del MLP quality-matched `20→26`. No se demostró ventaja de almacenamiento total, RAM runtime, latencia o energía. Detalle en `docs/SPRINT_2_REVIEW.md`.

## CURRENT BEST MODEL
En S2-T06 se seleccionó por validation el MLP `784→20→26→10` dentro de la familia prefijada; test se evaluó sólo después de congelarlo. Todos los modelos se evaluaron en época 25; no se eligió un checkpoint por mínimo de validation ni se declaró un mejor modelo global.

## CURRENT BASELINE
Sprint 0: MLP/ReLU 784→128→10 sobre MNIST diagnóstico, sin cambios. Sprint 1: vANN LeakyReLU 784→512→128→10 sobre Fashion-MNIST, 468874 parámetros; la pareja vANN/dANN-R tiene igual anchura oculta, no igual presupuesto efectivo. S2-T05: MLP denso LeakyReLU 784→13→17→10, 10623 parámetros y 10583 MAC principales, aproximadamente igualado a DSU-S por parámetros. S2-T06: MLP quality-matched en validation `784→20→26→10`, 16516 parámetros y 16460 MAC principales. Hardware CPU AMD Ryzen 5 5600G, Windows 11, float32, PyTorch 2.14.1+cpu.

## IMPORTANT NUMBERS
- Dedicación objetivo: 2–4 h/semana.
- Primer Go/No-Go: 6–8 semanas parciales.
- Objetivo aspiracional posterior para DSU-FFN:
  - >= 4x menos parámetros del bloque sustituido;
  - < 2% de degradación de calidad;
  - >= 1.25–1.5x de mejora de throughput o reducción equivalente de latencia.

## ACTIVE HYPOTHESIS
H-001 = SUPPORTED para Fashion-MNIST N=3 contra dANN-R de igual anchura. H-008 = SUPPORTED sólo dentro de la familia MLP densa y protocolo S2-T05/T06; el menor full match entre candidatos evaluados usó 1.553× parámetros entrenables y DSU-S mantuvo ventaja descriptiva de test. H-002 sigue UNTESTED para utilidad de memoria/runtime y cómputo físico; H-006 mantiene abierta la latencia real.

## OPEN QUESTIONS
- ¿Cuánto difiere la reproducción PyTorch de la semántica Keras de inicialización, máscara y shuffle?
- ¿Cuánto de la señal observada proviene del fan-in fijo frente a la organización dendrita→soma, inicialización y representación compacta?
- ¿Cuánto cuesta `G` en checkpoint y RAM runtime y qué cambiarían regeneración, índices compactos o topología implícita?
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
- R-007 cubre sólo una familia MLP, tres seeds y tolerancias heurísticas; no identifica el mejor MLP posible. El match de validation no eliminó la desventaja de test del seleccionado. Los 65536 bytes de topología de DSU-S hacen que su payload tensorial total supere los 66064 bytes de pesos del MLP seleccionado; no se infiere ventaja de almacenamiento total ni velocidad.
- S2-REVIEW confirma que la comparación DSU-S/dANN-R confunde fan-in variable vs fijo, topología, inicialización y representación. Sprint 3 debe diseñar controles antes de atribuir causalidad; el GO no autoriza modelos nuevos por sí mismo.

## BLOCKERS
Ninguno técnico.
El proyecto debe mantenerse como prioridad lateral frente a Facultad, empleo y proyectos principales.

## NEXT TASK
S3-T01 — Diseñar el control Fixed-dANN de fan-in 16 y la matriz causal de Sprint 3. No iniciada; tarea sólo de especificación, sin implementación ni entrenamiento.

## LAST UPDATE
2026-10-06
