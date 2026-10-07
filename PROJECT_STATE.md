# PROJECT_STATE

## CURRENT PHASE
Sprint 2 — DSU-S

## CURRENT SPRINT
S2

## CURRENT TASK
S2-REVIEW — Auditar los resultados completos de Sprint 2 y decidir si existe evidencia suficiente para avanzar a Sprint 3. Pendiente; no iniciada. Sprint 2 sigue abierto.

## LAST COMPLETED TASK
S2-T06: PASS técnico. Búsqueda MLP de dos capas ocultas con train+validation únicamente, escalera prefijada, sanity check y refinamiento completos. `784→20→26→10` seleccionado y congelado antes de abrir test. H-008 = SUPPORTED dentro de esa familia y protocolo. Gate REPRODUCTION_001 = REPRODUCED y Gate Sprint 1 = PASS permanecen vigentes.

## LAST RESULT
R-007, N=3: target DSU-S validation accuracy 88.488889 %, loss 0.320460462. MLP `20→26` fue el menor full match de validation: 88.167 ± 0.765 %, loss 0.336997 ± 0.012175, 16516 parámetros (1.553× DSU-S) y 16460 MAC principales (1.649×). Test final del seleccionado: 86.700 ± 0.191 %, loss 0.375776 ± 0.003194; DSU-S: 87.253 ± 0.277 %, loss 0.357825 ± 0.002485. Deltas pareados MLP−DSU-S: −0.553 ± 0.463 pp y +0.017952 ± 0.005227 loss. Detalle en `docs/EXPERIMENT_003_MLP_CAPACITY_SEARCH.md` y JSON locales `runs/mlp_capacity_s2_t06/`.

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
H-001 = SUPPORTED para Fashion-MNIST N=3 contra dANN-R de igual anchura. H-008 = SUPPORTED sólo dentro de la familia MLP densa y protocolo S2-T05/T06; el full match de validation necesitó 1.553× parámetros entrenables y DSU-S mantuvo ventaja descriptiva de test. H-002 sigue UNTESTED para eficiencia física/latencia de ejecución; H-006 mantiene abierta la latencia real.

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
- R-007 cubre sólo una familia MLP, tres seeds y tolerancias heurísticas; no identifica el mejor MLP posible. El match de validation no eliminó la desventaja de test del seleccionado. Los 65536 bytes de topología de DSU-S hacen que su payload tensorial total supere los 66064 bytes de pesos del MLP seleccionado; no se infiere ventaja de almacenamiento total ni velocidad.

## BLOCKERS
Ninguno técnico.
El proyecto debe mantenerse como prioridad lateral frente a Facultad, empleo y proyectos principales.

## NEXT TASK
S2-REVIEW — Auditar los resultados completos de Sprint 2 y decidir si existe evidencia suficiente para avanzar a Sprint 3. No iniciada; requiere tarea propia.

## LAST UPDATE
2026-10-06
