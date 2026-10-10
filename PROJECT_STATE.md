# PROJECT_STATE

## CURRENT PHASE
Sprint 3 — Topology and Causal Ablations

## CURRENT SPRINT
S3

## CURRENT TASK
S3-T05 completada. Siguiente diseño pendiente: S3-T06 — diagnosticar la ausencia de ventaja material de compartimentalización, fijando primero un control para parámetros extra y escala inicial. No iniciado.

## LAST COMPLETED TASK
S3-T05: PASS técnico. Doce corridas Fashion-MNIST N=3 con cuatro particiones nested de 64 conexiones input por soma y hashes pareados. H-011 = NEGLIGIBLE dentro del protocolo. Sprint 2: PASS; Gate REPRODUCTION_001 = REPRODUCED y Gate Sprint 1 = PASS permanecen vigentes.

## LAST RESULT
R-012: `4×16 − 1×64` con las mismas 64 entradas/soma produjo +0.023 ± 0.138 pp de test accuracy y −0.005051 ± 0.001043 de test loss en N=3. Sólo una seed favoreció 4×16 en accuracy. H-011 = NEGLIGIBLE; la diferencia de loss es pequeña y los parámetros difieren. Detalle en `docs/EXPERIMENT_007_DENDRITIC_COMPARTMENT_ABLATION.md`.

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
H-009 = NOT SUPPORTED / NEGLIGIBLE dentro de Fashion-MNIST N=5 y el protocolo dense-mask pareado. H-010 = SUPPORTED en el alcance estructural y funcional CPU float32 de S3-T02. H-011 (partición dendrítica) = NEGLIGIBLE en S3-T05 N=3. H-012 (storage B/C) = SUPPORTED; H-013 (runtime B/C) = MIXED / NOT SUPPORTED como ventaja general.

## OPEN QUESTIONS
- ¿Cuánto difiere la reproducción PyTorch de la semántica Keras de inicialización, máscara y shuffle?
- ¿Cuánto de la señal observada proviene del fan-in fijo frente a la organización dendrita→soma, inicialización y representación compacta?
- ¿Cuánto cuesta `G` en checkpoint y RAM runtime y qué cambiarían regeneración, índices compactos o topología implícita?
- ¿La ventaja paramétrica se convierte en ventaja real de memoria y latencia?
- ¿El estado temporal por dendrita aporta capacidad suficiente para justificar su coste?

## KNOWN PROBLEMS
- RAM pico de CPU no medida fiablemente; el benchmark S3-T04 registra explícitamente esta limitación.
- Los experimentos de cierre usan subconjuntos pequeños y no demuestran calidad final ni comparabilidad estadística.
- La latencia CPU mostró variación entre procesos; no usarla como comparación de rendimiento entre arquitecturas.
- R-003 N=5 difiere numéricamente del archivo Keras N=5; inicialización, shuffle y detalles de Adam no son bit a bit equivalentes. La pérdida media mantiene la dirección esperada, con inversión individual en seed 4. La pequeña DE inicial de accuracy vANN no provino de redondeo prematuro; N=5 presenta mayor dispersión.
- El tiempo de entrenamiento de R-002/R-003 incluye validación por época; `samples_per_second` divide muestras train por ese tiempo combinado. No se guardan pesos de checkpoints, sólo métricas y selección `final_epoch_25`.
- DSU-S v0 se entrenó en CPU con Fashion-MNIST en tres seeds; CUDA se omite limpiamente si no está disponible. No hay RAM pico CPU, comparación controlada de latencia ni evidencia de ventaja energética.
- El `state_dict` medido corresponde a una instancia recién inicializada con la misma estructura, no a los pesos finales: el harness no conserva pesos entrenados. N=3 y fan-in exacto frente a promedio impiden atribuir una diferencia de calidad sólo al almacenamiento compacto.
- R-006 compara sólo un MLP denso fijado y N=3; la ventaja de DSU-S no aísla causalmente la estructura dendrítica de inicialización, conectividad y número de activaciones. MAC principales teóricos no son FLOPs exactos ni tiempo medido. RAM de activaciones no medida.
- R-007 cubre sólo una familia MLP, tres seeds y tolerancias heurísticas; no identifica el mejor MLP posible. El match de validation no eliminó la desventaja de test del seleccionado. Los 65536 bytes de topología de DSU-S hacen que su payload tensorial total supere los 66064 bytes de pesos del MLP seleccionado; no se infiere ventaja de almacenamiento total ni velocidad.
- S2-REVIEW confirma que la comparación DSU-S/dANN-R confunde fan-in variable vs fijo, topología, inicialización y representación. Sprint 3 debe diseñar controles antes de atribuir causalidad; el GO no autoriza modelos nuevos por sí mismo.
- S3-T02 verificó equivalencia paired con datos sintéticos y cinco pasos Adam en CPU float32. S3-T04 añadió checkpoint, estado Adam y microbenchmarks CPU; CUDA, trayectoria larga y RAM pico siguen sin evidencia.
- S3-T03 N=5 encontró efecto medio despreciable de regularizar fan-in/topología a K=16. Las aristas A/B difieren y la dispersión entre seeds supera el delta medio; no generalizar fuera del protocolo.
- S3-T05 N=3 igualó conectividad por soma pero no parámetros totales, número de activaciones ni escala exacta de logits iniciales. `4×16` usó topología sin solapamiento distinta de DSU-S v0. La clasificación NEGLIGIBLE es exploratoria y no demuestra que la compartimentalización sea inútil en general.

## BLOCKERS
Ninguno técnico.
El proyecto debe mantenerse como prioridad lateral frente a Facultad, empleo y proyectos principales.

## NEXT TASK
S3-T06 — Diseñar, antes de ejecutarlo, un diagnóstico de por qué la compartimentalización no produjo ventaja material de accuracy en S3-T05. Priorizar control de parámetros extra y escala inicial; no extender seeds automáticamente.

## LAST UPDATE
2026-10-09
