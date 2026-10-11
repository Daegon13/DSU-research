# PROJECT_STATE

## CURRENT PHASE
Sprint 3 — Topology and Causal Ablations

## CURRENT SPRINT
S3

## CURRENT TASK
S3-T10 completada con PASS técnico y resultado prospectivo negativo para ER raw. Sprint 3 sigue abierto. S3-T11 no se inició; DSU-T tampoco.

## LAST COMPLETED TASK
S3-T10: PASS técnico. Nueva cohorte PointSparse iso-presupuesto de ocho H/K con 9610 parámetros, proxy forward-only en seeds 1–3 y ranking ER raw sellados en commit `5516417` antes de entrenar. Piloto train/validation de 3 épocas y seed 1 por brazo; test cerrado. ER raw falló prospectivamente frente a accuracy de época 3 y no superó ancho H. H-ZEROCOST-ER = FAILED dentro del piloto; H-WIDTH-FANIN = PRELIMINARY_SIGNAL. Shortlist futura: 80×109, 160×49 y anchor 128×64. No se ejecutaron 25 épocas ni N=3 completo. S3-T09 permanece como señal retrospectiva acotada; H-014/015/016 no cambian.

## LAST RESULT
R-015: mejor validation accuracy de época 3: PointSparse 80×109, 84.817%; mejor validation loss: 240×29, .432157. ER raw vs accuracy: Spearman −.190, Kendall −.214, 5/15 pares no near-tie; width H: −.048, .000, 6/15. Los ocho pilotos sumaron 24 epoch-equivalents, frente a 600 de búsqueda ingenua completa. `docs/EXPERIMENT_010_PROSPECTIVE_ZERO_COST_SCREEN.md` y `data/s3_t10_pilot_results.json`.

R-013: PointSparse obtuvo test accuracy 86.913 ± .125% y NLL .372309 ± .001408. `1×64 − PointSparse` dio −.273 pp/+ .003573 NLL; `4×16 − 1×64` dio −.397 pp/+ .011383 NLL, con loss adversa en 3/3 seeds. La tendencia de menor NLL con B de S3-T05 no reapareció. Detalle en `docs/EXPERIMENT_008_SPARSE_VS_COMPARTMENT_CAUSAL.md`.

S3-T08 no generó resultados de aprendizaje. Formalizó Línea A (sparse architecture) y Línea B (dynamic neuron), y reprodujo contabilidad conocida PointSparse 9610/9472, DSU-S 10634/9984/8192 índices, Fixed-dANN 468874 almacenados/10634 efectivos/468224 MAC GEMM y vANN 468874. Registro: `data/historical_experiments.json`; especificación: `docs/ANALYTICAL_SCREENING_FRAMEWORK.md`.

R-014: en S3-T07, effective rank y participation ratio ordenaron 9/9 pares no near-tie de accuracy/NLL por medias; en S3-T05 original, effective rank 2/3 y participation ratio 1/3 pares de accuracy restantes. El contraste PointSparse/MLP13 histórico se invierte al normalizar rank por ancho. Pattern logdet falló en ambos grupos. Proxies: `data/zero_cost_proxy_results.json`; análisis: `data/zero_cost_validation.json`; detalles: `docs/EXPERIMENT_009_ZERO_COST_FORWARD_PROXIES.md`.

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
H-014 (H-SPARSE) = SUPPORTED frente al MLP histórico dentro del alcance acotado: +1.030 pp, 0.752 del gap histórico retenido. H-015 (H-COMPARTMENT-PURE) = CONTRARY por deterioro consistente de NLL de 4×16 respecto de 1×64. H-016 (H-DENDRITIC-STAGE) = INCONCLUSIVE; 1×64 pierde frente a PointSparse sin alcanzar umbral contrario.
H-ZEROCOST-ER = FAILED en la cohorte y piloto prospectivos S3-T10, sin borrar el resultado retrospectivo S3-T09. H-WIDTH-FANIN = PRELIMINARY_SIGNAL sólo en 3 épocas/1 seed; no hay conclusión de capacidad final.

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
- S3-T06 iguala parámetros y media/DE del agregado en un lote train sin labels, pero no iguala distribución completa, correlaciones, logits ni número de operaciones de activación. Las constantes se congelan antes de entrenar; no hay resultado de calidad.
- S3-T07 N=3 muestra que PointSparse retiene gran parte del gap de accuracy histórico, pero MLP/DSU-S históricos no están completamente pareados con él. La familia neutral aumenta activaciones y operaciones fijas con B; NLL y top-1 empeoran al aumentar B, mientras ECE mejora. No se infiere causa única, menor FLOPs totales ni ventaja de runtime.
- S3-T09 usa targets de test históricos sólo después de sellar proxies, para auditoría retrospectiva. S3-T05 tiene varios near-ties; MLP y PointSparse pertenecen a protocolos distintos. La señal de effective rank no es prueba prospectiva ni causal y no justifica optimizar un score con las mismas filas.
- S3-T10 muestra correlación negativa de ER raw con validation accuracy temprana; 13/28 pares son near-ties. El piloto de una seed puede reflejar velocidad de aprendizaje, no calidad final; MAC y bytes de topología varían ligeramente entre H/K pese a parámetros iguales. No extrapolar a test, N=3 completo, latencia, memoria runtime ni energía.

## BLOCKERS
Ninguno técnico.
El proyecto debe mantenerse como prioridad lateral frente a Facultad, empleo y proyectos principales.

## NEXT TASK
S3-T11 — Confirmar la shortlist seleccionada por piloto (80×109, 160×49, anchor 128×64) y diseñar mejores proxies tras el fallo prospectivo de ER raw. Congelar el protocolo antes de entrenar; no reabrir test durante selección. Sprint 3 sigue abierto; no iniciar Línea B automáticamente.

## LAST UPDATE
2026-10-11
