# PROJECT_STATE

## CURRENT PHASE
Sprint 3 CLOSED. Track A permanece BACKLOG / DEPRIORITIZED. PointSparse IID `128×64` sigue como baseline histórico. Track B: B1-E01 y B1-E02 completados; H-B1-STATE SUPPORTED sólo en EASY y H-B1-INTERFERENCE SUPPORTED sólo en NOISE `.1`. B2 y DSU-T no iniciados.

## CURRENT SPRINT
Ningún sprint experimental activo; S3 = PASS / CLOSED. B1-E02 cerró el contraste preregistrado de interferencia de Line B. B2-DESIGN es la siguiente tarea recomendada, aún no iniciada.

## CURRENT TASK
B1-E02 completado: PASS técnico y H-B1-INTERFERENCE = SUPPORTED en `delayed_recall_noise_v1` bajo amplitud `.1`. Informe en `docs/EXPERIMENT_B1_E02_STATEFUL_INTERFERENCE.md`.

## LAST COMPLETED TASK
B1-E02: 24/24 corridas pareadas de 20 épocas bajo ruido interior IID `.1`; S0 50% y S1 100% en las tres seeds y cuatro delays. En Δ=16, +50.00 ± 0.00 pp y ΔNLL −.691904 ± .000386; criterio preregistrado SUPPORTED. La accuracy de S1 sigue en techo, con NLL peor que EASY. Oracle no activado. Sprint 3 sigue CLOSED y Track A despriorizado.

## LAST RESULT
R-020: B1-E02: H-B1-INTERFERENCE = SUPPORTED sólo para `delayed_recall_noise_v1` `.1`. En Δ=4/8/16/32, S0 50% y S1 100% de test accuracy en las tres seeds; NLL S1 `.000387/.000641/.001309/.004783` de media, siempre peor que EASY. 24 corridas, 480 corridas-época, 1674.08 s acumulados de train+validation, resume 24/24. `docs/EXPERIMENT_B1_E02_STATEFUL_INTERFERENCE.md`, `data/b1_e02_results.json`.

R-019: B1-E01: en Δ=4/8/16/32, S0 50% y S1 100% de test accuracy para las tres seeds; retención analítica .8409/.7071/.5/.25. En Δ=16, ΔNLL S1−S0 = −.692323 ± .000190 y H-B1-STATE = SUPPORTED según preregistro. 24 corridas, 480 corridas-época y 1281.48 s acumulados de train+validation. `docs/EXPERIMENT_B1_E01_STATEFUL_POINTSPARSE.md`, `data/b1_e01_results.json`.

R-018: A1-E01: A0/A1/A2/A3 test accuracy `86.913 ± .125` / `86.940 ± .144` / `86.977 ± .047` / `86.873 ± .199` %, NLL `.372309 ± .001408` / `.364639 ± .002804` / `.368247 ± .001863` / `.365156 ± .004636`. C1 = INCONCLUSIVE; C2/C3/C4 = NEGLIGIBLE; ningún efecto material. Nueve corridas nuevas, 225 epoch-equivalents. `docs/EXPERIMENT_A1_E01_POINTSPARSE_TOPOLOGY.md`, `data/a1_e01_results.json`.

R-017: revisión de S3-T01–T11; gate S3-REVIEW PASS y Sprint 3 CLOSED. Los resultados causales y prospectivos mantienen sus estados previos. `docs/SPRINT_3_REVIEW.md`.

R-016: PointSparse 80×109/160×49/128×64 obtuvo validation accuracy época 25 `88.178 ± .331%` / `88.267 ± .109%` / `88.172 ± .416%`; test accuracy `86.680 ± .105%` / `86.803 ± .115%` / `86.913 ± .125%`. 160×49 tuvo menor NLL test `.366624 ± .002695`; el anchor mayor test accuracy. El piloto T10 de época 3 no preservó ranking final (Spearman −.50 en la shortlist); early-pilot histórico es WEAK. T10+T11 ejecutaron 174 epoch-equivalents incrementales frente a 600 ingenuos, reutilizando 75 épocas históricas del anchor. `docs/EXPERIMENT_011_POINTSPARSE_SHORTLIST_CONFIRMATION.md`, `data/s3_t11_confirmation_results.json`.

R-015: mejor validation accuracy de época 3: PointSparse 80×109, 84.817%; mejor validation loss: 240×29, .432157. ER raw vs accuracy: Spearman −.190, Kendall −.214, 5/15 pares no near-tie; width H: −.048, .000, 6/15. Los ocho pilotos sumaron 24 epoch-equivalents, frente a 600 de búsqueda ingenua completa. `docs/EXPERIMENT_010_PROSPECTIVE_ZERO_COST_SCREEN.md` y `data/s3_t10_pilot_results.json`.

R-013: PointSparse obtuvo test accuracy 86.913 ± .125% y NLL .372309 ± .001408. `1×64 − PointSparse` dio −.273 pp/+ .003573 NLL; `4×16 − 1×64` dio −.397 pp/+ .011383 NLL, con loss adversa en 3/3 seeds. La tendencia de menor NLL con B de S3-T05 no reapareció. Detalle en `docs/EXPERIMENT_008_SPARSE_VS_COMPARTMENT_CAUSAL.md`.

S3-T08 no generó resultados de aprendizaje. Formalizó Línea A (sparse architecture) y Línea B (dynamic neuron), y reprodujo contabilidad conocida PointSparse 9610/9472, DSU-S 10634/9984/8192 índices, Fixed-dANN 468874 almacenados/10634 efectivos/468224 MAC GEMM y vANN 468874. Registro: `data/historical_experiments.json`; especificación: `docs/ANALYTICAL_SCREENING_FRAMEWORK.md`.

R-014: en S3-T07, effective rank y participation ratio ordenaron 9/9 pares no near-tie de accuracy/NLL por medias; en S3-T05 original, effective rank 2/3 y participation ratio 1/3 pares de accuracy restantes. El contraste PointSparse/MLP13 histórico se invierte al normalizar rank por ancho. Pattern logdet falló en ambos grupos. Proxies: `data/zero_cost_proxy_results.json`; análisis: `data/zero_cost_validation.json`; detalles: `docs/EXPERIMENT_009_ZERO_COST_FORWARD_PROXIES.md`.

## CURRENT BEST MODEL
En S2-T06 se seleccionó por validation el MLP `784→20→26→10` dentro de la familia prefijada; test se evaluó sólo después de congelarlo. Todos los modelos se evaluaron en época 25; no se eligió un checkpoint por mínimo de validation ni se declaró un mejor modelo global.

## CURRENT BASELINE
Para investigación futura A/B: **PointSparse `128×64`**, baseline experimental por defecto dentro del presupuesto probado: 9610 parámetros, 9472 MAC principales/muestra, N=3 a 25 épocas. No es óptimo global. Los controles históricos siguientes mantienen sus roles originales y no se sustituyen en sus experimentos.

Sprint 0: MLP/ReLU 784→128→10 sobre MNIST diagnóstico, sin cambios. Sprint 1: vANN LeakyReLU 784→512→128→10 sobre Fashion-MNIST, 468874 parámetros; la pareja vANN/dANN-R tiene igual anchura oculta, no igual presupuesto efectivo. S2-T05: MLP denso LeakyReLU 784→13→17→10, 10623 parámetros y 10583 MAC principales, aproximadamente igualado a DSU-S por parámetros. S2-T06: MLP quality-matched en validation `784→20→26→10`, 16516 parámetros y 16460 MAC principales. Hardware CPU AMD Ryzen 5 5600G, Windows 11, float32, PyTorch 2.14.1+cpu.

## IMPORTANT NUMBERS
- Dedicación objetivo: 2–4 h/semana.
- Primer Go/No-Go: 6–8 semanas parciales.
- Objetivo aspiracional posterior para DSU-FFN:
  - >= 4x menos parámetros del bloque sustituido;
  - < 2% de degradación de calidad;
  - >= 1.25–1.5x de mejora de throughput o reducción equivalente de latencia.

## ACTIVE HYPOTHESIS
La hipótesis de trabajo futura, aún no demostrada causalmente, es que el ancho sparse y la conectividad restringida explican parte de la eficiencia paramétrica observada. H-A1-TOPOLOGY = NEGLIGIBLE: a H/K y grados fijos, las reorganizaciones A2/A3 no cambiaron materialmente la calidad bajo A1-E01. C1 balance IID→regular queda INCONCLUSIVE por NLL/accuracy discordantes. H-WIDTH-FANIN sigue INCONCLUSIVE; no se atribuye el resultado a width. H-B1-STATE = SUPPORTED únicamente para EASY; H-B1-INTERFERENCE = SUPPORTED únicamente para NOISE `.1`. Ambas tareas quedaron en techo de accuracy; no se identificó límite de memoria ni eficiencia física. Los estados de S3 están consolidados en `docs/SPRINT_3_REVIEW.md`.

H-001 = SUPPORTED para Fashion-MNIST N=3 contra dANN-R de igual anchura. H-008 = SUPPORTED sólo dentro de la familia MLP densa y protocolo S2-T05/T06; el menor full match entre candidatos evaluados usó 1.553× parámetros entrenables y DSU-S mantuvo ventaja descriptiva de test. H-002 sigue UNTESTED para utilidad de memoria/runtime y cómputo físico; H-006 mantiene abierta la latencia real.
H-009 = NOT SUPPORTED / NEGLIGIBLE dentro de Fashion-MNIST N=5 y el protocolo dense-mask pareado. H-010 = SUPPORTED en el alcance estructural y funcional CPU float32 de S3-T02. H-011 (partición dendrítica) = NEGLIGIBLE en S3-T05 N=3. H-012 (storage B/C) = SUPPORTED; H-013 (runtime B/C) = MIXED / NOT SUPPORTED como ventaja general.
H-014 (H-SPARSE) = SUPPORTED frente al MLP histórico dentro del alcance acotado: +1.030 pp, 0.752 del gap histórico retenido. H-015 (H-COMPARTMENT-PURE) = CONTRARY por deterioro consistente de NLL de 4×16 respecto de 1×64. H-016 (H-DENDRITIC-STAGE) = INCONCLUSIVE; 1×64 pierde frente a PointSparse sin alcanzar umbral contrario.
H-ZEROCOST-ER = FAILED en la cohorte y piloto prospectivos S3-T10, sin borrar el resultado retrospectivo S3-T09. H-WIDTH-FANIN = INCONCLUSIVE para calidad final de la shortlist T11; la señal de T10 quedó preliminar. EARLY-PILOT-3E = WEAK como predictor entre familias históricas, con fracaso para ordenar la shortlist prospectiva cercana.

## OPEN QUESTIONS
- ¿Cuánto difiere la reproducción PyTorch de la semántica Keras de inicialización, máscara y shuffle?
- ¿Qué contribución causal separable tienen ancho, topología, solapamiento y fan-in restringido frente al MLP probado?
- ¿Cuánto cuesta `G` en checkpoint y RAM runtime y qué cambiarían regeneración, índices compactos o topología implícita?
- ¿Puede una implementación sparse obtener ventaja de memoria residente o latencia bajo batches declarados?
- ¿Generaliza la señal fuera de Fashion-MNIST y de la familia MLP estudiada?
- ¿El estado temporal por unidad sparse aporta capacidad suficiente para justificar su coste, antes de añadir compartimentos dinámicos?

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
- S3-T11 reutiliza resultados S3-T07 tras auditar equivalencia, pero T07 no congeló número de hilos CPU. Las curvas T10/T07 del anchor coinciden en accuracy y orden de minibatches; las pérdidas difieren levemente por aritmética. La shortlist sólo tiene tres brazos y todas sus medias de validation final son near-ties. No declarar predictor validado por correlaciones de tres puntos, ni ventaja global desde NLL, test accuracy o coste analítico aislados. El cálculo de ahorro incremental depende del trabajo histórico de 75 épocas del anchor.
- A1-E01 reutiliza A0 S3-T07 con threads históricos no fijados frente a un thread en corridas nuevas. El contraste C1 mezcló régimen de grados y aristas concretas: +.027 pp de accuracy media pero −.007670 de NLL, por lo que es INCONCLUSIVE, no evidencia causal pura de balance. C2/C3/C4 NEGLIGIBLE sólo acotan las topologías y tres seeds probadas; no prueban equivalencia universal de G ni resuelven width. No se midieron RAM, latencia ni energía.
- B1-E01 usa una tarea EASY con pasos de delay cero y último input idéntico para ambas clases. S0 tiene 50% por construcción, S1 alcanza 100% para todos los delays probados; ese techo impide inferir el horizonte límite. No se probó interferencia, otras tareas, LSTM/GRU, RAM pico, latencia de inferencia ni energía. Los 1281.48 s son duración acumulada de train+validation, no benchmark de inferencia.
- B1-E02 añade sólo ruido IID interior de amplitud `.1`, incluso en canales 0/1, y conserva el 100% de accuracy S1; no identifica límite de memoria ni umbral de interferencia. La NLL sube frente a EASY en 12/12 parejas. No se probaron distractores estructurados, modelos recurrentes estándar, tareas naturales, RAM pico, latencia de inferencia ni energía. Los 1674.08 s son train+validation acumulados, no benchmark de inferencia.

## BLOCKERS
Ninguno técnico.
El proyecto debe mantenerse como prioridad lateral frente a Facultad, empleo y proyectos principales.

## NEXT TASK
B2-DESIGN: preregistrar, en tarea separada y sin entrenamiento, un contraste entre escala temporal fija y escalas heterogéneas fijas con hipótesis que pueda revelar el tradeoff entre memoria corta y larga. Requiere gate propio; B1-E02 no autoriza implementar B2, DSU-T ni reabrir Track A.

## LAST UPDATE
2026-10-13
