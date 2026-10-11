# S3-REVIEW — Revisión y cierre de Sprint 3

**Fecha:** 2026-10-11. **Gate S3-REVIEW: PASS. Sprint 3: CLOSED.** Revisión documental de S3-T01–T11 y del [registro histórico](../data/historical_experiments.json). No se generaron entrenamientos, proxies ni benchmarks nuevos. Los estados de hipótesis conservan los criterios y nombres de sus experimentos; `PASS` evalúa la ejecución del sprint, no convierte resultados científicos negativos en positivos.

| Campo de tarea | Alcance de S3-REVIEW |
|---|---|
| TASK ID / OBJECTIVE / WHY | S3-REVIEW; cerrar Topology and Causal Ablations; decidir qué explica la señal y qué queda abierto sin prolongar el sprint por backlog futuro. |
| CONTEXT | S3-T01–T11 completadas; R-009–R-016 y preregistros existentes. |
| ALLOWED FILES | Este review, estado, resultados, hipótesis, roadmap, decisiones, programa A/B, mapa bioinspirado, framework, protocolo y recovery prompt. |
| FORBIDDEN FILES / OUT OF SCOPE | Código, tests, configs, `data/`, `runs/`, entrenamientos, proxies, benchmarks, arquitecturas nuevas y comienzo de Línea A/B. |
| REQUIREMENTS | Auditar fuentes, conservar estados y límites, separar calidad/costes, fijar baseline y cierre, registrar handoff. |
| ACCEPTANCE CRITERIA | Ledger, descomposición causal, claims, screening, líneas A/B y roadmap coherentes; decisión explícita de Sprint 3. |
| TESTS | Suite completa existente, enlaces internos, `git diff --check` y `git status`; sin tests nuevos para una revisión documental. |
| EXPECTED OUTPUT / EXPECTED COMMIT | Review y documentos de estado; `research: close Sprint 3 causal ablations`. |
| STOP CONDITIONS | Evidencia contradictoria no resuelta, hipótesis sin estado o modificación accidental de artefactos experimentales. |

## 1. EXECUTIVE CONCLUSION

DSU-S v0 representa una función sparse equivalente a Fixed-dANN con mucho menos payload tensorial y estado Adam en el par probado. La compactación no crea una ventaja de accuracy: B/C comparten función y pesos activos. Regularizar el fan-in a K=16 tuvo efecto despreciable en A/B; la compartimentalización estática no mostró una mejora causal de calidad bajo los controles estudiados. PointSparse, sin compartimentos, retuvo descriptivamente gran parte de la ventaja paramétrica histórica frente al MLP pequeño. Tres distribuciones H/K de igual presupuesto terminaron casi empatadas en validation y con métricas finales discordantes. Effective rank raw falló en la cohorte prospectiva T10; el piloto de tres épocas es un filtro exploratorio débil. Todo esto se limita a las implementaciones, tareas, seeds y presupuestos documentados.

## 2. HYPOTHESIS LEDGER

| ID y pregunta | Estado final | Evidencia principal / experimento | Alcance y qué no demuestra |
|---|---|---|---|
| H-009: ¿aporta regularizar fan-in a 16? | **NOT SUPPORTED / NEGLIGIBLE** | Fixed-dANN − dANN-R: −.042 ± .329 pp accuracy, +.001066 ± .007721 NLL; [S3-T03](EXPERIMENT_005_FIXED_FANIN_ABLATION.md). | Fashion-MNIST N=5, dense-mask pareado; cambian también aristas y distribución topológica. No prueba que K nunca importe. |
| H-010: ¿equivale la representación compacta? | **SUPPORTED** | Máximos 1.19e−7 forward, 3.73e−9 gradientes, 3.35e−8 tras cinco pasos Adam; [S3-T02](EXPERIMENT_004_FIXED_DANN_EQUIVALENCE.md). | Pareja con `G` y pesos activos idénticos, CPU float32. No prueba trayectorias largas, GPU ni mejora de calidad. |
| H-011: ¿mejora dividir 64 entradas en compartimentos? | **NEGLIGIBLE** | `4×16 − 1×64`: +.023 ± .138 pp accuracy y −.005051 ± .001043 NLL; [S3-T05](EXPERIMENT_007_DENDRITIC_COMPARTMENT_ABLATION.md). | N=3, cobertura y minibatches pareados, pero parámetros, activaciones y escala no iguales. No aísla sólo la no linealidad ni invalida toda dendrita. |
| H-012: ¿reduce storage B/C? | **SUPPORTED** | Fixed/DSU: 22.281× payload modelo, 21.737× checkpoint modelo, 30.752× checkpoint modelo+Adam; [S3-T04](EXPERIMENT_006_PHYSICAL_REPRESENTATION_BENCHMARK.md). | Sólo dense-mask Fixed-dANN frente a DSU-S con `G` int64; no implica menos RAM pico ni menos bytes que el MLP quality-matched. |
| H-013: ¿reduce latencia B/C? | **MIXED / NOT SUPPORTED** como ventaja general | DSU-S gana forward batch 1; Fixed gana forward batch 128/512; Adam batch 128 similar; [S3-T04](EXPERIMENT_006_PHYSICAL_REPRESENTATION_BENCHMARK.md). | CPU/PyTorch y modos medidos. No prueba aceleración general, GPU ni energía. |
| H-014 (H-SPARSE): ¿conserva PointSparse la señal paramétrica? | **SUPPORTED** en contraste histórico acotado | 9610 parámetros, 86.913 ± .125% test; +1.030 pp frente al MLP 13→17 y 0.752 del gap DSU-S−MLP retenido; [S3-T07](EXPERIMENT_008_SPARSE_VS_COMPARTMENT_CAUSAL.md). | MLP/DSU-S históricos no están totalmente pareados con PointSparse. No atribuye la ganancia a un mecanismo único. |
| H-015 (H-COMPARTMENT-PURE): ¿ayudan ramas sin parámetros extra? | **CONTRARY** | `4×16 − 1×64`: −.397 pp accuracy, +.011383 NLL; NLL adversa 3/3 seeds; [S3-T07](EXPERIMENT_008_SPARSE_VS_COMPARTMENT_CAUSAL.md). | Igual G64, 9610 parámetros, 9472 MAC principales y media/DE inicial calibradas; difieren activaciones, operaciones fijas, función y gradientes. |
| H-016 (H-DENDRITIC-STAGE): ¿ayuda una etapa con una rama? | **INCONCLUSIVE** | `1×64 − PointSparse`: −.273 pp accuracy, +.003573 NLL; [S3-T07](EXPERIMENT_008_SPARSE_VS_COMPARTMENT_CAUSAL.md). | N=3; dirección adversa sin alcanzar umbral contrario. No demuestra ausencia universal de utilidad. |
| H-ZEROCOST-ER: ¿ordena ER raw mejor que ancho? | **FAILED** en cohorte prospectiva T10 | ER raw: Spearman −.190 y 5/15 pares no near-tie; ancho: −.048 y 6/15; [S3-T10](EXPERIMENT_010_PROSPECTIVE_ZERO_COST_SCREEN.md). | Ocho PointSparse iso-presupuesto, target validation época 3/una seed. No borra la señal retrospectiva [T09](EXPERIMENT_009_ZERO_COST_FORWARD_PROXIES.md) ni prueba que todo proxy falle. |
| H-WIDTH-FANIN: ¿hay reparto H/K materialmente mejor? | **INCONCLUSIVE** para calidad final | Medias validation `80×109/160×49/128×64`: 88.178/88.267/88.172%, todas near-ties <.20 pp; test favorece anchor; [S3-T11](EXPERIMENT_011_POINTSPARSE_SHORTLIST_CONFIRMATION.md). | Sólo tres candidatos, 9610 parámetros, N=3; T10 tuvo señal temprana. No establece óptimo global ni equivalencia de MAC/topología. |
| EARLY-PILOT-3E: ¿predice la época 3 la calidad final? | **WEAK** | Favorable en S3-T07, parcial en MLP S2-T06, adverso en S3-T05; Spearman −.50 en shortlist T10→T11; [S3-T11](EXPERIMENT_011_POINTSPARSE_SHORTLIST_CONFIRMATION.md). | Grupos pequeños y shortlist cercana; no es oracle ni regla universal de descarte. |

H-001/H-008 siguen acotadas al [cierre de Sprint 2](SPRINT_2_REVIEW.md); H-002/H-006 mantienen abiertas RAM/runtime y eficiencia física general. H-003–H-007 son backlog, no hipótesis resueltas por Sprint 3.

## 3. CAUSAL DECOMPOSITION

| Transición / contraste | Factor examinado y control | Confusores que permanecen |
|---|---|---|
| dANN-R → Fixed-dANN (A/B) | Misma familia dense-mask, 8192 aristas input, inicialización densa, splits y minibatches pareados; fan-in variable frente a 16 exacto. | Cambian las aristas concretas y la distribución topológica con el fan-in; no es efecto puro del número 16. |
| Fixed-dANN → DSU-S (B/C) | Misma `G`, pesos activos, biases y función; dense-mask frente a almacenamiento compacto. | Diferencias numéricas CPU, kernels y temporales; no es una comparación de capacidad/accuracy. El B de A/B tiene otra escala inicial y no se mezclan sus cohortes. |
| DSU-S → ablation `1×64/…/8×8` | Misma cobertura de 64 entradas por soma y partición nested para estudiar B. | El `4×16` no es la G histórica de DSU-S; cambian parámetros, activaciones, agregación y escala. No es un par causal directo DSU-S/PointSparse. |
| Ablación original → controles neutrales S3-T06/T07 | G64 y valores base compartidos; 9610 parámetros y 9472 MAC principales por brazo; calibración sin labels de media/DE inicial. | Persisten número/disposición de no linealidades, operaciones fijas, distribución/correlaciones, logits y gradientes. La mejora pequeña de NLL de T05 no reapareció; no se identificó cuál confusor la causó. |
| PointSparse → `1×64` → `4×16` neutral | Primero etapa dendrita→soma con una rama; luego partición bajo presupuesto neutral. | La primera diferencia incluye composición de activaciones/calibración; la segunda no iguala operaciones totales. PointSparse vs MLP histórico tampoco es un par completamente controlado. |
| PointSparse `128×64` → H/K iso-presupuesto | 9610 parámetros, mismo protocolo final; T10 cribó ocho, T11 comparó tres. | Ancho, fan-in, aristas, inicialización efectiva, MAC principales y bytes de índices cambian juntos. Validation/test discrepan y N=3 no fija un óptimo. |

## 4. WHAT CURRENTLY EXPLAINS THE SIGNAL?

**SUPPORTED EVIDENCE.** DSU-S y PointSparse distribuyen un presupuesto de ~10k parámetros entre muchas unidades de conectividad restringida; PointSparse sin compartimentos obtuvo +1.030 pp de test frente al MLP pequeño histórico, reteniendo descriptivamente 0.752 del gap DSU-S−MLP. B/C demuestra que compactar la *misma función* explica bytes frente a dense-mask, no accuracy. A/B no halló efecto material del fan-in fijo; el control neutral no halló beneficio de multiplicar compartimentos. La explicación de trabajo más parsimoniosa para la señal de calidad por parámetro es la familia de arquitecturas sparse con muchas unidades, **sin atribución causal cerrada** de cuál de sus rasgos ayuda.

**PLAUSIBLE MECHANISMS, UNPROVEN.** Mayor ancho, fan-in restringido, sparsity estructurada, menor interferencia entre features, diversidad de representaciones o regularización podrían contribuir. No hay un control que los separe todos; el contraste PointSparse/MLP es histórico y la búsqueda H/K final quedó inconclusa.

## 5. POINTSPARSE BASELINE DECISION

**Baseline experimental por defecto dentro del presupuesto ensayado: PointSparse `128×64`.** Tiene 9610 parámetros, 9472 MAC principales/muestra y N=3 a 25 épocas con resultados por seed revalidados. Su test accuracy fue 86.913 ± .125% y test NLL .372309 ± .001408. Las alternativas T11 empataron casi en validation y ninguna cumplió el criterio preregistrado de mejora material coherente; `160×49` obtuvo mejor validation loss y test NLL, lo que queda registrado. La elección favorece un control simple y reutilizable, no declara que `128×64` sea globalmente óptimo ni mejor en todas las métricas. [ADR-014](DECISIONS.md).

## 6. STATIC DENDRITE DECISION

**STATIC DENDRITIC COMPARTMENTALIZATION: PAUSED / DEPRIORITIZED.** S3-T05 fue negligible en accuracy y S3-T07 neutral fue contrario para NLL al añadir ramas; H-016 sigue inconclusa. No hay ganancia causal suficiente bajo Fashion-MNIST estático, N=3 y estos controles para invertir más presupuesto en expandir ramas estáticas. Reabrir sólo ante una predicción específica en otra tarea, interacción con estado temporal, input contextual/apical, plasticidad local o argumento matemático nuevo, con control y coste prefijados. Esta decisión no sostiene que las dendritas nunca ayuden.

## 7. ANALYTICAL SCREENING FRAMEWORK REVIEW

| Capa | Estado tras S3 | Uso permitido y límite |
|---|---|---|
| Structural Cost Model | **USEFUL / VALIDATED FOR ACCOUNTING** | Parámetros almacenados/efectivos, conexiones, MAC principales, bytes de pesos/topología/estado y temporales lógicos; conteos conocidos verificados. Conservar. No llamar a temporales RAM pico ni a MAC tiempo/energía. |
| Signal Propagation | **PARTIAL / DIAGNOSTIC** | Fórmulas y diagnósticos iniciales ayudan a detectar escala/degeneración obvia; no hay validación de calidad final. Calibración de media/DE no iguala distribuciones, logits o gradientes. |
| Zero-cost proxies | **NO PRIMARY SELECTOR VALIDATED** | ER raw fue PROMISING retrospectivo en T09, pero FAILED prospectivo en T10; ER normalizado, PR, correlación, near-zero y entropía fueron WEAK; cobertura/Jaccard y escalas UNINFORMATIVE para ranking estudiado; `activation_pattern_logdet` MISLEADING. Uso sólo auxiliar y preregistrado. |
| Cheap pilot | **WEAK predictor; exploratory filter** | Tres épocas pueden mostrar aprendizaje o fallos gruesos con muchos candidatos, pero no preservaron el ranking de la shortlist ni garantizan calidad final. No descartar near-ties como si fueran diferencias sólidas. |

El fingerprint de hardware sigue siendo específico del CPU, batch y modo S3-T04; un predictor de latencia, RAM pico o energía no quedó validado. No se ejecutaron proxies backward.

## 8. NEW EXPERIMENTAL POLICY

El [protocolo](EXPERIMENT_PROTOCOL.md) mantiene etapas separadas y no autoriza experimentos por sí mismo: **Stage 0**, restricciones y costes analíticos; **Stage 1**, diagnósticos forward/zero-cost como filtros auxiliares de degeneración; **Stage 2**, piloto barato cuando hay muchos candidatos, sin asumir que tres épocas preservan rankings; **Stage 3**, shortlist N=3 y protocolo completo con test sólo después de selección congelada; **Stage 4**, N≥5 para efectos relevantes que necesiten confirmación. No promover ningún proxy a criterio primario hasta superar validación prospectiva preregistrada sobre cohorte nueva y target pertinente. Mantener métricas separadas, grupos de protocolo compatibles y resultados negativos.

## 9. LINE A — SPARSE ARCHITECTURE

Objetivo: entender y, sólo con evidencia nueva, mejorar arquitecturas sparse. Baseline: PointSparse `128×64`. Orden recomendado de preguntas, **sin promoverlas todavía a tareas**: (A1) aislar ancho frente a conectividad/fan-in con control causal mejor que el intercambio H/K T11; (A2) comparar topología aleatoria con alternativa prefijada; (A3) medir solapamiento de receptive fields; (A4) estudiar si `G` puede aprenderse o reorganizarse sin confundir presupuesto/tuning; (A5) medir implementación sparse físicamente eficiente, incluyendo RAM pico y latencia por batch; (A6) validar fuera de Fashion-MNIST. Empezar por un único diseño preregistrado si se decide abrir esta línea. No continuar ahora con grid searches H×K extensos: los deltas finales locales de T11 fueron pequeños y discordantes.

## 10. LINE B — DYNAMIC NEURON

Línea independiente para comprobar si el estado temporal aporta capacidad que justifique parámetros, estado residente y coste de ejecución. Primer control futuro: **PointSparse-static vs PointSparse-stateful** en tarea temporal, con secuencias/reset, presupuesto y baseline temporal razonable prefijados. Sólo si aparece señal: PointSparse-stateful vs dendritic-stateful para aislar el efecto de compartimentos. Orden preliminar B1 estado temporal simple, B2 escalas temporales heterogéneas, B3 excitabilidad adaptativa/AIS, B4 compartimentos dendríticos dinámicos, B5 plasticidad local/estructural. DSU-T y variantes bioinspiradas siguen siendo candidatos, no implementaciones autorizadas por este review.

## 11. WHAT WE CAN CLAIM

- En el par funcionalmente equivalente Fixed-dANN/DSU-S medido, DSU-S redujo 22.281× el payload tensorial del modelo y 30.752× el checkpoint de entrenamiento, con topología persistente incluida.
- La compactación preservó forward, gradientes y cinco pasos Adam dentro de tolerancia CPU float32 para los pesos/topología pareados.
- PointSparse mostró mejor calidad por parámetro que el MLP denso pequeño **probado** en Fashion-MNIST, como comparación histórica descriptiva entre protocolos.
- La compartimentalización estática adicional no produjo mejora causal de calidad bajo los controles y criterios de S3-T05/T07; en el control neutral `4×16` deterioró NLL frente a `1×64`.
- ER raw falló en la cohorte prospectiva y target temprano T10; tres épocas fueron un predictor débil en la evaluación T11.

## 12. WHAT WE CANNOT CLAIM

No afirmar que DSU sea universalmente más eficiente que MLP, que sparsity siempre sea mejor, que PointSparse sea globalmente óptimo, que las dendritas nunca ayuden, que DSU consuma menos energía, que sea generalmente más rápida, que ER prediga calidad arquitectónica en general, que un piloto de tres épocas prediga fiablemente la calidad final, que estos resultados generalicen fuera de Fashion-MNIST o que escalen a Transformers/modelos grandes. Tampoco llamar al payload RAM pico ni a MAC principales FLOPs o latencia medidos. DSU-S con `G` int64 tiene 108072 bytes de payload conocido frente a 66064 bytes de pesos del MLP quality-matched `20→26`; no existe claim de menor almacenamiento total frente a ese MLP.

## 13. SPRINT 3 SCORECARD

| Question | Result | Confidence | Next action |
|---|---|---|---|
| Fixed fan-in K=16 | Efecto negligible A/B | Alta dentro de N=5 y control | Cerrar H-009 en este alcance |
| Compact storage | Sí frente a dense-mask equivalente | Alta para bytes/función medidos | Mantener representación; medir otros comparadores por separado |
| Runtime compact | Mixto según batch/modo | Alta para este CPU; baja para generalizar | No prometer speedup |
| Static compartments | Sin beneficio material; control neutral adverso | Moderada dentro de Fashion-MNIST N=3 | Pausar rama |
| Sparse width/PointSparse | Señal paramétrica histórica | Moderada/descriptiva | Línea A con mejor control |
| H/K optimum | No identificado | Baja para óptimo global | No ampliar grid ahora |
| Zero-cost ER | Fallo prospectivo T10 | Alta para esa cohorte/target | No usar solo |
| Cheap pilot | WEAK | Moderada para grupos evaluados | Sólo filtro exploratorio |

## 14. DECISION: CLOSE OR CONTINUE SPRINT 3

**Sprint 3 = PASS / CLOSED.** El objetivo original de separar fan-in/topología, representación y compartimentalización recibió controles, resultados negativos y límites explícitos. El framework también fue probado retrospectiva y prospectivamente. Las preguntas que quedan (mecanismo exacto de PointSparse, mejores topologías, utilidad temporal y eficiencia física total) son investigación de Línea A o B, no condiciones faltantes para cerrar este sprint. El cierre no valida una arquitectura global ni inicia otra fase.

## 15. NEXT PHASE DECISION

Mantener dos tracks: **Research Track A — Sparse Architecture** y **Research Track B — Dynamic Neuron**. La próxima decisión recomendada es una tarea **A1 de sólo diseño**, que preregistre un contraste causal acotado para separar ancho de conectividad/topología usando PointSparse `128×64` y presupuesto explícito; no ejecutarla en este review. El [ROADMAP](ROADMAP.md) conserva **Sprint 4 = DSU-T** como reserva histórica para un eventual gate de Línea B, sujeto a diseño separado. Tareas A1/A2 pueden precederlo o correr independientemente sin llamarse Sprint 4 ni iniciar B automáticamente.

### Fuentes y gate

La trazabilidad numérica principal está en [RESULTS](RESULTS.md) R-009–R-016 y en los informes S3-T02–T11 enlazados arriba. El [registro histórico](../data/historical_experiments.json) contiene 30 filas con `protocol_family` y `paired_comparison_group`; faltantes son `null`, no cero. Esta revisión no modificó datos experimentales.

| Condición del gate | Evidencia de cumplimiento |
|---|---|
| 1. Hipótesis con estado | Ledger H-009–H-016, H-ZEROCOST-ER, H-WIDTH-FANIN y EARLY-PILOT-3E. |
| 2. Claims y cannot-claims | Secciones 11–12, con comparadores y límites. |
| 3. Descomposición causal | Sección 3, con factores aislados y confusores residuales. |
| 4. Baseline PointSparse | Sección 5 y ADR-014. |
| 5. Dendritas estáticas | Sección 6: PAUSED / DEPRIORITIZED. |
| 6. Screening honesto | Sección 7 y ADR-015; fallos prospectivos incluidos. |
| 7. Líneas A/B separadas | Secciones 9–10, distinto objetivo y primer control. |
| 8. Roadmap coherente | Sección 15; S4 sigue reservado para DSU-T. |
| 9. Sin experimentos | Sólo documentación; datos y runners intactos. |
| 10. Decisión final | Sección 14: Sprint 3 PASS / CLOSED. |
