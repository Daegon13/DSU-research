# EXPERIMENT_008 — Sparse width y compartimentos pareados (S3-T07)

## Protocolo congelado antes de observar entrenamiento

TASK ID: S3-T07. OBJECTIVE: contrastar H-014/H-015/H-016 con cinco brazos y tres seeds. WHY: S3-T05 mezcló parámetros y escala. CONTEXT: S3-T06 cerró controles y calibración forward. ALLOWED FILES: runner y tests experimentales, config, este informe, HYPOTHESES, RESULTS, diseño Sprint 3 y PROJECT_STATE. FORBIDDEN FILES: baselines, DSUSv0, arquitecturas S3-T06, splits y resultados históricos. REQUIREMENTS: 15 corridas reanudables con hashes pareados. OUT OF SCOPE: tuning, kernels, seeds 4–5, DSU-T. ACCEPTANCE CRITERIA: 15 JSON válidos, clasificación prefijada y auditoría. TESTS: suite, diff check y estado Git. EXPECTED OUTPUT: R-013 y artefactos. EXPECTED COMMIT: `experiment: isolate sparse width and dendritic compartment effects`. STOP CONDITIONS: fallo de pareo, fuga de labels/test a calibración o bug metodológico.

Fashion-MNIST, `54000/6000/10000`, `/255`, Adam `lr=.001`, betas `(.9,.999)`, epsilon `1e-7`, batch 128, CrossEntropy de logits, 25 épocas, CPU float32, `final_epoch_25`, seeds 1–3. Una sola G64, valores base input, salida, sesgos, split y orden de minibatches por seed. PointSparse y `1×64`, `2×32`, `4×16`, `8×8` tienen cada uno 9610 parámetros entrenables y 9472 MAC principales. Los B usan `sum(d)/sqrt(B)` y gain/offset de S3-T06 calibrados exactamente con los primeros 256 índices train del split, sin labels y antes del optimizador; no se recalibran. ECE exploratoria: 15 bins uniformes de confianza máxima, intervalos `[0,1/15]`, `(1/15,2/15]`, etc.; error absoluto entre accuracy y confianza medias, ponderado por fracción de ejemplos. Test sólo en época 25. Duración incluye train y validation, no mide latencia.

Se guardan por época train loss, validation loss y accuracy. En test: NLL, accuracy, media de norma L2 de logits, entropía predictiva, margen correcto `logit_y − max(logit_otros)`, fracción de márgenes positivos y ECE. Accuracy se almacena como proporción. Se publican mean ± DE muestral N=3 y deltas pareados en puntos porcentuales. Umbrales usan valores sin redondear. `loss no peor materialmente` se operacionaliza como delta medio ≤+0.005; deterioro fuerte consistente como delta loss ≥+0.01 y tres seeds adversas.

H-014 PointSparse vs MLP histórico 85.883% y DSU-S 87.253%: SUPPORTED si mejora MLP ≥0.75 pp y queda ≤0.50 pp debajo de DSU-S; WEAK SUPPORT si mejora MLP entre 0.30 y 0.75 pp o queda 0.50–1.00 pp debajo de DSU-S, siempre que no cumpla NOT SUPPORTED; NOT SUPPORTED si mejora MLP <0.30 pp; otros patrones INCONCLUSIVE. La fracción del gap histórico retenida es descriptiva, no causal exacta. Si loss y accuracy se contradicen materialmente, clasificar INCONCLUSIVE y declarar el conflicto.

H-016, `1×64 − PointSparse`: PRELIMINARY SUPPORT con accuracy ≥+0.50 pp, ≥2/3 victorias y delta loss ≤+0.005; WEAK SUPPORT con +0.15 a <+0.50 pp, ≥2 victorias y loss ≤+0.005; NEGLIGIBLE con |accuracy|<0.15 pp y |loss|<0.005; CONTRARY si accuracy ≤−0.50 pp o deterioro fuerte consistente; resto INCONCLUSIVE.

H-015, `4×16 − 1×64`: PRELIMINARY SUPPORT con accuracy ≥+0.50 pp, ≥2/3 victorias y loss ≤+0.005 (se reporta si fueron 3/3); WEAK SUPPORT con +0.15 a <+0.50 pp, ≥2 victorias y loss ≤+0.005; NEGLIGIBLE con |accuracy|<0.15 pp y |loss|<0.005; CONTRARY si accuracy ≤−0.50 pp o deterioro fuerte consistente; resto INCONCLUSIVE. Comparaciones B adyacentes exploratorias. N=3 no permite conclusión general.

## Ejecución y auditoría

Comando: `.venv/Scripts/python.exe -m dsu_research.sparse_compartment_experiment --config experiments/configs/sparse_compartment_fmnist_s3_t07.json --output runs/sparse_compartment_s3_t07`. Los 15 JSON completos y `summary.json` se guardan en `runs/sparse_compartment_s3_t07/` (ignorado por Git); cada corrida se escribe atómicamente al terminar. Una segunda invocación validó y reanudó los 15 archivos sin reentrenar. El runner vuelve a calcular la calibración y exige igualdad exacta con `runs/s3_t06_initial_diagnostics.json`, también regenerable mediante el comando de S3-T06. No se usaron labels para calibrar ni se usó validation/test antes de fijar coeficientes.

Dentro de cada seed coincidieron exactamente G64, valores base input, salida, sesgos, índices train/validation, los 256 índices de calibración y los 25 hashes de orden de minibatches. Prefijos SHA-256 de 12 caracteres para auditoría rápida (los hashes completos están en cada JSON):

| Seed | G64 | input base | output | train split | validation split | calibración | 25 minibatches |
|---:|---|---|---|---|---|---|---|
| 1 | fd7f2469a457 | e794167cbf9a | ab4f9c2d6a81 | f3838eee6ae9 | d363e9447f47 | 6e2834ad32df | 85b5a2cda9d1 |
| 2 | 982abdee19b1 | fc27b4493908 | ea55cee50c8e | acf787f92794 | 56fbaf61830f | 6eba57d8b916 | b71f47453629 |
| 3 | 67056be9e35b | 85cbfe595782 | c9ad0628d01a | ad16d1e1628f | 22a1e84521fc | fb8fc4e41fb6 | 63d93539e7bd |

Los cinco brazos conservaron 9610 parámetros, 8192 pesos input y 9472 MAC principales por muestra. El número de sumas fijas de ramas sí crece con B; tampoco están igualados activaciones, gather, FLOPs totales, latencia, RAM o energía. No se detectó bug metodológico durante entrenamiento. Suite final: 51 passed, 2 skipped; `git diff --check` limpio.

## Resultados por seed

Accuracy en porcentaje; NLL es cross-entropy de test. Norma, entropía y margen son medias por ejemplo. ECE usa 15 bins y es exploratoria. La fracción de márgenes positivos coincide con accuracy porque el margen es positivo exactamente cuando la clase correcta es argmax, salvo empates; los 15 artefactos guardan ambos valores.

| Seed | Brazo | Train loss | Val loss | Val acc. | Test NLL | Test acc. | Norma L2 | Entropía | Margen | ECE | Duración s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | PointSparse | .300638 | .322322 | 88.533 | .371451 | 86.91 | 27.307 | .3273 | 4.231 | .0133 | 165.7 |
| 1 | 1×64 | .288634 | .319045 | 88.833 | .373305 | 86.64 | 30.200 | .3039 | 4.709 | .0205 | 172.7 |
| 1 | 2×32 | .309352 | .338959 | 88.317 | .382386 | 86.48 | 30.866 | .3289 | 4.305 | .0164 | 165.6 |
| 1 | 4×16 | .327906 | .345870 | 88.033 | .389755 | 86.15 | 30.754 | .3560 | 4.095 | .0092 | 166.9 |
| 1 | 8×8 | .346942 | .359727 | 87.167 | .397288 | 85.81 | 30.985 | .3793 | 3.769 | .0085 | 177.8 |
| 2 | PointSparse | .301687 | .338940 | 87.717 | .371541 | 87.04 | 27.768 | .3357 | 4.125 | .0095 | 157.0 |
| 2 | 1×64 | .295305 | .345138 | 87.750 | .378173 | 86.75 | 33.126 | .3225 | 4.454 | .0123 | 157.8 |
| 2 | 2×32 | .309574 | .352926 | 87.250 | .386104 | 86.38 | 30.624 | .3391 | 4.243 | .0109 | 164.7 |
| 2 | 4×16 | .323925 | .355518 | 87.450 | .391633 | 86.07 | 30.509 | .3570 | 4.016 | .0096 | 167.0 |
| 2 | 8×8 | .344177 | .367712 | 87.150 | .402768 | 85.79 | 28.905 | .3819 | 3.731 | .0097 | 177.9 |
| 3 | PointSparse | .303372 | .329999 | 88.267 | .373933 | 86.79 | 27.541 | .3347 | 4.119 | .0126 | 161.7 |
| 3 | 1×64 | .291475 | .331820 | 88.283 | .376165 | 86.53 | 31.683 | .3155 | 4.638 | .0177 | 168.0 |
| 3 | 2×32 | .302660 | .333457 | 88.150 | .376932 | 86.86 | 29.908 | .3284 | 4.299 | .0126 | 172.9 |
| 3 | 4×16 | .318577 | .343822 | 87.967 | .380405 | 86.51 | 29.061 | .3481 | 4.129 | .0112 | 174.9 |
| 3 | 8×8 | .342871 | .359854 | 87.450 | .397852 | 85.80 | 28.592 | .3759 | 3.778 | .0073 | 184.9 |

## Agregados y contrastes

Media ± DE muestral, N=3. Accuracy y validation accuracy en porcentaje; ECE es fracción.

| Brazo | Test NLL | Test acc. % | Val loss | Val acc. % | Norma L2 | Entropía | Margen | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PointSparse | .372309 ± .001408 | 86.913 ± .125 | .330420 ± .008317 | 88.172 ± .416 | 27.539 ± .230 | .33254 ± .00458 | 4.158 ± .063 | .01181 ± .00202 |
| 1×64 | .375881 ± .002446 | 86.640 ± .110 | .332001 ± .013048 | 88.289 ± .542 | 31.669 ± 1.463 | .31396 ± .00942 | 4.600 ± .132 | .01682 ± .00414 |
| 2×32 | .381807 ± .004613 | 86.573 ± .253 | .341780 ± .010037 | 87.906 ± .574 | 30.466 ± .498 | .33211 ± .00604 | 4.282 ± .034 | .01330 ± .00284 |
| 4×16 | .387264 ± .006014 | 86.243 ± .234 | .348403 ± .006246 | 87.817 ± .319 | 30.108 ± .915 | .35370 ± .00487 | 4.080 ± .058 | .01001 ± .00105 |
| 8×8 | .399303 ± .003014 | 85.800 ± .010 | .362431 ± .004574 | 87.256 ± .169 | 29.494 ± 1.301 | .37906 ± .00301 | 3.759 ± .025 | .00850 ± .00118 |

Los deltas siguientes son `alto − bajo`, con accuracy en pp. Los tres valores entre paréntesis son las seeds 1/2/3. Una NLL positiva perjudica al brazo alto.

| Contraste | Δ accuracy media; por seed | Δ NLL media; por seed | Δ norma | Δ entropía | Δ margen | Δ ECE |
|---|---:|---:|---:|---:|---:|---:|
| 1×64 − PointSparse | −.273 (−.27, −.29, −.26) | +.003573 (+.001854, +.006632, +.002232) | +4.131 | −.01858 | +.442 | +.00501 |
| 2×32 − 1×64 | −.067 (−.16, −.37, +.33) | +.005926 (+.009080, +.007930, +.000767) | −1.204 | +.01816 | −.318 | −.00352 |
| 4×16 − 2×32 | −.330 (−.33, −.31, −.35) | +.005457 (+.007369, +.005529, +.003473) | −.358 | +.02159 | −.202 | −.00330 |
| 8×8 − 4×16 | −.443 (−.34, −.28, −.71) | +.012038 (+.007533, +.011135, +.017447) | −.614 | +.02536 | −.321 | −.00151 |
| 4×16 − 1×64 | −.397 (−.49, −.68, −.02) | +.011383 (+.016450, +.013460, +.004239) | −1.561 | +.03975 | −.520 | −.00682 |

**H-014 = SUPPORTED en el alcance histórico acotado.** PointSparse mejora al MLP `784→13→17→10` histórico (10623 parámetros, NLL `.398187 ± .007987`, accuracy `85.883 ± .287%`) por **+1.030 pp** y reduce NLL medio en `.025878`. Queda **−.340 pp** y `+.014484` NLL frente a DSU-S histórica (`87.253 ± .277%`, `.357825 ± .002485`, 10634 parámetros). Retiene descriptivamente **0.752** del gap de accuracy histórico de 1.370 pp. El MLP, DSU-S y PointSparse no son una comparación causal pareada completa: arquitectura, topología, inicialización y presupuesto difieren; no atribuir toda la diferencia a sparse width.

**H-016 = INCONCLUSIVE.** `1×64 − PointSparse` pierde `.273 pp` en las tres seeds y aumenta NLL `.003573`; no alcanza el umbral contrario de `.50 pp` ni el criterio de deterioro fuerte consistente. Su norma de logits y margen correcto medio suben, pero accuracy baja y ECE empeora; un margen medio mayor no implica más ejemplos bien clasificados. La etapa adicional y la calibración fija afectan la función, sobre todo para valores negativos.

**H-015 = CONTRARY bajo el criterio de NLL deteriorada consistentemente.** `4×16 − 1×64` pierde `.397 pp`; NLL aumenta `.011383` y es positiva en las tres seeds (`+.016450`, `+.013460`, `+.004239`). El umbral adverso de accuracy de `.50 pp` no se alcanza en la media; esta clasificación usa el criterio alternativo predefinido de deterioro fuerte y consistente de loss. ECE baja `.00682`, pero NLL y top-1 empeoran; ECE por bins de confianza máxima no mide toda la distribución predictiva. No hay evidencia de beneficio de compartimentos puros en este protocolo.

## Curvas y diagnóstico de confianza

Validation media por época, `loss / accuracy %`, manteniendo la época 25 como selección final:

| Época | PointSparse | 1×64 | 2×32 | 4×16 | 8×8 |
|---:|---|---|---|---|---|
| 1 | .5406 / 81.689 | .5277 / 82.194 | .5584 / 81.306 | .6035 / 79.900 | .6754 / 77.744 |
| 5 | .4026 / 85.989 | .3952 / 86.289 | .4089 / 85.733 | .4247 / 85.483 | .4450 / 84.806 |
| 10 | .3710 / 87.106 | .3651 / 87.356 | .3798 / 86.994 | .3930 / 86.561 | .4093 / 85.933 |
| 15 | .3526 / 87.511 | .3470 / 87.733 | .3610 / 87.428 | .3736 / 86.956 | .3893 / 86.322 |
| 20 | .3419 / 87.733 | .3385 / 87.878 | .3517 / 87.650 | .3613 / 87.267 | .3757 / 86.883 |
| 25 | .3304 / 88.172 | .3320 / 88.289 | .3418 / 87.906 | .3484 / 87.817 | .3624 / 87.256 |

`1×64` empieza ligeramente mejor en validation que PointSparse, pero su ventaja de loss se cierra hacia época 25 y en test no se sostiene. A mayor B, train loss final, validation loss y test NLL tienden a subir; la brecha visible desde época 1 no converge al final. A diferencia de S3-T05, aquí **no reaparece la mejora de NLL con B**. La caída de margen medio y accuracy y el aumento de entropía acompañan el empeoramiento de NLL; la norma L2 desciende suavemente desde B=1 y ECE mejora, lo que impide reducir el fenómeno a una sola medida de escala o calibración. El cambio de dirección respecto de S3-T05 es compatible con confusión previa por parámetros/escala, pero N=3 y las restantes diferencias de activación/operaciones no permiten probar esa causa.

## Gate, límites y siguiente decisión

**Gate técnico S3-T07: PASS.** Las 15 corridas terminaron y el resume reutilizó las 15; hashes y calibración coincidieron, los conteos permanecieron iguales y las métricas adicionales se calcularon con una definición única. No se detectó bug metodológico conocido ni hubo invalidaciones/repeticiones. Las conclusiones son descriptivas para Fashion-MNIST, CPU float32 y N=3. Los contrastes B modifican número y disposición de LeakyReLU y operaciones fijas; igualar parámetros/MAC principales y media/DE inicial no iguala función, gradientes ni coste físico. No se hicieron seeds 4–5, tuning, kernels ni DSU-T.

Recomendación científica: priorizar la hipótesis de **sparse topology / sparse width** como explicación de la ventaja paramétrica histórica y no ampliar por ahora la familia dendrítica estática. Si se necesita confirmar una señal antes de cerrar Sprint 3, una réplica N=5 de PointSparse frente al control de interés sería una tarea nueva y previamente especificada; S3-T07 no la inicia.
