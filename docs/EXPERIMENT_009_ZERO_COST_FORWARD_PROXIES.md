# S3-T09 — Forward-only zero-cost proxies

## Protocolo congelado antes de consultar calidad histórica

**Estado de esta sección:** definiciones y análisis fijados antes de ejecutar proxies o abrir los targets del registro. El código de colección no importa el registro histórico. La validación retrospectiva se ejecutará como fase separada después de escribir el JSON de proxies.

- CPU float32, modelos sin entrenar, `eval()` e `inference_mode()`; estadísticas en float64 local. Ningún backward, labels, época ni checkpoint entrenado.
- Seeds de inicialización/topología `1,2,3`. Para cada seed, Fashion-MNIST **train**: `np.random.default_rng(seed).shuffle(arange(60000))`; usar los primeros 256 índices del train de 54000. `ToTensor()` equivale a píxeles float32/255. Se guarda SHA-256 de índices little-endian int64 y de píxeles float32; no se abre validation/test para obtener el lote.
- La representación principal es la última capa oculta previa a logits: segunda hidden en MLP; activación somática en PointSparse, compartimentos y DSU-S; segunda activación en vANN/dANN/Fixed. La preactivación correspondiente se registra. Intermedias pueden auditarse, sin mezclarlas en ranking.
- PointSparse y familia neutral reutilizan exactamente `base_topology`, `paired_values` y calibración sin labels de S3-T06/T07 sobre el mismo lote. Familia original reutiliza `base_topology`, `shared_initialization` e `initialize_paired` de S3-T05. MLP y DSU-S se inicializan con sus constructores tras fijar la seed. S3-T03 usa `build_paired_models` para conservar la pareja dense-mask.
- Primera cohorte principal: cinco brazos `S3-T07` y cuatro `S3-T05`. Adapters adicionales: MLP 13→17 (S2-T05), MLP 20→26 (S2-T06), vANN/dANN-R (R-002, N=3), dANN-R/Fixed-dANN (S3-T03) y DSU-S (S2-T04). Los últimos tienen grupos de tamaño 1–2 o protocolos distintos y no sostienen rankings multiarquitectura fuertes. `S3-T06` sólo tiene diagnósticos sin calidad final; S0 usa MNIST y se excluye. Se evita duplicar las filas N=5 de R-003 con las N=3 de R-002.

### Definiciones numéricas

Sea `A∈R^{N×U}` la activación principal. Centrar cada columna y calcular `λ_i≥0`, autovalores de `A_cᵀA_c/(N−1)` en float64. Valores negativos de redondeo se truncan a cero. Con `p_i=λ_i/Σλ`, `effective_rank_raw=exp(−Σ_{p_i>0}p_i log p_i)` y `effective_rank_norm=raw/min(N,U)`. **Se usa distribución de autovalores, no singular values.** `participation_ratio_raw=(Σλ)^2/Σλ²` y `participation_ratio_norm=raw/min(N,U)`. Si toda varianza es cero, ambos raw y norm valen 0. `ε_rank=1e−12` sólo determina degeneración del denominador; los términos `p=0` se omiten.

`mean_abs_corr` y mediana son `|corr|` de pares de columnas fuera de diagonal con desviación >`1e−12`; se informa cuántas unidades constantes se excluyen. Sin pares válidos se devuelve `null`. La auditoría posterior precisó la mediana convencional para número par de pares (promedio de los dos centrales); la implementación inicial usaba el central inferior de `torch.median`. `near_zero_fraction=mean(|A|<1e−6)` y `exact_zero_fraction=mean(A==0)`. Escala de preactivación/activación: media, DE poblacional, RMS y máximo absoluto. Logits: media, DE poblacional, media de norma L2 por ejemplo y entropía predictiva media calculada con `log_softmax` estable.

Cobertura sparse usa conjuntos de índices **por soma**, con unión de ramas dentro de soma: features únicas globales, cociente sobre 784, media/DE poblacional y máximo de referencias por feature (incluyendo features no usadas), y media Jaccard de los 8128 pares de somas. MLP denso tiene cobertura 1; Jaccard y conteos de referencias se dejan `null` porque no son comparables con la topología sparse.

`activation_pattern_logdet`: `P=(A>0)` binaria para LeakyReLU; `K=P Pᵀ+(1−P)(1−P)ᵀ`; reportar `logdet(I_N+K/U)/N`. Es un descriptor inspirado en diversidad de patrones/NASWOT, **no** la implementación canónica. Su regularización `I_N`, divisor `U` y normalización por `N` se fijan aquí. Un `slogdet` no positivo o métrica no finita invalida la corrida.

Tiempo por brazo/seed: construcción e inicialización, calibración si aplica, y cómputo de proxies separados; total excluye carga de Fashion-MNIST compartida. `forward_passes` cuenta una colección principal más las dos evaluaciones internas de calibración neutral. `peak_tensor_size_bytes_estimate` es el mayor tensor lógico conocido (`inputs`, `A`, `pre`, `logits`, covariance, kernel), **no RAM pico**. La auditoría posterior de esta estimación se documenta en Ejecución y trazabilidad; no se cambió ninguna fórmula de ranking.

### Evaluación retrospectiva fijada

Unir por `architecture_id=experiment_id|architecture`. Comparar sólo filas con iguales `protocol_family` **y** `paired_comparison_group`; grupos con un modelo no producen correlación. Target primario: test accuracy histórica; secundario: `−test_nll`. Los targets sólo se leen después de persistir resultados proxy. La correlación Spearman usa rangos promedio para empates; Kendall τ-b se informa para `n≥3`. Para cada par, la orientación fijada es **mayor mejor** para rangos, participation ratio, cobertura y logdet; **menor mejor** para correlación, Jaccard entre unidades y fracción casi nula. Escalas y entropía usan orientación `mayor` sólo como diagnóstico mecánico, sin hipótesis de monotonicidad. Empates de proxy exactos reciben medio acierto.

Near-tie real: `|Δaccuracy|<0.001` (0.10 pp) o `|ΔNLL|<0.005`. Informar pairwise accuracy excluyéndolos y, adicionalmente, incluyendo cada near-tie con 0.5 crédito neutral; Spearman/Kendall usan todos los modelos. No ajustar orientación, epsilon, batch ni fórmula tras observar correlaciones. Clasificación descriptiva PROMISING / WEAK / UNINFORMATIVE / MISLEADING por consistencia de dirección, tests A/B y pares, sin umbral de significancia rígido. Test A (PointSparse vs MLP pequeño) es **histórico entre protocolos**, por tanto sólo descriptivo. Tests B/C usan S3-T07; D compara raw/norm entre anchos y grupos sin atribución causal.

## Ejecución y trazabilidad

`python -m dsu_research.analysis.zero_cost_collection` escribió [medidas por seed](../data/zero_cost_proxy_results.json) **antes** de que `python -m dsu_research.analysis.zero_cost_validation` abriera el [registro histórico](../data/historical_experiments.json). El archivo de proxies quedó sellado en el commit `9719b41`; el [análisis retrospectivo](../data/zero_cost_validation.json) guarda su SHA-256. Son 48 registros: 16 arquitecturas × tres seeds; 72 evaluaciones forward equivalentes, contando calibración neutral. Los hashes de índices train por seed son `6e2834ad32df`, `6eba57d8b916`, `fb8fc4e41fb6`; coinciden con S3-T06. Los SHA-256 completos de índices y píxeles están en JSON. Repetir la colección produjo **todos los fingerprints y proxies idénticos**; sólo cambió el timing.

CPU Ryzen 5 5600G, PyTorch 2.14.1+cpu, torchvision 0.29.1+cpu, NumPy 2.5.3; modelos float32 y estadísticas float64 con un hilo para esta colección. Tiempo medido total de construcción, calibración y proxies de los 48 registros: **1095 ms** (media 22.8 ms, rango 3.0–109.0 ms), más ~70 ms de lectura de los tres lotes train. No incluye arranque de proceso ni descarga del dataset. Un registro usa 1 forward principal; el compartimento neutral contabiliza otras dos evaluaciones internas de calibración. **Correcciones de auditoría tras abrir los targets:** la primera estimación de mayor tensor omitía el `gather` compacto y el peso dense-mask temporal. Se corrigió sólo ese campo: máximo conocido **8388608 bytes** para modelos compactos sparse y **1605632 bytes** para controles dense-mask; las métricas de ranking, fingerprints y tiempos quedaron idénticos. La mediana de correlaciones se corrigió después al punto medio convencional; cambiaron 45 valores por seed, pero **ningún Spearman ni par ordenado cambió**. Son máximos de un tensor lógico, no RAM pico. Comparado con entrenamientos previos de ~160 s **por seed y brazo** en S3-T07, el screen es muy barato en este hardware; son procedimientos diferentes, no una comparación de latencia de inferencia.

## Arquitecturas y proxies

Se incluyeron cinco brazos neutrales S3-T07, cuatro compartimentos originales S3-T05, MLP 13→17 y 20→26, vANN/dANN-R de R-002, dANN-R/Fixed-dANN S3-T03 y DSU-S: **16 arquitecturas**. MLP 20→26 y DSU-S quedan como diagnósticos individuales por sus grupos separados; la comparación PointSparse–MLP 13→17 se informa sólo como Test A histórico entre protocolos. No se incluyeron MNIST S0, duplicados N=5 de R-003, ni las filas S3-T06 sin target final. Los adapters reprodujeron las inicializaciones/topologías de sus tareas; no se cargó ningún checkpoint entrenado.

Implementados: effective rank y participation ratio raw/normalizados; correlación media/mediana; fracciones near-zero/exact-zero; escala de preactivaciones y activaciones; escala de logits/entropía; cobertura, referencias y Jaccard de topología; `activation_pattern_logdet` NASWOT-like. Las fórmulas exactas están en la sección congelada. `exact_zero_fraction` y escalas son diagnósticos, no selección.

Media sobre seeds 1–3. Test accuracy/NLL vienen del registro previo, **no** se calcularon con el lote proxy. ER y PR son raw; `ER/U` y `PR/U` usan `min(256,U)`.

| Grupo | Arquitectura | U | ER | ER/U | PR | PR/U | Corr. abs. media | Pattern logdet | Test acc. % | Proxy ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Histórico | MLP 13→17 | 17 | 4.358 | .2563 | 2.959 | .1740 | .363 | .0900 | 85.883 | 3.4 |
| Histórico | MLP 20→26 | 26 | 6.647 | .2557 | 4.145 | .1594 | .264 | .1223 | 86.700 | 3.1 |
| S3-T07 | PointSparse | 128 | 15.942 | .1245 | 6.333 | .0495 | .214 | .1846 | 86.913 | 15.8 |
| S3-T07 | neutral 1×64 | 128 | 13.504 | .1055 | 5.350 | .0418 | .162 | .1737 | 86.640 | 24.9 |
| S3-T07 | neutral 2×32 | 128 | 12.263 | .0958 | 4.733 | .0370 | .196 | .1899 | 86.573 | 28.6 |
| S3-T07 | neutral 4×16 | 128 | 10.260 | .0802 | 3.790 | .0296 | .228 | .1906 | 86.243 | 32.5 |
| S3-T07 | neutral 8×8 | 128 | 7.883 | .0616 | 2.942 | .0230 | .304 | .1878 | 85.800 | 41.2 |
| S3-T05 | original 1×64 | 128 | 8.736 | .0683 | 4.419 | .0345 | .215 | .1846 | 86.867 | 13.7 |
| S3-T05 | original 2×32 | 128 | 12.132 | .0948 | 5.853 | .0457 | .209 | .1835 | 86.830 | 14.9 |
| S3-T05 | original 4×16 | 128 | 14.208 | .1110 | 6.156 | .0481 | .221 | .1782 | 86.890 | 16.7 |
| S3-T05 | original 8×8 | 128 | 13.415 | .1048 | 5.477 | .0428 | .212 | .1728 | 86.993 | 22.4 |
| R-002 | vANN | 128 | 16.111 | .1259 | 6.134 | .0479 | .219 | .1748 | 89.433 | 12.8 |
| R-002 | dANN-R | 128 | 15.824 | .1236 | 6.596 | .0515 | .205 | .1834 | 87.237 | 47.4 |
| S3-T03 | dANN-R | 128 | 15.824 | .1236 | 6.596 | .0515 | .205 | .1834 | 87.216 | 33.8 |
| S3-T03 | Fixed-dANN | 128 | 17.215 | .1345 | 7.316 | .0572 | .204 | .1798 | 87.174 | 34.8 |
| S2-T04 | DSU-S | 128 | 17.435 | .1362 | 7.499 | .0586 | .195 | .1844 | 87.253 | 18.9 |

Las tablas completas por seed incluyen media y DE muestral para todas las métricas. La fila dANN-R de R-002 y la de S3-T03 reutilizan el mismo tipo de inicialización/topología por seed; **no** son dos observaciones independientes para correlación. Las filas entre grupos no se comparan causalmente.

## Ranking retrospectivo por grupo

Orientación de la tabla: proxy favorable en el sentido congelado; para NLL el target es `−NLL`. `Pares` excluye near-ties históricos (<0.10 pp accuracy, <0.005 NLL); `τ` es Kendall τ-b. La versión que **incluye** near-ties con 0.5 crédito está en el JSON. En S3-T07 hubo un near-tie de accuracy y NLL; en S3-T05 hubo tres y cuatro, respectivamente. Los valores `ρ=±1` de grupos de dos modelos no son evidencia de correlación general y se dejan sólo en el archivo de análisis.

| Proxy | S3-T07 acc. ρ / τ / pares | S3-T07 −NLL ρ / τ / pares | S3-T05 acc. ρ / τ / pares | S3-T05 −NLL ρ / τ / pares |
|---|---|---|---|---|
| ER raw | 1.00 / 1.00 / 9/9 | 1.00 / 1.00 / 9/9 | .60 / .33 / 2/3 | .80 / .67 / 2/2 |
| ER normalizado | 1.00 / 1.00 / 9/9 | 1.00 / 1.00 / 9/9 | .60 / .33 / 2/3 | .80 / .67 / 2/2 |
| PR raw y normalizado | 1.00 / 1.00 / 9/9 | 1.00 / 1.00 / 9/9 | .00 / .00 / 1/3 | .40 / .33 / 2/2 |
| Correlación absoluta media (menor) | .70 / .60 / 7/9 | .70 / .60 / 8/9 | −.40 / −.33 / 2/3 | .00 / .00 / 1/2 |
| Near-zero (menor) | .89 / .84 / 8/9 | .89 / .84 / 8/9 | .80 / .67 / 3/3 | 1.00 / 1.00 / 2/2 |
| Pattern logdet | −.60 / −.40 / 3/9 | −.60 / −.40 / 2/9 | −.80 / −.67 / 0/3 | −1.00 / −1.00 / 0/2 |

En S3-T07 los rangos ER y PR ordenan **9/9 pares no empatados en cada seed individual**. En S3-T05 ER raw ordena 3/3, 3/3 y 1/3 por seed; la señal es menos estable. Near-zero alcanza 8/9 por medias en S3-T07, pero por seed baja a 7.5/9, 4.5/9 y 6.5/9; los valores son ~0–2×10⁻⁵ allí, próximos al suelo impuesto por LeakyReLU/epsilon. Por eso una aparente correlación de medias no implica un proxy robusto. Cobertura es 1.0 para todos los brazos principales y Jaccard permanece constante dentro de cada topología pareada: ambos carecen de poder de ranking aquí.

El par R-002 vANN/dANN-R tiene `n=2`: vANN termina con mayor accuracy y dANN-R con menor NLL, así que las direcciones de targets difieren. El par S3-T03 dANN-R/Fixed-dANN difiere sólo 0.042 pp de accuracy y ~0.0011 NLL; ambos quedan como near-tie bajo las tolerancias fijadas y aportan **cero pares puntuables**. Los `ρ=±1` de esos pares no se usan para calificar proxies.

## Tests A–D fijados

**A — PointSparse frente a MLP 13→17.** Test histórico: 86.913 > 85.883 %. ER raw 15.942 > 4.358, PR raw 6.333 > 2.959, menor correlación .214 < .363 y pattern logdet .185 > .090 favorecen PointSparse. ER normalizado **.1245 < .2563** y PR normalizado **.0495 < .1740** favorecen al MLP. Es una comparación entre S3-T07 y S2-T05, con protocolos/inicialización diferentes: acierto descriptivo, no validación causal ni par incluido en correlaciones.

**B — tendencia neutral.** PointSparse→1×64→2×32→4×16→8×8 cae en accuracy 86.913→86.640→86.573→86.243→85.800 %. ER y PR caen monótonamente y ordenan los nueve pares no near-tie; 1×64 vs 2×32 es near-tie según la regla fijada. Pattern logdet no sigue la calidad.

**C — coste de compartimentalización.** En S3-T07, más B coincide con menor dimensionalidad efectiva somática y peor NLL/top-1 bajo presupuesto neutral. En S3-T05 original, ER sube desde B=1 a B=4 mientras accuracy permanece casi plana y NLL mejora con B. Por tanto el proxy advierte del deterioro **en el control neutral**, pero no es una ley general de B ni identifica por sí solo el mecanismo causal.

**D — sesgo por ancho.** PointSparse (U=128) supera ampliamente al MLP pequeño (U=17) en ER raw, mientras ER/U invierte la comparación; PR hace lo mismo. MLP 20→26 tiene ER raw 6.647 > 4.358 de MLP 13→17 pero ER normalizado .2557≈.2563. Dentro de S3-T05/T07 todos tienen U=128, así que raw y normalizado dan **exactamente el mismo ranking**. Normalizar elimina el techo dimensional directo, pero no convierte una comparación entre protocolos en causal ni garantiza un mejor predictor.

## Clasificación y límites

| Proxy | Clasificación descriptiva | Motivo |
|---|---|---|
| Effective rank raw | **PROMISING, acotado** | 9/9 S3-T07, señal parcial S3-T05 y Test A descriptivo; susceptible a ancho |
| Effective rank normalizado | WEAK | Conserva S3-T07, pero Test A se invierte; no discrimina distinto a raw con U fijo |
| Participation ratio raw / normalizado | WEAK | Perfecto S3-T07, inconsistente en S3-T05; normalizado invierte Test A |
| Correlación media / mediana | WEAK | 7–8/9 S3-T07; inconsistencias en S3-T05 y S1 |
| Near-zero fraction | WEAK | Tasas diminutas y orden por seed inestable pese a buenos promedios |
| Cobertura y Jaccard | UNINFORMATIVE | Constantes dentro de los pares con topología compartida |
| Escala de preactivación / activación y logits | UNINFORMATIVE | Direcciones incompatibles entre S3-T05 y S3-T07; diagnósticos de escala |
| Entropía predictiva | WEAK | 8/9 S3-T07 accuracy, 0/3 S3-T05; sin dirección causal prefijada |
| Activation-pattern logdet | **MISLEADING** | 3/9 S3-T07 y 0/3 S3-T05; variante NASWOT-like, no canónica |

El gate **S3-T09: PASS técnico**: proxies forward-only, fingerprints, raw/norm, rankings y resultados negativos están registrados; tests A–D respondidos; ningún entrenamiento, backward, search o cambio de arquitectura. La evidencia es retrospectiva y pequeña: dos grupos principales de 5 y 4 modelos, uno con numerosos near-ties; arquitectura y seed no son muestras independientes. No se eligió arquitectura nueva con estos proxies ni se ajustaron definiciones después de ver accuracy. **S3-T10 recomendado:** diseñar screening retrospectivo/cheap pilot con rango efectivo como candidato independiente y redundancia como diagnóstico, con nueva cohorte o tarea previamente congelada; no ajustar un score con pesos a estas mismas filas.
