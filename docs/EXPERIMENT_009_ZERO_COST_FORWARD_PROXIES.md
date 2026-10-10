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

`mean_abs_corr` y mediana son `|corr|` de pares de columnas fuera de diagonal con desviación >`1e−12`; se informa cuántas unidades constantes se excluyen. Sin pares válidos se devuelve `null`. `near_zero_fraction=mean(|A|<1e−6)` y `exact_zero_fraction=mean(A==0)`. Escala de preactivación/activación: media, DE poblacional, RMS y máximo absoluto. Logits: media, DE poblacional, media de norma L2 por ejemplo y entropía predictiva media calculada con `log_softmax` estable.

Cobertura sparse usa conjuntos de índices **por soma**, con unión de ramas dentro de soma: features únicas globales, cociente sobre 784, media/DE poblacional y máximo de referencias por feature (incluyendo features no usadas), y media Jaccard de los 8128 pares de somas. MLP denso tiene cobertura 1; Jaccard y conteos de referencias se dejan `null` porque no son comparables con la topología sparse.

`activation_pattern_logdet`: `P=(A>0)` binaria para LeakyReLU; `K=P Pᵀ+(1−P)(1−P)ᵀ`; reportar `logdet(I_N+K/U)/N`. Es un descriptor inspirado en diversidad de patrones/NASWOT, **no** la implementación canónica. Su regularización `I_N`, divisor `U` y normalización por `N` se fijan aquí. Un `slogdet` no positivo o métrica no finita invalida la corrida.

Tiempo por brazo/seed: construcción e inicialización, calibración si aplica, y cómputo de proxies separados; total excluye carga de Fashion-MNIST compartida. `forward_passes` cuenta una colección principal más las dos evaluaciones internas de calibración neutral. `peak_tensor_size_bytes_estimate` es el mayor tensor lógico conocido (`inputs`, `A`, `pre`, `logits`, covariance, kernel), **no RAM pico**.

### Evaluación retrospectiva fijada

Unir por `architecture_id=experiment_id|architecture`. Comparar sólo filas con iguales `protocol_family` **y** `paired_comparison_group`; grupos con un modelo no producen correlación. Target primario: test accuracy histórica; secundario: `−test_nll`. Los targets sólo se leen después de persistir resultados proxy. La correlación Spearman usa rangos promedio para empates; Kendall τ-b se informa para `n≥3`. Para cada par, la orientación fijada es **mayor mejor** para rangos, participation ratio, cobertura y logdet; **menor mejor** para correlación, Jaccard entre unidades y fracción casi nula. Escalas y entropía usan orientación `mayor` sólo como diagnóstico mecánico, sin hipótesis de monotonicidad. Empates de proxy exactos reciben medio acierto.

Near-tie real: `|Δaccuracy|<0.001` (0.10 pp) o `|ΔNLL|<0.005`. Informar pairwise accuracy excluyéndolos y, adicionalmente, incluyendo cada near-tie con 0.5 crédito neutral; Spearman/Kendall usan todos los modelos. No ajustar orientación, epsilon, batch ni fórmula tras observar correlaciones. Clasificación descriptiva PROMISING / WEAK / UNINFORMATIVE / MISLEADING por consistencia de dirección, tests A/B y pares, sin umbral de significancia rígido. Test A (PointSparse vs MLP pequeño) es **histórico entre protocolos**, por tanto sólo descriptivo. Tests B/C usan S3-T07; D compara raw/norm entre anchos y grupos sin atribución causal.

## Resultados y gate

Pendientes de la fase de colección forward-only y del análisis retrospectivo separado.
