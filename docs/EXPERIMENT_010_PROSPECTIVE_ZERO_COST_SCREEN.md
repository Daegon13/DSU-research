# S3-T10 — Validación prospectiva del screening zero-cost

## Preregistro (antes de entrenar)

TASK ID: S3-T10. OBJECTIVE: probar si effective rank raw ordena prospectivamente una nueva familia PointSparse de igual presupuesto. WHY: S3-T09 sólo validó retrospectivamente una señal acotada. CONTEXT: ER raw fue PROMISING en S3-T07, parcial en S3-T05; ER normalizado cambió el orden entre anchos. ALLOWED FILES: runner y tests S3-T10, adaptador de cobertura, JSON de preregistro/resultados, este informe y registros de estado. FORBIDDEN FILES: baselines, splits, seeds y métricas previas, `historical_experiments.json`. REQUIREMENTS: cohorte y proxies sellados en Git antes de entrenar, test cerrado, piloto 3 épocas/1 seed, comparación con ancho. OUT OF SCOPE: 25 épocas, N=3 de entrenamiento completo, DSU-T, Line B y score compuesto. ACCEPTANCE CRITERIA: ocho brazos con 9610 parámetros, tres proxy seeds, ocho pilotos completos, análisis contra width y shortlist. TESTS: tests S3-T10, suite relevante, `git diff --check`. EXPECTED OUTPUT: preregistro, ocho curvas y análisis. EXPECTED COMMIT: `analysis: preregister PointSparse iso-budget screening` seguido por `experiment: validate prospective zero-cost PointSparse screen`. STOP CONDITIONS: acceso a test, desajuste de presupuesto, preregistro sin commit, fuga de resultado al screening, discrepancia de fingerprints.

**Pregunta:** con 9610 parámetros entrenables PointSparse, ¿cómo cambia la calidad temprana al intercambiar ancho H por fan-in K, y aporta ER raw información sobre ordenar sólo por H?

`P(H,K)=HK+H+10H+10=H(K+11)+10=9610`; por tanto `K=9600/H−11`, entero en la cohorte. MAC principales por muestra: `H(K+10)`. Cada unidad elige K entradas distintas uniformemente sin reemplazo; las unidades pueden compartir entradas. `topology_seed` y `training_seed` se pasan por separado. El anchor `128×64` usa exactamente la inicialización y topología PointSparse de S3-T07. Los demás brazos usan la misma regla de varianza Kaiming para LeakyReLU 0.1, salida Xavier uniforme y biases cero, cada uno con su propio H/K. No hay emparejamiento de unidades entre brazos.

| H×K | Parámetros | MAC principales | Conexiones efectivas | Topología int64, bytes | Payload modelo estimado, bytes | Temporal `selected_inputs` batch 128, bytes |
|---|---:|---:|---:|---:|---:|---:|
| 64×139 | 9610 | 9536 | 9536 | 71168 | 109608 | 4 554 752 |
| 80×109 | 9610 | 9520 | 9520 | 69760 | 108200 | 4 464 640 |
| 96×89 | 9610 | 9504 | 9504 | 68352 | 106792 | 4 374 528 |
| 120×69 | 9610 | 9480 | 9480 | 66240 | 104680 | 4 239 360 |
| 128×64 | 9610 | 9472 | 9472 | 65536 | 103976 | 4 194 304 |
| 160×49 | 9610 | 9440 | 9440 | 62720 | 101160 | 4 014 080 |
| 192×39 | 9610 | 9408 | 9408 | 59904 | 98344 | 3 833 856 |
| 240×29 | 9610 | 9360 | 9360 | 55680 | 94120 | 3 563 520 |

`effective_connections` aquí coincide con MAC principales del modelo, incluidas las conexiones salida. Payload = 9610 pesos float32 (38440 bytes) + índices int64. Temporal es una estimación lógica del tensor de entradas reunidas, no RAM pico. Otros tensores temporales por brazo están en el JSON estructural. MAC principales no equivalen a FLOPs totales, latencia ni energía.

### Proxy forward-only congelado

Reutilizar las definiciones numéricas de [S3-T09](EXPERIMENT_009_ZERO_COST_FORWARD_PROXIES.md): modelos float32, estadísticas float64, `eval()` e `inference_mode()`, sin labels ni backward. Por cada seed 1, 2, 3: 256 imágenes **train**, selección `default_rng(seed).shuffle(arange(60000))[:256]`, píxeles `/255`; los hashes de índices y píxeles se guardan en [preregistro](../data/s3_t10_preregistered_screen.json). El lote y sus fingerprints son comunes entre brazos. Se registran ER raw/norm, PR raw/norm, correlación absoluta media, DE de activaciones/logits, entropía predictiva, fracción casi nula y cobertura. Cada proxy figura por seed y como media ± DE muestral. `ER/H` queda sólo como diagnóstico. `activation_pattern_logdet` no participa: S3-T09 lo clasificó MISLEADING.

**Proxy principal: ER raw, mayor mejor.** Los demás rankings son controles descriptivos; menor correlación absoluta se orienta como mejor. El ranking puramente por H, mayor mejor, queda congelado. Ningún resultado histórico participa en el cálculo ni en la elección. El JSON guarda commit fuente, timestamp, fingerprint y protocolo. Los proxies no son accuracy observada.

| H×K | ER raw, media ± DE | Rango ER raw |
|---|---:|---:|
| 160×49 | 21.209 ± 2.344 | 1 |
| 192×39 | 18.546 ± 1.182 | 2 |
| 240×29 | 17.901 ± 2.085 | 3 |
| 120×69 | 16.805 ± 1.446 | 4 |
| 128×64 | 15.942 ± 2.598 | 5 |
| 80×109 | 15.428 ± 1.909 | 6 |
| 96×89 | 14.912 ± 2.096 | 7 |
| 64×139 | 12.512 ± 1.371 | 8 |

**Predicción falsable:** `160×49 > 192×39 > 240×29 > 120×69 > 128×64 > 80×109 > 96×89 > 64×139`. Top-3: `160×49`, `192×39`, `240×29`. Bottom-2: `96×89`, `64×139`. Ranking de ancho: `240×29 > 192×39 > 160×49 > 128×64 > 120×69 > 96×89 > 80×109 > 64×139`. La diferencia entre ambos hace verificable si ER añade algo a H.

### Piloto y análisis prefijados

Sólo tras el primer commit: ocho arquitecturas × 3 épocas × training seed 1 y topology seed 1. Fashion-MNIST histórico: 54000 train / 6000 validation (shuffle NumPy seed 1, últimos 6000 validation), ToTensor; batch 128, Adam `lr=.001`, betas `.9/.999`, epsilon `1e-7`, weight decay 0, CrossEntropy, LeakyReLU 0.1. Registrar train loss, validation loss y accuracy **al final de cada época**. Target principal: validation accuracy de época 3; secundario: negativo de validation loss de época 3. No elegir best epoch ni abrir test. Cada brazo se persiste inmediatamente; el runner valida fingerprints y reanuda.

Comparar ER raw/norm, PR raw/norm y width H con ambos targets: Spearman, Kendall tau-b y precisión de ranking por pares. Near-tie primario: diferencia absoluta de accuracy **<0.20 puntos porcentuales**; el umbral secundario para loss se congela en `|Δloss|<0.005`, como diagnóstico. Reportar pares incluyendo near-ties (medio crédito) y excluyéndolos. Todos los brazos cuentan en Spearman/Kendall. Un proxy empatado recibe medio crédito. No modificar fórmulas ni umbrales tras el piloto.

ER raw será `VALIDATED_PROMISING` si Spearman accuracy ≥0.60, precisión de pares sin near-ties >70% y supera **ambas** métricas del baseline width H (Spearman y precisión de pares) estrictamente. Será `PARTIAL` si su señal es positiva sin alcanzar todo ello, o si no supera ancho; `FAILED` si correlación es nula/negativa o ranking materialmente engañoso. La comparación estricta con ancho operacionaliza “ER aporta información” y evita llamar mejora a un empate. Estos son criterios internos, no significancia estadística.

H-WIDTH-FANIN comienza `UNTESTED`: el piloto puede dar `PRELIMINARY_SIGNAL` si el rango de accuracy entre brazos es ≥0.20 pp, `NO_PRELIMINARY_SIGNAL` si <0.10 pp o `INCONCLUSIVE` entre ambos. Esta etiqueta describe sólo un filtro de 3 épocas/1 seed y nunca es SUPPORTED/REFUTED definitivo. H-ZEROCOST-ER comienza `UNTESTED` y seguirá la clasificación anterior.

Shortlist prefijada para Stage 3 futura: mejor accuracy de piloto, mejor ER raw si difiere y anchor `128×64`, eliminando duplicados; máximo tres. Una arquitectura mala sólo se conservaría como control documental si aporta valor científico, sin entrenarla aquí. Coste comparativo: 24 epoch-equivalents realizados frente a 600 para ocho brazos × 25 épocas × 3 seeds ingenuos; reducción de 96% **antes de seleccionar**, sin inferir joules ni ahorro total de una Stage 3 futura.

## Resultados del piloto

Pendiente hasta que el preregistro esté versionado en Git.
