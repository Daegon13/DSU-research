# DSU Analytical Screening Framework — especificación S3-T08

**Propósito:** descartar configuraciones estructuralmente pobres y priorizar candidatas antes de entrenamientos completos. No predice accuracy exacta ni combina métricas en un score con pesos arbitrarios. Entrada: `ArchitectureSpec` inmutable; salida: `StructuralCost` más, en tareas futuras, medidas de señal, proxies y clasificación cualitativa de hardware. El [registro histórico](../data/historical_experiments.json) conserva `experiment_id`, `protocol_family` y `paired_comparison_group`; `null` significa no medido/reportado, nunca cero. El snapshot está versionado. El generador `python -m dsu_research.analysis.experiment_registry` reconstruye las filas desde los JSON locales de `runs/`, que el repositorio ignora; el snapshot permite inspección en clones sin ellos.

## 1. Coste estructural implementado

Para MLP con anchuras `(n₀,…,nL)`: `P=Σ_l(nₗ₋₁+1)nₗ`, `MAC=Σ_l nₗ₋₁nₗ`. Para H somas, B ramas, K entradas/rama y O salidas, `C=HBK`:

| Familia | Parámetros efectivos | MAC principales por muestra | Almacenamiento |
|---|---:|---:|---|
| PointSparse (B=1) | `C+H+HO+O` | `C+HO` | compacto + `C` índices |
| DSU-S / compartimentos originales | `C+2HB+H+HO+O` | `C+HB+HO` | compacto + `C` índices |
| Compartimentos neutrales | `C+H+HO+O` | `C+HO` | compacto + `C` índices + 2 escalares de calibración |
| dANN-R / Fixed-dANN dense-mask | mismos efectivos que DSU-S | efectivos `C+HB+HO`; GEMM denso `DHB+HBH+HO` | `DHB+HBH+HO+HB+H+O` posiciones; máscaras bool densas; Fixed añade `C` índices |

Los compartimentos neutrales también hacen `H(B−1)` sumas fijas por muestra. `stored_parameter_bytes=P_stored×bytes_per_parameter`; `topology_bytes=C×bytes_per_index` para compactos, o bytes de máscaras y `G` persistentes para controles; `model_checkpoint_tensor_payload=stored_parameter_bytes+persistent_buffer_bytes`. Adam se estima con dos momentos del tamaño de los parámetros almacenados: `2×stored_parameter_bytes`, sin escalares step, gradientes, cabeceras ni serialización. `state_bytes=H×state_scalars_per_unit×batch×bytes_per_scalar`; el valor es cero en estos modelos estáticos y el campo sólo reserva contabilidad futura, no implementa una neurona dinámica. El diccionario de activaciones informa tamaños lógicos de tensores forward para un batch, **no** pico de RAM ni memoria de autograd. `effective_connections` incluye los pesos de los MAC principales, no sesgos. La fórmula de dense-mask presupone máscara de cable con B conexiones por soma; el control dANN-R tiene K promedio, no exacto por rama.

Validación exacta: PointSparse `9610/9472`; DSU-S `10634/9984`, `8192` índices y `108072` bytes de payload; Fixed-dANN `468874` almacenados, `10634` efectivos, `468224` MAC GEMM densos y `2407976` bytes de payload B; vANN `468874/468224`. El payload no equivale al tamaño de archivo `torch.save`, que incluye overhead. Véanse tests y [S3-T04](EXPERIMENT_006_PHYSICAL_REPRESENTATION_BENCHMARK.md).

## 2. Propagación de señal — diseño

Guardar por capa preactivación, activación y logits: media, varianza poblacional y fracción `|a|<ε` con ε fijado antes de ver resultados. Para `z=Σᵢwᵢxᵢ+b`, pesos independientes de media cero y varianza `σw²`, `Var(z)≈K σw² E[x²]+Var(b)`; la expresión exige independencia y no captura correlaciones de imagen/topología. Para LeakyReLU con pendiente α y `z~N(0,q)`, `E[φ(z)]=(1−α)√(q/(2π))` y `E[φ(z)²]=(1+α²)q/2`; `Var(φ(z))` se obtiene restando la media al cuadrado. Para suma de ramas `r=Σ_b c_b a_b`, `Var(r)=Σ_b c_b²Var(a_b)+2Σ_{i<j}cᵢcⱼCov(aᵢ,aⱼ)`; ignorar la covarianza sería incorrecto cuando comparten entradas. Medir empíricamente covarianzas en lote train sin labels y registrar desviación del modelo. Escala de gradiente sólo con un backward etiquetado o Jacobiano acotado; no llamarla predicción del entrenamiento.

## 3. Proxies zero-cost — protocolo S3-T09 propuesto

Congelar arquitecturas históricas y un lote train pequeño idéntico por seed, sin test ni entrenamiento; repetir inicializaciones para estabilidad. Primera tanda barata: rango efectivo `exp(−Σpᵢlog pᵢ)` y participation ratio `(Σλ)²/Σλ²` del espectro de covarianza de activaciones somáticas; correlación media absoluta entre unidades; cobertura de inputs (`|∪G|/D`) y redundancia de índices; fracción de activaciones casi nulas; varianza de logits y entropía predictiva. Una segunda tanda usa **un** backward sobre el mismo lote/labels train: norma y varianza de gradientes y fracción `|g|<εg` por grupo de parámetros efectivos. `ε` y `εg`, tamaño de lote, tratamiento de varianza cero y centro de covarianza se fijan antes de comparar. Lograr menos coste que varias épocas se verifica por tiempo real; no basta llamarlas “zero-cost”.

Jacobiano completo, espectro exacto para H grande y condition number quedan opcionales por coste. Si se usan, limitar a JVP/VJP aleatorios con presupuesto fijado. Comparar proxies con validation final **dentro de `protocol_family`/grupo pareado**, reportando rangos y contraejemplos; test sólo como auditoría descriptiva de selecciones ya congeladas. Preguntas retrospectivas: PointSparse/MLP pequeño, efecto de B, storage B/C y advertencia de latencia batch grande. No exigir acierto en todas ni ajustar pesos a estos mismos casos.

## 4. Hardware — diseño

Primer fingerprint empírico: AMD Ryzen 5 5600G, Windows 11, PyTorch CPU float32; S3-T04 midió B/C bajo modos de hilos y batches concretos. Registrar `{MAC ejecutados, bytes de parámetros/topología, bytes temporales lógicos, gathers por muestra, batch, hilos, dtype, backend}`. Estimar intensidad aritmética `~2×MAC/bytes_movidos` sólo si el denominador se mide o se etiqueta como cota/modelo; reutilización de caché y temporales la alteran. Etiquetas heurísticas: `compute-friendly` si dominan GEMM regulares y batch favorece reutilización; `gather-bound probable` si hay C índices/gather y batch alto; `memory-bound probable` si el tráfico estimado supera la reutilización computacional. Calibrar orden aproximado **sólo en este hardware** con S3-T04 y verificar por batch; no prometer CPU universal, energía o speedup desde MAC. Una implementación de predictor queda para tarea posterior.

## Política de decisión

Conservar métricas independientes y frente de Pareto condicionado a presupuesto: una arquitectura sólo se descarta por dominancia cuando sus costes comparables no mejoran y la incertidumbre de calidad/proxy está declarada. Nunca descartar por test retrospectivo, ni usar protocolos históricos distintos como pares causales. [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) define las etapas 0–4. S3-T09 implementará y validará la primera tanda de proxies; Sprint 3 no se cierra automáticamente.
