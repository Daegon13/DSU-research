# A1-DESIGN — topología causal de PointSparse

**Estado (2026-10-11):** diseño y generación estructural, sin entrenamiento. Track A posterior a Sprint 3, que permanece CLOSED. **Gate A1-DESIGN: PASS estructural**, condicionado para A1-E01 a revalidar los artefactos y la reutilización A0. H-A1-TOPOLOGY sigue UNTESTED.

TASK ID: A1-DESIGN. OBJECTIVE: preregistrar cuatro topologías a H/K fijos. WHY: medir cuánto depende PointSparse de G antes de añadir mecanismos neuronales dinámicos. CONTEXT: S3-REVIEW fijó PointSparse `128×64`; H-WIDTH-FANIN quedó INCONCLUSIVE. ALLOWED FILES: este diseño, estado/programa/roadmap/hipótesis/decisiones, artefacto estructural, módulo y tests de topología. FORBIDDEN FILES: PointSparse histórico, baselines, datasets, splits y runners de entrenamiento. REQUIREMENTS: grados, cobertura, hashes y protocolo futuro congelados. OUT OF SCOPE: entrenamiento, grid H×K, proxies, Line B/S4. ACCEPTANCE CRITERIA: invariantes por seed y contraste estructural prefijado. TESTS: tests estructurales y suite. EXPECTED OUTPUT: diseño y métricas sin labels. EXPECTED COMMIT: `research: design PointSparse topology causal test`. STOP CONDITIONS: falla de grados, cobertura, separación estructural, equivalencia A0 o test firewall.

## Pregunta y alcance causal

Con D=784, H=128, K=64, activación LeakyReLU 0.1, salida 128→10, parámetros, MAC y pesos iniciales iguales, ¿cambia la calidad al intervenir únicamente en la **organización de las aristas** de G? A0→A1 cambia el balance de grados de inputs junto con aristas concretas, por lo que es una intervención sobre el régimen de muestreo, no un efecto de grado puro. A1→A2/A3 mantiene incluso el vector de grados por input y aísla mejor la organización de G. No se estudia causalmente width ni se resuelve H-WIDTH-FANIN. Un resultado pequeño sólo acota las topologías probadas; no atribuye la señal a width.

## Matemática, brazos e invariantes

E=H·K=8192; E/D=10.44897959. Para balance exacto, 352 inputs tienen grado 11 y 432 tienen grado 10: `352·11 + 432·10 = 8192`. Cada hidden unit tiene 64 entradas únicas. El modelo tiene `8192+128+1280+10=9610` parámetros y `8192+1280=9472` MAC principales/muestra. G usa 8192 índices int64 = 65536 bytes; estos conteos no prueban FLOPs completos, latencia, RAM ni energía.

| Brazo | Generación | Grados de input | Propósito |
|---|---|---|---|
| A0 IID Random | `base_topology(seed)`: NumPy `default_rng`, 64 elecciones sin reemplazo por fila, independientes | Variables; cobertura observada, no forzada | Baseline histórico |
| A1 Balanced Random | Base balanceada, luego 100000 intentos de swaps aleatorios válidos | Mismo vector 10/11 para A1/A2/A3 | Balance de uso de features |
| A2 Balanced Low-Overlap | Copia de la misma A1, 60000 intentos de swaps que bajan Jaccard medio | Idénticos a A1 | Distribuir intersecciones más uniformemente |
| A3 Balanced High-Overlap | Copia de la misma A1, 60000 intentos de swaps que suben Jaccard medio | Idénticos a A1 | Concentrar intersecciones y formar grupos similares |

Para A1/A2/A3: coverage 784/784, E=8192, H=128, K=64, sin edges duplicadas y grados exactos. A0 comparte H/K/presupuesto/modelo, pero su grado de input y cobertura son emergentes. Todas las topologías se generan sin dataset ni labels.

## Generador, objetivo y auditoría

`pointsparse_topology.py` usa `topology_seed` entero de 32 bits. La base coloca 8192 stubs cíclicos sobre una permutación aleatoria de 784 features, lo que da grados 10/11 y evita duplicados por fila. Los 100000 intentos de swaps degree-preserving aleatorizan esa construcción sin optimizar overlap. Es una cadena finita de mezcla reproducible, **no** una muestra demostrablemente uniforme entre todos los grafos regulares. A2/A3 parten de una copia de esa misma base por seed; sus RNG de swaps están separados y congelados en el código. No se elige entre candidatos según accuracy.

Un swap toma `(u,x)` y `(v,y)` con `u≠v`; se acepta sólo si `(u,y)` y `(v,x)` no existen. Los grados de ambas particiones, el vector exacto de inputs, la cobertura y E permanecen invariantes. Para `c_ij=|G_i∩G_j|`, `J_ij=c_ij/(128-c_ij)` y el objetivo es el promedio de `J_ij` sobre los `C(128,2)=8128` pares. A2 acepta únicamente reducción estricta; A3 aumento estricto. Se registran intentos fijos, swaps aceptados, hash SHA-256 de base/final y objetivo inicial/final. Hash: bytes contiguos little-endian int64 de G en orden de filas. Serializar G explícitamente junto a seed, versión de código y hash para A1-E01; no depender sólo de la seed.

La suma de intersecciones está **fijada** al fijar grados: `Σ_{i<j} c_ij = Σ_x C(degree_x,2) = 352·55 + 432·45 = 38800`; el promedio es `38800/8128 ≈ 4.773`. El promedio de intersecciones no puede ser objetivo de optimización. Jaccard sí puede variar por su denominador no lineal. Si las intersecciones se pudieran repartir óptimamente como 1840 pares con 4 y 6288 con 5, el límite inferior de mean J sería `.0387505653`; A1 aleatorio ronda `.03902`. Por ello, incluso el máximo descenso teórico disponible es ~`.00027`. A2 logra ~`.00021` (aprox. 80% de ese margen), además de reducir la cola. Registrar coverage; media/DE/mín/máx y distribución de grados; hidden degree; media/mediana/máximo/DE/p90 de Jaccard; histograma completo de `c_ij`; y conectividad bipartita. Como diagnóstico simple de clustering, usar la cola (p90, máximo, histograma) y no inferir comunidades biológicas de un índice único.

**Separación mínima previa al aprendizaje, por cada seed 1–3:** `meanJ(A1)-meanJ(A2) ≥ 0.00015` (más de la mitad del margen teórico disponible), `meanJ(A3)-meanJ(A1) ≥ 0.001`, `maxJ(A1)-maxJ(A2) ≥ 0.02` y `maxJ(A3)-maxJ(A1) ≥ 0.20`. Si falla una seed, A1-E01 se detiene; no ajustar presupuestos ni umbrales según calidad. Es un umbral descriptivo de G, no una predicción de accuracy. La pequeña distancia absoluta de medias A1/A2 impide llamar a ese brazo una manipulación grande de Jaccard medio: el cambio claro es la cola y la proximidad al mínimo matemático.

**Medición estructural sin training:** el artefacto [a1_topology_structural.json](../data/a1_topology_structural.json) contiene por seed/brazo hashes completos, swaps, objetivos, grados, cobertura, Jaccard, histograma de intersecciones y conectividad. Resumen `meanJ / maxJ`:

| Seed | A0 | A1 | A2 | A3 |
|---|---|---|---|---|
| 1 | .042610 / .132743 | .039015 / .132743 | .038805 / .066667 | .040294 / .523810 |
| 2 | .042734 / .122807 | .039019 / .122807 | .038806 / .075630 | .040296 / .454545 |
| 3 | .042945 / .113043 | .039023 / .122807 | .038805 / .066667 | .040310 / .505882 |

En las tres seeds A2 se separa de A1 en media por ~.00021 y reduce el máximo; A3 aumenta la media por ~.00128 y eleva el máximo. Los tests fijan los umbrales descritos. La evidencia de A2 es una reducción de colas de overlap bajo grados fijos, más que un gran desplazamiento del promedio.

## Contrastes preregistrados y protocolo futuro A1-E01

1. A1−A0: balancear grados de input bajo H/K fijos.
2. A2−A1: disminuir solapamientos altos con vector de grados idéntico.
3. A3−A1: aumentar clustering/solapamientos altos con vector idéntico.
4. A3−A2: extremos organizativos con grados idénticos, contraste estructural más limpio.

Fashion-MNIST oficial, split 54000/6000/10000 por seed y píxeles `/255`; CPU float32; seeds de topología y entrenamiento **campos separados** (pareo propuesto: ambos 1,2,3). Mismo PointSparse sin alterar, `paired_values(training_seed)` para pesos y biases iniciales iguales por seed en los cuatro brazos; insertar sólo G distinta. Adam lr `.001`, betas `.9/.999`, epsilon `1e-7`, weight decay 0, CrossEntropy, batch 128, 25 épocas, sin scheduler, clipping, mixed precision ni early stopping. Mismos índices train/validation, orden de minibatches y evaluación de validation por época; test una sola vez **después de la época 25**, sin best epoch. Métricas: accuracy y NLL de validation/test, pares por seed, media/DE de deltas; diagnósticos históricos de test sólo secundarios. Guardar un JSON por corrida, hashes de G/estado inicial/split/orden, métricas por época, entorno y configuración; validar en resume sin reentrenar. Sin cheap pilot ni proxies como selector. Presupuesto nuevo con reuse A0: `3×3×25=225` epoch-equivalents; si reuse falla: `4×3×25=300`.

**Regla de reuse A0:** por cada seed, validar el registro S3-T07 y comparar generator, hash G, clase/forward, hashes de todos los tensores iniciales, topología/training seeds, split/transform, orden de 25 épocas, optimizador y sus hiperparámetros, batch, precisión, dispositivo, métrica y `final_epoch_25`. La reconstrucción actual dio coincidencia exacta de G e inicialización para seeds 1–3; los JSON históricos contienen split 54k/6k/10k, 25 hashes de batch y `final_epoch_25`. S3-T11 ya auditó el anchor y detectó número de threads CPU no congelado en T07; se debe declarar esa limitación numérica y no prometer trayectoria bit a bit. En A1-E01, revalidar hashes de artefactos y código; si cambia algo material, entrenar A0 de nuevo antes de comparar. El generador nuevo no cambia A0 ni PointSparse.

## Reglas de calidad y cierre, congeladas antes de A1-E01

Para cada contraste, delta = segundo menos primero en puntos porcentuales de test accuracy y en NLL. `MATERIAL` si `|media Δaccuracy| ≥ 0.50 pp`, signo consistente en ≥2/3 seeds y NLL compatible (NLL medio mejora en la dirección del brazo de mayor accuracy o `|ΔNLL| < .005`). `SMALL BUT REAL SIGNAL` si `0.20 ≤ |media Δaccuracy| < 0.50 pp` y signo consistente en ≥2/3. `NEGLIGIBLE` si `|media Δaccuracy| < 0.20 pp` y `|media ΔNLL| < .005`. Todo lo demás: `INCONCLUSIVE`. Los umbrales son descriptivos para N=3 y no sustituyen confirmación estadística. Validation se reporta, pero no se usa para mover umbrales, elegir seeds ni escoger un checkpoint.

H-A1-TOPOLOGY: «Con H, K, parámetros, MAC, cobertura y distribución de grados controlados, la organización de overlap/topología sparse cambia materialmente la calidad de PointSparse». **UNTESTED**. A0/A1 controla presupuesto pero no iguala grados ni cobertura por construcción; la hipótesis se juzga principalmente en A1/A2/A3.

Si A2/A3 muestran efecto material repetible, Track A podrá continuar sólo sobre la propiedad identificada y mediante tarea separada. Si todos los contrastes son SMALL o NEGLIGIBLE, registrar «PointSparse efficiency appears robust to tested topology details», mantener IID `128×64` como baseline y pasar Track A a backlog; priorizar una decisión separada de Line B/S4. Si son mixtos o inconclusos, permitir **máximo una** tarea diagnóstica acotada antes de esa decisión. No abrir una búsqueda general. Para futura Line B, comenzar con el baseline sparse más simple respaldado: IID si la topología no importa mucho; si una topología resulta claramente mejor, usarla como segundo control manteniendo IID para comparabilidad histórica. Este gate no inicia Line B.

## Gate A1-DESIGN

PASS estructural: cuatro generadores definidos, 3 seeds verificadas, 352×11 + 432×10 y cobertura 784/784 en A1/A2/A3, separación low/high cumplida, hashes registrados, protocolo y cierre preregistrados, A0 contrastado con los registros S3-T07, sin dataset ni entrenamiento nuevos. La suite local pasó (`105 passed, 2 skipped`). A1-E01 deberá comprobar otra vez hashes, entorno y reuse antes de cualquier corrida. Sprint 3 sigue CLOSED y Line B intacta.
