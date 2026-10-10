# S3-T01 — Diseño causal de Sprint 3

**Fecha:** 2026-10-08. **Estado:** especificación; sin modelos nuevos ni entrenamientos. **Gate S3-T01: PASS de diseño**, sujeto a verificación de implementación en S3-T02. Fuentes: [DSU_S_SPEC.md](DSU_S_SPEC.md), [REPRODUCTION_001.md](REPRODUCTION_001.md), [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) y [SPRINT_2_REVIEW.md](SPRINT_2_REVIEW.md).

## Ficha de tarea

| Campo | Valor |
|---|---|
| TASK ID | S3-T01 |
| OBJECTIVE | Definir Fixed-dANN y controles A/B/C para separar fan-in/topología de representación. |
| WHY | A y C difieren simultáneamente en distribución de conexiones, inicialización y almacenamiento. |
| CONTEXT | Sprint 2 PASS; H-001/H-008 apoyadas sólo en sus alcances; H-002 abierta. |
| ALLOWED FILES | Este diseño, `HYPOTHESES.md`, `ROADMAP.md`, `DECISIONS.md`, `PROJECT_STATE.md`. |
| FORBIDDEN FILES | Código, tests, configs y resultados experimentales. |
| REQUIREMENTS | Topología B=C exacta, política de pesos pareados, matriz causal, protocolo y límites de claims. |
| OUT OF SCOPE | Implementar Fixed-dANN, entrenar, cambiar DSU-S, abrir otras arquitecturas o datasets. |
| ACCEPTANCE CRITERIA | Responder las seis preguntas del gate al final de este documento. |
| TESTS | Suite existente, `git diff --check`, `git status`; no tests nuevos por ser documental. |
| EXPECTED OUTPUT | Diseño auditable y siguiente tarea S3-T02. |
| EXPECTED COMMIT | `docs: define sprint 3 causal ablation design` (si se crea un commit). |
| STOP CONDITIONS | Topología o mapeo B/C ambiguos; diferencias funcionales sin explicar; cambio silencioso de split, seeds o baseline. |

## Objetivo, confusores y sistemas

La señal de Sprint 2 es de capacidad por parámetro entrenable bajo un protocolo acotado. A vs C cambia fan-in, distribución de topología, escala inicial de pesos y representación. Las mediciones de calidad, conectividad, almacenamiento y rendimiento físico deben mantenerse separadas.

Convención: `G[s,b,k]` contiene índices de entrada, `s=0..127`, `b=0..3`, `k=0..15`; la dendrita linealizada es `j=4s+b`. En B y C, `G` se obtiene de **una sola** fuente canónica: el buffer `DSUSv0.G` de la seed de topología, o su mismo algoritmo y orden exacto (`np.random.default_rng(topology_seed).choice(784,16,replace=False)` repetido en orden `j`). Para checkpoints, el buffer cargado prevalece sobre la seed. Verificar igualdad elemento a elemento y SHA-256 de `G` antes de usar un par; igualdad de seed o de distribución por sí sola no basta.

| model | variable fan-in? | exact K | topology algorithm | topology G | dense storage? | compact storage? | stored params | effective params | topology metadata | expected mathematical function |
|---|---|---:|---|---|---|---|---:|---:|---|---|
| A — dANN-R existente | Sí | No; media 16 | 8192 posiciones globales sin reemplazo en matriz 784×512 | Máscara global propia, variable por dendrita | Sí, máscara fija | No | 468874 | 10634 | `input_mask` y `cable_mask` booleanas persistentes | LeakyReLU dendrita/soma y salida lineal bajo su máscara variable |
| B — Fixed-dANN por implementar | No | 16 | Muestreo canónico de DSU-S, 16 distintas por dendrita | **La misma `G` que C**, mismo orden `s,b,k` | Sí, máscara fija | No | 468874 | 10634 | Máscaras densas; `G` sólo si se decide persistirla además, contabilizándola | La misma función que C al mapear pesos y sesgos |
| C — DSU-S v0 existente | No | 16 | Muestreo canónico por dendrita | **La misma `G` que B** | No | Sí | 10634 | 10634 | Buffer `G` int64 persistente, 8192 índices = 65536 bytes | La misma función que B al mapear pesos y sesgos |

Los conteos `stored params` incluyen tensores entrenables, incluso posiciones muertas de A/B; no incluyen máscaras ni índices. A/B almacenan `784×512 + 512 + 512×128 + 128 + 128×10 + 10 = 468874` posiciones entrenables nominales. Sólo `8192 + 512 + 512 + 128 + 1280 + 10 = 10634` son efectivas. B es una excepción experimental explícita a ADR-003, no la implementación principal. Para B, los bytes de máscaras y cualquier `G` adicional se medirán y sumarán aparte de los 1 875 496 bytes de parámetros float32; no duplicar ni ocultar metadata.

**A vs B:** ambos usan dense-mask, igual ancho, exactamente 8192 conexiones de entrada, cuatro ramas por soma y salida idéntica. Cambia la *distribución* del fan-in y por ello la topología concreta. El efecto estimable es de regularizar fan-in **junto con** el algoritmo/distribución topológica; no equivale a aislar exclusivamente `K=16` ni a mantener las mismas aristas. Deben registrarse mínimo, máximo, histograma y solapamiento de conexiones. Una diferencia de inicialización invalidaría incluso esta lectura estrecha.

**B vs C:** con `G` y pesos efectivos iguales, los dos representan la misma función; cambian almacenamiento y ruta de operaciones dense-mask frente a gather/reducción. Calidad de clasificador no es la pregunta primaria. Diferencias pequeñas por orden de reducción o kernel son numéricas, no evidencia de efecto neuronal.

## Equivalencia B ↔ C: contrato para S3-T02

Para cada `G`, crear la máscara B con `mask[j,i] = 1` si y sólo si `i` aparece en `G[s,b,:]`. Exigir 16 entradas distintas por fila, 8192 unos y cable `s→[4s,4s+1,4s+2,4s+3]`. Copiar `W_d[s,b,k]` a `B.dendrites.weight[j,G[s,b,k]]`, `bias_d[s,b]` a `B.dendrites.bias[j]`, `W_s[s,b]` a `B.somata.weight[s,j]`, `bias_s[s]` a `B.somata.bias[s]`, y `W_o/bias_o` a la salida B. Poner pesos inactivos a cero; su valor almacenado es matemáticamente irrelevante si la máscara se aplica en cada forward. Verificar también la conversión inversa y los hashes topológicos tras serializar/cargar.

Con el mismo input float32 y modo evaluación, comparar **preactivaciones dendríticas**, activaciones dendríticas, preactivaciones somáticas, activaciones somáticas y logits. Usar tolerancia inicial `torch.testing.assert_close(rtol=1e-5, atol=1e-6)` en CPU float32 sobre lotes fijos que incluyan ceros, valores positivos y negativos; repetir con varias seeds/topologías. Comparar además gradientes efectivos de cada banco y gradientes de entrada con la misma loss escalar; tolerancia inicial igual. Registrar error absoluto/relativo máximo y revisar si alguna operación requiere tolerancia distinta por reducción. No relajarla sólo para obtener PASS: justificar con evidencia numérica. Igualdad bit a bit no es requisito.

El test debe fallar si las máscaras no corresponden exactamente a `G`, si una conexión extra actúa, si difieren los sesgos/activaciones o si los gradientes activos no coinciden dentro de tolerancia. **Sin PASS funcional y de gradientes, B vs C no es un aislamiento válido de representación.** Si no se puede exponer una activación intermedia sin tocar DSU-S, usar hooks o cálculo diagnóstico externo para S3-T02; no modificar DSU-S en esta tarea.

## Inicialización y trayectorias

Para **A vs B**, mantener la política de dANN-R: Glorot uniform sobre matrices densas `[512,784]`, `[128,512]`, `[10,128]`, sesgos cero, enmascarar pesos inactivos y volver a aplicar la máscara en cada forward. Emparejar `training_seed`, orden de inicialización, split y shuffle. Si la implementación B consume RNG distinto, copiar antes de enmascarar los tensores iniciales densos de A al control B y registrar el método. Así los pesos de aristas coincidentes tienen la misma escala y valor; las aristas diferentes siguen impidiendo emparejamiento peso a peso completo. La escala Glorot de A/B **difiere** de la inicialización vigente de C (`[512,16]`, `[128,4]`, `[10,128]`); no reutilizar resultados históricos A/C como prueba causal sin declarar este confusor.

Para **B vs C**, construir una pareja nueva con `G` compartida y una sola fuente de pesos activos. Preferencia: inicializar C con su política vigente, copiar los seis bancos a B por el mapeo anterior y poner posiciones inactivas en cero. Esta prueba no cambia la política de C y garantiza igualdad inicial. Para estudio de entrenamiento, empezar ambas parejas con esos valores y registrar hashes de los bancos activos; usar los mismos minibatches, CrossEntropy, Adam (`lr=0.001`, `betas=(0.9,0.999)`, `eps=1e-7`, `weight_decay=0`), opciones de kernel/device y precisión. Los pesos inactivos B deben tener gradiente **cero**, no participar en el forward ni alterar RNG de pesos activos. En Adam sin weight decay, momentos de posiciones inactivas empiezan y quedan en cero; verificar después de un paso y tras varios pasos que también permanezcan inactivas y que las actualizaciones activas coincidan dentro de tolerancia. Si se introduce weight decay, AdamW u otro regularizador, revisar de nuevo el control antes de interpretarlo.

E01 y E02 usan **cohortes de inicialización distintas** del mismo diseño B: B toma escala densa en E01 para compararse con A y pesos compactos transferidos en E02 para compararse con C. No mezclar sus métricas ni reutilizar los pesos de B entre contrastes.

Primero comparar forward y un paso de optimizador; después, en tarea experimental separada, comparar trayectorias a pasos prefijados, loss y logits con batch ordering idéntico. Diferencias pueden acumularse por orden de reducción, kernels, operaciones masked/gather, CPU/GPU y redondeo float32. Registrar divergencia y no llamarla diferencia de capacidad neuronal sin análisis adicional. La equivalencia funcional a pesos congelados es condición necesaria; igualdad exacta de trayectorias no está garantizada por álgebra real en hardware float32.

## S3-E01 — Fan-in/topología (ejecutado en S3-T03)

Comparar A y B con **N=5** seeds de entrenamiento/topología `1..5` si el coste es razonable, reportando cada trial y media ± DE de diferencias pareadas B−A para `final_epoch_25` en validation y test. Fashion-MNIST oficial, `ToTensor()` float32, 54000/6000/10000, split por `np.random.default_rng(training_seed)` y hashes iguales por pareja; 25 épocas, batch 128, Adam y parámetros de arriba, LeakyReLU 0.1, logits/cross-entropy. No seleccionar por mínimo de validation ni afinar B tras ver test. Mantener el código y resultados históricos de A intactos; si se reutilizan sus N=5 métricas, verificar config, hashes y condiciones comparables y explicar diferencias de inicialización. Si no, planificar ambas corridas con política común.

`training_seed` fija Python/NumPy/PyTorch/CUDA si aplica e inicialización; `split_seed = training_seed` fija únicamente la partición 54k/6k; `shuffle_seed = training_seed` fija el `torch.Generator` de DataLoader; `topology_seed = training_seed` en el plan de cinco parejas, pero es un campo lógico separado que B usa sólo para `G` y A para máscara global. Igual valor de seed **no** implica la misma topología A/B. Registrar los cuatro roles, hashes de split y máscara/`G`, entorno y orden de minibatches. Si se necesita separar fuentes de varianza, una futura matriz cruzada de training/topology seeds será otra tarea.

Resultado primario de calidad: diferencia pareada de loss y accuracy de test en época 25, con validation diagnóstica y resultados por seed; inspeccionar historia y dispersión. Una mejora B frente a A apoyaría el efecto **conjunto** de regularización de fan-in y nueva distribución topológica bajo dense-mask; empate o peor rendimiento también se registran. N=5 es descriptivo, no prueba de significancia ni de generalización.

**Resultado S3-T03 / S3-E01 (2026-10-08): PASS técnico; H-009 NOT SUPPORTED / NEGLIGIBLE.** Con inicialización densa y orden de minibatches pareados, B−A fue −0.042 ± 0.329 pp de test accuracy y +0.001066 ± 0.007721 de test loss. B ganó accuracy en 2/5 seeds y loss en 3/5. El efecto medio quedó dentro de los umbrales despreciables prefijados. Detalle en [EXPERIMENT_005_FIXED_FANIN_ABLATION.md](EXPERIMENT_005_FIXED_FANIN_ABLATION.md).

## S3-E02 — Representación (ejecutado en S3-T04)

Orden de gates: (1) identidad de `G` y mapeo de parámetros; (2) equivalencia de intermedios/logits/gradientes; (3) un paso y trayectoria Adam pareados; (4) accounting y mediciones físicas. El criterio primario es **equivalencia funcional**, no accuracy. Reportar errores numéricos máximos y trayectorias por paso; si (1)–(2) fallan, no interpretar B/C como ablación de representación.

Contabilizar por separado parámetros almacenados/efectivos, bytes de pesos, máscaras y `G`, payload tensorial total, bytes de checkpoint de `state_dict` con igual política de serialización, MAC principales teóricos, operaciones efectivamente ejecutadas, latencia controlada y RAM/VRAM pico **sólo cuando haya medición fiable**. B dense-mask puede ejecutar MAC de matrices densas pese a 9984 MAC de aristas activas; esos conteos teóricos estructurales no son FLOPs ejecutados. Medir inferencia batch 1 y un batch de entrenamiento fijado, warm-up, repeticiones, sincronización GPU, entorno y dispersión; perfilar gather/indexing separado si es posible. No inferir energía sin medirla. No afirmar que compactación mejora accuracy, ni almacenamiento total frente al MLP `20→26`: el C actual tiene 108072 bytes de payload conocido frente a 66064 de ese MLP.

**Resultado S3-T04 / S3-E02 (2026-10-09): PASS técnico.** B/C conservaron equivalencia de logits dentro de la tolerancia aprobada. DSU-S redujo payload tensorial del modelo 22.281×, estado Adam 44.080× y checkpoint de entrenamiento 30.752× frente a Fixed-dANN. La latencia cambió de signo según batch/modo: DSU-S favorecida en forward batch 1 y Fixed-dANN en forward batches 128/512. RAM pico CPU no se midió fiablemente. H-012 SUPPORTED; H-013 MIXED / NOT SUPPORTED como ventaja general. Protocolo y tablas: [EXPERIMENT_006_PHYSICAL_REPRESENTATION_BENCHMARK.md](EXPERIMENT_006_PHYSICAL_REPRESENTATION_BENCHMARK.md).

## Segunda fase: B ramas y K entradas

Tras cerrar E01/E02, considerar barrido de `B∈{1,2,4,8}` con K=16 y barrido de `K∈{8,16,32}` con B=4, cada uno una variable conceptual por vez. En ambos, fijar 128 somas, 784 entradas, diez salidas, activaciones, dataset/split, entrenamiento, seeds, algoritmo de muestreo y política de inicialización declarada. **No** atribuir cambios sólo a B/K si también cambian conexiones y parámetros; reportar los presupuestos. Para B distinto cambia el número de sesgos y pesos soma, y para K distinto cambia el número de aristas. No usar estos barridos como controles de presupuesto igualado.

**S3-T05 ejecutado: partición dendrítica con 64 conexiones por soma.** Se compararon `(B,K)=(1,64),(2,32),(4,16),(8,8)` con 128 somas y 8192 conexiones input→dendrita, 64 índices únicos por soma, `G64` compartida y partición nested. `B` altera número de sesgos dendríticos, pesos soma y activaciones; total de parámetros efectivos `9610 + 256B` (9866, 10122, 10634, 11658). Por ello **no es igualdad exacta de parámetros**, ni aísla solamente la no linealidad. La variante B=4 con conjunto de 64 sin solapamiento **no es** la topología DSU-S v0 original. Las cuatro se entrenaron de nuevo con seeds 1–3, inicialización base y minibatches pareados. El contraste principal `4×16 − 1×64` fue +0.023 ± 0.138 pp de test accuracy y −0.005051 ± 0.001043 de test loss. Gate técnico PASS; H-011 NEGLIGIBLE según los umbrales fijados. [Detalle](EXPERIMENT_007_DENDRITIC_COMPARTMENT_ABLATION.md).

## S3-T06 — controles posteriores a la ablación

S3-T06 quedó cerrado como PASS de diseño, implementación y diagnóstico forward, sin entrenamiento. PointSparse y la familia neutral B=1,2,4,8 reutilizan exactamente G64 nested y tienen 9610 parámetros y 9472 MAC principales cada uno. La agregación fija `sum(d)/sqrt(B)` y la calibración afín global sin labels se congelaron antes de S3-T07. El menor loss de S3-T05 puede reflejar probabilidades/márgenes, escala de logits, parámetros extra o compartimentalización; todavía no se distingue cuál. H-014 y H-015 siguen UNTESTED. [Especificación, tabla y procedimiento](S3_T06_SPARSE_VS_COMPARTMENT_CONTROLS.md). Siguiente tarea exacta: S3-T07 — ejecutar la comparación causal pareada.

## Hipótesis, límites y gate

- **H-009, fan-in:** NOT SUPPORTED / NEGLIGIBLE en S3-E01 para Fashion-MNIST N=5; regularizar a K=16 exacto no mostró efecto material bajo el protocolo dense-mask pareado. El contraste examina distribución de fan-in junto a distribución topológica.
- **H-010, representación:** SUPPORTED en el alcance estructural y funcional CPU float32 de S3-T02; S3-T04 midió almacenamiento y ejecución del par equivalente.
- **H-011, partición dendrítica:** NEGLIGIBLE para `4×16` frente a `1×64` bajo S3-T05 N=3; no se observó una mejora material de accuracy con las mismas 64 entradas por soma.

Amenazas: fan-in y topología inseparables en A/B; escala Glorot distinta entre reproducción y DSU-S; aristas A/B distintas aunque compartan seed; N pequeño; orden float32 y kernels; máscaras y metadata en el checkpoint; mediciones de tiempo dependientes de hardware; barrido B/K con sesgos y activaciones no igualados; sólo Fashion-MNIST. No investigar aún estado temporal, DSU-T, AIS, inhibición, rewiring, spikes, Transformer ni otros datasets.

**Backlog de topología, sin implementación:** regeneración desde `topology_seed`, checkpoint sin `G`, índices int32, índices int16 empaquetados, topología implícita y topología estructurada/por bloques. Para cada opción distinguir bytes persistidos de checkpoint de dtype/bytes y costo de indexación en runtime; regenerar o comprimir el archivo no implica ahorrar RAM.

**Gate S3-T01 = PASS de diseño:** A/B difieren en fan-in/distribución topológica; B/C sólo en representación si pasan igualdad de `G`, mapeo de pesos y prueba funcional; `G` se comparte desde buffer/algoritmo canónico y se verifica elemento a elemento; pesos iniciales B/C se transfieren, no se infieren de seeds; calidad corresponde a E01 y equivalencia/bytes/latencia/RAM a E02; 1×64/2×32/4×16/8×8 es la comparación prioritaria de partición bajo 64 conexiones por soma. PASS no valida todavía ninguna hipótesis experimental. **Siguiente tarea exacta: S3-T02 — Implementar Fixed-dANN y verificar equivalencia funcional paired con DSU-S.**
