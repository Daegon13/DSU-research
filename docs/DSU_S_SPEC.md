# DSU-S v0 — especificación S2-T01

**Estado:** especificación cerrada; implementación estructural verificada; primer benchmark de aprendizaje completado en [EXPERIMENT_001_DSU_S_LEARNING.md](EXPERIMENT_001_DSU_S_LEARNING.md). **Alcance:** una sola capa dendrítica espacial sobre Fashion-MNIST, seguida de salida lineal. DSU-S v0 usa exactamente la anchura oculta de la pareja fija de `REPRODUCTION_001`: 512 dendritas y 128 somas. No es una variante de dANN-R ni una reproducción de su máscara.

## Definición matemática

Convención: índices de entrada `i ∈ {0,…,783}`, soma `s ∈ {0,…,127}`, dendrita local `b ∈ {0,…,3}` y conexión local `k ∈ {0,…,15}`. Para cada ejemplo `x ∈ ℝ^784`, `G[s,b]` contiene **exactamente 16 índices distintos** de entrada. Cada conjunto se muestrea uniformemente sin reemplazo dentro de esa dendrita; conjuntos de dendritas diferentes son independientes y pueden solaparse. `G` es fijo durante todo el entrenamiento, no recibe gradientes y se genera con `topology_seed` explícita, separada de `training_seed`. Se registrarán ambas seeds y la topología o un hash de ella. Cambiar `topology_seed` constituye otra topología experimental, incluso si se conserva la seed de entrenamiento.

Con `φ(z) = LeakyReLU(z, negative_slope=0.1)`:

```text
d[s,b] = φ(Σ(k=0..15) W_d[s,b,k] · x[G[s,b,k]] + bias_d[s,b])
h[s]   = φ(Σ(b=0..3)  W_s[s,b]   · d[s,b]       + bias_s[s])
logits[o] = Σ(s=0..127) W_o[o,s] · h[s] + bias_o[o],  o ∈ {0,…,9}
```

Cada soma recibe sólo sus cuatro dendritas. La salida tiene 10 logits y **no** aplica softmax dentro del modelo; la futura pérdida puede usar cross-entropy sobre logits, como la reproducción PyTorch.

| Tensor | Shape conceptual | Función |
|---|---:|---|
| `x` | `[batch, 784]` | Entrada aplanada |
| `G` | `[128, 4, 16]` | Índices fijos, no entrenables |
| `x[G]` | `[batch, 128, 4, 16]` | Vista conceptual de gather |
| `W_d` | `[128, 4, 16]` | Pesos compactos de entrada |
| `bias_d` / `d` | `[128, 4]` / `[batch, 128, 4]` | Sesgos y salidas dendríticas |
| `W_s` | `[128, 4]` | Cuatro pesos propios por soma |
| `bias_s` / `h` | `[128]` / `[batch, 128]` | Sesgos y salidas somáticas |
| `W_o` / `bias_o` | `[10, 128]` / `[10]` | Proyección de salida y sesgo |
| `logits` | `[batch, 10]` | Salida sin softmax |

## Conteos independientes y almacenamiento

| Componente | Cálculo | Entrenables |
|---|---:|---:|
| Entrada→dendrita | `128 × 4 × 16` | 8192 |
| Sesgos dendríticos | `128 × 4` | 512 |
| Dendrita→soma | `128 × 4` | 512 |
| Sesgos somáticos | `128` | 128 |
| Soma→salida | `128 × 10` | 1280 |
| Sesgos de salida | `10` | 10 |
| **Total** | `8192 + 512 + 512 + 128 + 1280 + 10` | **10634** |

En DSU-S v0, `stored_trainable_parameters = effective_trainable_parameters = 10634`. Con float32, los tensores entrenables contienen **42536 bytes** (`10634 × 4`), sin contar gradientes, estado del optimizador, cabeceras ni asignaciones temporales.

La topología tiene `128 × 4 × 16 = 8192` índices, que son buffers o metadata, **no** parámetros entrenables. Si se guardan como enteros de 64 bits, ocupan **65536 bytes**; como enteros de 32 bits, **32768 bytes**. El tamaño concreto depende de la representación que admita la futura implementación. Los bytes de tensores del modelo serían, para float32 e índices int64, `42536 + 65536 = 108072` bytes; con índices int32, `75304` bytes. Estos son conteos nominales de payload, no mediciones de RAM ni tamaño de archivo. Si la topología se regenera determinísticamente desde `topology_seed`, el checkpoint puede omitir el tensor `G`, pero la topología debe existir durante la ejecución y su coste residente debe medirse. No se fija aún una optimización de checkpoint.

Las futuras métricas distinguirán **trainable parameter bytes**, **topology/index bytes**, **total model tensor bytes**, **checkpoint bytes** y **runtime memory**. Para las tres últimas se reportarán medidas reales de la implementación; el estado de Adam, gradientes, activaciones y `gather` pueden dominar la RAM de entrenamiento.

## Cómputo estructural y límite de interpretación

| Tramo | MAC principales por muestra |
|---|---:|
| Dendritas | `512 × 16 = 8192` |
| Somas | `128 × 4 = 512` |
| Salida | `128 × 10 = 1280` |
| **Total** | **9984** |

El vANN denso de la reproducción requiere `784×512 + 512×128 + 128×10 = 468224` MAC principales por muestra; el cociente teórico es `468224 / 9984 ≈ 46.90`. Se excluyen sesgos, activaciones, indexación, movimiento de datos y backward. Este cociente **no** estima la latencia, throughput, RAM ni energía de PyTorch. S2-T04 medirá el coste real de gather/indexing, reducciones y salida.

## Restricciones para S2-T02

La implementación almacenará los pesos conectados directamente en tensores compactos. El flujo conceptual es `input → indexed gather → [batch,128,4,16] → reducción ponderada → [batch,128,4] → reducción somática → [batch,128] → salida`. Este flujo no obliga a materializar cada tensor intermedio: esa decisión se medirá después. Quedan prohibidos como implementación principal `Linear(784,512) + mask`, matrices densas equivalentes llenas de ceros, parámetros muertos y pesos enmascarados que sigan almacenados. Se añadirán tests de forma, conectividad, conteo y forward/backward al implementar la unidad.

DSU-S v0 no contiene memoria temporal, `lambda`, estado recurrente, spikes, gating apical, contexto, atención, plasticidad local, topología aprendible, pruning, integración Transformer ni múltiples capas DSU. No se introducen optimizaciones complejas en esta especificación.

## Relación con dANN-R y primer experimento

`dANN-R` de `REPRODUCTION_001` elige **8192 posiciones globales sin reemplazo** de una matriz 784×512. Por ello tiene 16 entradas **en promedio** por dendrita, con fan-in variable. Usa matrices densas enmascaradas para entrada y cable: almacena 468874 parámetros aunque sólo 10634 sean efectivos. DSU-S v0 fija 16 entradas distintas **en cada** dendrita y almacena 10634 parámetros entrenables compactos. Conserva 512 dendritas, 128 somas, cuatro dendritas por soma, LeakyReLU 0.1 y salida 128→10. Los dos cambios —distribución local de conectividad y almacenamiento físico— se declararán juntos al interpretar la primera comparación; no se atribuirá una diferencia de calidad o velocidad sólo a uno de ellos sin una ablación posterior.

S2-T04 comparará inicialmente **vANN**, **dANN-R dense-mask** y **DSU-S v0 compact** sobre Fashion-MNIST con **N=3 seeds**, usando el mismo split y protocolo de `REPRODUCTION_001` cuando sea razonablemente posible. Las parejas compartirán split por seed; se registrarán por separado `training_seed` y `topology_seed` de DSU-S y sus hashes. Cualquier diferencia de inicialización o tratamiento de seeds se declarará antes de interpretar resultados. La comparación inicial iguala anchura oculta y protocolo de datos, **no** presupuesto de parámetros almacenados, MAC, tiempo o energía. No se ejecuta en S2-T01.

Una señal prometedora para S2-T04 sería aprendizaje estable, rendimiento cercano a dANN-R y sólo 10634 parámetros entrenables físicamente almacenados. El resultado de calidad ya consta en el informe S2-T04; la velocidad real requiere medición controlada posterior. Estos criterios de interpretación experimental no son un gate arbitrario de implementación.

## IMPLEMENTATION STATUS: IMPLEMENTED / VERIFIED (S2-T02/T03)

`DSUSv0` almacena seis tensores entrenables compactos y `G` como buffer `int64` persistente en `state_dict`; al cargar, el buffer guardado reemplaza la topología generada en el constructor. El modelo acepta imágenes que aplana a `[batch,784]` y devuelve diez logits. Cada fila de `G` se genera con `np.random.default_rng(topology_seed).choice(784,16,replace=False)`; el generador es local e independiente de la seed de entrenamiento. El hash SHA-256 de `G` se registra en las métricas estructurales.

La inicialización es Glorot uniform explícita para `W_d` como matriz `[512,16]`, `W_s` como `[128,4]` y `W_o` como `[10,128]`; los tres sesgos empiezan en cero. Esta decisión mantiene la familia de inicialización usada en los baselines, pero sus dimensiones compactas producen límites distintos de la matriz densa enmascarada de dANN-R. No se ajustó para mejorar accuracy.

Verificado estáticamente: `stored = trainable = effective = 10634`; bytes entrenables float32 `42536`; `G` tiene 8192 índices `torch.int64` y `65536` bytes; payload de tensores `108072` bytes; MAC principales `8192 + 512 + 1280 = 9984` por muestra. No son medidas de RAM, FLOPs totales ni latencia. Los tests inspeccionan shapes de parámetros y buffers, conectividad real, matemática del forward, gradientes, serialización y un batch sintético por el harness. La suite CPU pasó; CUDA queda sin ejecutar en un host sin GPU. Fashion-MNIST y S2-T04 se documentan en el informe de aprendizaje.
