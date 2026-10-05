# REPRODUCTION_001 — Fashion-MNIST, vANN frente a dANN-R

## Identificación y alcance

**TASK ID:** S1-T01 (protocolo), S1-T02 (implementación), S1-T03 (comparación) y S1-T04 (informe y gate). **Estado:** REPRODUCED para la pareja fija D=4, S=128 con N=5; ver R-003 en `docs/RESULTS.md`.

**Referencia:** Chavlis, S. y Poirazi, P. (2025), “Dendrites endow artificial neural networks with accurate, robust and parameter-efficient learning”, *Nature Communications* 16, 943. [DOI](https://doi.org/10.1038/s41467-025-56297-9), [texto completo](https://pmc.ncbi.nlm.nih.gov/articles/PMC11754790/), [repositorio oficial](https://github.com/Poirazi-Lab/dendritic_anns/tree/fadc3846174bc7a67b284a76b0989ed2cef2767e). Código inspeccionado en `fadc3846174bc7a67b284a76b0989ed2cef2767e`; datos en `DATA.zip` de esa revisión. La tabla fija de datos permite una comparación numérica concreta, aunque el ZIP no incluye una garantía verificable de la revisión exacta que generó cada ejecución.

**Pregunta:** En una pareja de redes con igual anchura de ambas capas ocultas, ¿dANN-R obtiene menor pérdida de test con menos conexiones entrenables efectivas que vANN sobre Fashion-MNIST? Es una prueba de la dirección del efecto de la Fig. 2b para **una sola configuración**, no la reproducción de la figura completa ni una afirmación de superioridad de accuracy.

## Arquitectura fijada

| Componente | vANN | dANN-R |
|---|---|---|
| Entrada | Imagen 28×28 a vector de 784 float32 | Igual |
| Capa oculta 1 | Dense 784→512, todas las conexiones | 512 dendritas = **4 por cada uno de 128 somas**; 8192 conexiones de entrada en total, distribuidas aleatoriamente entre las 784×512 posiciones |
| Capa oculta 2 | Dense 512→128, todas las conexiones | Cada grupo contiguo de 4 dendritas alimenta exclusivamente a su soma: 512 pesos de cable y 128 sesgos somáticos |
| Salida | Dense 128→10, completamente conectada, softmax | Igual |
| Activación oculta | LeakyReLU, pendiente negativa 0.1 tras cada capa | Igual, tras dendrita y soma |
| Regularización | Sin dropout, sin early stopping | Igual |

En el **texto** del paper, cada dendrita recibe 16 entradas. En el **código** de dANN-R, `make_masks()` llama a `random_connectivity(..., conns=16×D×S)`, que elige **8192 posiciones globales sin reemplazo**; no garantiza 16 por dendrita. Por tanto, 16 es el promedio de sinapsis por dendrita, con variación e incluso posibles dendritas sin entrada. La máscara dendrita→soma sí es fija y estructurada. Mantener esta semántica del código es necesario para reproducir el resultado archivado; imponer exactamente 16 entradas por dendrita sería otro experimento.

**Conteos esperados** (incluyen sesgos y salida):

- vANN: `(784+1)×512 + (512+1)×128 + (128+1)×10 = 468874` pesos y sesgos almacenados y entrenables.
- dANN-R: `(8192+512) + (512+128) + (128+1)×10 = 10634` conexiones y sesgos **efectivos entrenables**, 44.09× menos que vANN. El modelo Keras oficial almacena **468874** entradas en sus matrices densas, incluidas las enmascaradas; la cifra 10634 no representa su memoria física ni sus FLOPs reales.

La inicialización de las capas `keras.layers.Dense` carece de argumentos explícitos: por defecto **Glorot uniform** para kernels y **ceros** para sesgos; la máscara pone los pesos no conectados a cero y los gradientes correspondientes se enmascaran antes de Adam. Los pesos de cable son libres, sin normalización ni restricción de signo. La implementación PyTorch de S1-T02 aplica Glorot uniform explícitamente sobre matrices densas, aunque no reproduce las mismas muestras aleatorias que Keras.

## Dataset y entrenamiento

| Campo | Protocolo operativo del código oficial |
|---|---|
| Dataset | `keras.datasets.fashion_mnist`: 60000 imágenes train oficiales, 10000 test oficiales; 10 clases |
| Preprocesamiento | `float32`, división por 255, aplanado a 784; sin estandarización ni augmentación; `noise=True` con sigma 0.0, sin cambio de valores |
| Split | `np.random.default_rng(trial)` baraja los 60000 índices; últimos 6000 para validation, restantes 54000 para train; test separado; **mismo split entre vANN y dANN-R para cada trial**, pero distinto entre trials |
| Seeds | `trial=1,2,3,4,5` en el barrido; `keras.utils.set_random_seed(trial)` para inicialización; `default_rng(trial)` independiente para split y máscara; `tf.data.Dataset.shuffle` baraja train cada época, sin seed explícita en esa llamada |
| Optimizador | Adam, lr 0.001, beta1 0.9, beta2 0.999; restantes defaults Keras, sin scheduler, clipping ni weight decay explícitos |
| Pérdida | Sparse categorical cross-entropy sobre probabilidades softmax (`from_logits=False`); etiquetas enteras |
| Batch | 128; último batch incompleto se conserva; train rebarajado en cada época |
| Épocas | **25** en `codes_/main.py` y 25 filas por trial en el archivo de resultados, sin early stopping |
| Evaluación | Loss y accuracy de train/validation por época; test loss y accuracy **al final de la época 25**; no se restaura el mínimo de validation |

Para esta configuración fija no hay selección de checkpoint ni ajuste por validation; se registra el mínimo de validation como diagnóstico. El barrido de la Fig. 2 usa D∈{1,2,4,8,16,32,64}, S∈{32,64,128,256,512} y cinco trials; la Tabla 1 selecciona la mejor configuración de cada familia **por accuracy de test**, una práctica distinta de esta reproducción predefinida. No seleccionar nuestro modelo por test.

## Resultado publicado/archivado contra el que compararemos

Fuente numérica primaria para la pareja: `DATA.zip` → `DATA/results_fmnist_1_layer/output_all_final.pkl` → DataFrame `testing`, filtros `model∈{vanilla_ann,dend_ann_random}`, `num_dends=4`, `num_soma=128`, `sigma=0.0`. Valores después de 25 épocas:

| Modelo | Parámetros efectivos | Test accuracy, media ± DE (5) | Test loss, media ± DE (5) |
|---|---:|---:|---:|
| vANN | 468874 | 89.112 ± 0.621 % | 0.401243 ± 0.020207 |
| dANN-R | 10634 | 86.830 ± 0.184 % | 0.365028 ± 0.003595 |

La pérdida de dANN-R es 0.036215 menor (≈9.0 %) a pesar de tener menos parámetros efectivos; su accuracy es 2.282 puntos porcentuales menor. La Tabla 1 del artículo informa los **mejores de toda la familia** (dANN-R 89.612 ± 0.0870 %, loss 0.3245 ± 0.0028; vANN 89.288 ± 0.3654 %, loss 0.4040 ± 0.0066) y sirve sólo de contexto, no de objetivo para D=4, S=128. Las DE anteriores son muestrales (`ddof=1`) calculadas de los cinco valores archivados.

## Minimum Reproduction Experiment para S1-T02

1. Implementar **sólo** vANN y dANN-R de la tabla, sin DSU ni variantes RF. Usar la misma configuración de Fashion-MNIST, train/validation/test, máscaras, optimizer y evaluación; no ajustar hiperparámetros para mejorar resultados.
2. Ejecutar primero trials **1, 2 y 3** para el smoke científico. Después queda prevista la extensión **4 y 5**, con la misma configuración y sin repetir tuning, para comparar N=5 con el archivo oficial.
3. Registrar por trial: índices o hash del split, máscara o hash, seeds, versiones, hardware, configuración completa, pérdidas y accuracies por época y en test, conteos efectivos y almacenados, duración y memoria si se puede medir. Comparar **pérdida de test media** como efecto primario y accuracy como límite de interpretación.
4. Comparar las medias observadas contra la fila fija de arriba; mostrar diferencia absoluta y dispersión, sin aplicar los números de Tabla 1 a este par. La comparación es por **misma anchura oculta**, no por igualdad de parámetros, FLOPs o tiempo.

**Criterio S1-T02:** `PASS` si la implementación y datos cumplen la especificación, las tres ejecuciones completan 25 épocas con métricas válidas, los conteos coinciden y la media de test loss de dANN-R es menor que la de vANN; reportar magnitud y divergencia numérica frente al archivo oficial, sin atribuir equivalencia estadística a N=3. `PARTIAL` si el protocolo se ejecuta y los conteos coinciden pero la dirección del efecto no se sostiene; investigar y registrar, sin tuning oportunista. `FAIL` si no se logra una ejecución válida o la arquitectura/split/protocolo difieren materialmente. Extender a N=5 antes de reclamar reproducción completa del valor publicado.

**Comando real de S1-T02:** ver la sección de implementación abajo. El harness se extendió para admitir validation y Fashion-MNIST sin cambiar el recorrido del baseline de Sprint 0.

**Coste estimado:** seis entrenamientos de 25 épocas × 54000 ejemplos = 8.1 millones de ejemplos vistos para tres trials. En vANN, sólo el forward denso equivale aproximadamente a `3×25×54000×(784×512+512×128+128×10) ≈ 1.9×10¹²` MAC; backward, evaluación y la implementación concreta agregan coste. Es un **cálculo de operaciones**, no una medición de tiempo, RAM, FLOPs exactos o energía en el Ryzen 5 5600G CPU. Medir antes de prometer duración; el código oficial dANN-R ejecuta matrices densas y máscaras, así que su coste físico no cae 44×.

## Diferencias y límites de las fuentes

| Asunto | Paper | Código/datos oficiales | Decisión operativa |
|---|---|---|---|
| Épocas FMNIST | 20 en Methods | `main.py`: 25; 25 filas/ejecución en `output_all_final.pkl` | **25** para cotejo numérico con archivo; registrar discrepancia. Una corrida 20 épocas sería una reproducción textual distinta, sin el mismo objetivo numérico. |
| Entradas por dendrita | 16 | `random_connectivity` elige `16×D×S` aristas globales | Replicar **16 de promedio** según código; no forzar 16 exactas. |
| Parámetros | Número de conexiones efectivas | `Dense` completos con valores/gradientes enmascarados; `utils.num_trainable_params()` cuenta activos | Reportar ambos conteos; no inferir aceleración ni ahorro físico del Keras original. |
| Inicialización | No especifica distribución | Defaults de `Dense`: Glorot uniform y bias cero; se enmascara tras inicializar | Registrar semántica y diferencias de framework en S1-T02. |
| Selección | Fig. 2 y Tabla 1 resumen barrido | Test final tras 25 épocas; Tabla 1 escoge máximo de accuracy de test entre configuraciones | Fijar D=4/S=128 **antes** de entrenar; comparar con su fila archivada. |
| Split y semillas | 90/10 y 5 inicializaciones | Split barajado con `default_rng(trial)`; trials 1–5; train shuffle sin seed explícita | Repetir split por trial y documentar aleatoriedad restante. |

**Fuentes de código inspeccionadas:** `codes_/main.py`, `codes_/run_all_fig2.sh`, `codes_/opt.py` (`get_data`, `make_masks`, `get_model`, `custom_train_loop`), `codes_/receptive_fields.py` (`random_connectivity`, `connectivity`), `codes_/utils.py` (`num_trainable_params`), `codes_/analysis_model_evaluation.py`, `codes_/figure_2.py`; todas en la revisión enlazada. La documentación S1-T01 no modifica baselines ni resultados de Sprint 0.

## Implementación S1-T02

Comando real: `.venv/Scripts/python.exe -m dsu_research.run --config experiments/configs/reproduction_001_fmnist.json --output runs/reproduction_001 --seeds 1 2 3`. La lista de seeds admite luego `1 2 3 4 5` sin modificar código. Cada pareja por seed usa exactamente el mismo split. Los JSON individuales y el agregado se guardan en `runs/reproduction_001/` (ignorado por Git). El test usa siempre los pesos al final de la época 25; se registra el mínimo de validation sólo como diagnóstico.

`VanillaANN` y `DendriticANNRandom` almacenan 468874 parámetros. La segunda conserva matrices densas para reproducir la semántica original. Su máscara de entrada elige 8192 posiciones sin reemplazo con `np.random.default_rng(seed).choice(784*512, 8192, replace=False)` en orden input-major, y la máscara del cable conecta cuatro dendritas contiguas a cada soma. La máscara vive como buffer no entrenable; los pesos inactivos se ponen a cero al inicializar y el forward vuelve a multiplicar por ella, por lo que los gradientes inactivos son cero. El conteo efectivo se obtiene sumando las máscaras y los sesgos y pesos de salida; no se sustituye el conteo de PyTorch.

Ambas capas `Linear` reciben inicialización Glorot uniform y sesgos cero explícitos. La primera diferencia inevitable es que, aun con la misma seed numérica, el generador aleatorio de PyTorch no produce los mismos pesos que TensorFlow/Keras. La segunda es que el shuffle por época usa un `torch.Generator` explícito y DataLoader, mientras que el original usa `tf.data.Dataset.shuffle` sin seed local explícita. Se fijan Adam beta1=0.9, beta2=0.999, epsilon=1e-7 y weight decay=0 para acercarse a los defaults de Keras; el detalle numérico de implementación de Adam puede diferir entre frameworks. PyTorch devuelve logits y usa cross-entropy directa; Keras devuelve softmax y usa SparseCategoricalCrossentropy sobre probabilidades. Matemáticamente son equivalentes salvo diferencias de redondeo y clipping interno.

El campo `training.duration_seconds` del harness incluye las evaluaciones de validation al final de cada época y excluye test y benchmark de inferencia. Por ello `training.samples_per_second` divide sólo muestras de train por un tiempo que incluye validation; sirve como throughput del protocolo completo, no como velocidad pura del paso de entrenamiento. El benchmark de inferencia es batch 1, cinco warm-ups y 20 repeticiones, igual al harness S0. La RAM pico de CPU y el tamaño de checkpoint siguen sin medirse; la configuración no guarda pesos de checkpoints, pero sí registra `final_epoch_25` como estado usado en test.

### Auditoría de ejecución N=3

Las seis corridas registradas en `runs/reproduction_001/` tienen 25 filas de historia. Cada pareja usa hashes idénticos de índices train y validation y máscaras distintas entre trials. Las tres máscaras tienen exactamente 8192 conexiones de entrada, 512 de cable y 10634 parámetros efectivos incluyendo sesgos y salida. La pérdida de test de dANN-R es menor en las tres parejas. El resultado cuantitativo completo y las divergencias de referencia están en R-002. El objetivo primario se cumple para N=3, sin afirmar equivalencia estadística con los cinco trials archivados.

## Cierre S1-T03/S1-T04 — N=5

Se ejecutaron **sólo** seeds 4 y 5 con la misma configuración y el mismo comando de S1-T02, cambiando únicamente `--seeds 4 5`. Los seis JSON de seeds 1–3 no se reentrenaron. `runs/reproduction_001/summary_n3.json` conserva el agregado anterior y `summary.json` contiene ahora el agregado N=5 con deltas pareados. Ambos son artefactos locales ignorados por Git. La tabla completa por trial, los agregados y la comparación con la referencia están en R-003 de `docs/RESULTS.md`.

### Auditoría de seeds y precisión

- `seed_everything(seed)` fija `random.seed`, `np.random.seed`, `torch.manual_seed` y CUDA cuando está disponible; estas corridas CPU registran CUDA como `null`. Split y máscara usan sendos `np.random.default_rng(seed)`; el `DataLoader` de train recibe `torch.Generator().manual_seed(seed)` y baraja cada época. La inicialización Glorot usa el generador de PyTorch después de fijar la seed.
- Prueba directa sin entrenamiento con secuencia **1, 2, 1**: las muestras de Python, NumPy global, `default_rng`, PyTorch, los pesos iniciales de ambos modelos, la máscara y los primeros 12 índices barajados cambiaron entre 1 y 2 y fueron idénticos al repetir 1. Los hashes de pesos iniciales vANN fueron `a1f71a3b5308d16d` y `9e1f0480450ecd1b`; los de máscara, `7472cbf6a53e0d10` y `f4d4c849b2ad2386`. Los primeros 12 índices del DataLoader fueron `[24,78,89,4,15,61,29,94,48,22,56,63]` y `[1,55,90,66,22,2,70,10,86,26,39,33]`.
- En los diez JSON: cinco hashes distintos de train, cinco de validation y cinco de máscara; los hashes de train/validation coinciden dentro de cada pareja. Cada historia tiene 25 épocas y `final_epoch_25`; split 54000/6000/10000, 25×54000 ejemplos vistos, parámetros y configuración esperados. `test.accuracy` se guarda como cociente de aciertos sobre 10000, sin redondeo previo. En N=3, vANN registró 8943, 8943 y 8944 aciertos: su DE de **0.006 puntos porcentuales** se debe a esos conteos cercanos, no a pérdida de precisión en JSON. Su loss sí varió entre seeds; no se ha determinado una causa adicional de la baja variación de accuracy.

### Gate de reproducción

**REPRODUCED** para esta configuración y el comportamiento cualitativo: arquitectura y protocolo suficientemente fieles, cinco trials completos, rango de rendimiento comparable, dANN-R con menor test loss **en promedio**, menor accuracy en las cinco parejas y 44.09× menos parámetros/conexiones efectivos. La diferencia de loss individual se invierte en seed 4 (`+0.002618` dANN-R menos vANN); se conserva como resultado negativo. No hay discrepancia grave conocida en arquitectura, máscara, split o criterio de evaluación. La dirección media coincide con el archivo oficial, pero el tamaño de la ventaja de loss es menor y las medias no son idénticas. N=5 sólo permite una lectura descriptiva, no una afirmación fuerte de significancia.

**Diferencias conocidas con Keras y el paper:** el paper dice 20 épocas y 16 entradas por dendrita, mientras que los resultados oficiales de esta fila usan 25 épocas y 16 entradas **en promedio** por máscara global. PyTorch y TensorFlow no generan los mismos pesos con igual número de seed; `DataLoader` y `tf.data.Dataset.shuffle` producen órdenes distintos. Adam y la cross-entropy sobre logits pueden diferir numéricamente de la implementación Keras con softmax y pérdida sobre probabilidades. El ZIP oficial no garantiza qué revisión exacta produjo cada trial. Estas diferencias ayudan a explicar la divergencia numérica, sin aislar causalmente cada contribución.

**Demostrado:** reducción de conectividad y parámetros efectivos de dANN-R frente a vANN de igual anchura y la dirección de loss/accuracy para esta pareja fija de Fashion-MNIST. **No demostrado:** superioridad de DSU, ventaja general frente a MLP, reducción de memoria física, FLOPs, latencia o energía, ni rendimiento de una futura arquitectura. Ambos modelos almacenan 468874 parámetros densos; 10634 sólo cuenta entradas activas y sesgos de dANN-R.
