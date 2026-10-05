# REPRODUCTION_001 — Fashion-MNIST, vANN frente a dANN-R

## Identificación y alcance

**TASK ID:** S1-T01. **Estado:** especificación cerrada; no se han implementado ni ejecutado modelos de Sprint 1.

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

La inicialización de las capas `keras.layers.Dense` carece de argumentos explícitos: por defecto **Glorot uniform** para kernels y **ceros** para sesgos; la máscara pone los pesos no conectados a cero y los gradientes correspondientes se enmascaran antes de Adam. Los pesos de cable son libres, sin normalización ni restricción de signo. Una futura implementación compacta en PyTorch deberá preservar conexiones, función y protocolo, y registrar que su inicialización numérica no equivale automáticamente a Glorot/Keras sobre la matriz densa de origen.

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

**Comando futuro previsto:** `uv run dsu-run --config experiments/configs/reproduction_001_fmnist.json --output runs/reproduction_001/` (interfaz **propuesta**, aún no existe; S1-T02 debe fijar el comando real). El harness actual de Sprint 0 no admite todavía validation, este dataset o ambos modelos. No ejecutar este comando ahora.

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
