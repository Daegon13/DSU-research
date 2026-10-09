# S3-T04 / S3-E02 — Coste físico de Fixed-dANN y DSU-S

**Fecha:** 2026-10-09. **Gate técnico: PASS.** Pregunta: al representar exactamente la misma función con la misma topología, ¿qué cuestan la representación dense-mask y la compacta en almacenamiento y ejecución? No se entrenó Fashion-MNIST, no se modificaron arquitecturas ni se optimizaron kernels.

## Protocolo y equipo

Comando: `.venv/Scripts/python.exe -m dsu_research.physical_benchmark --output runs/physical_s3_t04.json`. Código y resultados de esta ejecución: `src/dsu_research/physical_benchmark.py` y el JSON local en `runs/` (ignorado por Git). Seed de modelo/topología `20261009`; seed de inputs sintéticos `104`; SHA-256 de `G`: `bab8e7f02d3a46d0b4c7ee953fd182f8b6f407565d9076be11a52076c4446ec9`. CPU AMD Ryzen 5 5600G with Radeon Graphics, 12 núcleos lógicos; Windows 11 build 26200; Python 3.12.13; PyTorch 2.14.1+cpu; CPU float32; CUDA no disponible. PyTorch: 6 threads intraop y 6 interop originales. Se midieron regímenes separados de 1 y 6 threads intraop, iguales para ambos modelos dentro de cada régimen. `platform.processor()` expone `AMD64 Family 25 Model 80 Stepping 0`; el nombre comercial se verificó con `Get-CimInstance Win32_Processor`.

`fixed_dann_from_dsu_s` copió `G`, los pesos activos y los sesgos de una DSU-S inicializada. Antes de cada par de mediciones se confirmó `G` idéntica y `torch.testing.assert_close(rtol=1e-5, atol=1e-6)` sobre los mismos inputs; error máximo inicial de logits para batch 128: **2.24e-8**. En forward y forward+backward se repitió la comprobación después. Los pasos Adam modifican pesos, por lo que cada modo/batch utiliza una pareja fresca. El contraste de gradientes y cinco pasos Adam de S3-T02 permanece documentado en [EXPERIMENT_004_FIXED_DANN_EQUIVALENCE.md](EXPERIMENT_004_FIXED_DANN_EQUIVALENCE.md).

Se precalienta cada llamada cinco veces. `torch.utils.benchmark.Timer.blocked_autorange(min_run_time=0.15)` se repite tres veces por modelo, modo, batch y régimen; la mediana reportada es la mediana de esas tres medianas. IQR se calcula sobre las observaciones combinadas de Timer. `num_threads` se fija explícitamente en Timer. Los inputs y targets se generan antes de cronometrar. Forward usa `eval()` e `inference_mode()`. Forward+backward hace `zero_grad(set_to_none=True)`, forward, CrossEntropy y backward, sin paso de optimizer. El paso Adam añade `optimizer.step()` y precalienta con estado ya inicializado. No se ejecutaron epochs.

## Almacenamiento medido

Todos los bytes de tensores se derivan de `numel() * element_size()`; los buffers son persistentes en `state_dict`. Checkpoints serializados con el mismo `torch.save` en `BytesIO`. Adam usa `lr=0.001` y un paso con el mismo batch sintético 128 para crear estados. Las diferencias en checkpoint incluyen cabeceras y metadatos de serialización.

| Medida | Fixed-dANN | DSU-S | Fixed / DSU |
|---|---:|---:|---:|
| Parámetros entrenables | 468874 | 10634 | 44.092× |
| Bytes de parámetros | 1875496 | 42536 | 44.092× |
| Buffers, elementos | 475136 | 8192 | 57.999× |
| Bytes de buffers | 532480 | 65536 | 8.125× |
| Payload tensorial del modelo | 2407976 | 108072 | 22.281× |
| Checkpoint sólo modelo | 2411273 | 110929 | 21.737× |
| Adam `exp_avg` | 1875496 | 42536 | 44.092× |
| Adam `exp_avg_sq` | 1875496 | 42536 | 44.092× |
| Adam otros tensores (`step`, seis escalares float32) | 24 | 24 | 1.000× |
| Estado tensorial Adam total | 3751016 | 85096 | 44.080× |
| Payload conocido modelo + estado Adam | 6158992 | 193168 | 31.884× |
| Checkpoint modelo + optimizer | 6166930 | 200536 | 30.752× |

Fixed persiste `G` 65536 bytes, `input_mask` 401408 bytes y `soma_mask` 65536 bytes. DSU-S persiste `G` 65536 bytes. Las máscaras booleanas de Fixed y su `G` duplican información topológica por diseño del control existente. Los momentos Adam de Fixed tienen el tamaño completo de matrices densas aun cuando las posiciones inactivas permanecen en cero; DSU-S guarda estados sólo para parámetros compactos existentes. El payload conocido no incluye gradientes, activaciones, allocator ni memoria del intérprete y **no es RAM pico**.

## Trabajo matemático

Ambos tienen **9984 MAC principales efectivos por muestra**: 8192 dendríticos + 512 somáticos + 1280 de salida. Fixed ejecuta multiplicaciones de matrices densas de tamaños `784×512 + 512×128 + 128×10 = 468224 MAC principales/muestra`; DSU-S computa 9984 MAC principales compactos/muestra. Cociente **46.897×**. Este conteo omite gather/indexación, máscaras, activaciones, reducciones, tráfico de memoria y backward; **cociente de MAC ≠ cociente de latencia**. No se midieron FLOPs reales de kernels.

## Latencia CPU

Cada celda de tiempo es `mediana [IQR]` en ms por batch. La última columna es latencia Fixed / DSU: mayor que 1 favorece DSU. Latencia por muestra = ms/batch dividido por batch; throughput = `1000 × batch / ms/batch`. Los valores por muestra, throughput, tres medianas independientes y cantidad de mediciones están en el JSON reproducible.

| Threads | Modo | Batch | Fixed ms | DSU ms | Fixed / DSU |
|---|---|---:|---:|---:|---:|
| 1 | Forward | 1 | 0.400 [0.110] | 0.094 [0.008] | 4.245 |
| 1 | Forward | 32 | 0.814 [0.288] | 1.106 [0.116] | 0.736 |
| 1 | Forward | 128 | 1.924 [0.148] | 4.522 [0.794] | 0.425 |
| 1 | Forward | 512 | 5.829 [0.954] | 23.119 [8.448] | 0.252 |
| 1 | Forward+backward | 32 | 2.214 [0.555] | 2.055 [0.701] | 1.078 |
| 1 | Forward+backward | 128 | 5.134 [0.672] | 6.250 [1.087] | 0.821 |
| 1 | Adam step | 128 | 7.283 [1.443] | 7.646 [1.502] | 0.953 |
| 6 | Forward | 1 | 0.277 [0.019] | 0.116 [0.007] | 2.381 |
| 6 | Forward | 32 | 0.480 [0.023] | 0.512 [0.017] | 0.937 |
| 6 | Forward | 128 | 0.782 [0.075] | 1.527 [0.257] | 0.512 |
| 6 | Forward | 512 | 2.076 [0.190] | 6.818 [0.333] | 0.304 |
| 6 | Forward+backward | 32 | 1.227 [0.108] | 1.102 [0.100] | 1.114 |
| 6 | Forward+backward | 128 | 2.151 [0.190] | 2.641 [0.207] | 0.815 |
| 6 | Adam step | 128 | 4.036 [0.452] | 3.862 [0.389] | 1.045 |

En forward batch 1, DSU es materialmente más rápida. Batch 32: Fixed es materialmente más rápido con 1 thread y similar con 6. Batches 128/512: Fixed es materialmente más rápido en ambos regímenes. Forward+backward batch 32 es similar con 1 thread y ligeramente favorable a DSU con 6; batch 128 favorece Fixed. El paso Adam batch 128 es similar en ambos. Las clases usan umbrales descriptivos internos: ratio ≥1.10 favorable a DSU, 0.90–1.10 similar, ≤0.90 favorable a Fixed; no son pruebas estadísticas formales. **No hay una conclusión global de velocidad** porque el signo cambia con batch y modo.

## Memoria temporal y límites

El forward actual de DSU-S materializa `selected = x[:, G.reshape(-1)]` y lo reorganiza a `[batch,128,4,16]`: 32768, 1048576, 4194304 y 16777216 bytes float32 para batch 1, 32, 128 y 512. La vista `reshape` puede compartir ese almacenamiento; multiplicación y reducción pueden generar otros temporales. Fixed materializa activaciones densas principales `[batch,512]` (2048, 65536, 262144, 1048576 bytes) y `[batch,128]` (512, 16384, 65536, 262144 bytes); máscaras aplicadas a pesos también pueden crear temporales. Estos tamaños son teóricos por shape, no pico medido. **CPU peak runtime memory = NOT MEASURED RELIABLY**: no había una medición RSS aislada y fiable disponible; `tracemalloc` no contabiliza adecuadamente las asignaciones nativas de PyTorch. No hay GPU en este host. No se afirma ahorro de RAM runtime ni energía.

Los tiempos son específicos de este CPU, versión, implementación PyTorch y estado del sistema; no implican ventaja de entrenamiento completo. El cuello de botella plausible en batch grande es gather/indexación y materialización temporal, reducciones pequeñas y la ventaja de GEMM denso optimizado. No se aisló causalmente cada coste ni se aplicaron optimizaciones post hoc. La siguiente investigación de rendimiento, si se autoriza, deberá perfilar esos componentes por separado. El benchmark actual mide implementación existente.

**H-012 storage: SUPPORTED** para este par B/C: bajan payload tensorial, checkpoint de modelo, estado Adam y checkpoint de entrenamiento. **H-013 runtime: MIXED / NOT SUPPORTED como ventaja general**: forward batch 1 favorece claramente a DSU y forward+backward batch 32 con 6 threads la favorece levemente; batches grandes de forward favorecen a Fixed. La ventaja de parámetros no predice velocidad. **Siguiente tarea exacta:** S3-T05 — diseñar y ejecutar la ablación de compartimentalización dendrítica con 64 conexiones por soma (`1×64`, `2×32`, `4×16`, `8×8`). No se inicia aquí.
