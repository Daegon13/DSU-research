# EXPERIMENT_003 — Búsqueda de capacidad MLP (S2-T06)

**Fecha:** 2026-10-06. **Pregunta:** dentro de una familia MLP densa fija de dos capas ocultas, ¿cuántos parámetros se necesitan para alcanzar aproximadamente la calidad de validation de DSU-S v0?

## Protocolo y objetivos fijados antes del search

DSU-S no se reentrena. Sus métricas de la época 25 en `runs/dsu_s_v0_s2_t04/dsu_s_seed{1,2,3}.json` son:

| Seed | Validation loss | Validation accuracy |
|---:|---:|---:|
| 1 | 0.315036451 | 88.500 % |
| 2 | 0.324821134 | 88.633 % |
| 3 | 0.321523801 | 88.333 % |
| Media | **0.320460462** | **88.488889 %** |

Se define **accuracy match** como media MLP ≥ **87.988889 %** (DSU menos 0.50 pp), **loss match** como media MLP ≤ **0.340460462** (DSU más 0.020), y **full quality match** como ambos simultáneamente. Son tolerancias heurísticas internas.

Familia: `784→h1→round(17/13*h1)→10`, LeakyReLU 0.1 después de cada capa oculta, logits sin softmax. Escalera prefijada: `13→17` reutilizado de S2-T05, seguido por `16→21`, `19→25`, `22→29`, `25→33`, `28→37`, `32→42`. Las seeds son 1, 2 y 3. El primer full match termina la escalera gruesa, salvo un candidato siguiente como sanity check. Se prueban después todos los `h1` enteros entre último fallo y primer pass; se elige el full match de menor número de parámetros.

Fashion-MNIST oficial: train/validation de 54000/6000, split y orden por seed iguales a S2-T04/S2-T05, `ToTensor()` float32/255, Adam lr 0.001, betas 0.9/0.999, epsilon 1e-7, batch 128, 25 épocas, CrossEntropyLoss y estado **final_epoch_25**. La búsqueda carga sólo train y validation; no crea el dataset test. Guarda los `state_dict` finales por seed/candidato en `runs/mlp_capacity_s2_t06/`, ignorado por Git. No hay selección de checkpoint por mínimo de validation, early stopping ni ajuste de hiperparámetros por tamaño.

Entorno: CPU AMD Ryzen 5 5600G, 33 686 745 088 bytes de RAM física reportados por Windows, Windows 11 (build 26200), PyTorch 2.14.1+cpu, torchvision 0.29.1+cpu, NumPy 2.5.3, float32. Cada seed se aplica a Python, NumPy y PyTorch; CUDA no se usó. Los artefactos locales incluyen 18 historias de 25 épocas y 18 checkpoints finales; todas las métricas de validation inspeccionadas fueron finitas. No se midió RAM pico del proceso.

Comando: `.venv/Scripts/python.exe -m dsu_research.capacity_search search`. El manifiesto `runs/mlp_capacity_s2_t06/search_manifest.json` se actualiza tras cada candidato completo. Una selección se registra primero en `frozen_selection.json`; sólo entonces `.venv/Scripts/python.exe -m dsu_research.capacity_search test` evalúa test exclusivamente en la arquitectura elegida y sus estados finales guardados. Si no hay full match hasta `32→42`, test permanece cerrado.

## Búsqueda en validation y selección congelada antes de test

Las métricas siguientes son de **validation en la época 25**, media ± DE muestral sobre seeds 1–3. El baseline `13→17` se reutilizó de S2-T05 y no se reentrenó. Cada candidato nuevo completó sus tres seeds antes de decidir. Los conteos de parámetros fueron verificados con el modelo instanciado, no inferidos únicamente de la tabla prevista.

| h1→h2 | Parámetros | Validation loss | Validation accuracy | Accuracy match | Loss match | Full match | Función |
|---|---:|---:|---:|---|---|---|---|
| 13→17 | 10623 | 0.367635 ± 0.015837 | 87.250 ± 0.627 % | No | No | No | S2-T05 reutilizado |
| 16→21 | 13137 | 0.345326 ± 0.015126 | 87.839 ± 0.761 % | No | No | No | Escalera |
| 19→25 | 15675 | 0.344374 ± 0.010663 | 87.833 ± 0.404 % | No | No | No | Escalera; último fallo |
| 22→29 | 18237 | 0.332518 ± 0.018729 | 88.239 ± 0.629 % | Sí | Sí | Sí | Escalera; primer pass |
| 25→33 | 20823 | 0.325550 ± 0.015100 | 88.444 ± 0.597 % | Sí | Sí | Sí | Único sanity check |
| 20→26 | 16516 | 0.336997 ± 0.012175 | 88.167 ± 0.765 % | Sí | Sí | Sí | Refinamiento; menor pass |
| 21→27 | 17359 | 0.338806 ± 0.019519 | 88.022 ± 0.611 % | Sí | Sí | Sí | Refinamiento |

La escalera se detuvo tras `22→29` y el sanity check `25→33`. El intervalo último fallo/primer pass fue `h1=19/22`; se probaron **todos** los enteros intermedios `h1=20,21`. La relación calidad/tamaño no fue perfectamente monótona (por ejemplo, `20→26` tuvo mejor loss medio que `21→27`), algo esperable en N=3; la selección se basó sólo en el criterio prefijado y el menor número de parámetros.

**SELECTED_MLP: `784→20→26→10`. PARAMETERS: 16516. PARAMETER_RATIO_VS_DSU: 1.553131465. MAC_RATIO_VS_DSU: 1.648637821.** MAC principales MLP: **16460/muestra** frente a **9984/muestra** DSU-S. Esta decisión fue registrada en `runs/mlp_capacity_s2_t06/frozen_selection.json` y en el manifiesto con fase `frozen_before_test`; desde este punto la arquitectura está **CONGELADA**. Ningún resultado de test intervino en la búsqueda o selección.

## Test final del modelo seleccionado

Después de escribir la selección congelada, se ejecutó `.venv/Scripts/python.exe -m dsu_research.capacity_search test`. Se cargaron únicamente los tres `state_dict` finales de `20→26`, sin reentrenamiento. `runs/mlp_capacity_s2_t06/selected_test.json` contiene las métricas completas. Los hashes de train/validation del seleccionado coinciden con DSU-S seed a seed.

| Seed | MLP test loss | MLP test accuracy | DSU-S test loss | DSU-S test accuracy | Δ loss MLP−DSU | Δ accuracy MLP−DSU |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.373320 | 86.81 % | 0.357590 | 87.18 % | +0.015730 | −0.37 pp |
| 2 | 0.379388 | 86.48 % | 0.355466 | 87.56 % | +0.023922 | −1.08 pp |
| 3 | 0.374621 | 86.81 % | 0.360419 | 87.02 % | +0.014202 | −0.21 pp |
| **Media ± DE muestral** | **0.375776 ± 0.003194** | **86.700 ± 0.191 %** | **0.357825 ± 0.002485** | **87.253 ± 0.277 %** | **+0.017952 ± 0.005227** | **−0.553 ± 0.463 pp** |

Los deltas agregados son estadísticas de las **diferencias pareadas**, no restas de dispersiones independientes. El MLP alcanzó el criterio interno en validation, pero su test accuracy quedó 0.553 pp por debajo de DSU-S; la ventaja de test de DSU-S tiene la misma dirección en las tres seeds. El umbral de match se fijó para validation y no se reinterpretó usando test.

## Eficiencia, interpretación y límites

El seleccionado tiene **16516 parámetros entrenables y almacenados**, **66064 bytes** de tensores de pesos float32, **16460 MAC principales/muestra** y **522.05 s** de train+validation sumando las tres seeds (150.28, 161.31 y 210.46 s). El search completo, sin el baseline reutilizado, sumó **2782.68 s** de train+validation. Estos tiempos no son una comparación controlada de latencia ni velocidad frente a DSU-S.

DSU-S tiene **10634 parámetros entrenables**, **42536 bytes** de pesos, **65536 bytes** de índices de topología y **9984 MAC principales/muestra**. El payload de pesos más topología de DSU-S es **108072 bytes**, superior a los 66064 bytes de pesos del MLP; por tanto, el resultado apoya eficiencia en **parámetros entrenables** y MAC principales teóricos dentro de esta comparación, pero **no** una ventaja del payload tensorial total de DSU-S frente al MLP seleccionado. MAC principales omiten gather, activaciones, tráfico de memoria, backward y overhead de ejecución.

Dentro de la familia MLP densa predefinida de dos capas ocultas y protocolo fijo, hicieron falta aproximadamente **1.55×** los parámetros entrenables de DSU-S para alcanzar los criterios de calidad de validation establecidos. La primera arquitectura gruesa que pasó fue `22→29`; el refinamiento encontró `20→26`. El resultado refuerza **H-008 = SUPPORTED dentro de este alcance**: el MLP casi igualado de S2-T05 quedó por debajo, el match de validation apareció a mayor tamaño y DSU-S mantuvo ventaja descriptiva de test en los tres pares. El sanity check `25→33` también pasó.

N=3 limita la precisión de la estimación y no constituye prueba de significancia. El search cubre sólo una familia y una relación fija entre anchos; no identifica el mejor MLP posible ni aísla causalmente el efecto dendrítico de inicialización, topología, cantidad de activaciones u otras diferencias. El match es heurístico: otra tolerancia puede cambiar el tamaño mínimo. No se demuestran menor RAM de ejecución, menor latencia, menor energía, FLOPs exactos, ventaja de almacenamiento total o generalización a otros datasets. No se hizo tuning posterior ni se evaluó test de candidatos descartados.

**Gate técnico S2-T06: PASS.** Test estuvo cerrado durante search, se respetaron escalera, sanity check y refinamiento, el seleccionado se congeló antes del test y se registraron resultados positivos y negativos. **Siguiente tarea exacta: S2-REVIEW — Auditar los resultados completos de Sprint 2 y decidir si existe evidencia suficiente para avanzar a Sprint 3.** Sprint 2 no queda cerrado automáticamente y Sprint 3 no se inicia aquí.
