# LITERATURE

## Propósito

Este archivo no es una bibliografía decorativa.

Cada paper debe responder:

- qué pregunta resolvió;
- qué mecanismo utilizó;
- qué resultados obtuvo;
- qué podemos reutilizar;
- qué sigue sin responder para DSU.

---

## 1. Beniaguev, Segev & London — Single cortical neurons as deep artificial neural networks

**Tema:** complejidad computacional de una neurona biológica.

### Relevancia
Mostró que reproducir el comportamiento de una neurona piramidal cortical detallada requiere una red artificial temporal profunda, sugiriendo que la neurona biológica individual posee capacidad computacional interna considerable.

### Uso para DSU
Justifica explorar unidades más ricas que una neurona puntual.

### No demuestra
Que copiar más biología mejore automáticamente una IA.

---

## 2. Chavlis & Poirazi — Dendritic artificial neural networks

**Referencia:** Spyridon Chavlis y Panayiota Poirazi, “Dendrites endow artificial neural networks with accurate, robust and parameter-efficient learning”, *Nature Communications* **16**, 943 (2025), DOI [10.1038/s41467-025-56297-9](https://doi.org/10.1038/s41467-025-56297-9). [Texto completo](https://pmc.ncbi.nlm.nih.gov/articles/PMC11754790/); [código y datos oficiales](https://github.com/Poirazi-Lab/dendritic_anns), revisión `fadc3846174bc7a67b284a76b0989ed2cef2767e` inspeccionada para S1-T01.

**Tema:** conectividad dendrita→soma y muestreo restringido de entradas en clasificación de imágenes.

### Pregunta y método

¿Mejora la calidad y la eficiencia en parámetros una red con dendritas frente a una red completamente conectada de las mismas dimensiones ocultas? Para Fashion-MNIST, la dANN-R toma entradas aleatorias y conecta cada dendrita sólo a su soma. La vANN conecta todas las entradas a la primera capa y todos los nodos de la primera capa a la segunda. Ambas tienen dos capas ocultas, activaciones LeakyReLU (pendiente 0.1) y salida softmax. El paper compara también otras variantes que quedan fuera de S1-T01.

### Resultados publicados relevantes

La Fig. 2 muestra pérdida y accuracy de test frente a parámetros efectivos, con cinco inicializaciones por configuración. Su leyenda declara intervalos de confianza del 95 %. La Tabla 1 reporta los **mejores modelos de cada familia** en Fashion-MNIST, no la pareja fija de nuestra reproducción: dANN-R, accuracy **89.612 ± 0.0870 %** y pérdida **0.3245 ± 0.0028**; vANN, **89.288 ± 0.3654 %** y **0.4040 ± 0.0066** (media ± desviación estándar, N=5). No usar esos números como objetivo numérico de una configuración distinta.

El archivo oficial `DATA.zip`, `DATA/results_fmnist_1_layer/output_all_final.pkl`, contiene para la pareja fija **D=4 dendritas/soma, S=128 somas**: dANN-R **86.830 ± 0.184 %** y pérdida **0.365028 ± 0.003595**, vANN **89.112 ± 0.621 %** y pérdida **0.401243 ± 0.020207** (media ± desviación estándar muestral, N=5; accuracy convertido a porcentaje). Corresponden a las cinco pruebas etiquetadas 1–5 y 25 épocas almacenadas. La comparación esperada para esta pareja es **menor pérdida** de dANN-R, no mayor accuracy.

### Qué reutilizamos y qué no demuestra

Reutilizamos la comparación controlada vANN/dANN-R, el dataset y el protocolo de entrenamiento. La cifra de parámetros de dANN-R cuenta conexiones activas; el código oficial materializa matrices densas enmascaradas, así que la reducción de parámetros efectivos no demuestra reducción de memoria, FLOPs, latencia ni energía. Tampoco demuestra ninguna propiedad de DSU. Las diferencias entre artículo y código, y la configuración reproducible exacta, están en [REPRODUCTION_001.md](REPRODUCTION_001.md).

---

## 3. Parametric / learnable LIF

**Tema:** constantes temporales entrenables.

### Relevancia
Permitir que la dinámica temporal neuronal sea aprendida puede mejorar eficiencia y rendimiento en SNN.

### Uso para DSU
Inspiración para \(\lambda_b\) entrenable por dendrita.

---

## 4. Closed-form Continuous-time Networks (CfC)

**Tema:** dinámica temporal continua eficiente.

### Relevancia
Muestra que se pueden modelar dinámicas temporales ricas sin integrar ODE costosas paso a paso.

### Uso para DSU
Inspiración para mantener estados temporales baratos.

---

## 5. Multi-compartment spiking neurons

**Tema:** múltiples compartimentos neuronales.

### Relevancia
Resultados recientes muestran ventajas de neuronas multicompartimento frente a modelos puntuales en algunas tareas temporales y de visión.

### Uso para DSU
Apoya estudiar estados independientes por rama.

---

## 6. Active dendrites / contextual modulation

**Tema:** modulación contextual dendrítica.

### Relevancia
Permite crear subredes dependientes del contexto y reducir interferencia entre tareas.

### Uso para DSU
Base conceptual de una futura DSU-A.

### Gate
No implementar antes de demostrar DSU-S y DSU-T.

---

## 7. Neuromorphic computing / Loihi

**Tema:** cómputo orientado a eventos.

### Relevancia
Demuestra que sparsity temporal y eventos pueden producir enormes ventajas con hardware adecuado.

### Uso para DSU
Posible fase futura.

### Restricción
No asumir que ventajas en Loihi se trasladan a GPU/CPU.

---

## 8. Research gaps que nos interesan

Buscar trabajos que combinen simultáneamente:

- dendritas estructuradas;
- estado temporal por rama;
- ejecución eficiente en CPU/GPU;
- reducción explícita de memory traffic;
- sustitución de FFN;
- benchmarks de latencia batch 1;
- lenguaje autoregresivo.

Antes de reclamar novedad debe realizarse una búsqueda bibliográfica formal y actualizada.

---

# Plantilla para nuevos papers

## TITLE

**Authors:**  
**Year:**  
**DOI/URL:**

### Question

### Method

### Main result

### What we can reuse

### What it does not answer

### Relevance to DSU

### Notes
