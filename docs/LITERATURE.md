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

**Tema:** dendritas artificiales y eficiencia paramétrica.

### Relevancia
Demuestra que conectividad dendrítica estructurada puede igualar o superar redes convencionales en algunos benchmarks usando muchos menos parámetros.

### Uso para DSU
Principal baseline conceptual de DSU-S.

### Pregunta abierta
¿Puede la ventaja convertirse en velocidad y menor tráfico de memoria en hardware convencional?

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
