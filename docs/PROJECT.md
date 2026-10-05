# PROJECT

## Nombre provisional

**DSU Research — Dendritic Stateful Unit**

El nombre es interno. No implica novedad académica ni disponibilidad legal.

## Problema

Las redes neuronales artificiales modernas suelen utilizar unidades simples y grandes matrices densas.

Esto puede producir:

- grandes cantidades de parámetros;
- alto tráfico de memoria;
- elevada RAM/VRAM;
- inferencia costosa en hardware local.

Las neuronas biológicas, en cambio, realizan computación local dentro de sus dendritas y mantienen estado dinámico.

El proyecto investiga si una abstracción mínima de esas propiedades puede mejorar la eficiencia de modelos artificiales.

## Hipótesis principal

Una unidad formada por:

- múltiples ramas dendríticas;
- conectividad local estructurada;
- integración somática;
- estado temporal aprendible opcional;

puede sustituir grupos de neuronas convencionales con mejor relación entre calidad y recursos.

## Objetivos

### Objetivo 1
Reproducir resultados relevantes de arquitecturas dendríticas publicadas.

### Objetivo 2
Construir DSU-S, una unidad espacial sin memoria.

### Objetivo 3
Medir si existe una ventaja de parámetros o memoria.

### Objetivo 4
Construir DSU-T añadiendo estado temporal sólo si DSU-S lo justifica.

### Objetivo 5
Optimizar la estructura para hardware convencional.

### Objetivo 6
Sólo con evidencia previa: sustituir el FFN de un Transformer pequeño.

## No objetivos iniciales

- construir hardware;
- fabricar chips;
- simular electroquímica completa;
- Hodgkin–Huxley detallado;
- memristores;
- STDP completo;
- neurotransmisores artificiales;
- LLM de gran escala;
- spikes como requisito inicial.

## Criterio científico

La investigación debe ser falsable.

No basta con demostrar que DSU aprende.

Hay que demostrar si aporta una ventaja frente a baselines razonables y por qué.

## Criterio de inversión

Cada nueva fase debe ganarse el derecho a existir mediante resultados de la fase anterior.

## Restricciones reales

Proyecto individual.

Recursos principales:

- PC doméstica;
- Python;
- PyTorch;
- datasets públicos;
- herramientas de profiling;
- tiempo limitado.

El diseño debe respetar estas restricciones desde el inicio.
