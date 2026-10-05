# BENCHMARKS

## Fase inicial

El objetivo no es usar datasets enormes.

Queremos experimentos baratos, rápidos y diagnósticos.

## Baselines mínimos

### Baseline A
MLP/ReLU convencional.

### Baseline B
MLP igualado aproximadamente por parámetros.

### Baseline C
MLP igualado aproximadamente por FLOPs.

### Baseline D
Arquitectura dendrítica publicada reproducida.

## Fase temporal

Añadir:
- RNN;
- GRU;
- LIF/PLIF si corresponde;
- CfC si corresponde.

## Fase Transformer

Comparar:
- FFN convencional;
- DSU-FFN.

Mantener:
- atención;
- tokenizer;
- dataset;
- entrenamiento;
- tamaño aproximado del resto del modelo;

idénticos cuando sea posible.

## Métricas obligatorias

### Calidad
- accuracy;
- loss;
- F1 si aplica;
- perplexity en lenguaje.

### Tamaño
- parámetros entrenables;
- parámetros totales;
- checkpoint en MB.

### Memoria
- RAM pico;
- VRAM pico;
- estado persistente.

### Velocidad
- training samples/s;
- inference samples/s;
- ms/muestra;
- tokens/s.

### Computación
- FLOPs/MACs aproximados;
- cantidad de operaciones especializadas.

## Métricas derivadas

- calidad/parámetro;
- calidad/MB;
- calidad/latencia;
- calidad/FLOP;
- calidad/byte movido cuando podamos estimarlo.

## Criterio de señal interesante

Para el primer Go/No-Go no exigimos velocidad superior.

Buscamos:
- reducción clara de parámetros;
- calidad comparable;
- reproducibilidad;
- explicación razonable.

## Objetivo aspiracional posterior

En DSU-FFN:

- >= 4x menos parámetros;
- < 2% degradación relativa de calidad;
- >= 1.25–1.5x mejora de throughput o latencia equivalente;
- >= 3x reducción de memoria del bloque sustituido.

No son garantías ni requisitos de la fase inicial.
