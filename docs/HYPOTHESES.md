# HYPOTHESES

Estados permitidos:

- UNTESTED
- SUPPORTED
- WEAK SUPPORT
- REFUTED
- INCONCLUSIVE

---

## H-001 — Eficiencia espacial dendrítica

**Status:** UNTESTED

### Statement
Una arquitectura dendrítica compacta puede mantener rendimiento comparable al baseline dANN-R de igual anchura usando muchos menos parámetros físicamente almacenados; comparaciones posteriores con MLP de presupuestos igualados evaluarán la ventaja más general.

### Reason
Cada dendrita procesa sólo un subconjunto de las entradas y realiza agregación local antes del soma.

### Experiment
S2-T04: comparar vANN, dANN-R dense-mask y DSU-S v0 compact en Fashion-MNIST con el mismo split y protocolo cuando sea posible, inicialmente N=3 seeds. S2-T05/T06: comparaciones posteriores igualadas por parámetros y por calidad.

### Prediction
DSU-S almacenará 10634 parámetros entrenables frente a 468874 de dANN-R y vANN; la similitud de calidad sigue por probar. El coste de índices se reportará aparte.

---

## H-002 — Utilidad física de la conectividad regular

**Status:** UNTESTED

### Statement
Una estructura compacta de fan-in fijo puede convertir la sparsity dendrítica en una reducción física de almacenamiento y un cómputo estructural utilizable.

### Reason
DSU-S v0 almacena sólo pesos conectados y 16 índices por dendrita; dANN-R almacena matrices densas enmascaradas. La indexación puede añadir coste y debe medirse.

### Experiment
S2-T04: medir bytes de parámetros, bytes de topología, bytes de checkpoint, RAM y latencia reales frente a dANN-R dense-mask, junto con calidad. La comparación de distintas topologías y kernels queda para Sprint 3/5, con variables controladas.

### Prediction
DSU-S reducirá almacenamiento de pesos entrenables; el resultado de latencia queda abierto. Menos parámetros o MAC teóricos no garantizan aceleración.

---

## H-003 — Permutación entre capas evita islas

**Status:** UNTESTED

### Statement
Una permutación barata de características entre capas aumentará capacidad sin necesidad de conectividad global densa.

### Experiment
DSU-S con y sin permutación.

### Prediction
La permutación mejorará calidad a coste computacional muy pequeño.

---

## H-004 — Memoria dendrítica mejora tareas temporales

**Status:** UNTESTED

### Statement
Permitir que cada dendrita mantenga estado temporal independiente mejora tareas secuenciales frente a DSU-S.

### Experiment
Comparar:
- DSU-S;
- DSU-T con lambda fija;
- DSU-T con lambda aprendible;
- RNN/GRU;
- modelos temporales relevantes.

### Prediction
DSU-T obtendrá mejor relación calidad/parámetros en tareas temporales.

---

## H-005 — Lambda aprendible > lambda fija

**Status:** UNTESTED

### Statement
Constantes temporales aprendibles permitirán que diferentes ramas representen distintas escalas de memoria.

### Prediction
Mejor calidad o igual calidad con menor tamaño.

---

## H-006 — Menos parámetros no implica necesariamente menor latencia

**Status:** UNTESTED

### Statement
Una DSU puede reducir parámetros sin ser más rápida en PyTorch si su patrón de acceso a memoria es ineficiente.

### Purpose
Esta hipótesis existe para prevenir conclusiones erróneas.

---

## H-007 — DSU puede sustituir un FFN pequeño

**Status:** UNTESTED

### Statement
Una DSU suficientemente madura puede sustituir parcial o totalmente el FFN de un Transformer pequeño manteniendo calidad comparable con menos memoria.

### Gate
No probar hasta completar los milestones previos.
