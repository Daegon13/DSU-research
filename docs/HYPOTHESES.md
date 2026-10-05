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
Una unidad con varias dendritas estructuradas puede alcanzar calidad comparable a un MLP convencional usando menos parámetros.

### Reason
Cada dendrita procesa sólo un subconjunto de las entradas y realiza agregación local antes del soma.

### Experiment
Comparar MLP vs DSU-S con:
- mismo dataset;
- mismo protocolo;
- igualdad aproximada por calidad;
- igualdad aproximada por parámetros.

### Prediction
DSU-S necesitará menos parámetros para alcanzar una calidad similar.

---

## H-002 — Sparsity estructurada > sparsity aleatoria

**Status:** UNTESTED

### Statement
La agrupación estructurada y regular de entradas será más eficiente que sparsity aleatoria de densidad equivalente.

### Reason
La estructura regular debería favorecer acceso de memoria, vectorización y kernels agrupados.

### Experiment
Comparar:
- grupos fijos;
- grupos aleatorios;
- máscara sparse irregular;
- operaciones block/grouped.

### Prediction
La versión estructurada tendrá latencia menor o más estable.

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
