# ARCHITECTURE

# 1. DSU-S

La especificación normativa de **DSU-S v0** está en [DSU_S_SPEC.md](DSU_S_SPEC.md). Fija `D=784`, `H=128`, `B=4`, `K=16` exacto por dendrita, topología aleatoria fija por `topology_seed`, LeakyReLU 0.1 en dendritas y somas, y salida lineal de 10 logits. `DSUSv0` almacena sus 10634 parámetros entrenables en tensores compactos y `G` como buffer entero persistente. Su estructura está verificada y el primer benchmark de aprendizaje figura en [EXPERIMENT_001_DSU_S_LEARNING.md](EXPERIMENT_001_DSU_S_LEARNING.md). Sigue sin estado temporal.

## Control Fixed-dANN de Sprint 3

`FixedDANN` es un baseline experimental dense-mask, no la implementación principal. Recibe exactamente la `G` de DSU-S, deriva máscaras `[512,784]` y `[128,512]`, y puede recibir los mismos pesos efectivos mediante `fixed_dann_from_dsu_s`. S3-T02 verificó equivalencia de forward, gradientes y cinco pasos Adam en CPU float32; véase [EXPERIMENT_004_FIXED_DANN_EQUIVALENCE.md](EXPERIMENT_004_FIXED_DANN_EQUIVALENCE.md).

---

# 2. DSU-T

Añade estado independiente por dendrita.

\[
d_{b,t}
=
\lambda_b d_{b,t-1}
+
(1-\lambda_b)\phi(z_{b,t})
\]

donde:

\[
0 \le \lambda_b < 1
\]

y \(\lambda_b\) puede ser:

- fija;
- entrenable;
- compartida;
- individual por rama.

## Estado somático opcional

\[
u_t =
\rho u_{t-1}
+
\sum_b c_b d_{b,t}
\]

\[
y_t = \psi(u_t)
\]

La primera implementación debe probar memoria dendrítica antes de añadir estado somático.

---

# 3. DSU-A — futura

Añade una vía contextual/apical.

\[
a_t = f(c_t, a_{t-1})
\]

\[
g_b = \sigma(q_b a_t)
\]

\[
\tilde d_b = g_b d_b
\]

El soma integra \(\tilde d_b\) en lugar de \(d_b\).

## Uso hipotético

- routing contextual;
- continual learning;
- adaptación durante inferencia;
- separación de tareas.

No implementar antes del gate correspondiente.

---

# 4. Topología

La topología de DSU-S v0 queda fijada en [DSU_S_SPEC.md](DSU_S_SPEC.md). En Sprint 3 se podrá variar, en experimentos separados:

- \(B \in \{2,4,8\}\)
- diferentes valores de \(K\);
- solapamiento vs no solapamiento;
- grupos fijos vs aleatorios;
- permutación entre capas.

---

# 5. Permutación

Entre bloques puede utilizarse:

\[
x' = P x
\]

donde \(P\) es una permutación fija.

No introduce parámetros entrenables.

Objetivo:

evitar que subconjuntos de características permanezcan aislados a lo largo de toda la red.

---

# 6. Restricción de implementación

No representar la arquitectura principal como una matriz densa multiplicada por una máscara llena de ceros, salvo como baseline experimental.

Preferir:

- tensores compactos;
- reshape;
- gather/indexing controlado;
- grouped operations;
- block structure;
- kernels pequeños y regulares.

---

# 7. Principio de diseño

Cada variable biológicamente inspirada debe justificar su coste.

Si una característica aumenta:

- memoria;
- FLOPs;
- latencia;
- complejidad;

sin mejorar un objetivo medible, debe eliminarse.
