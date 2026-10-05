# ARCHITECTURE

# 1. DSU-S

## Entrada

Sea:

\[
x \in \mathbb{R}^{D}
\]

Cada soma posee \(B\) dendritas.

Cada dendrita observa un subconjunto:

\[
G_b \subset \{1,\dots,D\}
\]

con:

\[
|G_b| = K,\quad K \ll D
\]

## Cálculo dendrítico

\[
z_b = W_b x_{G_b} + b_b
\]

\[
d_b = \phi(z_b)
\]

## Integración somática

\[
s = \psi\left(\sum_{b=1}^{B} c_b d_b + b_s\right)
\]

## Propiedades

- sin estado;
- diferenciable;
- compatible con backpropagation;
- conectividad estructurada;
- múltiples ramas por soma;
- coste controlado.

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

La primera familia de pruebas debería variar:

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
