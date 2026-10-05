# DSU Research

**Dendritic Stateful Unit — research project**

Proyecto experimental para investigar si una unidad neuronal artificial con:

- agregación dendrítica local,
- conectividad estructurada,
- estado temporal aprendible,
- integración somática,

puede ofrecer una mejor relación entre **calidad, parámetros, memoria y latencia** que capas densas convencionales ejecutadas en hardware común.

## Objetivo

La pregunta principal del proyecto es:

> ¿Puede una Dendritic Stateful Unit (DSU) proporcionar mejor calidad por parámetro y por byte de memoria que capas densas convencionales, manteniendo una inferencia eficiente en CPU/GPU?

Una pregunta posterior, sólo si la hipótesis inicial funciona, será:

> ¿Puede una DSU sustituir parcial o totalmente el FFN de un Transformer pequeño?

## Estado actual

El proyecto está en fase de fundación experimental.

**Prioridad:** proyecto lateral de investigación.  
**Dedicación recomendada:** 2–4 horas por semana.  
**Primer Go/No-Go:** después de 6–8 semanas parciales de trabajo.

## Documentación principal

- `PROJECT_STATE.md` — fuente de verdad operacional.
- `docs/PROJECT.md` — visión, alcance y objetivos.
- `docs/HYPOTHESES.md` — hipótesis registradas y su estado.
- `docs/ARCHITECTURE.md` — especificación técnica de DSU-S, DSU-T y futuras variantes.
- `docs/ROADMAP.md` — sprints, gates y milestones.
- `docs/EXPERIMENT_PROTOCOL.md` — reglas para que los experimentos sean comparables.
- `docs/BENCHMARKS.md` — baselines y métricas.
- `docs/LITERATURE.md` — mapa inicial del estado del arte.
- `docs/DECISIONS.md` — decisiones arquitectónicas.
- `docs/RESULTS.md` — resultados positivos, negativos e inconclusos.
- `templates/SESSION_TEMPLATE.md` — plantilla de cierre de sesión.
- `templates/EXPERIMENT_TEMPLATE.md` — plantilla para cada experimento.

## Regla principal

No escalar por entusiasmo.

Escalar por evidencia.

```text
Hipótesis
   ↓
Experimento pequeño
   ↓
Medición
   ↓
Ablación
   ↓
Decisión
```

## Sprint 0: ejecutar el baseline

Requiere Python 3.10+ y conexión para la primera descarga de MNIST. Con `uv`:

```powershell
uv sync --extra test
uv run pytest -q
uv run dsu-run --config experiments/configs/mnist_mlp_smoke.json
```

Alternativa con `pip`:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[test]"
.venv\Scripts\python -m pytest -q
.venv\Scripts\dsu-run --config experiments/configs/mnist_mlp_smoke.json
```

Cada ejecución escribe un JSON con timestamp UTC en `runs/` (ignorado por Git). Para una ruta fija, agregar `--output runs/mi_experimento.json`. El config, la seed y el comando permiten repetir el experimento. Para repetir con las mismas versiones, usar el `uv.lock` con `uv sync --locked --extra test`. El dataset se guarda automáticamente en `data/`.

El smoke test usa los primeros 1024 ejemplos del split oficial de entrenamiento y los primeros 256 del split oficial de test de MNIST. Son subconjuntos fijos de diagnóstico, sin validation ni búsqueda de hiperparámetros; no se debe inferir calidad final de ellos. El orden de entrenamiento depende de la seed. La latencia usa batch 1, 5 warm-ups y 20 repeticiones. Los tiempos dependen del hardware; el modo determinista puede reducir velocidad.

Estructura funcional:

```text
experiments/configs/        configuración versionada
src/dsu_research/config.py  carga y validación
src/dsu_research/data.py    dataset y splits
src/dsu_research/model.py   baseline MLP/ReLU
src/dsu_research/harness.py entrenamiento, evaluación y métricas
src/dsu_research/run.py     CLI y exportación JSON
tests/                      verificaciones sin red
```

Sprint 0 sigue abierto hasta revisar sus gates; no hay DSU implementada.
