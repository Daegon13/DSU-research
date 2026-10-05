# PROJECT_STATE

## CURRENT PHASE
Sprint 0 — Fundación experimental

## CURRENT SPRINT
S0

## CURRENT TASK
S0-T01 — Inicializar el repositorio con la documentación base y estructura de investigación.

## LAST COMPLETED TASK
Definición conceptual del proyecto y roadmap inicial.

## LAST RESULT
Se definió una arquitectura de investigación centrada en:
- dendritas estructuradas;
- estado temporal aprendible;
- integración somática;
- compatibilidad con CPU/GPU común;
- evaluación por parámetros, memoria, latencia y calidad.

## CURRENT BEST MODEL
Ninguno. Todavía no hay implementación.

## CURRENT BASELINE
Pendiente: MLP/ReLU convencional.

## IMPORTANT NUMBERS
- Dedicación objetivo: 2–4 h/semana.
- Primer Go/No-Go: 6–8 semanas parciales.
- Objetivo aspiracional posterior para DSU-FFN:
  - >= 4x menos parámetros del bloque sustituido;
  - < 2% de degradación de calidad;
  - >= 1.25–1.5x de mejora de throughput o reducción equivalente de latencia.

## ACTIVE HYPOTHESIS
H-001:
Una unidad con agregación dendrítica estructurada puede alcanzar calidad comparable a una capa densa usando menos parámetros.

## OPEN QUESTIONS
- ¿Qué dataset mínimo usaremos para reproducir primero una arquitectura dendrítica publicada?
- ¿Qué topología de grupos dendríticos será más eficiente en PyTorch?
- ¿La ventaja paramétrica se convierte en ventaja real de memoria y latencia?
- ¿El estado temporal por dendrita aporta capacidad suficiente para justificar su coste?

## KNOWN PROBLEMS
Ninguno todavía.

## BLOCKERS
Ninguno técnico.
El proyecto debe mantenerse como prioridad lateral frente a Facultad, empleo y proyectos principales.

## NEXT TASK
1. Crear repo.
2. Copiar esta documentación.
3. Crear entorno Python.
4. Implementar harness experimental mínimo.
5. No implementar DSU todavía.

## LAST UPDATE
2026-10-04
