# Política de seguridad

## Versiones soportadas

Este es un proyecto personal. Solo se mantiene la rama `main`.

## Cómo reportar una vulnerabilidad

**No abras un issue público.** Usa el reporte privado de GitHub:
[Report a vulnerability](https://github.com/MatteoMurcia/estatuto-rag/security/advisories/new)
(pestaña *Security* → *Report a vulnerability*).

Incluye, si puedes:

- Descripción del problema y su impacto.
- Pasos para reproducirlo (o una prueba de concepto).
- Versión o commit afectado.

Intentaré responder en un plazo de 7 días y, si se confirma, publicar la corrección
y el aviso de seguridad correspondiente dando crédito a quien lo reportó (si lo desea).

## Alcance

- El código de este repositorio (`src/`, `scripts/`, `.github/`).
- Las credenciales de `docker-compose.yml`, `.env.example` y la CI son de desarrollo
  local (`estatuto/estatuto`) y no protegen nada real: no son una vulnerabilidad.
- El texto del Estatuto de los Trabajadores procede del BOE y es público.
