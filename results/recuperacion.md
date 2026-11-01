### Recuperación (33 preguntas con respuesta)

| Estrategia | hit@1 | hit@5 | recall@5 | MRR@10 | p50 (ms) |
|---|---|---|---|---|---|
| vector | 0.73 | 0.85 | 0.82 | 0.80 | 48 |
| texto | 0.18 | 0.48 | 0.44 | 0.29 | 3 |
| hibrido | 0.45 | 0.85 | 0.80 | 0.61 | 53 |
| hibrido+rerank | 0.67 | 0.88 | 0.85 | 0.76 | 29472 |

### hit@5 por tipo de pregunta

| Estrategia | factual | parafraseada | multi |
|---|---|---|---|
| vector | 1.00 | 0.58 | 1.00 |
| texto | 0.67 | 0.17 | 0.67 |
| hibrido | 1.00 | 0.58 | 1.00 |
| hibrido+rerank | 1.00 | 0.67 | 1.00 |
