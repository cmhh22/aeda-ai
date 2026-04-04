# Módulo de Ingesta: Conceptos Clave (Explicado Fácil)

Este documento te explica, en lenguaje simple, qué son **analito**, **censura** y **QA/QC**, por qué son importantes y en qué parte de la ingesta se usan.

---

## 1) ¿Qué es un analito?

Un **analito** es la sustancia que quieres medir en la muestra.

Ejemplos en tu dataset ISOVIDA:
- `Pb_(ppm)` (plomo)
- `Cr_(ppm)` (cromo)
- `V_(ppm)` (vanadio)
- `Fe_(%)` (hierro)

Piensa así:
- **Muestra** = el “recipiente” (sedimento)
- **Analito** = lo que medimos dentro de ese recipiente (metales, elementos)

---

## 2) ¿Qué es censura (`<LOD`, `>LOQ`, `±`)?

La **censura** ocurre cuando el laboratorio no puede reportar un valor exacto normal.

Casos típicos:
- `<LOD` o `< 20`: está por debajo del límite de detección
- `>LOQ` o `> 1000`: está por encima del límite de cuantificación
- `91 ± 16`: valor con incertidumbre asociada

¿Qué significa esto en la práctica?
- El dato no está “mal”, pero necesita una regla para volverse usable en análisis.

---

## 3) ¿Qué es QA/QC?

**QA/QC** = Aseguramiento y Control de Calidad de datos.

En términos simples:
- **QA**: reglas para prevenir problemas (cómo deben venir los datos)
- **QC**: chequeos para detectar problemas (qué hacemos si hay faltantes, outliers, etc.)

Ejemplos de reglas QA/QC:
- Máximo de faltantes permitido por columna
- Regla para tratar outliers
- Qué hacer con duplicados, blancos o estándares
- Criterio de aprobación/rechazo del dataset

---

## 4) ¿Por qué hace falta saber esto?

Porque estas decisiones cambian el resultado final.

Si dos personas usan reglas distintas para `<LOD`:
- Persona A usa `LOD/2`
- Persona B usa `LOD`

Entonces cambian:
- medias y medianas
- correlaciones
- clusters
- modelos
- conclusiones científicas

Por eso hay que fijar reglas únicas y documentarlas.

---

## 5) ¿En qué parte de la ingesta impacta cada cosa?

## Etapa A: Lectura de archivo (`RAW`)

Archivo relevante:
- `src/ingestion/raw_data_ingestor.py`

Aquí se detecta:
- texto especial (`<`, `>`, `±`)
- columnas de analitos
- columnas de incertidumbre (`U_*`)

Si esta etapa falla, todo lo demás se contamina.

---

## Etapa B: Parsing y normalización

Archivo relevante:
- `src/ingestion/raw_data_ingestor.py`

Aquí se convierte:
- `"< 20"` -> valor numérico con bandera de calidad
- `"91 ± 16"` -> valor central + incertidumbre separada

También se crean indicadores de calidad por columna.

---

## Etapa C: Manejo de censura

Archivo relevante:
- `src/ingestion/censored_value_handler.py`

Aquí decides la regla oficial:
- `lod_half`
- `ros`
- `qmle`
- `percentile`

Esta es una etapa crítica para reproducibilidad.

---

## Etapa D: Unidades y conversiones

Archivo relevante:
- `src/ingestion/raw_data_ingestor.py`

Aquí se unifican unidades para que todo sea comparable.

Ejemplo:
- `Fe_(%)` y `Pb_(ppm)` no se interpretan igual si no defines unidad objetivo.

---

## Etapa E: QA/QC y reporte

Archivo relevante:
- `src/ingestion/data_quality_reporter.py`

Aquí se reporta:
- faltantes
- censura tratada
- banderas de parsing
- recomendaciones

Este reporte es lo que te ayuda a defender metodología frente al tutor/tribunal.

---

## 6) Ejemplo real (ISOVIDA)

Caso:
- `Pb_(ppm) = 91`
- `U_Pb_(ppm) = ± 16`

Interpretación correcta:
- valor central = `91`
- incertidumbre = `16` en columna separada
- no se mezclan en una sola celda para modelado

Otro caso:
- `Cr_(ppm) = < 20`

Necesita regla acordada:
- `LOD/2` -> `10`
- o `LOD` -> `20`

Si no acuerdas esto, no hay reproducibilidad.

---

## 7) Decisiones mínimas que debes cerrar con tu tutor

1. Regla oficial para `<LOD` por analito
2. Regla oficial para `>LOQ` por analito
3. Uso oficial de incertidumbre `±` y columnas `U_*`
4. Unidad final oficial por analito
5. Umbrales QA/QC de aprobación

---

## 8) Plantilla corta para la reunión

"Necesito cerrar 5 decisiones para reproducibilidad del módulo de ingesta:"

- Regla `<LOD`:
- Regla `>LOQ`:
- Tratamiento de `±` / `U_*`:
- Unidad oficial por analito:
- Criterios QA/QC de aprobación:

---

## 9) Regla de oro

Sin reglas cerradas de censura + unidades + QA/QC,
el pipeline puede correr bien técnicamente,
pero la conclusión científica puede ser inconsistente.
