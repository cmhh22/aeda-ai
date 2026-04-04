# Plan Operativo - ISOVIDA Rectificada + Respuesta del Tutor

Este documento convierte la respuesta del tutor en acciones directas sobre la base rectificada `BD_ISOVIDA_MANGLARES2023_rectificadaYBA_230326.xlsx`.

## 1) Qué cambió en la base rectificada (impacto en Módulo 1)

- Hoja principal de trabajo: `DATA`.
- Hojas nuevas de soporte metodológico: `Diccionario_DATA` y `Contexto`.
- La columna de analitos fue simplificada (ej. `V` en lugar de `V_(ppm)`).
- La incertidumbre ahora está en columnas explícitas `U_*`.
- En `DATA` no se observaron símbolos `±`, `<` ni `>` en celdas de analitos.

Implicación práctica:
- El parser de notación sigue disponible para robustez, pero en la base rectificada el foco principal pasa a QA/QC, unidades oficiales y trazabilidad.

## 2) Traducción de las 5 decisiones del tutor a reglas del pipeline

1. Censura analítica:
- Decisión del tutor para la base rectificada: LOD/LOQ no se incluyen en el trabajo del Módulo 1.
- El tratamiento de LOD/LOQ queda fuera de alcance operativo y se considera trabajo del investigador.
- Implementación en el pipeline rectificado: `apply_censored_handling = false`.

2. Unidades y normalización:
- Definir unidad canónica por analito con `Diccionario_DATA` como fuente oficial.
- Regla inicial: elementos mayores (`Na`, `Mg`, `Al`, `Si`, `K`, `Ca`, `Fe`) en `%`; trazas en `ppm`.

3. QA/QC:
- Umbrales operativos iniciales cargados en `config/params.yaml`:
	- `max_missing_per_column_pct = 30%`
	- `max_missing_per_row_pct = 35%`
	- Outliers: `flag_and_review`
- Estos umbrales son de trabajo y deben quedar validados por tutor.

4. Diccionario de datos:
- Usar `Diccionario_DATA` como contrato semántico del dataset.
- Cualquier columna fuera de diccionario debe quedar en log de validación.

5. Objetivo científico prioritario:
- Valor por defecto para esta fase: `gradient_by_depth`.
- Si cambia el objetivo científico, actualizar `config/params.yaml` en `isovida_rectificada.objective_priority`.

## 3) Estado actual de implementación

- Scripts actualizados a esquema rectificado:
	- `main_ingestion.py`
	- `ingestion_examples.py`
	- `tests/validation/validate_ingestion.py`
- Perfil de configuración añadido en `config/params.yaml` (`isovida_rectificada`).

## 4) Pendientes para cerrar con tutor

- Uso formal de incertidumbre en columnas `U_*`.
- Unidades finales oficiales por analito.
- Umbrales QA/QC definitivos de aceptación.
- Definición final para variables con valores repetidos constantes (baja variabilidad).

## 5) Próximo paso técnico sugerido

Implementar una validación automática contra `Diccionario_DATA` que verifique:
- presencia de columnas obligatorias,
- tipo de dato esperado,
- rango esperado,
- y consistencia de unidades por analito.
