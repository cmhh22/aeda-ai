# AEDA Framework - Guía Maestra de Tesis

Este README es el **mapa principal** del proyecto para que no te pierdas durante el desarrollo.

## 1) Objetivo de la tesis

Construir un framework reproducible para análisis de datos ambientales con IA, comenzando por ISOVIDA, que permita:
- Ingerir datos de laboratorio complejos (`RAW`) de forma robusta.
- Estandarizar y validar calidad de datos para uso científico.
- Explorar patrones ambientales (espaciales/profundidad/composición).
- Aplicar modelos explicables para apoyar interpretación científica.
- Generar entregables claros para investigación y toma de decisiones.

## 2) Producto final (qué debes tener al terminar)

### Producto técnico
- Pipeline completo ejecutable: `RAW -> Ingesta -> QA/QC -> EDA -> ML explicable -> Reporte`.
- Configurable por matriz (sedimento, agua, aire, suelo, biota).
- Reproducible (mismas reglas, mismos resultados).

### Producto científico
- Dataset limpio y trazable basado en ISOVIDA.
- Resultados analíticos defendibles con reglas metodológicas explícitas.
- Hallazgos principales alineados con una hipótesis prioritaria.

### Producto de tesis
- Metodología clara por módulos.
- Justificación de decisiones técnicas (censura, unidades, QA/QC).
- Resultados + discusión + limitaciones + trabajo futuro.

## 3) Estado actual (checkpoint)

- ✅ Módulo 1 (Ingesta universal) implementado y validado con ISOVIDA.
- ✅ Estructura ordenada: `docs/`, `tests/`, `src/`.
- ✅ Documentación base creada.
- 🔄 Pendiente: cerrar decisiones oficiales con tutor (censura, unidades, QA/QC, objetivo científico #1).

## 4) Ruta de trabajo por etapas

## Etapa 0 - Alineación metodológica (CRÍTICA)
**Meta:** cerrar reglas con tutor antes de seguir a EDA/ML.

Entregables:
- Regla oficial para `<LOD`, `>LOQ/AQL`, `±` por analito.
- Unidades oficiales y conversiones por analito/matriz.
- Umbrales QA/QC de aprobación del dataset.
- Objetivo científico #1 acordado.

## Etapa 1 - Ingesta y control de calidad (Módulo 1)
**Meta:** convertir `RAW` en dataset analítico reproducible.

Incluye:
- Parsing de notación analítica.
- Manejo de censura.
- Conversión de unidades.
- Reporte de calidad.

Salida esperada:
- CSV limpio + metadatos + reporte QA/QC.

## Etapa 2 - Exploración científica (Módulo 2)
**Meta:** entender estructura del problema y formular hipótesis refinadas.

Incluye:
- Estadística descriptiva y distribuciones.
- Relaciones entre variables (correlaciones).
- Reducción de dimensionalidad (ej. PCA/UMAP).
- Identificación de patrones por estación/profundidad.

## Etapa 3 - Modelado IA explicable (Módulo 3)
**Meta:** modelar sin perder interpretabilidad científica.

Incluye:
- Modelos base y robustos (árboles, ensamblados).
- Validación y comparación de desempeño.
- Explicabilidad (SHAP / importancia de variables).

## Etapa 4 - Integración y reporte final (Módulo 4)
**Meta:** empaquetar resultados para tesis y uso práctico.

Incluye:
- Reporte final metodológico + resultados.
- Visualizaciones clave.
- Conclusiones y recomendaciones.

## 5) Proceso completo (cómo funciona)

1. **Entrada:** archivo laboratorio (`RAW`) con símbolos, faltantes e incertidumbres.
2. **Ingesta:** parsing + normalización + censura + unidades.
3. **QA/QC:** aplicación de criterios de aceptación.
4. **Dataset validado:** versión base para análisis.
5. **EDA:** patrones y estructura de datos.
6. **ML explicable:** modelos + interpretación.
7. **Salida final:** evidencia científica y técnica reproducible.

## 6) Tecnologías de IA/analítica por módulo

- **Módulo 1:** reglas estadísticas para censura + validación de calidad.
- **Módulo 2:** técnicas de exploración (PCA/UMAP, clustering opcional).
- **Módulo 3:** modelos supervisados e interpretabilidad (SHAP).
- **Módulo 4:** automatización de reportes y visualización.

## 7) Rol del científico vs rol del desarrollo

## Dónde la opinión del científico es imprescindible
- Definir reglas oficiales de censura por analito.
- Confirmar unidades y conversiones válidas.
- Establecer criterios QA/QC de aprobación.
- Priorizar objetivo científico y validar interpretación.

## Dónde depende más de ti (desarrollo)
- Implementación del pipeline y arquitectura modular.
- Automatización de validaciones y reportes.
- Trazabilidad, reproducibilidad y pruebas.
- Integración técnica entre módulos.

## Decisiones compartidas
- Selección final de variables para modelado.
- Umbrales prácticos cuando no hay norma cerrada.
- Definición de entregables intermedios por hito.

## 8) Preguntas clave para reunión con tutor

1. ¿Regla oficial por analito para `<LOD`, `>LOQ/AQL` y `±`?
2. ¿Unidad final oficial por analito y reglas de conversión?
3. ¿Criterios QA/QC para aprobar dataset?
4. ¿Diccionario oficial de columnas ISOVIDA?
5. ¿Cuál es el objetivo científico #1 de esta fase?

## 9) Cómo usar este README (cada semana)

- Actualiza el estado de cada etapa (✅ / 🔄 / ⛔).
- Registra decisiones cerradas con tutor.
- Define 1 objetivo técnico y 1 objetivo científico por semana.
- No avances a la siguiente etapa si la anterior no está metodológicamente cerrada.

## 10) Archivos guía importantes

- `docs/INGESTA_CONCEPTOS_CLAVE_README.md`
- `docs/ISOVIDA_DATASET_README.md`
- `docs/INGESTION_MODULE_GUIDE.md`
- `docs/IMPLEMENTATION_SUMMARY.md`
- `tests/unit/`
- `tests/validation/`

## 11) Definición de éxito de la tesis

La tesis está bien cerrada cuando:
- El pipeline corre de inicio a fin con reglas explícitas.
- Las decisiones metodológicas son reproducibles y defendibles.
- Los resultados responden una pregunta científica prioritaria.
- Queda documentación suficiente para replicar el proceso.
