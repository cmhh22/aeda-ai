"""Internationalization (i18n) for the AEDA-AI interface.

Design: the *English string is the key*. ``t(s)`` returns the Spanish
translation when the active language is Spanish, and the original English
string otherwise. If a string has no Spanish entry, the English text is
returned unchanged — so a missing translation degrades gracefully (the user
sees that phrase in English) and never breaks the app.

Only *display* text is translated. Internal identifiers (session_state keys,
routing keys, dataframe column names, method names) are never touched.
"""
from __future__ import annotations

import streamlit as st

DEFAULT_LANG = "es"

# English -> Spanish. Grow this catalog as more views are migrated.
ES: dict[str, str] = {
    # --- sidebar / shell ---
    "Automated EDA for environmental data": "Análisis exploratorio automatizado de datos ambientales",
    "Navigation": "Navegación",
    "No dataset loaded": "Ningún conjunto de datos cargado",
    "Current dataset": "Conjunto de datos actual",
    "Samples": "Muestras",
    "Variables": "Variables",
    "Clusters": "Grupos",
    # --- navigation page labels ---
    "Upload & Configure": "Cargar y configurar",
    "Analysis Plan": "Plan de análisis",
    "Results": "Resultados",
    "Depth Profiles": "Perfiles de profundidad",
    "Audit": "Auditoría",
    "Advanced Configuration": "Configuración avanzada",
    # --- page: Depth Profiles ---
    "Concentration vs. depth — sediment cores read as temporal series.":
        "Concentración vs. profundidad — los núcleos de sedimento se leen como series temporales.",
    "In sediment cores, **deeper = older**. The plots show how concentration "
    "of each variable changes through time, revealing historical contamination trends "
    "and geochemical changes.":
        "En los núcleos de sedimento, **más profundo = más antiguo**. Los gráficos muestran cómo "
        "cambia la concentración de cada variable a lo largo del tiempo, revelando tendencias "
        "históricas de contaminación y cambios geoquímicos.",
    "Run an analysis first from the Upload page.":
        "Ejecute primero un análisis desde la página de carga.",
    "No depth column detected in this dataset. Depth profiles require a column like 'Profundidad' or 'Depth'.":
        "No se detectó una columna de profundidad en este conjunto de datos. Los perfiles de "
        "profundidad requieren una columna como 'Profundidad' o 'Depth'.",
    "View mode": "Modo de visualización",
    "Single variable": "Una variable",
    "Multi-variable grid": "Cuadrícula multivariable",
    "Variable": "Variable",
    "Separate by core ({core})": "Separar por núcleo ({core})",
    "When a site has multiple sediment cores (e.g. Core A, "
    "Core B), this draws each core as a separate line. "
    "Useful to check reproducibility between cores at the "
    "same site.":
        "Cuando un sitio tiene varios núcleos de sedimento (p. ej. Núcleo A, "
        "Núcleo B), dibuja cada núcleo como una línea separada. Útil para comprobar "
        "la reproducibilidad entre núcleos del mismo sitio.",
    "Sites to display": "Sitios a mostrar",
    "Deselect sites to simplify the plot.": "Deseleccione sitios para simplificar el gráfico.",
    "Variable statistics": "Estadísticos de la variable",
    "Per-site descriptive statistics for the selected column.":
        "Estadísticos descriptivos por sitio de la columna seleccionada.",
    "Compare several variables side-by-side. Each panel is **read top-to-bottom: "
    "0 cm is the most recent sediment, deeper rows are older**. A line that rises "
    "(toward the surface) means the concentration has increased over time at that "
    "site; a line that stays flat means the chemistry is stable. "
    "**Tip:** start with 3–4 variables to keep the figure readable, then add more.":
        "Compare varias variables lado a lado. Cada panel se **lee de arriba hacia abajo: "
        "0 cm es el sedimento más reciente, las filas más profundas son más antiguas**. Una línea "
        "que asciende (hacia la superficie) indica que la concentración ha aumentado con el tiempo "
        "en ese sitio; una línea plana indica que la química es estable. "
        "**Sugerencia:** empiece con 3–4 variables para mantener la figura legible y luego añada más.",
    "Heavy metals": "Metales pesados",
    "Major elements": "Elementos mayoritarios",
    "Ancillary variables": "Variables complementarias",
    "Custom selection": "Selección personalizada",
    "Preset": "Conjunto predefinido",
    "Pre-built variable groups based on your dataset's geochemistry. "
    "Pick **Custom** to choose any combination.":
        "Grupos de variables predefinidos según la geoquímica de su conjunto de datos. "
        "Elija **Selección personalizada** para combinar libremente.",
    "Variables to plot": "Variables a graficar",
    "2–9 variables work well; more than that and the panels get crowded.":
        "Entre 2 y 9 variables funcionan bien; más que eso y los paneles se saturan.",
    "Select at least 2 variables.": "Seleccione al menos 2 variables.",
    "Columns in grid": "Columnas en la cuadrícula",
    "Fewer columns = wider panels = easier to compare individual sites.":
        "Menos columnas = paneles más anchos = más fácil comparar sitios individuales.",
    # --- page: Analysis Plan ---
    "What the system decided to do with your dataset, and why.":
        "Lo que el sistema decidió hacer con su conjunto de datos, y por qué.",
    "Your dataset contains **{n} samples** and **{f} variables**. "
    "After preprocessing and dimensionality reduction, the system will work with approximately "
    "**{d} dimensions**. The following analysis was tailored to your data's "
    "statistical properties and suspected geochemical composition.":
        "Su conjunto de datos contiene **{n} muestras** y **{f} variables**. "
        "Tras el preprocesamiento y la reducción de dimensionalidad, el sistema trabajará con "
        "aproximadamente **{d} dimensiones**. El análisis siguiente se adaptó a las propiedades "
        "estadísticas de sus datos y a su composición geoquímica presunta.",
    "Dataset profile": "Perfil del conjunto de datos",
    "Effective dimensions": "Dimensiones efectivas",
    "Samples/features ratio": "Razón muestras/variables",
    "Skewed variables": "Variables asimétricas",
    "Missing data": "Datos faltantes",
    "Correlated pairs (|r|>0.6)": "Pares correlacionados (|r|>0,6)",
    "Outlier samples (IQR)": "Muestras atípicas (IQR)",
    "Geochemistry detected": "Geoquímica detectada",
    "Trace elements": "Elementos traza",
    "Granulometry": "Granulometría",
    "Mixed units detected (% and mg/kg). Scaling is mandatory before multivariate analysis.":
        "Se detectaron unidades mixtas (% y mg/kg). El escalado es obligatorio antes del análisis multivariado.",
    "Warnings": "Advertencias",
    "Recommended analysis scales": "Escalas de análisis recomendadas",
    "Method recommendations": "Recomendaciones de métodos",
    "Preprocessing": "Preprocesamiento",
    "Dimensionality reduction": "Reducción de dimensionalidad",
    "Clustering": "Agrupamiento",
    "Spatial analysis": "Análisis espacial",
    "Anomaly detection": "Detección de anomalías",
    "Correlations": "Correlaciones",
    "Feature analysis": "Análisis de variables",
    "methods": "métodos",
    "Primary": "Primario",
    "Alternative": "Alternativa",
    "Priority": "Prioridad",
    "Confidence": "Confianza",
    "Validation report": "Informe de validación",
    "Completeness": "Completitud",
    "Issues found": "Problemas encontrados",
    # --- page: Upload & Configure ---
    "Upload your environmental dataset and run the analysis with one click.":
        "Cargue su conjunto de datos ambientales y ejecute el análisis con un clic.",
    "1. File": "1. Archivo",
    "Select an Excel or CSV file": "Seleccione un archivo Excel o CSV",
    "The file should contain environmental measurements with samples as rows and variables as columns.":
        "El archivo debe contener mediciones ambientales con las muestras en filas y las variables en columnas.",
    "Upload a file to begin.": "Cargue un archivo para comenzar.",
    "2. Sheet": "2. Hoja",
    "Select sheet": "Seleccione la hoja",
    "Choose the sheet containing your measurement data.": "Elija la hoja que contiene sus datos de medición.",
    "Only sheet available: **{sheet}**": "Única hoja disponible: **{sheet}**",
    "CSV file selected — sheet selection not needed.": "Archivo CSV seleccionado — no hace falta elegir hoja.",
    "Could not read the uploaded file. Make sure it is a valid Excel/CSV format.":
        "No se pudo leer el archivo cargado. Asegúrese de que sea un Excel/CSV válido.",
    "Data preview ({n} columns, first 10 rows)": "Vista previa de datos ({n} columnas, primeras 10 filas)",
    "3. Columns to analyze": "3. Columnas a analizar",
    "By default, non-numeric columns (identifiers, codes, dates, sites) "
    "are excluded. Adjust below if needed.":
        "Por defecto se excluyen las columnas no numéricas (identificadores, códigos, fechas, sitios). "
        "Ajuste abajo si hace falta.",
    "Exclude these columns from the ML analysis": "Excluir estas columnas del análisis de aprendizaje automático",
    "These columns will be ignored during the ML analysis. Coordinates and depth are excluded from ML but used for metadata.":
        "Estas columnas se ignoran durante el análisis de aprendizaje automático. Las coordenadas y la "
        "profundidad se excluyen del análisis pero se usan como metadatos.",
    "4. Analysis options": "4. Opciones de análisis",
    "Sensible defaults work for most environmental datasets — for fine-grained "
    "control, use the Advanced Configuration page after the first run.":
        "Los valores por defecto sirven para la mayoría de los conjuntos ambientales; para un control "
        "detallado, use la página de Configuración avanzada tras la primera ejecución.",
    "Missing values strategy": "Estrategia para valores faltantes",
    "How to fill in or remove missing values.": "Cómo rellenar o eliminar los valores faltantes.",
    "Replaces empty cells with a plausible value so ML algorithms can "
    "process the data. **Median** is robust against extreme values.":
        "Reemplaza las celdas vacías con un valor plausible para que los algoritmos puedan procesar los "
        "datos. La **mediana** es robusta frente a valores extremos.",
    "Method used to compress the dataset into a smaller number of components.":
        "Método para comprimir el conjunto de datos en un número menor de componentes.",
    "Compresses many variables into a few summary axes (components) "
    "that capture the main patterns. **PCA** is the standard choice "
    "for environmental data.":
        "Comprime muchas variables en unos pocos ejes resumen (componentes) que capturan los patrones "
        "principales. El **PCA** es la opción estándar para datos ambientales.",
    "Clustering method": "Método de agrupamiento",
    "Algorithm used to group similar samples.": "Algoritmo para agrupar muestras similares.",
    "Groups samples with similar chemistry. **Auto** tries K-Means "
    "and DBSCAN and keeps the best one according to a quality score.":
        "Agrupa muestras con química similar. **Automático** prueba K-Means y DBSCAN y conserva el mejor "
        "según una puntuación de calidad.",
    "5. Run": "5. Ejecutar",
    "Run analysis": "Ejecutar análisis",
    "Loading data...": "Cargando datos...",
    "Loading and validating data...": "Cargando y validando datos...",
    "Running auto-selector...": "Ejecutando el autoselector...",
    "Preprocessing...": "Preprocesando...",
    "Generating results...": "Generando resultados...",
    "Done!": "¡Listo!",
    "Analysis complete! Navigate to the other pages to see results.":
        "¡Análisis completo! Navegue a las otras páginas para ver los resultados.",
    "Variables analyzed": "Variables analizadas",
    "PCA components": "Componentes PCA",
    "The pipeline could not complete. The dataset may have an unexpected format or missing required columns.":
        "El flujo no pudo completarse. El conjunto de datos puede tener un formato inesperado o faltarle columnas necesarias.",
    "How it works": "Cómo funciona",
    "1. Upload": "1. Cargar",
    "Upload an Excel or CSV with your environmental measurements.":
        "Cargue un Excel o CSV con sus mediciones ambientales.",
    "2. Configure": "2. Configurar",
    "Select which columns to exclude and choose analysis options.":
        "Seleccione qué columnas excluir y elija las opciones de análisis.",
    "3. Explore": "3. Explorar",
    "Browse interactive plots: PCA biplot, clusters, correlations, depth profiles.":
        "Explore gráficos interactivos: biplot de PCA, grupos, correlaciones, perfiles de profundidad.",
    "Supported formats: .xlsx, .xls, .csv — Datasets tested with FRX geochemistry, granulometry, and sediment data.":
        "Formatos admitidos: .xlsx, .xls, .csv — Probado con datos de geoquímica por FRX, granulometría y sedimentos.",
    # option values (kept in English as logic values; shown translated)
    "median": "mediana", "mean": "media", "knn": "k-NN", "drop_rows": "eliminar filas",
    "pca": "PCA", "auto": "automático", "kmeans": "K-Means", "dbscan": "DBSCAN", "hierarchical": "jerárquico",
    # --- component: errors / params ---
    "Technical details (for debugging)": "Detalles técnicos (para depuración)",
    "(no parameters)": "(sin parámetros)",
    # --- Results: Surface (spatial) tab ---
    "Surface-layer inter-site analysis": "Análisis entre sitios de la capa superficial",
    "Compares sites using **only their surface sediment** (the most "
    "recent deposition). This avoids mixing different historical "
    "periods across sites, which is the standard approach for "
    "present-day spatial contamination studies.":
        "Compara los sitios usando **solo su sedimento superficial** (la deposición más reciente). "
        "Así se evita mezclar períodos históricos distintos entre sitios, que es el enfoque estándar "
        "para los estudios de contaminación espacial actual.",
    "Surface analysis was not executed. It requires the dataset to "
    "have both a **site column** and a **depth column**. "
    "Check the Analysis Plan or Audit page to see which were detected.":
        "El análisis de superficie no se ejecutó. Requiere que el conjunto de datos tenga una "
        "**columna de sitio** y una **columna de profundidad**. Revise la página de Plan de análisis "
        "o de Auditoría para ver cuáles se detectaron.",
    "Surface depth (cm)": "Profundidad superficial (cm)",
    "Defines the cutoff between 'surface' (recent) and 'deep' "
    "(older) sediment. Yoelvis (LEA-CEAC) recommends 10 cm as "
    "the default; 5 cm or 20 cm are common alternatives in "
    "different authors.":
        "Define el corte entre el sedimento 'superficial' (reciente) y el 'profundo' (antiguo). "
        "Yoelvis (LEA-CEAC) recomienda 10 cm por defecto; 5 cm o 20 cm son alternativas comunes "
        "en distintos autores.",
    "Could not recompute the surface analysis at {depth} cm.":
        "No se pudo recalcular el análisis de superficie a {depth} cm.",
    "No samples fall within the top {depth} cm. Try a deeper threshold.":
        "No hay muestras dentro de los primeros {depth} cm. Pruebe un umbral más profundo.",
    "Depth threshold": "Umbral de profundidad",
    "Sites with surface data": "Sitios con datos de superficie",
    "Surface samples": "Muestras superficiales",
    "Site groups": "Grupos de sitios",
    "Number of clusters of sites with similar surface chemistry. "
    "Computed by hierarchical Ward clustering.":
        "Número de grupos de sitios con química superficial similar. "
        "Calculado por agrupamiento jerárquico de Ward.",
    "Site × variable heatmap (Z-score per variable)":
        "Mapa de calor sitio × variable (puntuación Z por variable)",
    "Each variable is standardized across sites. **Red** = this site is high "
    "in that variable relative to other sites; **blue** = low. Sites are "
    "ordered by cluster, so members of the same group appear together.":
        "Cada variable se estandariza entre sitios. **Rojo** = este sitio es alto en esa variable "
        "respecto a los demás; **azul** = bajo. Los sitios se ordenan por grupo, de modo que los "
        "miembros del mismo grupo aparecen juntos.",
    "No site means to display.": "No hay medias por sitio para mostrar.",
    "No variables show variance across sites.": "Ninguna variable muestra varianza entre sitios.",
    "Site": "Sitio",
    "Z-score": "Puntuación Z",
    "Geographic distribution of sites": "Distribución geográfica de los sitios",
    "Each point is a site, positioned by its average coordinates. "
    "Color encodes the cluster from the surface-chemistry analysis: "
    "geographically close sites with the same color have similar "
    "surface chemistry.":
        "Cada punto es un sitio, ubicado por sus coordenadas medias. El color codifica el grupo del "
        "análisis de química superficial: los sitios geográficamente cercanos con el mismo color "
        "tienen química superficial similar.",
    "Longitude": "Longitud",
    "Latitude": "Latitud",
    "Clustering was skipped (only {n} site(s)). It needs at least 3 sites with surface samples.":
        "Se omitió el agrupamiento (solo {n} sitio(s)). Necesita al menos 3 sitios con muestras superficiales.",
    "Cluster": "Grupo",
    "Sites": "Sitios",
    "Count": "Conteo",
    "Cluster composition": "Composición de los grupos",
    "Sites in the same cluster have similar surface chemistry. The "
    "groupings are computed only over the surface layer, so they "
    "reflect *current* spatial patterns rather than the full core history.":
        "Los sitios del mismo grupo tienen química superficial similar. Los grupos se calculan solo "
        "sobre la capa superficial, de modo que reflejan los patrones espaciales *actuales* y no toda "
        "la historia del núcleo.",
}

ES.update({
    # ===== page: Results =====
    "Interactive dashboard: PCA, correlations, clusters and anomalies.":
        "Tablero interactivo: PCA, correlaciones, grupos y anomalías.",
    "Anomalies": "Anomalías",
    "Dimensionality ({m})": "Dimensionalidad ({m})",
    "Surface (spatial)": "Superficie (espacial)",
    "Dimensionality reduction was not executed or failed. "
    "Check the Analysis Plan for details.":
        "La reducción de dimensionalidad no se ejecutó o falló. "
        "Revise el Plan de análisis para más detalles.",
    "PCA biplot": "Biplot de PCA",
    "Each point is one sample. Samples close together have a similar "
    "chemical fingerprint. **Arrows** show how each variable pulls the "
    "samples — arrows pointing the same way mean those variables move "
    "together. Long arrows = the variable contributes a lot to the "
    "separation; short arrows = it contributes little.":
        "Cada punto es una muestra. Las muestras cercanas tienen una huella química similar. "
        "Las **flechas** muestran cómo cada variable tira de las muestras — flechas en la misma "
        "dirección indican que esas variables se mueven juntas. Flechas largas = la variable "
        "contribuye mucho a la separación; flechas cortas = contribuye poco.",
    "Color by": "Colorear por",
    "Categorical columns (Site_Name, Core) show group "
    "separation. Numeric columns (Profundidad) show a "
    "continuous gradient — useful to see how chemistry "
    "changes with depth.":
        "Las columnas categóricas (Site_Name, Core) muestran la separación entre grupos. "
        "Las columnas numéricas (Profundidad) muestran un gradiente continuo — útil para ver "
        "cómo cambia la química con la profundidad.",
    "Loading arrows": "Flechas de carga",
    "Principal component to show on this axis. Components are "
    "ordered by how much variability they capture: PC1 is the "
    "most informative, then PC2, etc. Changing the axis shows "
    "the dataset from a different angle.":
        "Componente principal a mostrar en este eje. Los componentes se ordenan por la variabilidad "
        "que capturan: PC1 es el más informativo, luego PC2, etc. Cambiar el eje muestra el conjunto "
        "de datos desde otro ángulo.",
    "X axis": "Eje X",
    "Y axis": "Eje Y",
    "Scree plot (variance explained)": "Gráfico de sedimentación (varianza explicada)",
    "Each bar shows how much of the total variability is captured by that "
    "component. Use this to pick a sensible number of components without "
    "losing meaningful structure.":
        "Cada barra muestra cuánta de la variabilidad total captura ese componente. Úselo para "
        "elegir un número sensato de componentes sin perder estructura relevante.",
    "Loadings table": "Tabla de cargas",
    "How strongly each variable contributes to each principal component. "
    "Same-sign loadings mean variables move together; opposite signs mean "
    "they move in opposite directions.":
        "Con qué fuerza contribuye cada variable a cada componente principal. Cargas del mismo "
        "signo indican que las variables se mueven juntas; signos opuestos, en direcciones opuestas.",
    "{m} embedding": "Proyección {m}",
    "{m} is a non-linear projection: distances on the plot reflect "
    "local similarity rather than global variance. Useful for visualizing groups, "
    "but not for biplot-style interpretation (no loadings or scree).":
        "{m} es una proyección no lineal: las distancias en el gráfico reflejan la similitud local "
        "más que la varianza global. Útil para visualizar grupos, pero no para una interpretación "
        "tipo biplot (sin cargas ni sedimentación).",
    "{m} 2D embedding": "Proyección 2D {m}",
    "Correlation analysis was not executed.": "El análisis de correlación no se ejecutó.",
    "Correlation matrix": "Matriz de correlación",
    "**Red** = the two variables tend to move together (rise and fall in sync). "
    "**Blue** = they move in opposite directions. **White** = no relationship. "
    "With **cluster-reorder axes** ON, similar variables are grouped along the "
    "diagonal, making blocks of co-varying elements (e.g. lithogenic vs. "
    "anthropogenic) easier to spot visually.":
        "**Rojo** = las dos variables tienden a moverse juntas (suben y bajan a la vez). "
        "**Azul** = se mueven en direcciones opuestas. **Blanco** = sin relación. Con "
        "**reordenar ejes por agrupamiento** activado, las variables similares se agrupan en la "
        "diagonal, lo que facilita ver bloques de elementos que covarían (p. ej. litogénicos vs. "
        "antropogénicos).",
    "Cluster-reorder axes": "Reordenar ejes por agrupamiento",
    "Significant correlations ({s} strong, {m} moderate)":
        "Correlaciones significativas ({s} fuertes, {m} moderadas)",
    "Variable pairs sorted by the strength of their association. "
    "**Strong** (|r| ≥ 0.7) indicates a robust relationship; "
    "**Moderate** (0.5 ≤ |r| < 0.7) suggests weaker but consistent links.":
        "Pares de variables ordenados por la fuerza de su asociación. **Fuerte** (|r| ≥ 0,7) indica "
        "una relación robusta; **Moderada** (0,5 ≤ |r| < 0,7) sugiere vínculos más débiles pero "
        "consistentes.",
    "Nonlinear relationship candidates ({n})": "Candidatos a relación no lineal ({n})",
    "Pairs where Spearman rank correlation substantially exceeds Pearson — "
    "this suggests a monotonic but non-linear relationship worth visual follow-up.":
        "Pares donde la correlación de rangos de Spearman supera notablemente a la de Pearson — "
        "esto sugiere una relación monótona pero no lineal que conviene revisar visualmente.",
    "Heavy metals vs. grain size": "Metales pesados vs. granulometría",
    "Spearman correlation between each heavy metal and each grain-size "
    "fraction. Strong positive correlations of metals with fine fractions "
    "may indicate particulate contamination.":
        "Correlación de Spearman entre cada metal pesado y cada fracción granulométrica. "
        "Correlaciones positivas fuertes de los metales con las fracciones finas pueden indicar "
        "contaminación particulada.",
    "Clustering was not executed.": "El agrupamiento no se ejecutó.",
    "Dimensionality reduction needed for cluster visualization.":
        "Se necesita la reducción de dimensionalidad para visualizar los grupos.",
    "Cluster analysis": "Análisis de agrupamiento",
    "**Left:** samples colored by the chemical groups the algorithm found "
    "automatically. **Right:** the same samples colored by a known label "
    "(e.g. site). If both panels look similar, chemistry and the chosen "
    "label agree — useful evidence that site-level differences are real. "
    "If they look different, chemistry is driven by something other than "
    "that label (depth, contamination, mineralogy).":
        "**Izquierda:** muestras coloreadas por los grupos químicos que el algoritmo encontró "
        "automáticamente. **Derecha:** las mismas muestras coloreadas por una etiqueta conocida "
        "(p. ej. el sitio). Si ambos paneles se parecen, la química y la etiqueta concuerdan — "
        "evidencia útil de que las diferencias entre sitios son reales. Si difieren, la química está "
        "dominada por otra cosa (profundidad, contaminación, mineralogía).",
    "Compare clusters with": "Comparar grupos con",
    "Number of groups the algorithm found in the data.":
        "Número de grupos que el algoritmo encontró en los datos.",
    "Silhouette score": "Coeficiente de silueta",
    "How well separated the clusters are. Range -1 to 1; "
    "higher is better. Above 0.5 = good separation, "
    "0.25–0.5 = moderate, below 0.25 = weak.":
        "Qué tan separados están los grupos. Rango de -1 a 1; más alto es mejor. Por encima de "
        "0,5 = buena separación, 0,25–0,5 = moderada, por debajo de 0,25 = débil.",
    "Average ratio of within-cluster to between-cluster "
    "distances. Lower is better. Below 1.0 = good, "
    "above 2.0 = weak.":
        "Razón media entre las distancias dentro de los grupos y entre los grupos. Más bajo es mejor. "
        "Por debajo de 1,0 = bueno, por encima de 2,0 = débil.",
    "How each cluster is composed in terms of the comparison label. "
    "If a cluster is dominated by a single site, location likely explains "
    "the chemical grouping; if mixed, chemistry may be driven by another "
    "factor (depth, contamination, or mineralogy).":
        "Cómo se compone cada grupo según la etiqueta de comparación. Si un grupo está dominado por "
        "un solo sitio, la ubicación probablemente explica la agrupación química; si está mezclado, "
        "la química puede estar dominada por otro factor (profundidad, contaminación o mineralogía).",
    "Variables that most discriminate between clusters":
        "Variables que más discriminan entre los grupos",
    "Variables ranked by how useful they are to tell clusters apart. "
    "High-ranking variables are the best clues when interpreting what each "
    "cluster represents — look at these first when assigning a label.":
        "Variables ordenadas por su utilidad para distinguir los grupos. Las variables mejor ubicadas "
        "son las mejores pistas al interpretar qué representa cada grupo — mírelas primero al asignar "
        "una etiqueta.",
    "Importance": "Importancia",
    "Anomaly detection was not executed.": "La detección de anomalías no se ejecutó.",
    "Samples flagged as **unusually different** from the rest of the dataset. "
    "These are not automatically 'contaminated' or 'wrong' — just statistical "
    "outliers in the multivariate chemical space. Each one deserves a look: "
    "they may be hotspots (real contamination), measurement errors, or "
    "samples from a chemically distinct sub-environment.":
        "Muestras señaladas como **inusualmente diferentes** del resto del conjunto. No son "
        "automáticamente 'contaminadas' ni 'erróneas' — solo valores atípicos en el espacio químico "
        "multivariado. Cada una merece revisión: pueden ser focos (contaminación real), errores de "
        "medición o muestras de un subambiente químicamente distinto.",
    "Anomalies detected": "Anomalías detectadas",
    "Anomaly": "Anomalía",
    "Anomalies in PCA space": "Anomalías en el espacio PCA",
    "Anomalous samples": "Muestras anómalas",
    "All the columns from the original Excel for each flagged sample. "
    "Use this table to inspect the full record, check suspect values, "
    "and cross-check against laboratory notes.":
        "Todas las columnas del Excel original para cada muestra señalada. Use esta tabla para "
        "inspeccionar el registro completo, comprobar valores sospechosos y contrastar con las "
        "notas de laboratorio.",
    "None": "Ninguno",
    # ===== page: Audit =====
    "Trace of every decision the pipeline made on this dataset. "
    "Use this page to verify the methodology and defend each choice.":
        "Traza de cada decisión que el flujo tomó sobre este conjunto de datos. Use esta página "
        "para verificar la metodología y defender cada elección.",
    "Overview": "Resumen", "Decisions": "Decisiones", "Interpretation": "Interpretación",
    "Technical": "Técnico",
    "Run summary": "Resumen de la ejecución",
    "File": "Archivo", "Columns": "Columnas", "Measurement variables": "Variables de medición",
    "site column **{col}** ({n} sites)": "columna de sitio **{col}** ({n} sitios)",
    "depth column **{col}**": "columna de profundidad **{col}**",
    "coordinates **{cols}**": "coordenadas **{cols}**",
    "Metadata detected:": "Metadatos detectados:",
    "No site, depth, or coordinate columns were detected.":
        "No se detectaron columnas de sitio, profundidad ni coordenadas.",
    "Input validation": "Validación de entrada",
    "Validation report not available.": "Informe de validación no disponible.",
    "Errors / warnings": "Errores / advertencias",
    "No data quality issues were detected.": "No se detectaron problemas de calidad de datos.",
    "Issue details ({n})": "Detalle de problemas ({n})",
    "No analysis plan available.": "No hay plan de análisis disponible.",
    "What the system chose to do, and why. Each entry shows the chosen "
    "method, the rationale, and the evidence from your dataset that "
    "supported the choice.":
        "Lo que el sistema decidió hacer, y por qué. Cada entrada muestra el método elegido, la "
        "justificación y la evidencia de su conjunto de datos que respaldó la elección.",
    "Plan-level warnings ({n})": "Advertencias del plan ({n})",
    "Data preparation": "Preparación de datos",
    "Variable summarization": "Resumen de variables",
    "Sample grouping": "Agrupamiento de muestras",
    "Variable relationships": "Relaciones entre variables",
    "Most informative variables": "Variables más informativas",
    "**{name}** — chose **{method}** ({conf} confidence)":
        "**{name}** — eligió **{method}** (confianza {conf})",
    "Why:": "Por qué:",
    "(no reason recorded)": "(sin razón registrada)",
    "Evidence from your data:": "Evidencia de sus datos:",
    "Parameters chosen by the auto-selector:": "Parámetros elegidos por el autoselector:",
    "{n} alternative method(s) were considered:":
        "Se consideraron {n} método(s) alternativo(s):",
    "The interpretation layer (EF, TEL/PEL, Birch) was not executed — "
    "either no heavy metals were detected, the reference element was "
    "missing, or no depth column was available for the baseline.":
        "La capa de interpretación (EF, TEL/PEL, Birch) no se ejecutó — o no se detectaron metales "
        "pesados, o faltaba el elemento de referencia, o no había columna de profundidad para el baseline.",
    "not used": "no usado",
    "✓ EF was computed against a **single global baseline** "
    "(deepest sample in the dataset).":
        "✓ El EF se calculó contra un **único baseline global** (la muestra más profunda del conjunto).",
    "✓ EF was computed against **per-site baselines** "
    "({n} sites with their own deepest sample).":
        "✓ El EF se calculó contra **baselines por sitio** ({n} sitios con su propia muestra más profunda).",
    "Metals analyzed:": "Metales analizados:",
    "Toxicological classification (NOAA TEL/PEL):": "Clasificación toxicológica (NOAA TEL/PEL):",
    "Each cell shows the number of samples in that toxicological category, "
    "per metal. Buchman (2008) and Long & MacDonald (1998).":
        "Cada celda muestra el número de muestras en esa categoría toxicológica, por metal. "
        "Buchman (2008) y Long & MacDonald (1998).",
    "Enrichment classification (Birch 2003):": "Clasificación de enriquecimiento (Birch 2003):",
    "Number of samples in each enrichment band, per metal. "
    "EF computed relative to the deepest core section.":
        "Número de muestras en cada banda de enriquecimiento, por metal. EF calculado respecto a la "
        "sección más profunda del núcleo.",
    "Enrichment factor (EF) descriptive statistics per metal":
        "Estadísticos descriptivos del factor de enriquecimiento (EF) por metal",
    "Preprocessing trace": "Traza de preprocesamiento",
    "Every transformation applied to the raw data, in order. "
    "This is the audit trail for reproducibility.":
        "Cada transformación aplicada a los datos crudos, en orden. Este es el registro de auditoría "
        "para la reproducibilidad.",
    "No preprocessing steps were recorded.": "No se registraron pasos de preprocesamiento.",
    "Step {i}: **{name}**": "Paso {i}: **{name}**",
    "(no parameters recorded)": "(sin parámetros registrados)",
    "ML quality metrics": "Métricas de calidad del aprendizaje automático",
    "These metrics evaluate how well the chosen models fit the data. "
    "They are useful for the analyst, not strictly necessary for "
    "scientific interpretation.":
        "Estas métricas evalúan qué tan bien se ajustan los modelos elegidos a los datos. Son útiles "
        "para el analista, no estrictamente necesarias para la interpretación científica.",
    "Components retained": "Componentes retenidos",
    "Cumulative variance": "Varianza acumulada",
    "Silhouette": "Silueta",
    "Range -1 to 1. Higher is better. Above 0.5 is good.":
        "Rango de -1 a 1. Más alto es mejor. Por encima de 0,5 es bueno.",
    "Lower is better. Measures intra/inter-cluster ratio.":
        "Más bajo es mejor. Mide la razón intra/inter-grupo.",
    "In auto mode the system compared:": "En modo automático el sistema comparó:",
    "Anomalies flagged": "Anomalías señaladas",
    "Correlation analysis": "Análisis de correlación",
    "Feature importance": "Importancia de variables",
    "did not produce a result": "no produjo un resultado",
    "did not run (clusters are available)": "no se ejecutó (los grupos están disponibles)",
    "Pipeline step status": "Estado de los pasos del flujo",
    "All pipeline steps completed successfully.":
        "Todos los pasos del flujo se completaron correctamente.",
    "These steps were skipped or failed silently during the run. "
    "Check the application logs for the underlying error.":
        "Estos pasos se omitieron o fallaron en silencio durante la ejecución. Revise los registros "
        "de la aplicación para ver el error subyacente.",
    # ===== page: Advanced Configuration =====
    "Re-run the analysis on the currently loaded dataset with custom "
    "parameters. Useful for sensitivity analysis and for the scientific "
    "tutor to validate alternative methodological choices.":
        "Vuelva a ejecutar el análisis sobre el conjunto de datos cargado con parámetros "
        "personalizados. Útil para el análisis de sensibilidad y para que el tutor científico valide "
        "elecciones metodológicas alternativas.",
    "Upload a dataset from the Upload page first. "
    "Once an initial run has been completed, this page will let you "
    "re-run the same analysis with different settings.":
        "Cargue primero un conjunto de datos desde la página de carga. Una vez completada una "
        "ejecución inicial, esta página le permitirá volver a ejecutar el mismo análisis con otra "
        "configuración.",
    "Dataset": "Conjunto de datos",
    "Expert overrides (fine-grained ML parameters)":
        "Anulaciones de experto (parámetros finos de ML)",
    "Unlock manual control over algorithm-specific parameters. "
    "Leave off to keep the system in fully automatic mode.":
        "Habilita el control manual de los parámetros específicos de cada algoritmo. Déjelo "
        "desactivado para mantener el sistema en modo totalmente automático.",
    "Changes vs. last run ({n} parameter(s))":
        "Cambios respecto a la última ejecución ({n} parámetro(s))",
    "No changes vs. the last run.": "Sin cambios respecto a la última ejecución.",
    "Re-run pipeline with these settings": "Volver a ejecutar el flujo con esta configuración",
    "Pipeline configuration": "Configuración del flujo",
    "Analysis methods": "Métodos de análisis",
    "Missing value strategy": "Estrategia para valores faltantes",
    "Scaling method": "Método de escalado",
    "Robust scaling resists outliers; standard is the typical default.":
        "El escalado robusto resiste los valores atípicos; el estándar es el habitual por defecto.",
    "CLR transform (compositional)": "Transformación CLR (composicional)",
    "Apply Centered Log-Ratio transform for compositional data "
    "(XRF, granulometry). This transform is manual (opt-in).":
        "Aplica la transformación de razón logarítmica centrada para datos composicionales "
        "(FRX, granulometría). Esta transformación es manual (opcional).",
    "PCA is the standard choice for environmental EDA.":
        "El PCA es la opción estándar para el AED ambiental.",
    "'auto' picks the best between K-Means and DBSCAN by silhouette.":
        "'automático' elige el mejor entre K-Means y DBSCAN por silueta.",
    "Correlation method": "Método de correlación",
    "'compare' computes both Pearson and Spearman.":
        "'comparar' calcula tanto Pearson como Spearman.",
    "Anomaly contamination rate": "Tasa de contaminación de anomalías",
    "Expected fraction of anomalous samples in the dataset.":
        "Fracción esperada de muestras anómalas en el conjunto de datos.",
    "Environmental interpretation": "Interpretación ambiental",
    "Configure how enrichment factors and toxicological classifications "
    "are computed. The reference element should be conservative "
    "(typically Al or Fe) and present in the dataset.":
        "Configure cómo se calculan los factores de enriquecimiento y las clasificaciones "
        "toxicológicas. El elemento de referencia debe ser conservativo (típicamente Al o Fe) y "
        "estar presente en el conjunto de datos.",
    "Run environmental interpretation (EF, TEL/PEL, Birch)":
        "Ejecutar la interpretación ambiental (EF, TEL/PEL, Birch)",
    "Conservative element used as a normalizer in the EF formula. "
    "Al and Fe are the most common choices for sediment studies.":
        "Elemento conservativo usado como normalizador en la fórmula del EF. El Al y el Fe son las "
        "opciones más comunes en estudios de sedimentos.",
    "deepest: per-site deepest sample (recommended when sites are present). "
    "global_min_depth: single deepest sample for the whole dataset. "
    "user: provide your own baseline values.":
        "deepest: muestra más profunda por sitio (recomendado cuando hay sitios). global_min_depth: "
        "única muestra más profunda de todo el conjunto. user: usted proporciona sus propios valores "
        "de baseline.",
    "Custom baseline": "Baseline personalizado",
    "Edit baseline values as a table. The reference element and all "
    "metals to be analyzed must be present.":
        "Edite los valores del baseline como una tabla. El elemento de referencia y todos los metales "
        "a analizar deben estar presentes.",
    "Baseline parsed correctly.": "Baseline interpretado correctamente.",
    "Error parsing baseline: {e}": "Error al interpretar el baseline: {e}",
    "Expert overrides": "Anulaciones de experto",
    "These parameters override the automatic choices of each ML method. "
    "Leave them at their default to keep the system in auto mode.":
        "Estos parámetros anulan las elecciones automáticas de cada método de ML. Déjelos en su valor "
        "por defecto para mantener el sistema en modo automático.",
    "Dimensionality reduction (PCA / t-SNE / UMAP)":
        "Reducción de dimensionalidad (PCA / t-SNE / UMAP)",
    "Number of components (0 = automatic)": "Número de componentes (0 = automático)",
    "Clustering parameters": "Parámetros de agrupamiento",
    "n_clusters (0 = automatic)": "n_clusters (0 = automático)",
    "Number of clusters for K-Means and Hierarchical.":
        "Número de grupos para K-Means y jerárquico.",
    "k_range for auto-K (silhouette search)":
        "k_range para K automático (búsqueda por silueta)",
    "eps (0 = automatic via k-NN knee)": "eps (0 = automático vía codo k-NN)",
    "Hierarchical": "Jerárquico",
    "Linkage method": "Método de enlace",
    "Anomaly detection parameters": "Parámetros de detección de anomalías",
    "n_neighbors (LOF, 0 = default)": "n_neighbors (LOF, 0 = por defecto)",
    "random_state (Isolation Forest, -1 = none)":
        "random_state (Isolation Forest, -1 = ninguno)",
    "Parameter": "Parámetro", "Old Value": "Valor anterior", "New Value": "Valor nuevo",
    "(none)": "(ninguno)",
    "Initializing pipeline...": "Inicializando el flujo...",
    "Running analysis with the new settings...":
        "Ejecutando el análisis con la nueva configuración...",
    "Storing new results...": "Guardando los nuevos resultados...",
    "Pipeline re-executed successfully. "
    "New results are now visible in Results, Audit and the other pages.":
        "Flujo vuelto a ejecutar correctamente. Los nuevos resultados ya están visibles en "
        "Resultados, Auditoría y las demás páginas.",
    "The pipeline could not complete with these settings. Please review the parameter configuration.":
        "El flujo no pudo completarse con esta configuración. Revise la configuración de parámetros.",
    # ===== option values used as logic (shown translated via format_func) =====
    "drop_cols": "eliminar columnas",
    "standard": "estándar", "minmax": "min-max", "robust": "robusto",
    "off": "desactivado", "on": "activado",
    "tsne": "t-SNE", "umap": "UMAP",
    "isolation_forest": "Isolation Forest", "lof": "LOF", "zscore": "z-score", "iqr": "IQR",
    "compare": "comparar", "pearson": "Pearson", "spearman": "Spearman",
    "deepest": "más profunda", "global_min_depth": "mínima profundidad global", "user": "usuario",
    "ward": "Ward", "complete": "completo", "average": "promedio", "single": "simple",
})

ES.update({
    # --- page: Export ---
    "Export": "Exportar",
    "Download the result tables as Excel or CSV.":
        "Descargue las tablas de resultados en Excel o CSV.",
    "No tables available to export yet.": "Aún no hay tablas para exportar.",
    "All tables (Excel)": "Todas las tablas (Excel)",
    "One workbook with every result table as a separate sheet.":
        "Un libro con cada tabla de resultados en una hoja aparte.",
    "Download Excel (.xlsx)": "Descargar Excel (.xlsx)",
    "Could not build the Excel workbook.": "No se pudo generar el libro de Excel.",
    "Tip: to download a single table on its own, hover over it on the "
    "Results or Audit page and use the table's download button.":
        "Consejo: para descargar una sola tabla por separado, pase el cursor sobre "
        "ella en la página de Resultados o Auditoría y use su botón de descarga.",
    "Tables included: {names}": "Tablas incluidas: {names}",
    # export table / sheet names
    "Raw data": "Datos crudos",
    "Processed data": "Datos procesados",
    "Sample classification": "Clasificación de muestras",
    "Anomaly score": "Puntuación de anomalía",
    "PCA loadings": "Cargas del PCA",
    "PCA coordinates": "Coordenadas del PCA",
    "Component": "Componente",
    "Explained variance": "Varianza explicada",
    "PCA explained variance": "Varianza explicada del PCA",
    "Correlation ({m})": "Correlación ({m})",
    "TEL/PEL classification": "Clasificación TEL/PEL",
    "EF classification": "Clasificación de EF",
    "Enrichment factors (EF)": "Factores de enriquecimiento (EF)",
    "Surface site means": "Medias por sitio (superficie)",
    "Site coordinates": "Coordenadas de los sitios",
})

ES.update({
    # --- page: Export (PDF report) ---
    "Analysis report (PDF)": "Informe de análisis (PDF)",
    "A readable report: decisions, validation, key results, interpretation and parameters.":
        "Un informe legible: decisiones, validación, resultados clave, interpretación y parámetros.",
    "Download report (.pdf)": "Descargar informe (.pdf)",
    "Could not build the PDF report.": "No se pudo generar el informe PDF.",
    # --- PDF report contents ---
    "AEDA-AI analysis report": "Informe de análisis AEDA-AI",
    "Automated exploratory data analysis for environmental data":
        "Análisis exploratorio automatizado de datos ambientales",
    "Generated": "Generado",
    "Depth column": "Columna de profundidad",
    "yes": "sí",
    "Analysis decisions": "Decisiones del análisis",
    "For each step, the method the system chose and the reasoning behind it.":
        "Para cada paso, el método que el sistema eligió y el razonamiento detrás.",
    "Warnings": "Advertencias",
    "Data validation": "Validación de datos",
    "Severity": "Severidad",
    "Column": "Columna",
    "Detail": "Detalle",
    "Key results": "Resultados clave",
    "cumulative variance": "varianza acumulada",
    "Significant correlations": "Correlaciones significativas",
    "{s} strong, {m} moderate": "{s} fuertes, {m} moderadas",
    "Toxicological classification (NOAA TEL/PEL)": "Clasificación toxicológica (NOAA TEL/PEL)",
    "Enrichment factor (EF) — mean per metal": "Factor de enriquecimiento (EF) — media por metal",
    "Mean EF": "EF medio",
    "Methodology and parameters": "Metodología y parámetros",
    "Effective settings used in this run (for reproducibility).":
        "Configuración efectiva usada en esta ejecución (para reproducibilidad).",
    "Report generated by AEDA-AI. Figures are available in the application.":
        "Informe generado por AEDA-AI. Las figuras están disponibles en la aplicación.",
})

ES.update({
    # --- PDF report figures (Phase 2) ---
    "Key figures": "Figuras clave",
    "PCA biplot (PC1 vs PC2)": "Biplot de PCA (PC1 vs PC2)",
    "Correlation matrix (Pearson)": "Matriz de correlación (Pearson)",
    "Enrichment factor (EF) per metal": "Factor de enriquecimiento (EF) por metal",
})

LANGUAGES = {"es": "Español", "en": "English"}


def get_lang() -> str:
    lang = st.session_state.get("lang", DEFAULT_LANG)
    # Keep the engine layer's language in sync with the UI so engine-generated
    # messages (recommendation reasons, warnings, validation) match the UI.
    try:
        from aeda.i18n import set_lang as _engine_set_lang
        _engine_set_lang(lang)
    except Exception:
        pass
    return lang


def t(s: str) -> str:
    """Translate a display string. English is the key; falls back to English."""
    if get_lang() == "en":
        return s
    return ES.get(s, s)


def language_toggle() -> None:
    """Minimal ES · EN switch, right-aligned at the top of the page.

    The active language is shown in bold (plain text); the other is a small
    button. Lives in the main area (not the sidebar) so it reads as a subtle
    top-right corner control.
    """
    cur = st.session_state.get("lang", DEFAULT_LANG)
    _, c_es, c_sep, c_en = st.columns([24, 1, 1, 1])
    with c_es:
        if cur == "es":
            st.markdown("**ES**")
        elif st.button("ES", key="lang_es"):
            st.session_state.lang = "es"
            st.rerun()
    with c_sep:
        st.markdown("·")
    with c_en:
        if cur == "en":
            st.markdown("**EN**")
        elif st.button("EN", key="lang_en"):
            st.session_state.lang = "en"
            st.rerun()
