"""
Page 1: Upload & Configure

The user uploads an Excel/CSV file, selects which columns to exclude from
analysis (IDs, codes, coordinates), configures basic options, and runs
the pipeline. Results are stored in session_state for the other pages.

The page is structured as a numbered 1→5 workflow so the user always knows
where they are in the process.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import tempfile
import time

from app.i18n import t


def render():
    from app.components.page_header import page_header

    page_header(
        title=t("Upload & Configure"),
        description=t("Upload your environmental dataset and run the analysis with one click."),
        icon=":material/upload:",
    )

    # ---- Step 1: File upload ----
    st.subheader(t("1. File"))
    uploaded_file = st.file_uploader(
        t("Select an Excel or CSV file"),
        type=["xlsx", "xls", "csv"],
        label_visibility="collapsed",
        help=t("The file should contain environmental measurements with samples as rows and variables as columns."),
    )

    if uploaded_file is None:
        st.info(t("Upload a file to begin."))
        _show_example()
        return

    # Save to temp file for the pipeline
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # ---- Step 2: Sheet selection (Excel only) ----
    st.subheader(t("2. Sheet"))
    sheet_name = None
    if suffix in (".xlsx", ".xls"):
        xls = pd.ExcelFile(tmp_path)
        if len(xls.sheet_names) > 1:
            sheet_name = st.selectbox(
                t("Select sheet"),
                options=xls.sheet_names,
                label_visibility="collapsed",
                help=t("Choose the sheet containing your measurement data."),
            )
        else:
            sheet_name = xls.sheet_names[0]
            st.caption(t("Only sheet available: **{sheet}**").format(sheet=sheet_name))
    else:
        st.caption(t("CSV file selected — sheet selection not needed."))

    # ---- Preview data ----
    try:
        if suffix == ".csv":
            preview_df = pd.read_csv(tmp_path, nrows=10)
        else:
            preview_df = pd.read_excel(tmp_path, sheet_name=sheet_name, nrows=10)
    except Exception as e:
        from app.components.errors import show_error
        show_error(
            t("Could not read the uploaded file. Make sure it is a valid Excel/CSV format."),
            exc=e,
        )
        return

    with st.expander(t("Data preview ({n} columns, first 10 rows)").format(n=preview_df.shape[1]), expanded=False):
        st.dataframe(preview_df, use_container_width=True, height=300)

    # ---- Step 3: Column exclusion ----
    st.subheader(t("3. Columns to analyze"))
    st.caption(
        t(
            "By default, non-numeric columns (identifiers, codes, dates, sites) "
            "are excluded. Adjust below if needed."
        )
    )

    # Auto-detect likely non-measurement columns
    all_cols = preview_df.columns.tolist()
    non_numeric = preview_df.select_dtypes(exclude="number").columns.tolist()

    # Common metadata column names across Spanish and English datasets.
    # Note: short names like "X" and "Y" are ambiguous (Y is also the symbol
    # for Yttrium in geochemistry). We only flag them as metadata when there
    # is evidence that the dataset uses them as coordinates — specifically
    # when BOTH X and Y appear together, or when explicit UTM variants exist.
    METADATA_COLUMN_NAMES = {
        # Explicit coordinate names (unambiguous)
        "Latitud", "Longitud", "Latitude", "Longitude", "Lat", "Lon", "Lng",
        "X_UTM", "Y_UTM", "UTM_X", "UTM_Y",
        "Easting", "Northing", "Coord_X", "Coord_Y",
        # Row numbers / sample IDs that may be numeric
        "No", "N", "ID", "Id", "Sample_ID", "SampleID", "Sample_No", "Sample",
        "Order", "Row", "Index",
        # Sampling depth (structural: used for depth/surface analysis, not as a variable)
        "Profundidad", "Depth", "depth", "Prof", "Profundidad_cm", "Depth_cm",
    }
    numeric_metadata = [c for c in all_cols if c in METADATA_COLUMN_NAMES]

    # Ambiguous coordinate names: X and Y. Only treat them as metadata when
    # they appear *together* (typical of UTM coord pairs). This preserves
    # the ability to analyze a standalone column named "Y" (e.g., Yttrium)
    # when it isn't paired with an X coordinate.
    if "X" in all_cols and "Y" in all_cols:
        numeric_metadata.extend([c for c in ("X", "Y") if c in all_cols])

    suggested_exclude = sorted(set(non_numeric) | set(numeric_metadata))

    exclude_cols = st.multiselect(
        t("Exclude these columns from the ML analysis"),
        options=all_cols,
        default=suggested_exclude,
        label_visibility="collapsed",
        help=t("These columns will be ignored during the ML analysis. Coordinates and depth are excluded from ML but used for metadata."),
    )

    # ---- Step 4: Analysis options ----
    st.subheader(t("4. Analysis options"))
    st.caption(
        t(
            "Sensible defaults work for most environmental datasets — for fine-grained "
            "control, use the Advanced Configuration page after the first run."
        )
    )
    col1, col2, col3 = st.columns(3)

    with col1:
        # NOTE: option values stay in English (passed to the pipeline); shown
        # translated via format_func.
        impute = st.selectbox(
            t("Missing values strategy"),
            options=["median", "mean", "knn", "drop_rows"],
            index=0,
            format_func=lambda o: t(o),
            label_visibility="collapsed",
            help=t("How to fill in or remove missing values."),
        )
        st.caption(
            t(
                "Replaces empty cells with a plausible value so ML algorithms can "
                "process the data. **Median** is robust against extreme values."
            )
        )

    with col2:
        dim_method = st.selectbox(
            t("Dimensionality reduction"),
            options=["pca", "auto"],
            index=0,
            format_func=lambda o: t(o),
            label_visibility="collapsed",
            help=t("Method used to compress the dataset into a smaller number of components."),
        )
        st.caption(
            t(
                "Compresses many variables into a few summary axes (components) "
                "that capture the main patterns. **PCA** is the standard choice "
                "for environmental data."
            )
        )

    with col3:
        cluster_method = st.selectbox(
            t("Clustering method"),
            options=["auto", "kmeans", "dbscan", "hierarchical"],
            index=0,
            format_func=lambda o: t(o),
            label_visibility="collapsed",
            help=t("Algorithm used to group similar samples."),
        )
        st.caption(
            t(
                "Groups samples with similar chemistry. **Auto** tries K-Means "
                "and DBSCAN and keeps the best one according to a quality score."
            )
        )

    # ---- Step 5: Run pipeline ----
    st.subheader(t("5. Run"))
    if st.button(t("Run analysis"), type="primary", use_container_width=True):
        _run_pipeline(tmp_path, sheet_name, exclude_cols, impute, dim_method, cluster_method, uploaded_file.name)


def _run_pipeline(filepath, sheet_name, exclude_cols, impute, dim_method, cluster_method, filename):
    """Execute the AEDA pipeline with a progress bar."""
    from aeda.pipeline.runner import AEDAPipeline

    progress = st.progress(0, text=t("Loading data..."))

    try:
        # Settings used in this run — persisted so the Advanced Configuration
        # page can pre-fill its controls with the values that were actually used.
        settings = {
            "scale_method": "auto",
            "impute_strategy": impute,
            "dim_method": dim_method,
            "clustering_method": cluster_method,
            "anomaly_method": "auto",
            "correlation_method": "compare",
            "apply_clr": False,
            "contamination": 0.05,
            "run_interpretation": True,
            "reference_element": "Al",
            "baseline_strategy": "deepest",
            "custom_baseline": None,
            "dim_kwargs": {},
            "clustering_kwargs": {},
            "anomaly_kwargs": {},
        }

        pipeline = AEDAPipeline(
            impute_strategy=impute,
            dim_method=dim_method,
            clustering_method=cluster_method,
            apply_clr=False,
        )

        progress.progress(10, text=t("Loading and validating data..."))
        time.sleep(0.3)

        progress.progress(20, text=t("Running auto-selector..."))
        time.sleep(0.2)

        progress.progress(30, text=t("Preprocessing..."))
        results = pipeline.run(
            filepath,
            exclude_cols=exclude_cols,
            sheet_name=sheet_name,
        )

        progress.progress(80, text=t("Generating results..."))
        time.sleep(0.3)

        # Store in session state
        st.session_state.results = results
        st.session_state.raw_df = results.raw_data
        st.session_state.filename = filename
        st.session_state.run_context = {
            "tmp_path": filepath,
            "sheet_name": sheet_name,
            "exclude_cols": exclude_cols,
            # Persist the *effective* settings (what the auto-selector actually
            # resolved and applied), not the raw form inputs. This is what the
            # Advanced page reads to pre-fill its controls.
            "settings": results.effective_settings or settings,
        }

        progress.progress(100, text=t("Done!"))
        time.sleep(0.5)
        progress.empty()

        st.success(t("Analysis complete! Navigate to the other pages to see results."))

        # Quick summary
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(t("Samples"), results.raw_data.shape[0])
        col2.metric(t("Variables analyzed"), results.processed_data.shape[1] if results.processed_data is not None else "—")
        if results.dim_reduction:
            col3.metric(t("PCA components"), results.dim_reduction.n_components_selected)
        if results.clustering:
            col4.metric(t("Clusters"), results.clustering.n_clusters)

    except Exception as e:
        progress.empty()
        from app.components.errors import show_error
        show_error(
            t("The pipeline could not complete. The dataset may have an unexpected format or missing required columns."),
            exc=e,
        )


def _show_example():
    """Show usage instructions when no file is uploaded."""
    st.divider()
    st.subheader(t("How it works"))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**{t('1. Upload')}**")
        st.write(t("Upload an Excel or CSV with your environmental measurements."))
    with col2:
        st.markdown(f"**{t('2. Configure')}**")
        st.write(t("Select which columns to exclude and choose analysis options."))
    with col3:
        st.markdown(f"**{t('3. Explore')}**")
        st.write(t("Browse interactive plots: PCA biplot, clusters, correlations, depth profiles."))

    st.divider()
    st.caption(t("Supported formats: .xlsx, .xls, .csv — Datasets tested with FRX geochemistry, granulometry, and sediment data."))
