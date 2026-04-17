"""
Page 1: Upload & Configure

The user uploads an Excel/CSV file, selects which columns to exclude from
analysis (IDs, codes, coordinates), configures basic options, and runs
the pipeline. Results are stored in session_state for the other pages.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import tempfile
import time


def render():
    st.header("Upload & Configure")
    st.write("Upload your environmental dataset and configure the analysis.")

    # ---- File upload ----
    uploaded_file = st.file_uploader(
        "Select an Excel or CSV file",
        type=["xlsx", "xls", "csv"],
        help="The file should contain environmental measurements with samples as rows and variables as columns.",
    )

    if uploaded_file is None:
        st.info("Upload a file to begin.")
        _show_example()
        return

    # Save to temp file for the pipeline
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # ---- Sheet selection (Excel only) ----
    sheet_name = None
    if suffix in (".xlsx", ".xls"):
        xls = pd.ExcelFile(tmp_path)
        if len(xls.sheet_names) > 1:
            sheet_name = st.selectbox(
                "Select sheet",
                options=xls.sheet_names,
                help="Choose the sheet containing your measurement data.",
            )
        else:
            sheet_name = xls.sheet_names[0]
            st.caption(f"Sheet: **{sheet_name}**")

    # ---- Preview data ----
    try:
        if suffix == ".csv":
            preview_df = pd.read_csv(tmp_path, nrows=10)
        else:
            preview_df = pd.read_excel(tmp_path, sheet_name=sheet_name, nrows=10)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return

    st.subheader("Data preview")
    st.dataframe(preview_df, use_container_width=True, height=300)
    st.caption(f"{preview_df.shape[1]} columns detected")

    # ---- Column exclusion ----
    st.subheader("Columns to exclude from analysis")
    st.write("Select columns that are identifiers, codes, dates, or metadata — not measurements.")

    # Auto-detect likely non-measurement columns
    all_cols = preview_df.columns.tolist()
    non_numeric = preview_df.select_dtypes(exclude="number").columns.tolist()
    suggested_exclude = non_numeric

    exclude_cols = st.multiselect(
        "Exclude these columns",
        options=all_cols,
        default=suggested_exclude,
        help="These columns will be ignored during the ML analysis. Coordinates and depth are excluded from ML but used for metadata.",
    )

    # ---- Analysis options ----
    st.subheader("Analysis options")
    col1, col2, col3 = st.columns(3)

    with col1:
        impute = st.selectbox(
            "Missing values strategy",
            options=["median", "mean", "knn", "drop_rows"],
            index=0,
            help="How to handle remaining missing values after filtering.",
        )

    with col2:
        dim_method = st.selectbox(
            "Dimensionality reduction",
            options=["pca", "auto"],
            index=0,
            help="PCA is recommended for most environmental datasets.",
        )

    with col3:
        cluster_method = st.selectbox(
            "Clustering method",
            options=["auto", "kmeans", "dbscan", "hierarchical"],
            index=0,
            help="'auto' tries K-Means and DBSCAN, picks the best.",
        )

    # ---- Run pipeline ----
    st.divider()

    if st.button("Run analysis", type="primary", use_container_width=True):
        _run_pipeline(tmp_path, sheet_name, exclude_cols, impute, dim_method, cluster_method, uploaded_file.name)


def _run_pipeline(filepath, sheet_name, exclude_cols, impute, dim_method, cluster_method, filename):
    """Execute the AEDA pipeline with a progress bar."""
    from aeda.pipeline.runner import AEDAPipeline

    progress = st.progress(0, text="Loading data...")

    try:
        pipeline = AEDAPipeline(
            impute_strategy=impute,
            dim_method=dim_method,
            clustering_method=cluster_method,
        )

        progress.progress(10, text="Loading and validating data...")
        time.sleep(0.3)

        progress.progress(20, text="Running auto-selector...")
        time.sleep(0.2)

        progress.progress(30, text="Preprocessing...")
        results = pipeline.run(
            filepath,
            exclude_cols=exclude_cols,
            sheet_name=sheet_name,
        )

        progress.progress(80, text="Generating results...")
        time.sleep(0.3)

        # Store in session state
        st.session_state.results = results
        st.session_state.raw_df = results.raw_data
        st.session_state.filename = filename

        progress.progress(100, text="Done!")
        time.sleep(0.5)
        progress.empty()

        st.success("Analysis complete! Navigate to the other pages to see results.")

        # Quick summary
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Samples", results.raw_data.shape[0])
        col2.metric("Variables analyzed", results.processed_data.shape[1] if results.processed_data is not None else "—")
        if results.dim_reduction:
            col3.metric("PCA components", results.dim_reduction.n_components_selected)
        if results.clustering:
            col4.metric("Clusters", results.clustering.n_clusters)

    except Exception as e:
        progress.empty()
        st.error(f"Pipeline failed: {e}")
        st.exception(e)


def _show_example():
    """Show usage instructions when no file is uploaded."""
    st.divider()
    st.subheader("How it works")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**1. Upload**")
        st.write("Upload an Excel or CSV with your environmental measurements.")
    with col2:
        st.markdown("**2. Configure**")
        st.write("Select which columns to exclude and choose analysis options.")
    with col3:
        st.markdown("**3. Explore**")
        st.write("Browse interactive plots: PCA biplot, clusters, correlations, depth profiles.")

    st.divider()
    st.caption("Supported formats: .xlsx, .xls, .csv — Datasets tested with FRX geochemistry, granulometry, and sediment data.")
